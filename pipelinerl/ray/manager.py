from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Type

from pipelinerl.ray.backend import RayExecutionBackend, RolloutExecutionBackend, SyncExecutionBackend
from pipelinerl.ray.ray_actor import RolloutActorImpl
from pipelinerl.ray.worker import RolloutWorker
from pipelinerl.rollouts import RolloutRequest, RolloutResult

logger = logging.getLogger(__name__)


@dataclass
class CompletedRollout:
    request: RolloutRequest
    result: RolloutResult
    worker_index: int


@dataclass
class _WorkerSlot:
    index: int
    actor: Any
    active_ref: Any | None = None
    active_request: RolloutRequest | None = None
    retiring: bool = False
    ready: bool = False

    @property
    def idle(self) -> bool:
        return self.active_ref is None




class RolloutExecutionError(RuntimeError):
    def __init__(self, request: RolloutRequest, worker_index: int, cause: BaseException):
        super().__init__(f"rollout worker {worker_index} failed request {request.request_id}: {cause}")
        self.request = request
        self.worker_index = worker_index
        self.__cause__ = cause


class RayRolloutManager:
    def __init__(
        self,
        *,
        worker_cls: Type[RolloutWorker],
        worker_config: dict[str, Any],
        num_workers: int,
        ray_options: dict[str, Any] | None = None,
        execution_backend: str = "ray",
        log_collector: Any | None = None,
        context_extras: dict[str, Any] | None = None,
        worker_name_prefix: str = "ray_rollout_worker",
        max_pending: int | None = None,
    ) -> None:
        if num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        self.worker_cls = worker_cls
        self.worker_config = worker_config
        self.ray_options = ray_options or {}
        self.backend = self._make_backend(execution_backend)
        self.log_collector = log_collector
        self.context_extras = context_extras or {}
        self.worker_name_prefix = worker_name_prefix
        self.max_pending = max_pending
        self._slots: list[_WorkerSlot] = []
        self._next_worker_index = 0
        self._target_workers = 0
        self.set_target_workers(num_workers)

    def _make_backend(self, execution_backend: str) -> RolloutExecutionBackend:
        backend_name = str(execution_backend).strip().lower()
        if backend_name == "ray":
            return RayExecutionBackend()
        if backend_name == "sync":
            return SyncExecutionBackend()
        raise ValueError(f"Unknown rollout execution backend: {execution_backend}")

    @property
    def target_workers(self) -> int:
        return self._target_workers

    @property
    def active_count(self) -> int:
        return sum(1 for slot in self._slots if not slot.idle)

    @property
    def idle_count(self) -> int:
        return sum(1 for slot in self._slots if slot.idle and not slot.retiring)

    @property
    def worker_count(self) -> int:
        return len(self._slots)

    @property
    def pending_refs(self) -> list[Any]:
        return [slot.active_ref for slot in self._slots if slot.active_ref is not None]

    def set_submission_cap(self, max_pending: int | None) -> None:
        self.max_pending = max_pending

    def set_target_workers(self, num_workers: int) -> None:
        if num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        self._target_workers = num_workers
        self._start_all_worker()
        self._retire_excess_idle_workers()

    def _start_one_worker(self, lazy: bool = False) -> None:
        worker_index = self._next_worker_index
        self._next_worker_index += 1
        worker_name = f"{self.worker_name_prefix}_{worker_index}"
        config = copy.deepcopy(self.worker_config)
        config.setdefault("worker_id", worker_index)
        config.setdefault("worker_name", worker_name)

        actor = self._make_actor(config, worker_index, worker_name)
        slot = _WorkerSlot(index=worker_index, actor=actor)
        self._slots.append(slot)
        if lazy:
            return
        self._setup_slot(slot)
        logger.info("Started Ray rollout worker %s", slot.index)

    def _start_all_worker(self) -> None:
        logger.info("Creating Ray rollout ...")
        while len(self._slots) < self._target_workers:
            self._start_one_worker(lazy=True)
        self._setup_slots([slot for slot in self._slots if not slot.ready])
        logger.info(f"All {len(self._slots)} Ray rollout workers ready")

    def _setup_slot(self, slot: _WorkerSlot) -> None:
        self._setup_slots([slot])

    def _setup_slots(self, slots: list[_WorkerSlot]) -> None:
        if not slots:
            return
        results = self.backend.get([slot.actor.setup.remote() for slot in slots])
        for slot, ok in zip(slots, results):
            if not ok:
                raise RuntimeError(f"Ray rollout worker {slot.index} setup failed")
            slot.ready = True

    def _retire_excess_idle_workers(self) -> None:
        excess = len(self._slots) - self._target_workers
        if excess <= 0:
            return
        for slot in list(reversed(self._slots)):
            if excess <= 0:
                break
            if slot.idle:
                self._close_slot(slot)
                self._slots.remove(slot)
                excess -= 1
        if excess > 0:
            for slot in list(reversed(self._slots)):
                if excess <= 0:
                    break
                if not slot.retiring:
                    slot.retiring = True
                    excess -= 1

    def _close_slot(self, slot: _WorkerSlot) -> None:
        self.backend.close_actor(slot.actor, logger, f"Ray rollout worker {slot.index}")
        self.backend.kill_actor(slot.actor, logger, f"Ray rollout worker {slot.index}")

    def _select_idle_slot(self) -> _WorkerSlot | None:
        candidates = [slot for slot in self._slots if slot.ready and slot.idle and not slot.retiring]
        if not candidates:
            return None
        return min(candidates, key=lambda slot: slot.index)

    def try_submit(self, request: RolloutRequest) -> bool:
        if self.max_pending is not None and self.active_count >= self.max_pending:
            return False
        if len(self._slots) < self._target_workers:
            self._start_one_worker()
        slot = self._select_idle_slot()
        if slot is None:
            return False
        ref = slot.actor.generate.remote(request)
        slot.active_ref = ref
        slot.active_request = request
        return True

    def wait_completed(self, *, timeout_s: float = 0.01, num_returns: int = 1) -> list[CompletedRollout]:
        refs = self.pending_refs
        if not refs:
            return []
        done_refs, _ = self.backend.wait(refs, num_returns=min(num_returns, len(refs)), timeout=timeout_s)
        completed: list[CompletedRollout] = []
        for ref in done_refs:
            slot = self._slot_for_ref(ref)
            request = slot.active_request
            if request is None:
                raise RuntimeError("completed Ray ref has no active request")
            try:
                payload = self.backend.get(ref)
                result = RolloutResult.model_validate(payload)
                completed.append(CompletedRollout(request=request, result=result, worker_index=slot.index))
            except Exception as exc:
                slot.active_ref = None
                slot.active_request = None
                if slot.retiring:
                    self._close_slot(slot)
                    self._slots.remove(slot)
                self._retire_excess_idle_workers()
                raise RolloutExecutionError(request, slot.index, exc) from exc
            slot.active_ref = None
            slot.active_request = None
            if slot.retiring:
                self._close_slot(slot)
                self._slots.remove(slot)
        self._retire_excess_idle_workers()
        return completed

    def _make_actor(
        self,
        config,
        worker_index,
        worker_name,
    ):
        args = (
            self.worker_cls,
            config,
            worker_index,
            worker_name,
            self.context_extras,
            self.log_collector,
        )

        return self.backend.create_actor(RolloutActorImpl, args, self.ray_options)

    def _slot_for_ref(self, ref: Any) -> _WorkerSlot:
        for slot in self._slots:
            if slot.active_ref == ref:
                return slot
        raise KeyError("unknown Ray object ref")

    def health(self) -> list[bool]:
        return self.backend.get([slot.actor.health.remote() for slot in self._slots])

    def close(self) -> None:
        for slot in list(self._slots):
            self._close_slot(slot)
        self._slots = []
