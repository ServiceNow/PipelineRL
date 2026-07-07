from __future__ import annotations

import copy
import logging
import time
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
    active_start_time: float | None = None
    setup_ref: Any | None = None
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
        rollout_wall_timeout_s: float | None = None,
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
        self.rollout_wall_timeout_s = (
            float(rollout_wall_timeout_s) if rollout_wall_timeout_s and rollout_wall_timeout_s > 0 else None
        )
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

    def _start_one_worker(self, lazy: bool = False) -> _WorkerSlot:
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
            return slot
        self._setup_slot(slot)
        logger.info("Started Ray rollout worker %s", slot.index)
        return slot

    def _start_all_worker(self) -> None:
        logger.info("Creating Ray rollout ...")
        while len(self._slots) < self._target_workers:
            self._start_one_worker(lazy=True)
        self._setup_slots([slot for slot in self._slots if not slot.ready])
        logger.info(f"All {len(self._slots)} Ray rollout workers ready")

    def _setup_slot(self, slot: _WorkerSlot) -> None:
        self._setup_slots([slot])

    def _setup_slots(self, slots: list[_WorkerSlot]) -> None:
        """Fire setup.remote() for the given slots and block until all finish."""
        if not slots:
            return
        for slot in slots:
            if slot.setup_ref is None and not slot.ready:
                slot.setup_ref = slot.actor.setup.remote()
        pending = [s for s in slots if s.setup_ref is not None and not s.ready]
        if not pending:
            return
        results = self.backend.get([s.setup_ref for s in pending])
        for slot, ok in zip(pending, results):
            slot.setup_ref = None
            if not ok:
                raise RuntimeError(f"Ray rollout worker {slot.index} setup failed")
            slot.ready = True

    def _spawn_missing_workers_async(self) -> None:
        """Create new actors for every missing slot and fire setup.remote() without blocking.

        Callers should later invoke `_reap_pending_setups()` (non-blocking) or
        `_setup_slots(...)` (blocking) to promote the slot to ready.
        """
        while len(self._slots) < self._target_workers:
            slot = self._start_one_worker(lazy=True)
            slot.setup_ref = slot.actor.setup.remote()
            logger.info("Spawning Ray rollout worker %s (setup in flight)", slot.index)

    def _reap_pending_setups(self) -> None:
        """Non-blocking: mark slots ready whose setup.remote() has resolved."""
        pending = [s for s in self._slots if s.setup_ref is not None and not s.ready]
        if not pending:
            return
        refs = [s.setup_ref for s in pending]
        done_refs, _ = self.backend.wait(refs, num_returns=len(refs), timeout=0)
        if not done_refs:
            return
        ref_to_slot = {ref: slot for ref, slot in zip(refs, pending)}
        for ref in done_refs:
            slot = ref_to_slot.get(ref)
            if slot is None:
                continue
            try:
                ok = self.backend.get(ref)
            except Exception:
                logger.exception("Ray rollout worker %s setup failed; will be respawned", slot.index)
                slot.setup_ref = None
                if slot in self._slots:
                    self._slots.remove(slot)
                continue
            if not ok:
                logger.error("Ray rollout worker %s setup returned False; will be respawned", slot.index)
                slot.setup_ref = None
                if slot in self._slots:
                    self._slots.remove(slot)
                continue
            slot.setup_ref = None
            slot.ready = True
            logger.info("Ray rollout worker %s ready", slot.index)

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
        self._reap_pending_setups()
        if len(self._slots) < self._target_workers:
            self._spawn_missing_workers_async()
        slot = self._select_idle_slot()
        if slot is None:
            return False
        ref = slot.actor.generate.remote(request)
        slot.active_ref = ref
        slot.active_request = request
        slot.active_start_time = time.monotonic()
        return True

    def wait_completed(self, *, timeout_s: float = 0.01, num_returns: int = 1) -> list[CompletedRollout]:
        self._reap_pending_setups()
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
                slot.active_start_time = None
                if slot.retiring:
                    self._close_slot(slot)
                    self._slots.remove(slot)
                self._retire_excess_idle_workers()
                raise RolloutExecutionError(request, slot.index, exc) from exc
            slot.active_ref = None
            slot.active_request = None
            slot.active_start_time = None
            if slot.retiring:
                self._close_slot(slot)
                self._slots.remove(slot)
        self._retire_excess_idle_workers()
        if not completed:
            self._enforce_wall_timeout()
        return completed

    def _enforce_wall_timeout(self) -> None:
        """Kill any worker whose in-flight rollout has exceeded the wall-clock timeout.

        Raises RolloutExecutionError so the actor loop's existing retry path handles it.
        Only the first offending slot is raised for; remaining offenders (if any) will
        be caught on subsequent calls.
        """
        if self.rollout_wall_timeout_s is None:
            return
        now = time.monotonic()
        for slot in list(self._slots):
            if slot.active_ref is None or slot.active_start_time is None:
                continue
            elapsed = now - slot.active_start_time
            if elapsed <= self.rollout_wall_timeout_s:
                continue
            request = slot.active_request
            worker_index = slot.index
            request_id = request.request_id if request is not None else None
            logger.warning(
                "Rollout worker %s exceeded wall-clock timeout of %.1fs (elapsed %.1fs); killing worker (request_id=%s)",
                worker_index,
                self.rollout_wall_timeout_s,
                elapsed,
                request_id,
            )
            self._cancel_and_drop_slot(slot)
            self._retire_excess_idle_workers()
            # Kick off replacement setup immediately so its boot time overlaps
            # with the actor loop's retry plumbing rather than being paid
            # sequentially inside a later try_submit call.
            self._spawn_missing_workers_async()
            if request is None:
                return
            cause = TimeoutError(
                f"rollout exceeded wall-clock timeout of {self.rollout_wall_timeout_s:.1f}s "
                f"(elapsed {elapsed:.1f}s)"
            )
            raise RolloutExecutionError(request, worker_index, cause)

    def _cancel_and_drop_slot(self, slot: _WorkerSlot) -> None:
        ref = slot.active_ref
        if ref is not None:
            try:
                import ray as _ray
                _ray.cancel(ref, force=True, recursive=True)
            except Exception:
                logger.debug("ray.cancel failed for worker %s", slot.index, exc_info=True)
        slot.active_ref = None
        slot.active_request = None
        slot.active_start_time = None
        self.backend.kill_actor(slot.actor, logger, f"Ray rollout worker {slot.index} (timeout)")
        if slot in self._slots:
            self._slots.remove(slot)

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
