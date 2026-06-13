from __future__ import annotations

from typing import Type

from pipelinerl.ray.logging import (
    configure_ray_worker_logging,
    reset_worker_rollout_log_context,
    start_worker_rollout_log_context,
)
from pipelinerl.ray.worker import RolloutWorker, RolloutWorkerContext
from pipelinerl.rollouts import RolloutRequest


class RolloutActorImpl:
    def __init__(
        self,
        worker_cls: Type[RolloutWorker],
        config: dict,
        worker_id: int,
        worker_name: str | None = None,
        context_extras: dict | None = None,
        llm_router=None,
        log_collector=None,
    ):
        self.worker_id = worker_id
        self.worker_name = worker_name or f"ray_rollout_worker_{worker_id}"
        self.context_extras = context_extras or {}
        self.worker = worker_cls(config)
        actor_cfg = config.get("actor", {}) if isinstance(config, dict) else {}
        configure_ray_worker_logging(
            worker_name=self.worker_name,
            log_collector=log_collector,
            log_level=actor_cfg.get("ray_worker_log_level", self.context_extras.get("ray_worker_log_level", "ERROR")),
            litellm_log_level=actor_cfg.get(
                "ray_worker_litellm_log_level",
                self.context_extras.get("ray_worker_litellm_log_level", "WARNING"),
            ),
        )
        self.context = RolloutWorkerContext(
            worker_id=worker_id,
            worker_name=self.worker_name,
            llm_router=llm_router,
            extras=self.context_extras,
        )
        self.ready = False

    def setup(self) -> bool:
        self.worker.setup(self.context)
        self.ready = True
        return True

    def generate(self, request: RolloutRequest | dict) -> dict:
        if not self.ready:
            raise RuntimeError(f"{self.worker_name} has not been set up")
        if isinstance(request, dict):
            request = RolloutRequest.model_validate(request)
        tokens = start_worker_rollout_log_context(request)
        try:
            return self.worker.generate(request).model_dump()
        finally:
            reset_worker_rollout_log_context(tokens)

    def health(self) -> bool:
        if not self.ready:
            return False
        return bool(self.worker.health())

    def close(self) -> None:
        self.worker.close()
