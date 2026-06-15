from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from cube_harness.rl.event_publisher import EventPublisher

from pipelinerl.domains.cube.result_builder import (
    apply_rollout_rewards,
    build_payload,
    rollout_result_from_events,
    run_rollout,
    write_rollout_artifact,
)
from pipelinerl.ray.worker import RolloutWorker, RolloutWorkerContext
from pipelinerl.rollouts import RolloutRequest, RolloutResult

logger = logging.getLogger(__name__)


class CubeRolloutWorker(RolloutWorker):
    """Runs one cube-harness episode per request via `RolloutTaskRunner`."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        actor_cfg = config.get("actor", {})
        self._buffer_tokens = int(actor_cfg.get("buffer_tokens", 0))
        self._discount_factor = float(actor_cfg.get("discount_factor", 1.0))
        artifact_cfg = config.get("rollout_artifacts") or {}
        self._persist_rollout_artifacts = bool(artifact_cfg.get("enabled", False))
        artifact_dir = artifact_cfg.get("path")
        self._rollout_artifact_dir = (
            Path(artifact_dir) if artifact_dir else Path(config["output_dir"]) / "actor" / "rollout_artifacts"
        )

    def setup(self, context: RolloutWorkerContext) -> None:
        try:
            super().setup(context)
            logger.info("%s ready", self.worker_name)
        except Exception:
            self.ready = False
            logger.exception("%s failed during setup", self.worker_name)
            raise

    def generate(self, request: RolloutRequest) -> RolloutResult:
        item = request.dataset_item
        task_id = str(item["task_id"])
        domain = item.get("domain")
        if not self.ready:
            raise RuntimeError(f"{self.worker_name} not ready")

        rollout_dir = self._rollout_artifact_dir / request.request_id
        payload = build_payload(
            request=request,
            item=item,
            llm=request.extras.get("llm"),
            output_dir=rollout_dir,
        )

        logger.info("%s starting rollout for domain=%s task_id=%s", self.worker_name, domain, task_id)
        publisher = EventPublisher()
        start = time.perf_counter()
        try:
            run_rollout(payload, publisher)
        except Exception:
            logger.exception("%s rollout failed for domain=%s task_id=%s", self.worker_name, domain, task_id)
            raise

        events = publisher.events_from(0)
        result = rollout_result_from_events(
            events,
            latency=time.perf_counter() - start,
            dataset=item.get("dataset"),
            domain=domain,
        )
        apply_rollout_rewards(
            result.training_texts,
            agent_config=item["agent_cfg"],
            buffer_tokens=self._buffer_tokens,
            discount_factor=self._discount_factor,
        )
        if self._persist_rollout_artifacts:
            write_rollout_artifact(
                events=events,
                artifact_dir=self._rollout_artifact_dir,
                worker_name=self.worker_name,
                trajectory_id=request.request_id,
                task_id=task_id,
                domain=domain,
                dataset=item.get("dataset"),
            )
        return result

    def close(self) -> None:
        return None
