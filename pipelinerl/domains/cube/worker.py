from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from cube_harness.rl.event_publisher import EventPublisher

from pipelinerl.domains.cube.result_builder import (
    apply_reward_shaping,
    build_payload,
    rollout_result_from_events,
    run_rollout,
)
from pipelinerl.domains.cube.reward_shaping import RewardShapingConfig
from pipelinerl.ray.worker import RolloutWorker, RolloutWorkerContext
from pipelinerl.rollouts import RolloutRequest, RolloutResult

logger = logging.getLogger(__name__)


class CubeRolloutWorker(RolloutWorker):
    """Runs one cube-harness episode per request via `RolloutTaskRunner`."""

    def __init__(self, worker_config: dict[str, Any]):
        super().__init__(worker_config)
        actor_cfg = worker_config.get("actor", {})
        self._buffer_tokens = int(actor_cfg.get("buffer_tokens", 0))
        self._discount_factor = float(actor_cfg.get("discount_factor", 1.0))
        self._debug_env_response = bool(actor_cfg.get("debug_env_response", False))
        self._reward_shaping_config = RewardShapingConfig.from_mapping(
            worker_config.get("reward_shaping")
        )
        self.output_dir = Path(worker_config.get("output_dir"))

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

        payload = build_payload(
            request=request,
            item=item,
            llm=request.extras.get("llm"),
            output_dir=self.output_dir,
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
            task_id=task_id,
            reward_shaping_config=self._reward_shaping_config,
            debug_env_response=self._debug_env_response,
        )
        apply_reward_shaping(
            result.training_texts,
            agent_config=item["agent_cfg"],
            buffer_tokens=self._buffer_tokens,
            discount_factor=self._discount_factor,
            reward_shaping_config=self._reward_shaping_config,
        )
        return result

    def close(self) -> None:
        return None
