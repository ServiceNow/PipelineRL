from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra

from pipelinerl.domains.cube.episode_runner import CubeEpisodeRunner
from pipelinerl.rollouts import RolloutRequest, RolloutResult
from pipelinerl.ray.worker import RolloutWorker, RolloutWorkerContext

logger = logging.getLogger(__name__)


def set_agent_llm_config(agent_config: Any, llm: dict) -> None:
    llm_config = getattr(agent_config, "llm_config")

    llm_config.api_base = llm["base_url"]
    if not llm_config.api_base.endswith("/v1"):
        llm_config.api_base += "/v1"

    llm_config.api_key = "EMPTY"
    llm_config.model_name = llm.get("served_model_name") or llm["model_name"]
    llm_config.tokenizer_name = llm.get("tokenizer_name", llm_config.model_name)
    if not llm_config.model_name.startswith("openai/"):
        llm_config.model_name = f"openai/{llm_config.model_name}"

    llm_config.logprobs = llm["collect_logprobs"]
    if llm_config.logprobs:
        llm_config.include_stop_str_in_output = True
        llm_config.skip_special_tokens = False

    llm_parameters = llm.get("parameters", {})
    for param_name, param_value in llm_parameters.items():
        if hasattr(llm_config, param_name):
            setattr(llm_config, param_name, param_value)
        else:
            logger.warning("Cube-harness Agent LLM parameters does not have attribute '%s', skipping", param_name)


class CubeRolloutWorker(RolloutWorker):
    """CUBE rollout worker with lazy benchmark materialization."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        if self.llm is None:
            raise ValueError("CubeRolloutWorker requires an llm config")
        self._current_cube_id: str | None = None
        self._benchmark = None
        self._agent = None
        self._runtime_context = None
        self._container_backend = None
        self._runner: CubeEpisodeRunner | None = None

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

    def _close_current_cube(self) -> None:
        if self._benchmark is not None:
            try:
                self._benchmark.close()
            except Exception as exc:
                logger.warning("%s failed to close cube %s: %s", self.worker_name, self._current_cube_id, exc)
        self._current_cube_id = None
        self._benchmark = None
        self._agent = None
        self._runtime_context = None
        self._container_backend = None
        self._runner = None

    def _lazy_cube_init(self, item: dict[str, Any]) -> None:
        cube_id = str(item["cube_id"])
        if self._current_cube_id == cube_id:
            return

        self._close_current_cube()
        benchmark_obj = hydra.utils.instantiate(item["benchmark_cfg"])
        benchmark_obj.install()
        benchmark_obj.setup()

        self._runtime_context = getattr(benchmark_obj, "_runtime_context", None)
        self._container_backend = getattr(benchmark_obj, "container_backend", None)
        agent_template = hydra.utils.instantiate(item["agent_cfg"])
        set_agent_llm_config(agent_template, self.llm)
        self._agent = agent_template

        self._current_cube_id = cube_id
        self._benchmark = benchmark_obj

        logger.info("%s prepared cube %s", self.worker_name, cube_id)

    def _episode_runner(self) -> CubeEpisodeRunner:
        if self.llm_tokenizer is None:
            raise RuntimeError(f"{self.worker_name} tokenizer is not loaded")
        if self.llm is None:
            raise RuntimeError(f"{self.worker_name} llm config is not set")
        if self._runner is None:
            self._runner = CubeEpisodeRunner(
                llm_tokenizer=self.llm_tokenizer,
                llm_config=self.llm,
                llm_router=self.llm_router,
                worker_name=self.worker_name,
                runtime_context=self._runtime_context,
                container_backend=self._container_backend,
                buffer_tokens=self._buffer_tokens,
                discount_factor=self._discount_factor,
                persist_rollout_artifacts=self._persist_rollout_artifacts,
                rollout_artifact_dir=self._rollout_artifact_dir,
            )
        return self._runner

    def generate(self, request: RolloutRequest) -> RolloutResult:
        item = request.dataset_item
        cube_id = str(item["cube_id"])
        task_id = str(item["task_id"])
        try:
            if not self.ready:
                raise RuntimeError(f"{self.worker_name} not ready")
            self._lazy_cube_init(item)
            if self._benchmark is None or self._agent is None:
                raise RuntimeError(f"{self.worker_name} cube is not initialized")

            logger.error("%s starting rollout for cube_id=%s task_id=%s", self.worker_name, cube_id, task_id)

            task = {
                "task_config": self._benchmark.get_task_config(task_id),
                "max_steps": request.max_steps,
                "domain": item.get("domain", None),
                "dataset": item.get("dataset", None),
            }

            return self._episode_runner().run_terminal(
                task=task,
                agent_config=self._agent,
                rollout_key=request.request_id,
            )
        except Exception:
            logger.exception("%s rollout failed for cube_id=%s task_id=%s", self.worker_name, cube_id, task_id)
            raise

    def close(self) -> None:
        self._close_current_cube()
