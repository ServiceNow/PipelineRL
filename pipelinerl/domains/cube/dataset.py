from __future__ import annotations

import logging
from typing import Any
from pathlib import Path
from omegaconf import OmegaConf
from pipelinerl.ray.worker import RolloutDataset
from cube_harness.episode import MAX_STEPS
from cube_harness.rl import RolloutConfig, RolloutEngine

logger = logging.getLogger(__name__)

class CubeRolloutDataset(RolloutDataset):
    def __init__(self, engine: RolloutEngine):
        self.engine = engine
        self.task_configs = engine.task_configs()

    def __len__(self) -> int:
        return len(self.task_configs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.task_configs[index]

def _as_plain_dict(value: Any) -> dict[str, Any]:
    resolved = OmegaConf.to_container(value, resolve=True)
    return resolved

def load_datasets(dataset_names: list[str], **loader_kwargs: Any) -> list[RolloutDataset] | None:
    if dataset_names is None:
        dataset_names = []

    for name in dataset_names:
        assert name in loader_kwargs, f"Missing config for dataset {name}"

    ds = []
    for name in dataset_names:
        dataset_params = loader_kwargs[name]
        benchmark_data = _as_plain_dict(dataset_params.get("benchmark"))
        agent_data = _as_plain_dict(dataset_params.get("agent"))

        # later filled by pipelinerl actors, but we need to set dummy values here to create the RolloutConfig
        agent_data["llm_config"]["api_key"] = "DUMMY_KEY"
        agent_data["llm_config"]["api_base"] = "DUMMY_BASE"

        rollout_config = RolloutConfig(
            name=f"rl_trainer_{name}",
            output_dir=Path("dummy"),
            persist_rollout=False,
            benchmark_config=benchmark_data,
            agent_config=agent_data,
            max_steps=MAX_STEPS,
            execution_mode="local",
        )

        engine = RolloutEngine(config=rollout_config)
        ds.append(CubeRolloutDataset(engine=engine))

    # for now we only support a single dataset, so just return the first one. In the future we can extend this to support multiple datasets.
    return ds[0] if len(ds) > 0 else None