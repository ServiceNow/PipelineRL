from __future__ import annotations

import logging
from typing import Any
from pathlib import Path
from omegaconf import OmegaConf
from pipelinerl.ray.worker import RolloutDataset
from cube_harness.episode import MAX_STEPS
from cube_harness.rl import RolloutConfig

logger = logging.getLogger(__name__)

class CubeRolloutDataset(RolloutDataset):
    def __init__(self, rollout_config: RolloutConfig):
        self.config = rollout_config
        self._closed = False
        self.config.benchmark_config.install()
        self.benchmark = self.config.benchmark_config.make(self.config.infra)
        self.runtime_context = getattr(self.benchmark, "_runtime_context", None)
        self.dataset_name = self.config.benchmark_config.benchmark_metadata.name
        self.domain = self.config.name
        self.task_configs = list(self.config.benchmark_config.get_task_configs())

    def __len__(self) -> int:
        return len(self.task_configs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        task_config = self.task_configs[index]
        task_id = str(task_config.task_id)
        return {
            "task_config": task_config,
            "runtime_context": self.runtime_context,
            "agent_cfg": self.config.agent_config,
            "task_id": task_id,
            "domain": self.domain,
            "dataset": self.dataset_name,
        }

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.benchmark.close()
        except Exception:
            logger.warning("Cube benchmark close failed for %s", self.domain, exc_info=True)
        
        self._closed = True

class ConcatCubeRolloutDataset(RolloutDataset):
    def __init__(self, cube_rollout_datasets: list[CubeRolloutDataset]):
        self.cube_rollout_datasets = cube_rollout_datasets
        self.task_configs = []
        self.dataset_configs = []
        for idx, dataset in enumerate(self.cube_rollout_datasets):
            self.task_configs.extend(dataset.task_configs)
            self.dataset_configs.extend([idx] * len(dataset.task_configs))

        self._closed = False

    def __len__(self) -> int:
        return len(self.task_configs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        task_config = self.task_configs[index]
        current_dataset = self.cube_rollout_datasets[self.dataset_configs[index]]
        task_id = str(task_config.task_id)
        return {
            "task_config": task_config,
            "runtime_context": current_dataset.runtime_context,
            "agent_cfg": current_dataset.config.agent_config,
            "task_id": task_id,
            "domain": current_dataset.domain,
            "dataset": current_dataset.dataset_name
        }

    def close(self) -> None:
        if self._closed:
            return
        for dataset in self.cube_rollout_datasets:
            dataset.close()
    
        self._closed = True

def _as_plain_dict(value: Any) -> dict[str, Any]:
    return OmegaConf.to_container(value, resolve=True)

def load_datasets(dataset_names: list[str], **loader_kwargs: Any) -> RolloutDataset | ConcatCubeRolloutDataset | None:
    if dataset_names is None:
        dataset_names = []

    for name in dataset_names:
        assert name in loader_kwargs, f"Missing config for dataset {name}"

    ds = []
    for name in dataset_names:
        dataset_params = loader_kwargs[name]
        benchmark_data = _as_plain_dict(dataset_params.get("benchmark") or dataset_params.get("benchmark_config"))
        agent_data = _as_plain_dict(dataset_params.get("agent") or dataset_params.get("agent_config"))

        # later filled by pipelinerl actors, but we need to set dummy values here to create the RolloutConfig
        agent_data["llm_config"]["api_key"] = "DUMMY_KEY"
        agent_data["llm_config"]["api_base"] = "DUMMY_BASE"

        rollout_config = RolloutConfig(
            name=name,
            output_dir=Path("dummy"),
            persist_rollout=False,
            benchmark_config=benchmark_data,
            agent_config=agent_data,
            max_steps=MAX_STEPS,
            execution_mode="local",
        )

        dataset = CubeRolloutDataset(rollout_config=rollout_config)
        ds.append(dataset)
        logger.info("Loaded %d rollout items from %s", len(dataset), dataset.domain)

    if len(ds) == 0:
        return None
    if len(ds) == 1:
        return ds[0]
    
    logger.info("Concatenating %d datasets", len(ds))
    return ConcatCubeRolloutDataset(cube_rollout_datasets=ds)