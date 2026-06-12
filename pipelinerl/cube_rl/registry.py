from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CubeTaskRef:
    cube_id: str
    task_id: str


@dataclass(frozen=True)
class CubeSpec:
    cube_id: str
    split: str
    benchmark_cfg: dict[str, Any]
    agent_cfg: dict[str, Any]
    task_ids: tuple[str, ...]
    hint_condition: str | None = None

    def runtime_payload(self) -> dict[str, Any]:
        return {
            "cube_id": self.cube_id,
            "split": self.split,
            "benchmark_cfg": self.benchmark_cfg,
            "agent_cfg": self.agent_cfg,
            "hint_condition": self.hint_condition,
        }

    def task_refs(self) -> list[CubeTaskRef]:
        return [CubeTaskRef(cube_id=self.cube_id, task_id=task_id) for task_id in self.task_ids]


@dataclass(frozen=True)
class CubeRegistry:
    specs: tuple[CubeSpec, ...]
    train_tasks: tuple[CubeTaskRef, ...]
    test_tasks: tuple[CubeTaskRef, ...]

    def runtime_payloads(self) -> list[dict[str, Any]]:
        return [spec.runtime_payload() for spec in self.specs]


def _as_plain_dict(value: Any, *, name: str) -> dict[str, Any]:
    resolved = OmegaConf.to_container(value, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError(f"{name} must resolve to a dictionary")
    return resolved


def _cube_id(cube_cfg: Any) -> str:
    cube_id = getattr(cube_cfg, "id", None) or getattr(cube_cfg, "name", None)
    if not cube_id:
        raise ValueError("Each cube in cube_params.cubes must define id or name")
    return str(cube_id)


def _cube_split(cube_cfg: Any) -> str:
    split = str(getattr(cube_cfg, "split", "train"))
    if split not in {"train", "test"}:
        raise ValueError(f"Cube {_cube_id(cube_cfg)!r} has invalid split {split!r}; expected 'train' or 'test'")
    return split


def _cube_hint_condition(cube_cfg: Any) -> str | None:
    """Optional per-cube hint condition consumed by the in-training hint refresh."""
    hint_condition = getattr(cube_cfg, "hint_condition", None)
    if hint_condition is None:
        return None
    hint_condition = str(hint_condition)
    if hint_condition not in {"good", "none", "distractor"}:
        raise ValueError(
            f"Cube {_cube_id(cube_cfg)!r} has invalid hint_condition {hint_condition!r}; "
            "expected 'good', 'none' or 'distractor'"
        )
    return hint_condition


def _metadata_dataset(benchmark_obj: Any, task_id: str) -> str:
    task_metadata = getattr(benchmark_obj, "task_metadata", {}).get(task_id)
    extra_info = getattr(task_metadata, "extra_info", None)
    if isinstance(extra_info, dict) and extra_info.get("dataset"):
        return str(extra_info["dataset"])
    return ""


def _cube_dataset_filter(cube_cfg: Any) -> set[str] | None:
    dataset_names = getattr(cube_cfg, "dataset_names", None)
    dataset_name = getattr(cube_cfg, "dataset_name", None)
    if dataset_names is None and dataset_name is None:
        return None
    names: list[str] = []
    if dataset_name is not None:
        names.append(str(dataset_name))
    if dataset_names is not None:
        names.extend(str(name) for name in dataset_names)
    return set(names)


def _discover_task_ids(
    cube_id: str,
    benchmark_cfg: dict[str, Any],
    *,
    dataset_filter: set[str] | None,
) -> tuple[str, ...]:
    benchmark_obj = hydra.utils.instantiate(benchmark_cfg)
    task_ids = []
    for task_config in benchmark_obj.get_task_configs():
        task_id = str(task_config.task_id)
        if dataset_filter is not None and _metadata_dataset(benchmark_obj, task_id) not in dataset_filter:
            continue
        task_ids.append(task_id)
    try:
        benchmark_obj.close()
    except Exception:
        logger.debug("Ignoring close failure after metadata discovery for cube %s", cube_id, exc_info=True)
    if not task_ids:
        raise ValueError(
            f"Cube {cube_id!r} produced no task configs during metadata discovery. "
            "PipelineRL now expects task metadata to be available without worker setup/install."
        )
    return tuple(str(task_id) for task_id in task_ids)


def build_cube_registry(cfg: DictConfig) -> CubeRegistry:
    cube_params = cfg.get("cube_params", None)
    if cube_params is None or not getattr(cube_params, "cubes", None):
        raise ValueError("cube worker launcher requires cube_params.cubes")

    specs: list[CubeSpec] = []
    seen: set[str] = set()
    for cube_cfg in cube_params.cubes:
        cube_id = _cube_id(cube_cfg)
        if cube_id in seen:
            raise ValueError(f"Duplicate cube id in cube_params.cubes: {cube_id}")
        seen.add(cube_id)

        benchmark_cfg = _as_plain_dict(cube_cfg.benchmark, name=f"cube {cube_id}.benchmark")
        agent_cfg = _as_plain_dict(cube_cfg.agent, name=f"cube {cube_id}.agent")
        split = _cube_split(cube_cfg)
        task_ids = _discover_task_ids(
            cube_id,
            benchmark_cfg,
            dataset_filter=_cube_dataset_filter(cube_cfg),
        )
        specs.append(
            CubeSpec(
                cube_id=cube_id,
                split=split,
                benchmark_cfg=benchmark_cfg,
                agent_cfg=agent_cfg,
                task_ids=task_ids,
                hint_condition=_cube_hint_condition(cube_cfg),
            )
        )

    train_tasks = tuple(task for spec in specs if spec.split == "train" for task in spec.task_refs())
    test_tasks = tuple(task for spec in specs if spec.split == "test" for task in spec.task_refs())
    logger.info(
        "Discovered %d cube specs with %d train tasks and %d test tasks",
        len(specs),
        len(train_tasks),
        len(test_tasks),
    )
    return CubeRegistry(specs=tuple(specs), train_tasks=train_tasks, test_tasks=test_tasks)
