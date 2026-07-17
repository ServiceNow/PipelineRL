import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


_REQUIRED_ROW_KEYS = {
    "config",
    "task",
    "seed",
    "evaluation_type",
    "responses_create_params",
}


def load_tau2_problems(
    dataset_names: Sequence[str],
    *,
    data_files: Mapping[str, str] | DictConfig,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    if isinstance(data_files, DictConfig):
        data_files = OmegaConf.to_container(data_files, resolve=True)
    if not isinstance(data_files, Mapping):
        raise ValueError("Tau2 data_files must map dataset names to prepared JSONL paths")

    problems: list[dict[str, Any]] = []
    for dataset_name in dataset_names:
        if dataset_name not in data_files:
            raise ValueError(f"No prepared Tau2 JSONL configured for dataset {dataset_name!r}")
        path = Path(str(data_files[dataset_name]))
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                missing = sorted(_REQUIRED_ROW_KEYS - row.keys())
                if missing:
                    raise ValueError(f"{path}:{line_number} is missing keys {missing}")
                task = row["task"]
                if not isinstance(task, dict) or task.get("id") is None:
                    raise ValueError(f"{path}:{line_number} has no task.id")
                params = row["responses_create_params"]
                if not isinstance(params, dict) or "input" not in params or "tools" not in params:
                    raise ValueError(f"{path}:{line_number} has invalid responses_create_params")

                problem = dict(row)
                problem["dataset"] = str(dataset_name)
                problem["domain"] = "tau2"
                problem["task_id"] = str(task["id"])
                problems.append(problem)

    if seed is not None:
        random.Random(seed).shuffle(problems)
    return problems
