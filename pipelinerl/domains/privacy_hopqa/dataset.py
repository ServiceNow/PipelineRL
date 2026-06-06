"""Dataset loading for Privacy HopQA training.

MosaicProject builds and validates the dataset. PipelineRL consumes the
materialized JSONL splits produced by that process.
"""

import json
from pathlib import Path
from typing import Any

DOMAIN_NAME = "privacy_hopqa"


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL row in {path} at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path} at line {line_number}")
            rows.append(row)
    return rows


def _load_from_materialized_jsonl(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path).expanduser()
    problems: list[dict[str, Any]] = []
    for index, row in enumerate(_load_jsonl(dataset_path)):
        problem = dict(row)
        problem.setdefault("domain", DOMAIN_NAME)
        problem.setdefault("dataset", dataset_path.stem)
        problem.setdefault("problem_id", f"{DOMAIN_NAME}_{problem.get('chain_id', index)}")
        problems.append(problem)
    return problems


def _configured_dataset_path(dataset_name: str, configured_paths: dict[str, str | Path | None]) -> str | Path:
    aliases = {
        "final": "final_dataset_path",
        "privacy_hopqa_final": "final_dataset_path",
        "final_train": "final_train_dataset_path",
        "privacy_hopqa_final_train": "final_train_dataset_path",
        "train": "final_train_dataset_path",
        "final_val": "final_val_dataset_path",
        "privacy_hopqa_final_val": "final_val_dataset_path",
        "val": "final_val_dataset_path",
        "validation": "final_val_dataset_path",
        "final_test": "final_test_dataset_path",
        "privacy_hopqa_final_test": "final_test_dataset_path",
        "test": "final_test_dataset_path",
    }
    config_key = aliases.get(dataset_name)
    if config_key is None:
        raise ValueError(
            f"Unsupported privacy_hopqa dataset '{dataset_name}'. "
            "Pass a JSONL path or one of: final, train, val, test."
        )

    configured_path = configured_paths.get(config_key)
    if not configured_path:
        raise ValueError(
            f"Dataset alias '{dataset_name}' requires dataset_loader_params.{config_key}."
        )
    return configured_path


def load_problems(
    dataset_names: list[str] | str | None = None,
    final_dataset_path: str | Path | None = None,
    final_train_dataset_path: str | Path | None = None,
    final_val_dataset_path: str | Path | None = None,
    final_test_dataset_path: str | Path | None = None,
    max_examples: int | None = None,
    **_: Any,  # absorbs the rest of dataset_loader_params (helper_service_url, knobs, etc.)
) -> list[dict[str, Any]]:
    if dataset_names is None:
        return []
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    configured_paths = {
        "final_dataset_path": final_dataset_path,
        "final_train_dataset_path": final_train_dataset_path,
        "final_val_dataset_path": final_val_dataset_path,
        "final_test_dataset_path": final_test_dataset_path,
    }

    problems: list[dict[str, Any]] = []
    for dataset_name in dataset_names:
        candidate_path = Path(str(dataset_name)).expanduser()
        if candidate_path.is_file():
            path: str | Path = candidate_path
        else:
            path = _configured_dataset_path(str(dataset_name).strip(), configured_paths)
        problems.extend(_load_from_materialized_jsonl(path))

    # One contiguous id across everything we loaded, so ids stay unique even when
    # several split files are concatenated.
    for index, problem in enumerate(problems):
        problem["id"] = index
        problem.setdefault("domain", DOMAIN_NAME)

    if max_examples is not None:
        problems = problems[: int(max_examples)]
    return problems
