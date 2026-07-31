"""Strict preparation and loading for Tau2 benchmark rows."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict

from pipelinerl.domains.tau2.client import NEMO_GYM_SHA, TAU2_DATA_SHA


TAU2_DATA_DOMAINS = ("airline", "retail", "telecom")
TAU2_DATA_REPOSITORY = "https://github.com/bxyu-nvidia/tau2-bench.git"
NEMO_GYM_REPOSITORY = "https://github.com/NVIDIA-NeMo/Gym.git"
TAU2_PREPARED_DATA_SCHEMA_VERSION = 1
TAU2_NORMALIZATION_CONTRACT = "pinned_gym_normalize_row_except_nl_assertion_v1"
TAU2_PREPARED_MANIFEST_FILENAME = "tau2_prepared_manifest.json"
TAU2_PREPARED_FILENAMES = {
    dataset: f"tau2_{dataset}.jsonl" for dataset in TAU2_DATA_DOMAINS
}
TAU2_EXPECTED_ROW_COUNTS = {
    "airline": 50,
    "retail": 114,
    "telecom": 114,
}
TAU2_EXPECTED_NL_REWARD_ROWS = {
    "airline": 0,
    "retail": 112,
    "telecom": 0,
}
TAU2_EXPECTED_NONEMPTY_NL_ASSERTION_ROWS = {
    "airline": 50,
    "retail": 40,
    "telecom": 0,
}

_REQUIRED_ROW_KEYS = {
    "config",
    "task",
    "seed",
    "evaluation_type",
    "responses_create_params",
}


class Tau2PreparedFileIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: Literal["airline", "retail", "telecom"]
    filename: str
    sha256: str
    row_count: int
    task_split_name: Literal["base"]
    evaluation_type: Literal["all"]
    timeout: None = None
    nl_reward_basis_rows: int
    nonempty_nl_assertion_rows: int
    task_identity_sha256: str


class Tau2PreparedDataManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    source_repository: str
    source_revision: str
    reference_normalizer_repository: str
    reference_normalizer_revision: str
    normalization_contract: str
    files: list[Tau2PreparedFileIdentity]
    total_row_count: int
    composite_identity_sha256: str


@dataclass(frozen=True)
class ValidatedTau2PreparedData:
    manifest: Tau2PreparedDataManifest
    data_files: dict[str, Path]
    rows_by_dataset: dict[str, list[dict[str, Any]]]


@dataclass(frozen=True)
class _RowSummary:
    task_ids: list[str]
    nl_reward_basis_rows: int
    nonempty_nl_assertion_rows: int


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _canonical_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) for row in rows)


def _identity_digest(identities: Sequence[tuple[str, str]]) -> str:
    return _sha256_bytes(_canonical_json_bytes([list(item) for item in identities]))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not an object")
    return value


def _validate_tau2_row(
    row: Mapping[str, Any],
    dataset: str,
    label: str,
) -> tuple[str, bool, bool]:
    missing = sorted(_REQUIRED_ROW_KEYS - row.keys())
    if missing:
        raise ValueError(f"{label} is missing keys {missing}")

    config = _mapping(row["config"], f"{label} config")
    if config.get("domain") != dataset:
        raise ValueError(
            f"{label} config.domain={config.get('domain')!r}, expected {dataset!r}"
        )
    if config.get("task_split_name") != "base":
        raise ValueError(f"{label} must use task_split_name='base'")
    if "timeout" not in config or config["timeout"] is not None:
        raise ValueError(f"{label} must carry config.timeout=None")
    if row.get("evaluation_type") != "all":
        raise ValueError(f"{label} must use evaluation_type='all'")

    task = _mapping(row["task"], f"{label} task")
    if task.get("id") is None:
        raise ValueError(f"{label} has no task.id")
    task_id = str(task["id"])

    params = _mapping(
        row["responses_create_params"],
        f"{label} responses_create_params",
    )
    if "input" not in params or "tools" not in params:
        raise ValueError(
            f"{label} responses_create_params must include input and tools"
        )

    criteria = _mapping(
        task.get("evaluation_criteria"),
        f"{label} task.evaluation_criteria",
    )
    reward_basis = criteria.get("reward_basis")
    if not isinstance(reward_basis, list) or not all(
        isinstance(item, str) for item in reward_basis
    ):
        raise ValueError(f"{label} reward_basis is not a string list")
    nl_occurrences = reward_basis.count("NL_ASSERTION")
    if nl_occurrences not in (0, 1):
        raise ValueError(
            f"{label} carries {nl_occurrences} NL_ASSERTION occurrences"
        )
    if nl_occurrences and dataset != "retail":
        raise ValueError(f"{label} has NL_ASSERTION outside retail")

    nl_assertions = criteria.get("nl_assertions")
    if nl_assertions is not None and not isinstance(nl_assertions, list):
        raise ValueError(f"{label} nl_assertions is not a list or null")
    return task_id, bool(nl_occurrences), bool(nl_assertions)


def _summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    dataset: str,
    label: str,
) -> _RowSummary:
    task_ids: list[str] = []
    nl_reward_rows = 0
    nonempty_nl_rows = 0
    for index, row in enumerate(rows, start=1):
        task_id, has_nl_reward, has_nl_assertions = _validate_tau2_row(
            row,
            dataset,
            f"{label}:{index}",
        )
        task_ids.append(task_id)
        nl_reward_rows += has_nl_reward
        nonempty_nl_rows += has_nl_assertions

    if len(task_ids) != TAU2_EXPECTED_ROW_COUNTS[dataset]:
        raise ValueError(
            f"{dataset} row count={len(task_ids)}, "
            f"expected {TAU2_EXPECTED_ROW_COUNTS[dataset]}"
        )
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"{dataset} contains duplicate task identities")
    if nl_reward_rows != TAU2_EXPECTED_NL_REWARD_ROWS[dataset]:
        raise ValueError(
            f"{dataset} NL reward rows={nl_reward_rows}, "
            f"expected {TAU2_EXPECTED_NL_REWARD_ROWS[dataset]}"
        )
    expected_nonempty = TAU2_EXPECTED_NONEMPTY_NL_ASSERTION_ROWS[dataset]
    if nonempty_nl_rows != expected_nonempty:
        raise ValueError(
            f"{dataset} nonempty NL assertion rows={nonempty_nl_rows}, "
            f"expected {expected_nonempty}"
        )
    return _RowSummary(
        task_ids=task_ids,
        nl_reward_basis_rows=nl_reward_rows,
        nonempty_nl_assertion_rows=nonempty_nl_rows,
    )


def _normalize_row(
    row: Mapping[str, Any],
    dataset: str,
    label: str,
    upstream_normalize_row: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    _validate_tau2_row(row, dataset, label)
    raw = copy.deepcopy(dict(row))
    normalized = copy.deepcopy(raw)
    normalized["config"]["save_to"] = ""
    normalized["config"].get("llm_args_user", {}).pop("temperature", None)

    upstream = upstream_normalize_row(copy.deepcopy(raw))
    if not isinstance(upstream, dict):
        raise ValueError("Pinned Gym normalize_row did not return an object")
    raw_basis = raw["task"]["evaluation_criteria"]["reward_basis"]
    upstream_basis = upstream["task"]["evaluation_criteria"]["reward_basis"]
    expected_upstream_basis = [
        item for item in raw_basis if item != "NL_ASSERTION"
    ]
    if upstream_basis != expected_upstream_basis:
        raise ValueError(
            f"{label} pinned Gym normalization changed reward_basis unexpectedly"
        )

    expected = copy.deepcopy(upstream)
    expected["task"]["evaluation_criteria"]["reward_basis"] = raw_basis
    if normalized != expected:
        raise ValueError(
            f"{label} differs from pinned Gym normalization outside reward_basis"
        )
    has_delta = normalized != upstream
    if has_delta != ("NL_ASSERTION" in raw_basis):
        raise ValueError(f"{label} has an invalid normalization differential")
    return normalized, has_delta


def _prepared_file_identity(
    path: Path,
    dataset: str,
    rows: Sequence[Mapping[str, Any]],
) -> Tau2PreparedFileIdentity:
    summary = _summarize_rows(rows, dataset, str(path))
    identities = [(dataset, task_id) for task_id in summary.task_ids]
    return Tau2PreparedFileIdentity(
        dataset=dataset,
        filename=path.name,
        sha256=_sha256_bytes(path.read_bytes()),
        row_count=len(rows),
        task_split_name="base",
        evaluation_type="all",
        timeout=None,
        nl_reward_basis_rows=summary.nl_reward_basis_rows,
        nonempty_nl_assertion_rows=summary.nonempty_nl_assertion_rows,
        task_identity_sha256=_identity_digest(identities),
    )


def write_tau2_prepared_data(
    raw_root: Path,
    output_dir: Path,
    upstream_normalize_row: Callable[[dict[str, Any]], dict[str, Any]],
) -> Path:
    normalized_by_dataset: dict[str, list[dict[str, Any]]] = {}
    differential_counts: dict[str, int] = {}
    all_identities: list[tuple[str, str]] = []

    for dataset in TAU2_DATA_DOMAINS:
        paths = sorted((raw_root / dataset).glob("*.json"))
        if len(paths) != TAU2_EXPECTED_ROW_COUNTS[dataset]:
            raise ValueError(
                f"{dataset} raw files={len(paths)}, "
                f"expected {TAU2_EXPECTED_ROW_COUNTS[dataset]}"
            )
        normalized_rows = []
        differential_count = 0
        for path in paths:
            row = json.loads(path.read_text())
            if not isinstance(row, dict):
                raise ValueError(f"{path} is not a JSON object")
            normalized, has_delta = _normalize_row(
                row,
                dataset,
                str(path),
                upstream_normalize_row,
            )
            normalized_rows.append(normalized)
            differential_count += has_delta

        normalized_rows.sort(
            key=lambda item: (
                str(item["config"]["domain"]),
                str(item["task"]["id"]),
            )
        )
        summary = _summarize_rows(
            normalized_rows,
            dataset,
            f"normalized {dataset}",
        )
        normalized_by_dataset[dataset] = normalized_rows
        differential_counts[dataset] = differential_count
        all_identities.extend((dataset, task_id) for task_id in summary.task_ids)

    if differential_counts != TAU2_EXPECTED_NL_REWARD_ROWS:
        raise ValueError(
            "Pinned Gym normalization differential counts "
            f"{differential_counts}, expected {TAU2_EXPECTED_NL_REWARD_ROWS}"
        )
    if len(all_identities) != len(set(all_identities)):
        raise ValueError("Prepared Tau2 data contains duplicate composite identities")

    output_dir.mkdir(parents=True, exist_ok=False)
    files = []
    for dataset in TAU2_DATA_DOMAINS:
        path = output_dir / TAU2_PREPARED_FILENAMES[dataset]
        rows = normalized_by_dataset[dataset]
        path.write_bytes(_canonical_jsonl_bytes(rows))
        files.append(_prepared_file_identity(path, dataset, rows))

    manifest = Tau2PreparedDataManifest(
        schema_version=TAU2_PREPARED_DATA_SCHEMA_VERSION,
        source_repository=TAU2_DATA_REPOSITORY,
        source_revision=TAU2_DATA_SHA,
        reference_normalizer_repository=NEMO_GYM_REPOSITORY,
        reference_normalizer_revision=NEMO_GYM_SHA,
        normalization_contract=TAU2_NORMALIZATION_CONTRACT,
        files=files,
        total_row_count=len(all_identities),
        composite_identity_sha256=_identity_digest(all_identities),
    )
    manifest_path = output_dir / TAU2_PREPARED_MANIFEST_FILENAME
    manifest_path.write_bytes(
        _canonical_json_bytes(manifest.model_dump(mode="json"))
    )
    return manifest_path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line:
            raise ValueError(f"{path}:{line_number} is blank")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(row)
    return rows


def _coerce_data_files(
    data_files: Mapping[str, str] | DictConfig | None,
) -> Mapping[str, str] | None:
    if isinstance(data_files, DictConfig):
        data_files = OmegaConf.to_container(data_files, resolve=True)
    if data_files is not None and not isinstance(data_files, Mapping):
        raise ValueError("Tau2 data_files must map dataset names to JSONL paths")
    return data_files


def validate_tau2_prepared_data(
    manifest_path: Path,
    *,
    data_files: Mapping[str, str] | DictConfig | None = None,
) -> ValidatedTau2PreparedData:
    payload = json.loads(manifest_path.read_text())
    manifest = Tau2PreparedDataManifest.model_validate(payload)
    if manifest.schema_version != TAU2_PREPARED_DATA_SCHEMA_VERSION:
        raise ValueError(
            f"Tau2 prepared-data schema {manifest.schema_version} is unsupported"
        )
    if (
        manifest.source_repository != TAU2_DATA_REPOSITORY
        or manifest.source_revision != TAU2_DATA_SHA
        or manifest.reference_normalizer_repository != NEMO_GYM_REPOSITORY
        or manifest.reference_normalizer_revision != NEMO_GYM_SHA
        or manifest.normalization_contract != TAU2_NORMALIZATION_CONTRACT
    ):
        raise ValueError("Tau2 prepared-data source or normalization identity drift")
    canonical_manifest = _canonical_json_bytes(manifest.model_dump(mode="json"))
    if manifest_path.read_bytes() != canonical_manifest:
        raise ValueError("Tau2 prepared-data manifest is not canonical")

    datasets = [item.dataset for item in manifest.files]
    if datasets != list(TAU2_DATA_DOMAINS):
        raise ValueError(
            f"Tau2 prepared-data files are ordered {datasets}, "
            f"expected {list(TAU2_DATA_DOMAINS)}"
        )

    configured = _coerce_data_files(data_files)
    if configured is not None and set(configured) != set(TAU2_DATA_DOMAINS):
        raise ValueError(
            "Tau2 data_files must contain exactly airline, retail, and telecom"
        )

    resolved_files: dict[str, Path] = {}
    rows_by_dataset: dict[str, list[dict[str, Any]]] = {}
    all_identities: list[tuple[str, str]] = []
    for expected, recorded in zip(TAU2_DATA_DOMAINS, manifest.files, strict=True):
        expected_filename = TAU2_PREPARED_FILENAMES[expected]
        if recorded.filename != expected_filename:
            raise ValueError(
                f"{expected} filename={recorded.filename!r}, "
                f"expected {expected_filename!r}"
            )
        path = (manifest_path.parent / recorded.filename).resolve()
        if configured is not None:
            configured_path = Path(str(configured[expected])).resolve()
            if configured_path != path:
                raise ValueError(
                    f"{expected} data file {configured_path} does not match {path}"
                )
        if not path.is_file():
            raise ValueError(f"Tau2 prepared data file does not exist: {path}")
        if _sha256_bytes(path.read_bytes()) != recorded.sha256:
            raise ValueError(f"{expected} prepared JSONL SHA256 mismatch")

        rows = _load_jsonl(path)
        if path.read_bytes() != _canonical_jsonl_bytes(rows):
            raise ValueError(f"{expected} prepared JSONL is not canonical")
        actual = _prepared_file_identity(path, expected, rows)
        if actual != recorded:
            raise ValueError(f"{expected} prepared-data identity mismatch")
        resolved_files[expected] = path
        rows_by_dataset[expected] = rows
        all_identities.extend(
            (expected, str(row["task"]["id"])) for row in rows
        )

    if len(all_identities) != len(set(all_identities)):
        raise ValueError("Tau2 prepared data contains duplicate composite identities")
    if manifest.total_row_count != len(all_identities):
        raise ValueError("Tau2 prepared-data total row count mismatch")
    if manifest.composite_identity_sha256 != _identity_digest(all_identities):
        raise ValueError("Tau2 prepared-data composite identity mismatch")

    return ValidatedTau2PreparedData(
        manifest=manifest,
        data_files=resolved_files,
        rows_by_dataset=rows_by_dataset,
    )


def load_tau2_problems(
    dataset_names: Sequence[str],
    *,
    data_files: Mapping[str, str] | DictConfig,
    prepared_data_manifest: str,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    validated = validate_tau2_prepared_data(
        Path(prepared_data_manifest),
        data_files=data_files,
    )

    problems: list[dict[str, Any]] = []
    for dataset_name in dataset_names:
        if dataset_name not in TAU2_DATA_DOMAINS:
            raise ValueError(f"Unknown Tau2 dataset {dataset_name!r}")
        for row in validated.rows_by_dataset[dataset_name]:
            problem = dict(row)
            problem["dataset"] = str(dataset_name)
            problem["domain"] = "tau2"
            problem["task_id"] = str(row["task"]["id"])
            problems.append(problem)

    if seed is not None:
        random.Random(seed).shuffle(problems)
    return problems
