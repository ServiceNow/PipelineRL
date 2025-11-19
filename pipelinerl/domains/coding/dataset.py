from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, DownloadMode, load_dataset
from datasets.exceptions import DatasetGenerationError
from huggingface_hub import snapshot_download
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

DOMAIN_NAME = "coding"
BASE_DATASET_NAME = "mixed-training-text-datasets"
DATASET_ALIASES = frozenset({DOMAIN_NAME, "code"})
SUPPORTED_DATASET_NAMES = frozenset({BASE_DATASET_NAME})
DEFAULT_DATASET_ID = "ServiceNow-AI/mixed-training-text-datasets"
DEFAULT_DATASET_CONFIG = "80k-if-math-coding-fncalling-stem"
DEFAULT_SPLIT_ORDER = ("train", "validation", "test")
DEFAULT_SPLIT_RATIOS = tuple((name, ratio) for name, ratio in zip(DEFAULT_SPLIT_ORDER, (0.9, 0.05, 0.05)))
DEFAULT_CALL_TYPES = ("assert", "std")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    split: str = "train"


@dataclass
@dataclass(frozen=True)
class ResolvedDatasetEntry:
    spec: DatasetSpec
    label: str


@dataclass(frozen=True)
class DatasetResolution:
    requested: list[str]
    entries: list[ResolvedDatasetEntry]


@dataclass
class DatasetOptions:
    dataset_id: str = DEFAULT_DATASET_ID
    dataset_config: str | None = DEFAULT_DATASET_CONFIG
    split_ratios: Sequence[tuple[str, float]] = DEFAULT_SPLIT_RATIOS
    trust_remote_code: bool = True
    max_examples_per_split: int | None = None
    allowed_call_types: Sequence[str] = DEFAULT_CALL_TYPES
    huggingface_token: str | None = None
    ability_filter: str = "code"


CacheKey = tuple[str, str | None, bool, str | None]
_DATASET_CACHE: dict[CacheKey, Dataset] = {}


def _normalize_loader_options(loader_kwargs: Dict[str, Any]) -> DatasetOptions:
    def _to_native(value: Any) -> Any:
        if isinstance(value, DictConfig):
            return OmegaConf.to_container(value, resolve=True)
        return value

    options = DatasetOptions()
    if loader_kwargs:
        raw_ratios = _to_native(loader_kwargs.get("split_ratios"))
        if isinstance(raw_ratios, dict):
            options.split_ratios = tuple(raw_ratios.items())
        dataset_config = loader_kwargs.get("dataset_config")
        if dataset_config:
            options.dataset_config = str(dataset_config)
        dataset_id = loader_kwargs.get("dataset_id")
        if dataset_id:
            options.dataset_id = str(dataset_id)
        if "max_examples_per_split" in loader_kwargs:
            value = loader_kwargs["max_examples_per_split"]
            options.max_examples_per_split = int(value) if value is not None else None
        if "trust_remote_code" in loader_kwargs:
            options.trust_remote_code = bool(loader_kwargs["trust_remote_code"])
        call_types = _to_native(loader_kwargs.get("allowed_call_types"))
        if isinstance(call_types, Iterable) and not isinstance(call_types, (str, bytes)):
            options.allowed_call_types = tuple(str(item) for item in call_types)
        token = loader_kwargs.get("huggingface_token") or loader_kwargs.get("hf_token")
        if token:
            options.huggingface_token = str(token)
        ability = loader_kwargs.get("ability_filter") or loader_kwargs.get("ability")
        if ability:
            options.ability_filter = str(ability)
    return options


def parse_dataset_name(entry: str) -> DatasetSpec:
    text = entry.strip()
    if "@" not in text:
        return DatasetSpec(name=text)
    name, split = text.split("@", 1)
    return DatasetSpec(name=name.strip(), split=split.strip() or "train")


def _normalize_dataset_names(dataset_names: List[str] | str | None) -> list[str]:
    if dataset_names is None:
        return []
    if isinstance(dataset_names, str):
        return [dataset_names]
    return [str(entry) for entry in dataset_names]


def _resolve_dataset_requests(dataset_names: List[str] | str | None) -> DatasetResolution:
    requested = _normalize_dataset_names(dataset_names)
    entries: list[ResolvedDatasetEntry] = []
    for entry in requested:
        spec = parse_dataset_name(entry)
        base_name = BASE_DATASET_NAME if spec.name in DATASET_ALIASES else spec.name
        if base_name not in SUPPORTED_DATASET_NAMES:
            raise ValueError(f"Unsupported coding dataset '{spec.name}'")
        resolved_spec = DatasetSpec(name=base_name, split=spec.split)
        entries.append(ResolvedDatasetEntry(spec=resolved_spec, label=f"{spec.name}@{spec.split}"))
    return DatasetResolution(requested=requested, entries=entries)


def _load_dataset(options: DatasetOptions) -> Dataset:
    ability = options.ability_filter
    cache_key: CacheKey = (
        options.dataset_id,
        options.dataset_config,
        options.trust_remote_code,
        ability,
    )
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    def _materialize_dataset(**extra_kwargs: Any) -> Dataset:
        return load_dataset(
            options.dataset_id,
            options.dataset_config,
            split="train",
            trust_remote_code=options.trust_remote_code,
            token=options.huggingface_token,
            **extra_kwargs,
        )

    try:
        ds = _materialize_dataset()
    except DatasetGenerationError as exc:
        logger.warning(
            "load_dataset failed for %s (%s): %s. Forcing re-download.",
            options.dataset_id,
            options.dataset_config,
            exc,
        )
        try:
            ds = _materialize_dataset(download_mode=DownloadMode.FORCE_REDOWNLOAD)
        except DatasetGenerationError as redownload_exc:
            logger.warning(
                "Forced re-download also failed for %s (%s): %s. Falling back to streaming mode.",
                options.dataset_id,
                options.dataset_config,
                redownload_exc,
            )
            stream = _materialize_dataset(streaming=True)
            try:
                ds = Dataset.from_list(list(stream))
            except OSError as stream_exc:
                logger.warning(
                    "Streaming fallback also failed for %s (%s): %s. Downloading snapshot locally.",
                    options.dataset_id,
                    options.dataset_config,
                    stream_exc,
                )
                ds = _load_snapshot(options)

    ds = ds.filter(lambda sample: sample.get("ability") == ability)
    _DATASET_CACHE[cache_key] = ds
    logger.info(
        "Loaded %s (%s) with %d coding samples",
        options.dataset_id,
        options.dataset_config,
        len(ds),
    )
    return ds


def _load_snapshot(options: DatasetOptions) -> Dataset:
    if not options.dataset_config:
        raise RuntimeError("Snapshot fallback requires a dataset_config but none was provided.")

    snapshot_dir = snapshot_download(
        repo_id=options.dataset_id,
        repo_type="dataset",
        token=options.huggingface_token,
        allow_patterns=(f"{options.dataset_config}/*", "dataset_infos.json"),
    )
    config_dir = Path(snapshot_dir) / options.dataset_config
    parquet_files = sorted(config_dir.glob("train-*.parquet"))
    if not parquet_files:
        raise RuntimeError(
            f"Snapshot for {options.dataset_id} ({options.dataset_config}) contained no train parquet shards"
        )

    tables: list[pa.Table] = []
    for shard in parquet_files:
        try:
            tables.append(pq.read_table(shard))
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Skipping corrupted parquet shard %s: %s", shard.name, exc)
    if not tables:
        raise RuntimeError(
            f"All locally downloaded shards failed to load for {options.dataset_id} ({options.dataset_config})."
        )

    table = pa.concat_tables(tables)
    logger.info(
        f"Loaded {table.num_rows} rows via snapshot fallback ({len(tables)} shards)",
    )
    return Dataset.from_table(table)


def _normalized_split_sequence(ratios: Sequence[tuple[str, float]]) -> list[tuple[str, float]]:
    if not ratios:
        return [("train", 1.0)]
    values = [(name, float(portion)) for name, portion in ratios if float(portion) > 0]
    total = sum(portion for _, portion in values)
    if math.isclose(total, 0.0):
        return [("train", 1.0)]
    return [(name, portion / total) for name, portion in values]


def _slice_indices(count: int, split: str, ratios: Sequence[tuple[str, float]], seed: int | None) -> list[int]:
    normalized = _normalized_split_sequence(ratios)
    if split not in {name for name, _ in normalized}:
        raise ValueError(f"Split '{split}' is not defined in split_ratios {normalized}")
    indices = list(range(count))
    rng = random.Random(seed or 0)
    rng.shuffle(indices)
    cumulative = 0.0
    start = 0
    selected: dict[str, list[int]] = {}
    for idx, (name, portion) in enumerate(normalized):
        cumulative += portion
        end = count if idx == len(normalized) - 1 else min(count, int(round(cumulative * count)))
        selected[name] = indices[start:end]
        start = end
    return selected.get(split, [])


def _decode_extra_info(raw_extra: Any) -> dict[str, Any]:
    if isinstance(raw_extra, dict):
        return raw_extra
    if isinstance(raw_extra, str) and raw_extra.strip():
        try:
            return json.loads(raw_extra)
        except json.JSONDecodeError:
            logger.debug("Failed to decode extra_info: %s", raw_extra[:128])
    return {}


def _build_record(sample: dict, dataset_label: str, allowed_call_types: Sequence[str]) -> dict | None:
    reward_model = sample.get("reward_model") or {}
    reward_raw = reward_model.get("ground_truth")
    if reward_raw is None:
        return None
    try:
        reward_context = json.loads(reward_raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if allowed_call_types and reward_context.get("call_type") not in set(allowed_call_types):
        return None

    prompt_messages = sample.get("prompt") or []
    if not prompt_messages:
        return None
    task = prompt_messages[0].get("content")
    if not task:
        return None

    extra_info = _decode_extra_info(sample.get("extra_info"))
    return {
        "dataset": dataset_label,
        "task": task,
        "reward_context": reward_context,
        "extra_info": extra_info,
    }


def _load_split(
    spec: DatasetSpec,
    *,
    options: DatasetOptions,
    seed: int | None,
    dataset_label: str | None = None,
) -> list[dict]:
    dataset = _load_dataset(options)
    indices = _slice_indices(len(dataset), spec.split, options.split_ratios, seed)
    if not indices:
        logger.warning("Requested split '%s' produced zero samples", spec.split)
        return []
    subset = dataset.select(indices)
    samples: list[dict] = []
    label = dataset_label or f"{spec.name}@{spec.split}"
    for sample in subset:
        record = _build_record(sample, label, options.allowed_call_types)
        if record is None:
            continue
        samples.append(record)
        if options.max_examples_per_split and len(samples) >= options.max_examples_per_split:
            break
    logger.info(
        "Loaded %d samples for %s",
        len(samples),
        label,
    )
    return samples


def _attach_ids(samples: list[dict]) -> list[dict]:
    for idx, sample in enumerate(samples):
        sample["id"] = idx
    return samples


def load_datasets(dataset_names: List[str] | str | None, seed: int | None = None, **loader_kwargs: Any) -> List[Dict]:
    resolution = _resolve_dataset_requests(dataset_names)
    if not resolution.entries:
        if dataset_names:
            logger.warning("No coding dataset entries were resolved for %s", dataset_names)
        return []

    options = _normalize_loader_options(loader_kwargs)
    aggregated: list[dict] = []
    for entry in resolution.entries:
        aggregated.extend(
            _load_split(entry.spec, options=options, seed=seed, dataset_label=entry.label),
        )

    if not aggregated:
        logger.warning("No coding datasets were loaded for entries %s", resolution.requested)

    return _attach_ids(aggregated)


def load_problems(dataset_names: List[str] | str | None, **loader_kwargs: dict) -> List[Dict]:
    """Hydra entrypoint that mirrors the math domain loader style."""

    seed = loader_kwargs.pop("seed", None)
    return load_datasets(dataset_names, seed=seed, **loader_kwargs)


__all__ = ["load_datasets", "load_problems", "parse_dataset_name", "DOMAIN_NAME"]
