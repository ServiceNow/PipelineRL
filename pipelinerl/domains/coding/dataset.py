from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import Dataset, load_dataset
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

DOMAIN_NAME = "coding"
DEFAULT_DATASET_ID = "PrimeIntellect/INTELLECT-3-RL"
DEFAULT_DATASET_CONFIG = "code"
DEFAULT_SPLIT = "train"
# Column used for difficulty filtering
DEFAULT_DIFFICULTY_COLUMN = "avg@8_qwen3_4b_instruct_2507"
# Default difficulty range: problems with 10-90% solve rate
DEFAULT_MIN_DIFFICULTY = 0.1
DEFAULT_MAX_DIFFICULTY = 0.9
DEFAULT_TRAIN_RATIO = 0.9

_DATASET_CACHE: dict[str, Dataset] = {}


@dataclass
class DatasetOptions:
    dataset_id: str = DEFAULT_DATASET_ID
    dataset_config: str = DEFAULT_DATASET_CONFIG
    split: str = DEFAULT_SPLIT
    subset: str = "train"
    train_ratio: float = DEFAULT_TRAIN_RATIO
    max_examples: int | None = None
    min_difficulty: float | None = DEFAULT_MIN_DIFFICULTY
    max_difficulty: float | None = DEFAULT_MAX_DIFFICULTY
    difficulty_column: str = DEFAULT_DIFFICULTY_COLUMN
    huggingface_token: str | None = None
    seed: int | None = None
    # Source filtering: None means all sources, or specify e.g. ["primeintellect"]
    sources: list[str] | None = None


def _to_native(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _normalize_options(loader_kwargs: Dict[str, Any]) -> DatasetOptions:
    options = DatasetOptions()
    if not loader_kwargs:
        return options

    if "dataset_id" in loader_kwargs:
        options.dataset_id = str(loader_kwargs["dataset_id"])
    if "dataset_config" in loader_kwargs:
        options.dataset_config = str(loader_kwargs["dataset_config"])
    if "split" in loader_kwargs:
        options.split = str(loader_kwargs["split"])
    if "subset" in loader_kwargs:
        options.subset = str(loader_kwargs["subset"])
    if "train_ratio" in loader_kwargs:
        val = loader_kwargs["train_ratio"]
        options.train_ratio = float(val) if val is not None else DEFAULT_TRAIN_RATIO
    if "max_examples" in loader_kwargs:
        val = loader_kwargs["max_examples"]
        options.max_examples = int(val) if val is not None else None
    if "min_difficulty" in loader_kwargs:
        val = loader_kwargs["min_difficulty"]
        options.min_difficulty = float(val) if val is not None else None
    if "max_difficulty" in loader_kwargs:
        val = loader_kwargs["max_difficulty"]
        options.max_difficulty = float(val) if val is not None else None
    if "difficulty_column" in loader_kwargs:
        options.difficulty_column = str(loader_kwargs["difficulty_column"])
    if "huggingface_token" in loader_kwargs or "hf_token" in loader_kwargs:
        token = loader_kwargs.get("huggingface_token") or loader_kwargs.get("hf_token")
        options.huggingface_token = str(token) if token else None
    if "seed" in loader_kwargs:
        val = loader_kwargs["seed"]
        options.seed = int(val) if val is not None else None
    if "sources" in loader_kwargs:
        val = loader_kwargs["sources"]
        if val is not None:
            if isinstance(val, str):
                options.sources = [val]
            else:
                options.sources = list(val)

    return options


def _load_raw_dataset(options: DatasetOptions) -> Dataset:
    cache_key = f"{options.dataset_id}:{options.dataset_config}:{options.split}"
    if cache_key in _DATASET_CACHE:
        logger.debug("Using cached dataset for %s", cache_key)
        return _DATASET_CACHE[cache_key]

    logger.info(
        "Loading dataset %s (config=%s, split=%s)...",
        options.dataset_id,
        options.dataset_config,
        options.split,
    )
    ds = load_dataset(
        options.dataset_id,
        options.dataset_config,
        split=options.split,
        token=options.huggingface_token,
    )
    _DATASET_CACHE[cache_key] = ds
    logger.info("Loaded %d samples from %s/%s", len(ds), options.dataset_id, options.dataset_config)
    return ds


def _parse_tests(info_str: str | None) -> dict[str, Any] | None:
    if info_str is None:
        return None

    try:
        info = json.loads(info_str)
    except json.JSONDecodeError:
        return None

    if not isinstance(info, dict):
        return None

    tests_raw = info.get("tests")
    if tests_raw is None:
        return None

    # Parse inner tests JSON if string
    if isinstance(tests_raw, str):
        try:
            tests = json.loads(tests_raw)
        except json.JSONDecodeError:
            return None
    else:
        tests = tests_raw

    if not isinstance(tests, dict):
        return None

    return tests


def _build_record(sample: dict, idx: int) -> dict | None:
    prompt = sample.get("question")
    if not prompt:
        return None

    tests = _parse_tests(sample.get("info"))
    if tests is None:
        return None

    inputs = tests.get("inputs", [])
    outputs = tests.get("outputs", [])
    if not inputs or not outputs:
        return None

    fn_name = tests.get("fn_name")

    # Determine call type based on whether fn_name is present
    # and whether inputs are strings (stdin) or lists (function args)
    if fn_name:
        call_type = "fn"
    elif inputs and isinstance(inputs[0], str) and "\n" in str(inputs[0]):
        # Multi-line string input suggests stdin/stdout style
        call_type = "std"
    else:
        # Single-line stdin problems (no fn_name, no newlines in input)
        call_type = "std"

    # Build reward_context in the format expected by verifier
    reward_context = {
        "inputs": inputs,
        "outputs": outputs,
        "call_type": call_type,
    }
    if fn_name:
        reward_context["fn_name"] = fn_name

    # Extract source info
    try:
        info = json.loads(sample.get("info", "{}"))
        source = info.get("source", "unknown")
    except json.JSONDecodeError:
        source = "unknown"

    return {
        "id": idx,
        "problem_id": f"intellect3_{idx}",
        "task": prompt,
        "reward_context": reward_context,
        "dataset": f"intellect3-rl-code@{source}",
        "domain": DOMAIN_NAME,
        "source": source,
    }


def _split_dataset(
    dataset: Dataset,
    options: DatasetOptions,
) -> Dataset:
    if options.subset not in ("train", "test", "all"):
        logger.warning("Invalid subset '%s', defaulting to 'train'", options.subset)
        options.subset = "train"

    # Return full dataset for "all" subset
    if options.subset == "all":
        logger.info("Using all data: %d samples", len(dataset))
        return dataset

    split_seed = 42
    total = len(dataset)
    indices = list(range(total))

    rng = random.Random(split_seed)
    rng.shuffle(indices)

    train_end = int(total * options.train_ratio)

    if options.subset == "train":
        selected_indices = indices[:train_end]
        logger.info(
            "Using train subset: %d samples (%.1f%% of %d)",
            len(selected_indices),
            options.train_ratio * 100,
            total,
        )
    else:  # test
        selected_indices = indices[train_end:]
        logger.info(
            "Using test subset: %d samples (%.1f%% of %d)",
            len(selected_indices),
            (1 - options.train_ratio) * 100,
            total,
        )

    return dataset.select(selected_indices)


def _filter_by_difficulty(
    dataset: Dataset,
    options: DatasetOptions,
) -> Dataset:
    if options.min_difficulty is None and options.max_difficulty is None:
        return dataset

    col = options.difficulty_column
    if col not in dataset.column_names:
        logger.warning(
            "Difficulty column '%s' not found in dataset. Skipping difficulty filter. "
            "Available columns: %s",
            col,
            dataset.column_names,
        )
        return dataset

    def in_range(sample: dict) -> bool:
        val = sample.get(col)
        if val is None:
            return True  # Keep samples without difficulty info
        if options.min_difficulty is not None and val < options.min_difficulty:
            return False
        if options.max_difficulty is not None and val > options.max_difficulty:
            return False
        return True

    filtered = dataset.filter(in_range)
    logger.info(
        "Difficulty filter (%s in [%s, %s]): %d -> %d samples",
        col,
        options.min_difficulty,
        options.max_difficulty,
        len(dataset),
        len(filtered),
    )
    return filtered


def _filter_by_source(
    dataset: Dataset,
    options: DatasetOptions,
) -> Dataset:
    """Filter dataset by source (e.g., primeintellect, lcbv5)."""
    if options.sources is None:
        return dataset

    allowed_sources = set(options.sources)

    def matches_source(sample: dict) -> bool:
        try:
            info = json.loads(sample.get("info", "{}"))
            source = info.get("source", "unknown")
            return source in allowed_sources
        except json.JSONDecodeError:
            return False

    filtered = dataset.filter(matches_source)
    logger.info(
        "Source filter (sources=%s): %d -> %d samples",
        options.sources,
        len(dataset),
        len(filtered),
    )
    return filtered


def load_datasets(
    dataset_names: List[str] | str | None = None,
    seed: int | None = None,
    **loader_kwargs: Any,
) -> List[Dict]:
    if loader_kwargs.get("seed") is None and seed is not None:
        loader_kwargs["seed"] = seed

    options = _normalize_options(loader_kwargs)

    raw_dataset = _load_raw_dataset(options)

    split_dataset = _split_dataset(raw_dataset, options)

    filtered_dataset = _filter_by_difficulty(split_dataset, options)

    # Filter by source if specified
    filtered_dataset = _filter_by_source(filtered_dataset, options)

    # Convert to our format
    samples: list[dict] = []
    for idx, sample in enumerate(filtered_dataset):
        record = _build_record(sample, idx)
        if record is not None:
            samples.append(record)

    logger.info("Built %d valid coding samples", len(samples))

    if options.seed is not None:
        rng = random.Random(options.seed)
        rng.shuffle(samples)

    if options.max_examples is not None and len(samples) > options.max_examples:
        samples = samples[: options.max_examples]
        logger.info("Limited to %d samples", len(samples))

    # Re-assign IDs after filtering/shuffling
    for idx, sample in enumerate(samples):
        sample["id"] = idx

    return samples


def load_problems(
    dataset_names: List[str] | str | None = None,
    **loader_kwargs: Any,
) -> List[Dict]:
    seed = loader_kwargs.pop("seed", None)
    return load_datasets(dataset_names, seed=seed, **loader_kwargs)
