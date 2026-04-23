from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import Dataset, load_dataset
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

DOMAIN_NAME = "logic"
DEFAULT_DATASET_ID = "PrimeIntellect/INTELLECT-3-RL"
DEFAULT_DATASET_CONFIG = "logic"
DEFAULT_SPLIT = "train"
DEFAULT_DIFFICULTY_COLUMN = "avg@16_qwen3_4b_instruct_2507"
DEFAULT_MIN_DIFFICULTY = 0.0
DEFAULT_MAX_DIFFICULTY = 1.0
DEFAULT_TRAIN_RATIO = 0.9

# Tasks to skip (as recommended by i3-logic defaults)
DEFAULT_TASKS_TO_SKIP = ["arc_agi", "arc_agi_2", "buggy_tables"]

_DATASET_CACHE: dict[str, Dataset] = {}


@dataclass
class DatasetOptions:
    dataset_id: str = DEFAULT_DATASET_ID
    dataset_config: str = DEFAULT_DATASET_CONFIG
    split: str = DEFAULT_SPLIT
    subset: str = "train"
    train_ratio: float = DEFAULT_TRAIN_RATIO
    test_size: int | None = None  # Fixed test size (overrides train_ratio if set)
    max_examples: int | None = None
    min_difficulty: float | None = DEFAULT_MIN_DIFFICULTY
    max_difficulty: float | None = DEFAULT_MAX_DIFFICULTY
    difficulty_column: str = DEFAULT_DIFFICULTY_COLUMN
    tasks_to_skip: list[str] | None = None
    huggingface_token: str | None = None
    seed: int | None = None


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
    if "test_size" in loader_kwargs:
        val = loader_kwargs["test_size"]
        options.test_size = int(val) if val is not None else None
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
    if "tasks_to_skip" in loader_kwargs:
        val = loader_kwargs["tasks_to_skip"]
        options.tasks_to_skip = list(val) if val is not None else None
    if "huggingface_token" in loader_kwargs or "hf_token" in loader_kwargs:
        token = loader_kwargs.get("huggingface_token") or loader_kwargs.get("hf_token")
        options.huggingface_token = str(token) if token else None
    if "seed" in loader_kwargs:
        val = loader_kwargs["seed"]
        options.seed = int(val) if val is not None else None

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


def _parse_info(info_str: str | None) -> dict[str, Any] | None:
    if info_str is None:
        return None
    try:
        return json.loads(info_str)
    except json.JSONDecodeError:
        return None


def _build_record(sample: dict, idx: int, tasks_to_skip: set[str]) -> dict | None:
    prompt = sample.get("question")
    if not prompt:
        return None

    info = _parse_info(sample.get("info"))
    if info is None:
        return None

    task = info.get("task")
    if not task:
        return None

    # Skip certain tasks
    if task in tasks_to_skip:
        return None

    # Get game data for verification
    game_data = info.get("game_data_str") or info.get("game_data")
    if game_data is None:
        return None

    # Build reward_context in the format expected by verifier
    reward_context = {
        "task": task,
        "game_data": game_data,
    }

    return {
        "id": idx,
        "problem_id": f"logic_{idx}",
        "task": prompt,
        "reward_context": reward_context,
        "dataset": f"intellect3-rl-logic@{task}",
        "domain": DOMAIN_NAME,
        "logic_task": task,
    }


def _split_dataset(
    dataset: Dataset,
    options: DatasetOptions,
) -> Dataset:
    if options.subset not in ("train", "test"):
        logger.warning("Invalid subset '%s', defaulting to 'train'", options.subset)
        options.subset = "train"

    split_seed = 42
    total = len(dataset)
    indices = list(range(total))

    rng = random.Random(split_seed)
    rng.shuffle(indices)

    # Determine split point
    if options.test_size is not None and options.test_size > 0:
        # Use fixed test size
        test_count = min(options.test_size, total)
        train_end = total - test_count
    else:
        # Use ratio
        train_end = int(total * options.train_ratio)

    if options.subset == "train":
        selected_indices = indices[:train_end]
        logger.info(
            "Using train subset: %d samples (%.1f%% of %d)",
            len(selected_indices),
            len(selected_indices) / total * 100,
            total,
        )
    else:
        selected_indices = indices[train_end:]
        logger.info(
            "Using test subset: %d samples (%.1f%% of %d)",
            len(selected_indices),
            len(selected_indices) / total * 100,
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
            return True
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


def load_datasets(
    dataset_names: List[str] | str | None = None,
    seed: int | None = None,
    **loader_kwargs: Any,
) -> List[Dict]:
    if loader_kwargs.get("seed") is None and seed is not None:
        loader_kwargs["seed"] = seed

    options = _normalize_options(loader_kwargs)

    # Determine tasks to skip
    tasks_to_skip = set(options.tasks_to_skip or DEFAULT_TASKS_TO_SKIP)

    # Load raw dataset
    raw_dataset = _load_raw_dataset(options)

    # Split into train/test
    split_dataset = _split_dataset(raw_dataset, options)

    # Filter by difficulty
    filtered_dataset = _filter_by_difficulty(split_dataset, options)

    # Convert to our format
    samples: list[dict] = []
    for idx, sample in enumerate(filtered_dataset):
        record = _build_record(sample, idx, tasks_to_skip)
        if record is not None:
            samples.append(record)

    logger.info("Built %d valid logic samples", len(samples))

    # Shuffle if seed provided
    if options.seed is not None:
        rng = random.Random(options.seed)
        rng.shuffle(samples)

    # Limit samples if requested
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

    # Normalize dataset_names to list
    if dataset_names is None:
        names = []
    elif isinstance(dataset_names, str):
        names = [dataset_names]
    else:
        names = list(dataset_names)

    # Dispatch based on dataset name
    if "logic_test" in names:
        loader_kwargs["subset"] = "test"
    elif "logic" in names or "logic_train" in names:
        if "subset" not in loader_kwargs:
            loader_kwargs["subset"] = "train"

    return load_datasets(dataset_names, seed=seed, **loader_kwargs)
