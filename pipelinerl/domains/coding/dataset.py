"""Combined dataset loader for BAAI/TACO + codeparrot/apps."""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# Patterns to extract function names from prompts (e.g., GeeksForGeeks style)
_FN_NAME_PATTERNS = [
    re.compile(r"Complete the function\s+(\w+)\s*\(", re.IGNORECASE),
    re.compile(r"function\s+named\s+['\"]?(\w+)['\"]?", re.IGNORECASE),
    re.compile(r"implement\s+(?:the\s+)?function\s+['\"]?(\w+)['\"]?", re.IGNORECASE),
    re.compile(r"write\s+(?:a\s+)?function\s+['\"]?(\w+)['\"]?\s*\(", re.IGNORECASE),
]


def _extract_fn_name_from_prompt(prompt: str) -> str | None:
    """Try to extract function name from prompt text."""
    for pattern in _FN_NAME_PATTERNS:
        match = pattern.search(prompt)
        if match:
            return match.group(1)
    return None

DOMAIN_NAME = "coding"

# TACO difficulties to exclude by default
TACO_EXCLUDED_DIFFICULTIES = {"HARD", "VERY_HARD"}


@dataclass
class CombinedDatasetOptions:
    taco_split: str = "train"
    apps_split: str = "train"
    subset: str = "train"  # "train", "test", or "all"
    train_ratio: float = 0.9
    max_examples: int | None = None
    max_tests_per_problem: int = 50  # Limit test cases to avoid memory issues
    taco_excluded_difficulties: set[str] = field(default_factory=lambda: TACO_EXCLUDED_DIFFICULTIES.copy())
    huggingface_token: str | None = None
    seed: int | None = None
    skip_apps: bool = False  # Skip loading APPS dataset (has inconsistent I/O formats)


def _to_native(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _normalize_options(loader_kwargs: Dict[str, Any]) -> CombinedDatasetOptions:
    options = CombinedDatasetOptions()
    if not loader_kwargs:
        return options

    if "taco_split" in loader_kwargs:
        options.taco_split = str(loader_kwargs["taco_split"])
    if "apps_split" in loader_kwargs:
        options.apps_split = str(loader_kwargs["apps_split"])
    if "subset" in loader_kwargs:
        options.subset = str(loader_kwargs["subset"])
    if "train_ratio" in loader_kwargs:
        val = loader_kwargs["train_ratio"]
        options.train_ratio = float(val) if val is not None else 0.9
    if "max_examples" in loader_kwargs:
        val = loader_kwargs["max_examples"]
        options.max_examples = int(val) if val is not None else None
    if "max_tests_per_problem" in loader_kwargs:
        val = loader_kwargs["max_tests_per_problem"]
        options.max_tests_per_problem = int(val) if val is not None else 50
    if "taco_excluded_difficulties" in loader_kwargs:
        val = _to_native(loader_kwargs["taco_excluded_difficulties"])
        if val is not None:
            if isinstance(val, (list, tuple)):
                options.taco_excluded_difficulties = set(val)
            elif isinstance(val, set):
                options.taco_excluded_difficulties = val
            else:
                options.taco_excluded_difficulties = set()
        else:
            options.taco_excluded_difficulties = set()
    if "huggingface_token" in loader_kwargs or "hf_token" in loader_kwargs:
        token = loader_kwargs.get("huggingface_token") or loader_kwargs.get("hf_token")
        options.huggingface_token = str(token) if token else None
    if "seed" in loader_kwargs:
        val = loader_kwargs["seed"]
        options.seed = int(val) if val is not None else None
    if "skip_apps" in loader_kwargs:
        options.skip_apps = bool(loader_kwargs["skip_apps"])

    return options


def _parse_input_output(io_data: str | dict | None) -> dict | None:
    """Parse input_output field (can be JSON string or dict)."""
    if io_data is None:
        return None

    if isinstance(io_data, dict):
        return io_data

    if isinstance(io_data, str):
        try:
            return json.loads(io_data)
        except (json.JSONDecodeError, ValueError):
            # ValueError can occur for extremely large integers exceeding Python limits
            return None

    return None


def _build_taco_record(sample: dict, idx: int, max_tests: int = 50) -> dict | None:
    """Convert TACO sample to internal format."""
    prompt = sample.get("question")
    if not prompt:
        return None

    io_data = _parse_input_output(sample.get("input_output"))
    if io_data is None:
        return None

    inputs = io_data.get("inputs", [])
    outputs = io_data.get("outputs", [])
    fn_name = io_data.get("fn_name")

    # If fn_name not in data, try to extract from prompt (e.g., GeeksForGeeks style)
    if not fn_name:
        fn_name = _extract_fn_name_from_prompt(prompt)

    if not inputs or not outputs:
        return None

    # Limit test cases to avoid memory issues
    if max_tests and len(inputs) > max_tests:
        inputs = inputs[:max_tests]
        outputs = outputs[:max_tests]

    call_type = "fn" if fn_name else "std"
    reward_context = {"inputs": inputs, "outputs": outputs, "call_type": call_type}
    if fn_name:
        reward_context["fn_name"] = fn_name

    return {
        "id": idx,
        "problem_id": f"taco_{idx}",
        "task": prompt,
        "reward_context": reward_context,
        "dataset": f"taco@{sample.get('source', 'unknown')}",
        "domain": DOMAIN_NAME,
        "source": sample.get("source", "unknown"),
        "difficulty": sample.get("difficulty", "unknown"),
    }


def _build_apps_record(sample: dict, idx: int, max_tests: int = 50) -> dict | None:
    """Convert APPS sample to internal format."""
    prompt = sample.get("question")
    if not prompt:
        return None

    io_data = _parse_input_output(sample.get("input_output"))
    if io_data is None:
        return None

    inputs = io_data.get("inputs", [])
    outputs = io_data.get("outputs", [])
    fn_name = io_data.get("fn_name")

    # If fn_name not in data, try to extract from prompt
    if not fn_name:
        fn_name = _extract_fn_name_from_prompt(prompt)

    if not inputs or not outputs:
        return None

    # Limit test cases to avoid memory issues
    if max_tests and len(inputs) > max_tests:
        inputs = inputs[:max_tests]
        outputs = outputs[:max_tests]

    call_type = "fn" if fn_name else "std"
    reward_context = {"inputs": inputs, "outputs": outputs, "call_type": call_type}
    if fn_name:
        reward_context["fn_name"] = fn_name

    return {
        "id": idx,
        "problem_id": f"apps_{sample.get('problem_id', idx)}",
        "task": prompt,
        "reward_context": reward_context,
        "dataset": "apps",
        "domain": DOMAIN_NAME,
        "source": "apps",
        "difficulty": sample.get("difficulty", "unknown"),
    }


def _split_dataset(
    samples: List[Dict],
    options: CombinedDatasetOptions,
) -> List[Dict]:
    """Split samples into train/test subsets."""
    if options.subset not in ("train", "test", "all"):
        logger.warning("Invalid subset '%s', defaulting to 'train'", options.subset)
        options.subset = "train"

    if options.subset == "all":
        logger.info("Using all data: %d samples", len(samples))
        return samples

    split_seed = 42
    total = len(samples)
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

    return [samples[i] for i in selected_indices]


def load_datasets(
    dataset_names: List[str] | str | None = None,
    seed: int | None = None,
    **loader_kwargs: Any,
) -> List[Dict]:
    """Load combined TACO + APPS dataset.

    Args:
        dataset_names: Ignored (for API compatibility).
        seed: Random seed for shuffling.
        **loader_kwargs: Additional options (taco_split, apps_split, etc.).

    Returns:
        List of problem records with task, reward_context, etc.
    """
    if loader_kwargs.get("seed") is None and seed is not None:
        loader_kwargs["seed"] = seed

    options = _normalize_options(loader_kwargs)

    samples: List[Dict] = []

    # Load TACO from parquet files via hf:// URLs (dataset script no longer supported)
    logger.info("Loading TACO dataset (split=%s)...", options.taco_split)
    taco_data_url = f"hf://datasets/BAAI/TACO/ALL/{options.taco_split}-*.parquet"
    taco = load_dataset(
        "parquet",
        data_files=taco_data_url,
        split="train",  # data_files creates a "train" split by default
        token=options.huggingface_token,
    )
    taco_filtered = 0
    taco_invalid = 0
    for sample in taco:
        difficulty = sample.get("difficulty", "")
        if options.taco_excluded_difficulties and difficulty in options.taco_excluded_difficulties:
            taco_filtered += 1
            continue
        record = _build_taco_record(sample, len(samples), max_tests=options.max_tests_per_problem)
        if record:
            samples.append(record)
        else:
            taco_invalid += 1
    logger.info(
        "TACO: loaded %d samples (filtered %d by difficulty, %d invalid)",
        len(samples),
        taco_filtered,
        taco_invalid,
    )

    # Load APPS from jsonl files via hf:// URLs (dataset script no longer supported)
    if not options.skip_apps:
        taco_count = len(samples)
        logger.info("Loading APPS dataset (split=%s)...", options.apps_split)
        apps_data_url = f"hf://datasets/codeparrot/apps/{options.apps_split}.jsonl"
        apps = load_dataset(
            "json",
            data_files=apps_data_url,
            split="train",  # data_files creates a "train" split by default
            token=options.huggingface_token,
        )
        apps_invalid = 0
        for sample in apps:
            record = _build_apps_record(sample, len(samples), max_tests=options.max_tests_per_problem)
            if record:
                samples.append(record)
            else:
                apps_invalid += 1
        logger.info("APPS: loaded %d samples (%d invalid)", len(samples) - taco_count, apps_invalid)
    else:
        logger.info("Skipping APPS dataset (skip_apps=True)")

    # Split into train/test
    samples = _split_dataset(samples, options)

    # Shuffle if seed provided
    if options.seed is not None:
        rng = random.Random(options.seed)
        rng.shuffle(samples)

    # Limit examples if specified
    if options.max_examples is not None and len(samples) > options.max_examples:
        samples = samples[: options.max_examples]
        logger.info("Limited to %d samples", len(samples))

    # Re-assign IDs after filtering/shuffling
    for idx, sample in enumerate(samples):
        sample["id"] = idx

    logger.info("Final combined dataset: %d samples", len(samples))
    return samples


def load_problems(
    dataset_names: List[str] | str | None = None,
    **loader_kwargs: Any,
) -> List[Dict]:
    """Hydra entrypoint for loading problems.

    Dispatches to appropriate loader based on dataset_names:
    - "livecodebench*": Load LiveCodeBench evaluation benchmark
    - "taco", "taco@train", "taco@test", "taco@all": Load TACO only (skip APPS)
    - "coding", "coding@all", etc.: Load TACO + APPS training data
    """
    seed = loader_kwargs.pop("seed", None)

    # Normalize dataset_names to list
    if dataset_names is None:
        names = []
    elif isinstance(dataset_names, str):
        names = [dataset_names]
    else:
        names = list(dataset_names)

    # Dispatch to LiveCodeBench if any livecodebench variant is requested
    livecodebench_names = [n for n in names if n.startswith("livecodebench")]
    if livecodebench_names:
        from . import livecodebench
        return livecodebench.load_datasets(dataset_names, seed=seed, **loader_kwargs)

    # Check if "taco" is requested (skip APPS, use TACO only)
    # Handles: "taco", "taco@train", "taco@test", "taco@all"
    taco_only = any(n.startswith("taco") for n in names)
    if taco_only:
        loader_kwargs["skip_apps"] = True
        # Map taco subset to standard subset names
        for i, n in enumerate(names):
            if n == "taco":
                names[i] = "coding@all"
            elif n.startswith("taco@"):
                names[i] = "coding@" + n.split("@", 1)[1]

    return load_datasets(names, seed=seed, **loader_kwargs)
