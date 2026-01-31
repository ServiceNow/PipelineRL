"""Dataset loader for IFEval instruction following domain.

Training data: allenai/IF_multi_constraints_upto5 (95k samples with up to 5 constraints)
Supports train/test splitting similar to coding domain.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import load_dataset

logger = logging.getLogger(__name__)

DOMAIN_NAME = "ifeval"
DATASET_ID = "allenai/IF_multi_constraints_upto5"

# Default test size similar to Google IFEval benchmark (541 samples)
DEFAULT_TEST_SIZE = 550
DEFAULT_TRAIN_RATIO = 0.994  # ~95k train, ~550 test

# Lazy-loaded tokenizer for prompt length filtering
_tokenizer = None


def _get_tokenizer():
    """Get tiktoken tokenizer for token counting."""
    global _tokenizer
    if _tokenizer is None:
        import tiktoken
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def _count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(_get_tokenizer().encode(text))


@dataclass
class DatasetOptions:
    max_examples: int | None = None
    seed: int | None = None
    huggingface_token: str | None = None
    # Limit number of constraints per sample
    max_constraints: int | None = None
    # Train/test split options
    subset: str = "train"  # "train", "test", or "all"
    train_ratio: float = DEFAULT_TRAIN_RATIO
    test_size: int | None = DEFAULT_TEST_SIZE  # Fixed test size (overrides train_ratio if set)
    # Filter out prompts exceeding this token count (to avoid context length errors)
    max_prompt_tokens: int | None = None


def _normalize_options(loader_kwargs: Dict[str, Any]) -> DatasetOptions:
    """Parse loader kwargs into DatasetOptions."""
    options = DatasetOptions()
    if not loader_kwargs:
        return options

    if "max_examples" in loader_kwargs:
        val = loader_kwargs["max_examples"]
        options.max_examples = int(val) if val is not None else None
    if "seed" in loader_kwargs:
        val = loader_kwargs["seed"]
        options.seed = int(val) if val is not None else None
    if "huggingface_token" in loader_kwargs or "hf_token" in loader_kwargs:
        token = loader_kwargs.get("huggingface_token") or loader_kwargs.get("hf_token")
        options.huggingface_token = str(token) if token else None
    if "max_constraints" in loader_kwargs:
        val = loader_kwargs["max_constraints"]
        options.max_constraints = int(val) if val is not None else None
    if "subset" in loader_kwargs:
        options.subset = str(loader_kwargs["subset"])
    if "train_ratio" in loader_kwargs:
        val = loader_kwargs["train_ratio"]
        options.train_ratio = float(val) if val is not None else DEFAULT_TRAIN_RATIO
    if "test_size" in loader_kwargs:
        val = loader_kwargs["test_size"]
        options.test_size = int(val) if val is not None else None
    if "max_prompt_tokens" in loader_kwargs:
        val = loader_kwargs["max_prompt_tokens"]
        options.max_prompt_tokens = int(val) if val is not None else None

    return options


def _parse_ground_truth(gt_str: str | None) -> tuple[list[str], list[dict | None]]:
    """Parse ground_truth JSON from AllenAI training data.

    Returns (instruction_ids, kwargs_list).
    """
    if not gt_str:
        return [], []

    try:
        gt = json.loads(gt_str.replace("'", '"'))
        if isinstance(gt, list) and len(gt) > 0:
            item = gt[0]
            instruction_ids = item.get("instruction_id", [])
            kwargs_list = item.get("kwargs", [])
            # Ensure lists are same length
            if len(kwargs_list) < len(instruction_ids):
                kwargs_list.extend([None] * (len(instruction_ids) - len(kwargs_list)))
            return instruction_ids, kwargs_list
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.debug(f"Failed to parse ground_truth: {e}")

    return [], []


def _build_record(sample: dict, idx: int, options: DatasetOptions) -> dict | None:
    """Convert an AllenAI IF training sample to our internal format."""
    messages = sample.get("messages", [])
    if not messages:
        return None

    # Get prompt from first message
    prompt = messages[0].get("content", "") if messages else ""
    if not prompt:
        return None

    # Parse verification info from ground_truth
    ground_truth = sample.get("ground_truth", "")
    instruction_ids, kwargs_list = _parse_ground_truth(ground_truth)

    if not instruction_ids:
        return None

    # Optionally limit number of constraints
    if options.max_constraints and len(instruction_ids) > options.max_constraints:
        instruction_ids = instruction_ids[:options.max_constraints]
        kwargs_list = kwargs_list[:options.max_constraints]

    # Build reward_context for verifier
    reward_context = {
        "instruction_id_list": instruction_ids,
        "kwargs": kwargs_list,
    }

    constraint = sample.get("constraint", "")
    constraint_type = sample.get("constraint_type", "single")

    return {
        "id": idx,
        "problem_id": sample.get("key", f"ifeval_{idx}"),
        "task": prompt,
        "reward_context": reward_context,
        "dataset": f"ifeval@{constraint_type}",
        "domain": DOMAIN_NAME,
        "constraint": constraint,
        "num_constraints": len(instruction_ids),
    }


def _split_dataset(
    samples: List[Dict],
    options: DatasetOptions,
) -> List[Dict]:
    """Split samples into train/test subsets."""
    if options.subset not in ("train", "test", "all"):
        logger.warning("Invalid subset '%s', defaulting to 'train'", options.subset)
        options.subset = "train"

    if options.subset == "all":
        logger.info("Using all data: %d samples", len(samples))
        return samples

    # Use fixed seed for reproducible splits
    split_seed = 42
    total = len(samples)
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
    else:  # test
        selected_indices = indices[train_end:]
        logger.info(
            "Using test subset: %d samples (%.1f%% of %d)",
            len(selected_indices),
            len(selected_indices) / total * 100,
            total,
        )

    return [samples[i] for i in selected_indices]


def load_datasets(
    dataset_names: List[str] | str | None = None,
    seed: int | None = None,
    **loader_kwargs: Any,
) -> List[Dict]:
    """Load IFEval training data from AllenAI IF_multi_constraints_upto5.

    Args:
        dataset_names: Ignored for compatibility.
        seed: Random seed for shuffling.
        **loader_kwargs: Additional options (max_examples, max_constraints, etc.)

    Returns:
        List of problem dictionaries with keys:
        - id: int
        - problem_id: str
        - task: str (the prompt with constraint instructions)
        - reward_context: dict with instruction_id_list and kwargs
        - dataset: str
        - domain: str
    """
    if loader_kwargs.get("seed") is None and seed is not None:
        loader_kwargs["seed"] = seed

    options = _normalize_options(loader_kwargs)

    logger.info("Loading IFEval training data from %s...", DATASET_ID)
    try:
        ds = load_dataset(
            DATASET_ID,
            split="train",
            token=options.huggingface_token,
        )
        logger.info("Loaded %d samples", len(ds))
    except Exception as e:
        logger.error("Failed to load training data: %s", e)
        return []

    samples: list[dict] = []
    filtered_long = 0
    for idx, sample in enumerate(ds):
        record = _build_record(sample, idx, options)
        if record is not None:
            # Filter out prompts that are too long
            if options.max_prompt_tokens is not None:
                token_count = _count_tokens(record["task"])
                if token_count > options.max_prompt_tokens:
                    filtered_long += 1
                    continue
            samples.append(record)

    logger.info("Built %d valid samples", len(samples))
    if filtered_long > 0:
        logger.info("Filtered out %d samples exceeding %d tokens", filtered_long, options.max_prompt_tokens)

    # Split into train/test subsets
    samples = _split_dataset(samples, options)

    # Shuffle if seed provided (after split to maintain split consistency)
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
    """Hydra entrypoint that mirrors other domain loader style.

    Dispatches based on dataset_names:
    - "ifeval_test": Load test subset from AllenAI data (~550 samples)
    - "ifeval" or "ifeval_train": Load train subset from AllenAI data (~94k samples)
    - "google_ifeval": Load Google IFEval benchmark (541 held-out eval samples)
    """
    seed = loader_kwargs.pop("seed", None)

    # Normalize dataset_names to list
    if dataset_names is None:
        names = []
    elif isinstance(dataset_names, str):
        names = [dataset_names]
    else:
        names = list(dataset_names)

    # Dispatch based on dataset name
    if "google_ifeval" in names:
        from . import google_ifeval
        return google_ifeval.load_datasets(dataset_names, seed=seed, **loader_kwargs)

    # Check for test subset request
    if "ifeval_test" in names:
        loader_kwargs["subset"] = "test"
    elif "ifeval" in names or "ifeval_train" in names:
        # Default to train subset
        if "subset" not in loader_kwargs:
            loader_kwargs["subset"] = "train"

    return load_datasets(dataset_names, seed=seed, **loader_kwargs)
