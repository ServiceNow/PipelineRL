"""Dataset loader for BFCL v3 function calling domain."""

from __future__ import annotations

import hashlib
import logging
import random
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

DOMAIN_NAME = "fn_calling"

# BFCL v3 test categories (single-turn only for now)
SINGLE_TURN_CATEGORIES = [
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "java",
    "javascript",
    "relevance",
    "irrelevance",
]

# Default categories to load (excludes multi-turn for simpler training)
DEFAULT_CATEGORIES = ["simple", "multiple", "parallel", "parallel_multiple"]


def _lazy_import_bfcl():
    """Lazily import BFCL modules to avoid import overhead."""
    try:
        from bfcl_eval.utils import (
            load_dataset_entry,
            load_ground_truth_entry,
            is_relevance_or_irrelevance,
        )
        from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
        from bfcl_eval.model_handler.utils import convert_to_tool
        from bfcl_eval.constants.enums import ModelStyle
        return {
            "load_dataset_entry": load_dataset_entry,
            "load_ground_truth_entry": load_ground_truth_entry,
            "is_relevance_or_irrelevance": is_relevance_or_irrelevance,
            "GORILLA_TO_OPENAPI": GORILLA_TO_OPENAPI,
            "convert_to_tool": convert_to_tool,
            "ModelStyle": ModelStyle,
        }
    except ImportError as e:
        logger.error(f"Failed to import bfcl_eval: {e}")
        raise ImportError(
            "bfcl_eval package required for fn_calling domain. "
            "Install with: pip install bfcl-eval"
        ) from e


def _load_category(
    test_category: str,
    max_examples: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load samples from a single BFCL test category."""
    bfcl = _lazy_import_bfcl()

    # Load dataset entries (prompts + function definitions)
    dataset_entries = bfcl["load_dataset_entry"](test_category, include_language_specific_hint=False)
    dataset_entries_with_hints = bfcl["load_dataset_entry"](test_category, include_language_specific_hint=True)

    # Load ground truth for non-relevance tests
    is_relevance = bfcl["is_relevance_or_irrelevance"](test_category)
    if not is_relevance:
        ground_truth_entries = bfcl["load_ground_truth_entry"](test_category)
    else:
        ground_truth_entries = [None] * len(dataset_entries)

    assert len(dataset_entries_with_hints) == len(dataset_entries) == len(ground_truth_entries)

    samples = []
    for i, (entry, entry_with_hints, ground_truth) in enumerate(
        zip(dataset_entries, dataset_entries_with_hints, ground_truth_entries)
    ):
        # Convert function definitions to OpenAI tool format
        functions_with_hints = entry_with_hints["function"]
        oai_tools = bfcl["convert_to_tool"](
            functions_with_hints,
            bfcl["GORILLA_TO_OPENAPI"],
            bfcl["ModelStyle"].OPENAI_COMPLETIONS,
        )

        # Build the prompt (BFCL uses a list of questions for multi-turn)
        question = entry["question"]
        if isinstance(question, list):
            prompt_text = question[0] if question else ""
        else:
            prompt_text = str(question)

        sample = {
            "task": prompt_text,
            "domain": DOMAIN_NAME,
            "dataset": f"{DOMAIN_NAME}@{test_category}",
            "reward_context": {
                "category": test_category,
                "function": entry["function"],  # Original function defs (no hints)
                "ground_truth": ground_truth,
                "is_relevance": is_relevance,
            },
            "extra_info": {
                "id": entry.get("id", f"{test_category}_{i}"),
                "oai_tools": oai_tools,
                "function_with_hints": functions_with_hints,
            },
        }
        samples.append(sample)

    # Apply max_examples limit
    if max_examples is not None and max_examples > 0:
        samples = samples[:max_examples]

    logger.info(f"Loaded {len(samples)} samples from BFCL category '{test_category}'")
    return samples


def _split_samples(
    samples: List[Dict[str, Any]],
    subset: Literal["train", "test"],
    train_ratio: float,
    seed: int,
) -> List[Dict[str, Any]]:
    """Split samples into train/test based on deterministic hashing.

    Uses a hash of the sample ID to ensure consistent splits across runs.
    """
    if train_ratio <= 0.0:
        return [] if subset == "train" else samples
    if train_ratio >= 1.0:
        return samples if subset == "train" else []

    result = []
    for sample in samples:
        # Use sample ID for deterministic split
        sample_id = sample.get("extra_info", {}).get("id", "")
        hash_input = f"{sample_id}:{seed}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        # Normalize to [0, 1)
        normalized = (hash_value % 10000) / 10000.0

        is_train = normalized < train_ratio
        if (subset == "train" and is_train) or (subset == "test" and not is_train):
            result.append(sample)

    return result


def load_datasets(
    dataset_names: List[str] | str | None = None,
    seed: int = 42,
    categories: Optional[List[str]] = None,
    max_examples_per_category: Optional[int] = None,
    max_examples: Optional[int] = None,
    subset: Literal["train", "test"] = "train",
    train_ratio: float = 0.9,
    **loader_kwargs: Any,
) -> List[Dict[str, Any]]:
    """Load BFCL v3 function calling datasets.

    Args:
        dataset_names: Ignored (for API compatibility).
        seed: Random seed for shuffling and splitting.
        categories: List of BFCL categories to load. Defaults to DEFAULT_CATEGORIES.
        max_examples_per_category: Maximum samples per category.
        max_examples: Maximum total samples.
        subset: Which split to return ("train" or "test").
        train_ratio: Fraction of data for training (default 0.9).
        **loader_kwargs: Additional kwargs (ignored for compatibility).

    Returns:
        List of problem dictionaries.
    """
    if categories is None:
        categories = DEFAULT_CATEGORIES

    all_samples = []
    for category in categories:
        if category not in SINGLE_TURN_CATEGORIES:
            logger.warning(f"Unknown or unsupported BFCL category '{category}', skipping")
            continue

        try:
            samples = _load_category(
                category,
                max_examples=max_examples_per_category,
                seed=seed,
            )
            all_samples.extend(samples)
        except Exception as e:
            logger.error(f"Failed to load category '{category}': {e}")
            continue

    # Apply train/test split
    all_samples = _split_samples(all_samples, subset, train_ratio, seed)

    # Shuffle with seed for reproducibility
    rng = random.Random(seed)
    rng.shuffle(all_samples)

    # Apply total max_examples limit
    if max_examples is not None and max_examples > 0:
        all_samples = all_samples[:max_examples]

    if not all_samples:
        logger.warning(f"fn_calling loader returned zero samples for subset='{subset}'")
    else:
        logger.info(f"Loaded {len(all_samples)} BFCL samples for subset='{subset}'")

    return all_samples


def load_problems(
    dataset_names: List[str] | str | None = None,
    **loader_kwargs: Any,
) -> List[Dict[str, Any]]:
    """Load problems (alias for load_datasets for API compatibility)."""
    return load_datasets(dataset_names, **loader_kwargs)
