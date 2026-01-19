"""Dataset loader for IFEval instruction following domain.

Training data: allenai/IF_multi_constraints_upto5 (95k samples with up to 5 constraints)
This loader is for TRAINING only. For eval, use google_ifeval domain.
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


@dataclass
class DatasetOptions:
    max_examples: int | None = None
    seed: int | None = None
    huggingface_token: str | None = None
    # Limit number of constraints per sample
    max_constraints: int | None = None


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
    for idx, sample in enumerate(ds):
        record = _build_record(sample, idx, options)
        if record is not None:
            samples.append(record)

    logger.info("Built %d valid samples", len(samples))

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
    """Hydra entrypoint that mirrors other domain loader style."""
    seed = loader_kwargs.pop("seed", None)
    return load_datasets(dataset_names, seed=seed, **loader_kwargs)
