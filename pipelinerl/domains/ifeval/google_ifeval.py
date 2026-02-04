from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import load_dataset

logger = logging.getLogger(__name__)

DOMAIN_NAME = "ifeval"  # Same domain for rollout routing
DATASET_ID = "google/IFEval"


@dataclass
class DatasetOptions:
    max_examples: int | None = None
    seed: int | None = None
    huggingface_token: str | None = None


def _normalize_options(loader_kwargs: Dict[str, Any]) -> DatasetOptions:
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

    return options


def _build_record(sample: dict, idx: int) -> dict | None:
    prompt = sample.get("prompt", "")
    if not prompt:
        return None

    instruction_ids = sample.get("instruction_id_list", [])
    kwargs_list = sample.get("kwargs", [])

    if not instruction_ids:
        return None

    reward_context = {
        "instruction_id_list": instruction_ids,
        "kwargs": kwargs_list,
    }

    return {
        "id": idx,
        "problem_id": f"google_ifeval_{sample.get('key', idx)}",
        "task": prompt,
        "reward_context": reward_context,
        "dataset": "google_ifeval",
        "domain": DOMAIN_NAME,
        "num_constraints": len(instruction_ids),
    }


def load_datasets(
    dataset_names: List[str] | str | None = None,
    seed: int | None = None,
    **loader_kwargs: Any,
) -> List[Dict]:
    if loader_kwargs.get("seed") is None and seed is not None:
        loader_kwargs["seed"] = seed

    options = _normalize_options(loader_kwargs)

    logger.info("Loading Google IFEval benchmark from %s...", DATASET_ID)
    try:
        ds = load_dataset(
            DATASET_ID,
            split="train",  # Google IFEval only has 'train' split
            token=options.huggingface_token,
        )
        logger.info("Loaded %d samples", len(ds))
    except Exception as e:
        logger.error("Failed to load Google IFEval: %s", e)
        return []

    samples: list[dict] = []
    for idx, sample in enumerate(ds):
        record = _build_record(sample, idx)
        if record is not None:
            samples.append(record)

    logger.info("Built %d valid samples", len(samples))

    if options.seed is not None:
        rng = random.Random(options.seed)
        rng.shuffle(samples)

    if options.max_examples is not None and len(samples) > options.max_examples:
        samples = samples[: options.max_examples]
        logger.info("Limited to %d samples", len(samples))

    for idx, sample in enumerate(samples):
        sample["id"] = idx

    return samples


def load_problems(
    dataset_names: List[str] | str | None = None,
    **loader_kwargs: Any,
) -> List[Dict]:
    seed = loader_kwargs.pop("seed", None)
    return load_datasets(dataset_names, seed=seed, **loader_kwargs)
