from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

DOMAIN_NAME = "coding"

TACO_EXCLUDED_DIFFICULTIES = {"HARD", "VERY_HARD"}


@dataclass
class CombinedDatasetOptions:
    taco_split: str = "train"
    apps_split: str = "train"
    subset: str = "train"  # "train", "test", or "all"
    train_ratio: float = 0.9
    max_examples: int | None = None
    max_tests_per_problem: int = 50  # Limit test cases to avoid context/mem issues
    taco_excluded_difficulties: set[str] = field(default_factory=lambda: TACO_EXCLUDED_DIFFICULTIES.copy())
    huggingface_token: str | None = None
    seed: int | None = None
    skip_apps: bool = False


def _to_native(value: Any) -> Any:
    if OmegaConf.is_config(value):
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
    prompt = sample.get("question")
    if not prompt:
        return None

    io_data = _parse_input_output(sample.get("input_output"))
    if io_data is None:
        return None

    inputs = io_data.get("inputs", [])
    outputs = io_data.get("outputs", [])
    # meant for stdin simulation, not actual function arguments.
    fn_name = io_data.get("fn_name")

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
    prompt = sample.get("question")
    if not prompt:
        return None

    io_data = _parse_input_output(sample.get("input_output"))
    if io_data is None:
        return None

    inputs = io_data.get("inputs", [])
    outputs = io_data.get("outputs", [])
    # Only use fn_name if explicitly provided in io_data.
    fn_name = io_data.get("fn_name")

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
    if loader_kwargs.get("seed") is None and seed is not None:
        loader_kwargs["seed"] = seed

    options = _normalize_options(loader_kwargs)

    samples: List[Dict] = []

    # load from parquet files via hf:// URLs (dataset script no longer supported)
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

    if not options.skip_apps:
        taco_count = len(samples)
        logger.info("Loading APPS dataset (split=%s)...", options.apps_split)
        apps_data_url = f"hf://datasets/codeparrot/apps/{options.apps_split}.jsonl"
        apps = load_dataset(
            "json",
            data_files=apps_data_url,
            split="train",
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

    samples = _split_dataset(samples, options)

    if options.seed is not None:
        rng = random.Random(options.seed)
        rng.shuffle(samples)

    if options.max_examples is not None and len(samples) > options.max_examples:
        samples = samples[: options.max_examples]
        logger.info("Limited to %d samples", len(samples))

    # re-assign IDs after filtering/shuffling
    for idx, sample in enumerate(samples):
        sample["id"] = idx

    logger.info("Final combined dataset: %d samples", len(samples))
    return samples


def load_problems(
    dataset_names: List[str] | str | None = None,
    **loader_kwargs: Any,
) -> List[Dict]:
    seed = loader_kwargs.pop("seed", None)

    # normalize dataset_names
    if dataset_names is None:
        names = []
    elif isinstance(dataset_names, str):
        names = [dataset_names]
    else:
        names = list(dataset_names)

    livecodebench_names = [n for n in names if n.startswith("livecodebench")]
    if livecodebench_names:
        from . import livecodebench
        return livecodebench.load_datasets(dataset_names, seed=seed, **loader_kwargs)

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
