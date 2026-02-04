"""Dataset loader for LiveCodeBench evaluation benchmark."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import load_dataset

logger = logging.getLogger(__name__)

DOMAIN_NAME = "coding"
DEFAULT_DATASET_ID = "livecodebench/code_generation_lite"
DEFAULT_VERSION = "release_latest"


@dataclass
class LiveCodeBenchOptions:
    dataset_id: str = DEFAULT_DATASET_ID
    version: str = DEFAULT_VERSION
    max_examples: int | None = None
    max_tests_per_problem: int | None = None
    seed: int | None = None
    huggingface_token: str | None = None


def _normalize_options(loader_kwargs: Dict[str, Any]) -> LiveCodeBenchOptions:
    """Parse loader kwargs into LiveCodeBenchOptions."""
    options = LiveCodeBenchOptions()
    if not loader_kwargs:
        return options

    if "dataset_id" in loader_kwargs:
        options.dataset_id = str(loader_kwargs["dataset_id"])
    if "version" in loader_kwargs:
        options.version = str(loader_kwargs["version"])
    if "max_examples" in loader_kwargs:
        val = loader_kwargs["max_examples"]
        options.max_examples = int(val) if val is not None else None
    if "max_tests_per_problem" in loader_kwargs:
        val = loader_kwargs["max_tests_per_problem"]
        options.max_tests_per_problem = int(val) if val is not None else None
    if "seed" in loader_kwargs:
        val = loader_kwargs["seed"]
        options.seed = int(val) if val is not None else None
    if "huggingface_token" in loader_kwargs or "hf_token" in loader_kwargs:
        token = loader_kwargs.get("huggingface_token") or loader_kwargs.get("hf_token")
        options.huggingface_token = str(token) if token else None

    return options


def _parse_test_cases(test_cases_str: str | None) -> List[Dict[str, str]]:
    """Parse test cases JSON string into list of input/output dicts."""
    if not test_cases_str:
        return []
    try:
        cases = json.loads(test_cases_str)
        if isinstance(cases, list):
            return cases
        return []
    except json.JSONDecodeError:
        return []


def _build_record(sample: dict, idx: int, options: LiveCodeBenchOptions) -> dict | None:
    """Convert a LiveCodeBench sample to our internal format."""
    question_content = sample.get("question_content")
    if not question_content:
        return None

    question_id = sample.get("question_id", f"lcb_{idx}")
    question_title = sample.get("question_title", "")
    platform = sample.get("platform", "unknown")
    difficulty = sample.get("difficulty", "unknown")
    starter_code = sample.get("starter_code", "")

    # Parse test cases - combine public and private
    public_tests = _parse_test_cases(sample.get("public_test_cases"))
    private_tests = _parse_test_cases(sample.get("private_test_cases"))
    all_tests = public_tests + private_tests

    if not all_tests:
        logger.debug(f"Skipping problem {question_id}: no test cases")
        return None

    # Limit test cases if specified
    if options.max_tests_per_problem and len(all_tests) > options.max_tests_per_problem:
        # Keep first few public tests, then sample from private
        n_public = min(len(public_tests), options.max_tests_per_problem // 2)
        n_private = options.max_tests_per_problem - n_public
        selected_tests = public_tests[:n_public]
        if n_private > 0 and private_tests:
            rng = random.Random(42 + idx)
            selected_private = rng.sample(private_tests, min(n_private, len(private_tests)))
            selected_tests.extend(selected_private)
        all_tests = selected_tests

    # Extract inputs and outputs
    inputs = [t.get("input", "") for t in all_tests]
    outputs = [t.get("output", "") for t in all_tests]

    # Build prompt with starter code if available
    prompt = question_content
    if starter_code and starter_code.strip():
        prompt += f"\n\nStarter code:\n```python\n{starter_code}\n```"

    # Build reward_context for verifier
    reward_context = {
        "inputs": inputs,
        "outputs": outputs,
        "call_type": "std",  # LiveCodeBench uses stdin/stdout
    }

    return {
        "id": idx,
        "problem_id": question_id,
        "task": prompt,
        "reward_context": reward_context,
        "dataset": f"livecodebench@{platform}",
        "domain": DOMAIN_NAME,
        "platform": platform,
        "difficulty": difficulty,
        "question_title": question_title,
    }


def load_datasets(
    dataset_names: List[str] | str | None = None,
    seed: int | None = None,
    **loader_kwargs: Any,
) -> List[Dict]:
    """Load LiveCodeBench problems for evaluation.

    Args:
        dataset_names: Ignored for compatibility.
        seed: Random seed for shuffling.
        **loader_kwargs: Additional options (version, max_examples, etc.)

    Returns:
        List of problem dictionaries with keys:
        - id: int
        - problem_id: str
        - task: str (the problem prompt)
        - reward_context: dict with inputs, outputs, call_type
        - dataset: str
        - domain: str
        - platform: str
        - difficulty: str
    """
    if loader_kwargs.get("seed") is None and seed is not None:
        loader_kwargs["seed"] = seed

    options = _normalize_options(loader_kwargs)

    logger.info(
        "Loading LiveCodeBench (version=%s)...",
        options.version,
    )

    # Load from HuggingFace
    ds = load_dataset(
        options.dataset_id,
        options.version,
        split="test",
        token=options.huggingface_token,
        trust_remote_code=True,
    )

    logger.info("Loaded %d samples from LiveCodeBench", len(ds))

    # Convert to our format
    samples: list[dict] = []
    for idx, sample in enumerate(ds):
        record = _build_record(sample, idx, options)
        if record is not None:
            samples.append(record)

    logger.info("Built %d valid LiveCodeBench samples", len(samples))

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
