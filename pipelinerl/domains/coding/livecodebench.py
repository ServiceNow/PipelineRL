"""Dataset loader for LiveCodeBench evaluation benchmark."""

from __future__ import annotations

import base64
import json
import logging
import pickle
import random
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset

logger = logging.getLogger(__name__)

DOMAIN_NAME = "coding"
DEFAULT_DATASET_ID = "livecodebench/code_generation_lite"
DEFAULT_VERSION = "release_latest"
FORMATTED_LCB_PROMPT_PREFIX = "lcb_"

_LCB_VERSION_TO_FILES: dict[str, list[str]] = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
    "release_latest": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
    "v1": ["test.jsonl"],
    "v2": ["test2.jsonl"],
    "v3": ["test3.jsonl"],
    "v4": ["test4.jsonl"],
    "v5": ["test5.jsonl"],
    "v6": ["test6.jsonl"],
}


def _resolve_lcb_version_tag(version: str | None) -> str:
    if version is None:
        return "release_latest"
    tag = version.strip()
    return tag if tag else "release_latest"


def _find_local_lcb_snapshot() -> Path | None:
    repo_rel = "huggingface/hub/datasets--livecodebench--code_generation_lite/snapshots"
    candidates = [
        Path("/home/toolkit") / repo_rel,
        Path.home() / repo_rel,
    ]

    for root in candidates:
        if not root.exists():
            continue
        snapshot_dirs = [p for p in root.iterdir() if p.is_dir()]
        if not snapshot_dirs:
            continue
        return max(snapshot_dirs, key=lambda p: p.stat().st_mtime)
    return None


def _load_lcb_from_local_snapshot(version_tag: str) -> tuple[Any, Path] | None:
    file_list = _LCB_VERSION_TO_FILES.get(version_tag)
    if not file_list:
        return None

    snapshot_dir = _find_local_lcb_snapshot()
    if snapshot_dir is None:
        return None

    data_files = [snapshot_dir / name for name in file_list]
    if not all(path.exists() for path in data_files):
        return None

    ds = load_dataset(
        "json",
        data_files=[str(path) for path in data_files],
        split="train",
    )
    return ds, snapshot_dir


@dataclass
class LiveCodeBenchOptions:
    dataset_id: str = DEFAULT_DATASET_ID
    version: str | None = DEFAULT_VERSION
    max_examples: int | None = None
    max_tests_per_problem: int | None = None
    seed: int | None = None
    huggingface_token: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    formatted_prompts_json: str | None = None


def _normalize_options(loader_kwargs: Dict[str, Any]) -> LiveCodeBenchOptions:
    """Parse loader kwargs into LiveCodeBenchOptions."""
    options = LiveCodeBenchOptions()
    if not loader_kwargs:
        return options

    if "dataset_id" in loader_kwargs:
        options.dataset_id = str(loader_kwargs["dataset_id"])
    if "version" in loader_kwargs:
        val = loader_kwargs["version"]
        if val is None:
            options.version = None
        else:
            version = str(val).strip()
            options.version = None if version.lower() in {"", "none", "default"} else version
    if "max_examples" in loader_kwargs:
        val = loader_kwargs["max_examples"]
        options.max_examples = int(val) if val is not None else None
    if "max_tests_per_problem" in loader_kwargs:
        val = loader_kwargs["max_tests_per_problem"]
        if val is None:
            options.max_tests_per_problem = None
        else:
            parsed = int(val)
            options.max_tests_per_problem = parsed if parsed > 0 else None
    if "seed" in loader_kwargs:
        val = loader_kwargs["seed"]
        options.seed = int(val) if val is not None else None
    if "huggingface_token" in loader_kwargs or "hf_token" in loader_kwargs:
        token = loader_kwargs.get("huggingface_token") or loader_kwargs.get("hf_token")
        options.huggingface_token = str(token) if token else None
    if "start_date" in loader_kwargs:
        val = loader_kwargs["start_date"]
        if val is not None:
            options.start_date = str(val).strip() or None
    if "end_date" in loader_kwargs:
        val = loader_kwargs["end_date"]
        if val is not None:
            options.end_date = str(val).strip() or None
    if "formatted_prompts_json" in loader_kwargs:
        val = loader_kwargs["formatted_prompts_json"]
        if val is not None:
            options.formatted_prompts_json = str(val).strip() or None

    return options


def _normalize_contest_date(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if len(value) < 10:
        return None
    date_part = value[:10]
    if len(date_part) != 10 or date_part[4] != "-" or date_part[7] != "-":
        return None
    return date_part


def _sample_in_date_window(sample: dict, options: LiveCodeBenchOptions) -> bool:
    if options.start_date is None and options.end_date is None:
        return True

    contest_date = _normalize_contest_date(sample.get("contest_date"))
    if contest_date is None:
        raise ValueError("LiveCodeBench sample is missing contest_date required for date-window filtering")

    if options.start_date is not None and contest_date < options.start_date:
        return False
    if options.end_date is not None and contest_date > options.end_date:
        return False
    return True


def _parse_metadata(metadata: Any) -> Dict[str, Any]:
    if isinstance(metadata, dict):
        return metadata
    if not isinstance(metadata, str) or not metadata.strip():
        return {}
    try:
        parsed = json.loads(metadata)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _decode_test_cases_payload(test_cases_raw: Any) -> List[Dict[str, Any]]:
    if isinstance(test_cases_raw, list):
        return [case for case in test_cases_raw if isinstance(case, dict)]
    if not isinstance(test_cases_raw, str):
        return []
    payload = test_cases_raw.strip()
    if not payload:
        return []

    loaders = (
        lambda raw: json.loads(raw),
        lambda raw: json.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8")))),
        lambda raw: json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8"))))
        ),
    )
    for load_cases in loaders:
        try:
            cases = load_cases(payload)
        except Exception:
            continue
        if isinstance(cases, list):
            return [case for case in cases if isinstance(case, dict)]
    return []


def _resolve_formatted_prompts_path(path_str: str) -> Path:
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate

    repo_root = Path(__file__).resolve().parents[3]
    repo_candidate = repo_root / candidate
    if repo_candidate.exists():
        return repo_candidate

    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate

    raise FileNotFoundError(f"Formatted LiveCodeBench prompts JSON not found: {path_str}")


def _load_formatted_prompt_entries(path_str: str) -> list[dict]:
    path = _resolve_formatted_prompts_path(path_str)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    entries: Any = payload
    if isinstance(payload, dict):
        if isinstance(payload.get("LiveCodeBenchModified"), list):
            entries = payload["LiveCodeBenchModified"]
        else:
            list_values = [value for value in payload.values() if isinstance(value, list)]
            if len(list_values) == 1:
                entries = list_values[0]

    if not isinstance(entries, list):
        raise ValueError(f"Unexpected formatted LiveCodeBench prompt structure in {path}")

    return entries


def _formatted_entry_question_id(entry: dict) -> str | None:
    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        question_id = metadata.get("question_id")
        if isinstance(question_id, str) and question_id.strip():
            return question_id.strip()

    entry_id = entry.get("id")
    if isinstance(entry_id, str) and entry_id.strip():
        normalized = entry_id.strip()
        if normalized.startswith(FORMATTED_LCB_PROMPT_PREFIX):
            return normalized[len(FORMATTED_LCB_PROMPT_PREFIX):]
        return normalized

    return None


def _formatted_entry_prompt(entry: dict) -> str:
    messages = entry.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Formatted LiveCodeBench prompt entry is missing messages list")

    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content

    raise ValueError("Formatted LiveCodeBench prompt entry does not contain a non-empty user message")


def _formatted_entry_in_date_window(entry: dict, options: LiveCodeBenchOptions) -> bool:
    metadata = entry.get("metadata")
    if not isinstance(metadata, dict):
        return options.start_date is None and options.end_date is None
    return _sample_in_date_window(metadata, options)


def _merge_formatted_prompts(
    canonical_samples: list[dict],
    options: LiveCodeBenchOptions,
) -> list[dict]:
    if not options.formatted_prompts_json:
        return canonical_samples

    entries = _load_formatted_prompt_entries(options.formatted_prompts_json)
    canonical_by_id = {sample["problem_id"]: sample for sample in canonical_samples}

    merged_samples: list[dict] = []
    seen_question_ids: set[str] = set()
    missing_from_canonical: list[str] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if not _formatted_entry_in_date_window(entry, options):
            continue

        question_id = _formatted_entry_question_id(entry)
        if question_id is None:
            raise ValueError("Formatted LiveCodeBench prompt entry is missing question_id/id")

        canonical = canonical_by_id.get(question_id)
        if canonical is None:
            missing_from_canonical.append(question_id)
            continue

        merged_record = dict(canonical)
        merged_record["task"] = _formatted_entry_prompt(entry)
        merged_record["preformatted_prompt"] = True
        merged_record["prompt_source"] = "formatted_json"
        merged_samples.append(merged_record)
        seen_question_ids.add(question_id)

    extra_canonical = sorted(set(canonical_by_id) - seen_question_ids)
    if missing_from_canonical or extra_canonical:
        raise RuntimeError(
            "Formatted LiveCodeBench prompts do not align with canonical test cases. "
            f"Missing in canonical: {sorted(missing_from_canonical)[:10]} "
            f"(total={len(missing_from_canonical)}). "
            f"Missing in formatted prompts: {extra_canonical[:10]} "
            f"(total={len(extra_canonical)})."
        )

    logger.info(
        "Loaded %d formatted LiveCodeBench prompts from %s",
        len(merged_samples),
        options.formatted_prompts_json,
    )
    return merged_samples


def _build_record(sample: dict, idx: int, options: LiveCodeBenchOptions) -> dict | None:
    """Convert a LiveCodeBench sample to our internal format."""
    question_content = sample.get("question_content")
    if question_content:
        question_id = sample.get("question_id", f"lcb_{idx}")
        question_title = sample.get("question_title", "")
        platform = sample.get("platform", "unknown")
        difficulty = sample.get("difficulty", "unknown")
        starter_code = sample.get("starter_code", "")

        # Parse test cases - combine public and private
        public_tests = _decode_test_cases_payload(sample.get("public_test_cases"))
        private_tests = _decode_test_cases_payload(sample.get("private_test_cases"))
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

        metadata = _parse_metadata(sample.get("metadata"))
        fn_name_raw = metadata.get("func_name")
        fn_name = fn_name_raw.strip() if isinstance(fn_name_raw, str) else ""
        call_type = "fn" if fn_name else "std"

        # Build reward_context for verifier
        reward_context = {
            "inputs": inputs,
            "outputs": outputs,
            "call_type": call_type,
        }
        if fn_name:
            reward_context["fn_name"] = fn_name

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
            "contest_date": _normalize_contest_date(sample.get("contest_date")),
            "preformatted_prompt": False,
            "prompt_source": "canonical_dataset",
        }

    prompt = sample.get("prompt")
    verification_info = sample.get("verification_info")
    if not prompt or not verification_info:
        return None

    try:
        parsed = json.loads(verification_info) if isinstance(verification_info, str) else verification_info
    except Exception:
        return None

    if not isinstance(parsed, dict):
        return None

    reward_context = parsed.get("reward_context", parsed)
    if not isinstance(reward_context, dict):
        return None

    inputs = reward_context.get("inputs")
    outputs = reward_context.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or not inputs or not outputs:
        return None

    if options.max_tests_per_problem is not None:
        n = min(options.max_tests_per_problem, len(inputs), len(outputs))
        reward_context = dict(reward_context)
        reward_context["inputs"] = inputs[:n]
        reward_context["outputs"] = outputs[:n]

    if "call_type" not in reward_context:
        reward_context["call_type"] = "std"

    problem_id = sample.get("problem_id", f"lcb_{idx}")
    task_type = sample.get("task_type", "unknown")

    return {
        "id": idx,
        "problem_id": problem_id,
        "task": prompt,
        "reward_context": reward_context,
        "dataset": "livecodebench@primeintellect",
        "domain": DOMAIN_NAME,
        "platform": sample.get("platform", "unknown"),
        "difficulty": sample.get("difficulty", "unknown"),
        "question_title": sample.get("question_title", ""),
        "task_type": task_type,
        "preformatted_prompt": False,
        "prompt_source": "canonical_dataset",
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
    ds = None
    load_errors: list[str] = []
    dataset_id_normalized = options.dataset_id.strip().lower()
    version_tag = _resolve_lcb_version_tag(options.version)
    for split in ("test", "train"):
        try:
            if dataset_id_normalized == DEFAULT_DATASET_ID:
                load_kwargs = {
                    "split": split,
                    "token": options.huggingface_token,
                    "version_tag": version_tag,
                }
                ds = load_dataset(options.dataset_id, **load_kwargs)
            elif options.version is None:
                ds = load_dataset(
                    options.dataset_id,
                    split=split,
                    token=options.huggingface_token,
                )
            else:
                ds = load_dataset(
                    options.dataset_id,
                    options.version,
                    split=split,
                    token=options.huggingface_token,
                )
            logger.info("Loaded split '%s' for dataset_id=%s version=%s", split, options.dataset_id, options.version)
            break
        except TypeError as exc:
            # Backward compatibility path if datasets version does not support version_tag.
            if dataset_id_normalized == DEFAULT_DATASET_ID:
                try:
                    ds = load_dataset(
                        options.dataset_id,
                        version_tag,
                        split=split,
                        token=options.huggingface_token,
                    )
                    logger.info(
                        "Loaded split '%s' with config fallback for dataset_id=%s version=%s",
                        split,
                        options.dataset_id,
                        options.version,
                    )
                    break
                except Exception as fallback_exc:
                    load_errors.append(f"split={split}: version_tag TypeError={exc}; config fallback={fallback_exc}")
                    continue
            load_errors.append(f"split={split}: {exc}")
        except Exception as exc:
            load_errors.append(f"split={split}: {exc}")

    if ds is None and dataset_id_normalized == DEFAULT_DATASET_ID:
        local_result = _load_lcb_from_local_snapshot(version_tag)
        if local_result is not None:
            ds, snapshot_dir = local_result
            logger.warning(
                "Falling back to local LiveCodeBench snapshot for version=%s at %s",
                version_tag,
                snapshot_dir,
            )

    if ds is None:
        raise RuntimeError(
            f"Failed to load dataset_id={options.dataset_id} version={options.version}. Errors: {load_errors}"
        )

    logger.info("Loaded %d samples from LiveCodeBench", len(ds))

    # Convert to our format
    samples: list[dict] = []
    for idx, sample in enumerate(ds):
        if not _sample_in_date_window(sample, options):
            continue
        record = _build_record(sample, idx, options)
        if record is not None:
            samples.append(record)

    samples = _merge_formatted_prompts(samples, options)

    logger.info("Built %d valid LiveCodeBench samples", len(samples))
    if options.start_date is not None or options.end_date is not None:
        logger.info(
            "Applied LiveCodeBench date window start=%s end=%s (inclusive)",
            options.start_date,
            options.end_date,
        )

    # Preserve the formatted prompt ordering when an explicit prompt file is provided.
    if options.formatted_prompts_json:
        logger.info("Preserving formatted LiveCodeBench prompt order from %s", options.formatted_prompts_json)
    elif options.seed is not None:
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
