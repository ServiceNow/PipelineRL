import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from datasets import load_dataset

logger = logging.getLogger(__name__)

DOMAIN = "coding"
DEFAULT_DATASET_ID = "livecodebench/code_generation"
DEFAULT_LANGUAGE = "python"
DEFAULT_MAX_PUBLIC_TESTS = 10
DEFAULT_APPS_DATASET_ID = "codeparrot/apps"
DEFAULT_CODECONTESTS_DATASET_ID = "deepmind/code_contests"
DEFAULT_DS1000_DATASET_ID = "xlangai/DS-1000"

ENTRY_POINT_PATTERN = re.compile(
    r"class\s+Solution\s*:(?:.|\n)*?def\s+(?P<method>[a-zA-Z_][\w]*)\s*\(",
    re.MULTILINE,
)
FUNCTION_PATTERN = re.compile(r"def\s+(?P<func>[a-zA-Z_][\w]*)\s*\(")

DEFAULT_CODECONTESTS_BUCKETS = {
    "lt1800": {"max_rating": 1800},
    "1800_2200": {"min_rating": 1800, "max_rating": 2200},
    "gt2200": {"min_rating": 2200},
}


@dataclass
class DatasetSelector:
    raw: str
    dataset_key: str
    config_name: str | None
    split_name: str
    alias: str


def _infer_alias(dataset_key: str) -> str:
    key = (dataset_key or "livecodebench").split("/", 1)[0]
    return key.lower() or "livecodebench"


def _parse_selector(raw: str, *, default_split: str) -> DatasetSelector:
    dataset_key = ""
    config_name: str | None = None
    split_name = default_split

    selector = raw.strip()
    if selector:
        dataset_part = selector
        split_part: str | None = None
        if "@" in selector:
            dataset_part, split_part = selector.split("@", 1)
        dataset_part = dataset_part.strip()
        if dataset_part:
            if "#" in dataset_part:
                dataset_hint, config_part = dataset_part.split("#", 1)
                dataset_hint = dataset_hint.strip()
                config_part = config_part.strip()
                if dataset_hint:
                    dataset_key = dataset_hint
                if config_part:
                    config_name = config_part
            else:
                dataset_key = dataset_part
        if split_part is None:
            split_part = ""
        split_part = split_part.strip()
        if split_part:
            if "/" in split_part:
                config_candidate, split_candidate = split_part.split("/", 1)
                config_candidate = config_candidate.strip()
                split_candidate = split_candidate.strip()
                if config_candidate:
                    config_name = config_candidate if config_name is None else config_name
                if split_candidate:
                    split_name = split_candidate
            else:
                split_name = split_part

    if not dataset_key:
        dataset_key = DEFAULT_DATASET_ID
    split_name = split_name or default_split
    return DatasetSelector(
        raw=selector or default_split,
        dataset_key=dataset_key,
        config_name=config_name,
        split_name=split_name,
        alias=_infer_alias(dataset_key),
    )


def _ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _guess_entry_point(starter_code: str | None) -> str | None:
    if not starter_code:
        return None
    class_match = ENTRY_POINT_PATTERN.search(starter_code)
    if class_match:
        method = class_match.group("method")
        if method != "__init__":
            return method
    func_match = FUNCTION_PATTERN.search(starter_code)
    if func_match:
        candidate = func_match.group("func")
        if candidate != "__init__":
            return candidate
    return None


def _parse_test_blob(blob: str | None, *, max_tests: int | None) -> list[dict]:
    if not blob:
        return []
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        logger.warning("Failed to decode test blob")
        return []
    if isinstance(parsed, dict) and "tests" in parsed:
        parsed = parsed["tests"]
    if not isinstance(parsed, list):
        return []

    cases: list[dict] = []
    for idx, raw in enumerate(parsed):
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = {"input": raw}
        if not isinstance(raw, dict):
            continue
        test_type = str(raw.get("testtype", raw.get("type", "functional"))).lower()
        if test_type not in {"functional", "stdin"}:
            continue
        input_text = raw.get("input", "")
        output_text = raw.get("output", "")
        case = {
            "id": idx,
            "type": test_type,
            "input": _ensure_text(input_text),
            "output": _ensure_text(output_text),
            "timeout": raw.get("timeout"),
        }
        cases.append(case)
        if max_tests is not None and len(cases) >= max_tests:
            break
    return cases


def _build_io_tests(
    pairs: Iterable[Any] | None,
    *,
    max_tests: int | None,
    start_id: int = 0,
    case_type: str = "functional",
) -> list[dict]:
    if not pairs:
        return []
    cases: list[dict] = []
    for idx, raw in enumerate(pairs):
        case_id = start_id + idx
        raw_case: dict[str, Any]
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                raw_case = parsed if isinstance(parsed, dict) else {"input": raw}
            except json.JSONDecodeError:
                raw_case = {"input": raw}
        elif isinstance(raw, dict):
            raw_case = raw
        else:
            raw_case = {"input": raw}
        resolved_type = str(raw_case.get("type", case_type or "functional")).lower()
        input_value = raw_case.get("input", raw_case.get("inputs", raw_case.get("stdin", "")))
        output_value = raw_case.get("output", raw_case.get("outputs", raw_case.get("stdout", "")))
        case = {
            "id": case_id,
            "type": resolved_type,
            "input": _ensure_text(input_value),
            "output": _ensure_text(output_value),
        }
        timeout = raw_case.get("timeout")
        if timeout is not None:
            case["timeout"] = timeout
        cases.append(case)
        if max_tests is not None and len(cases) >= max_tests:
            break
    return cases


def _filter_language(sample_language: str | None, desired: str | None) -> bool:
    if desired is None or not desired:
        return True
    return (sample_language or "").lower() == desired.lower()


def _load_livecodebench_split(
    dataset_id: str,
    split_name: str,
    *,
    config_name: str | None,
    language: str | None,
    max_public_tests: int | None,
    include_private_tests: bool,
    max_examples: int | None,
    trust_remote_code: bool,
) -> list[dict]:
    dataset = load_dataset(
        dataset_id,
        name=config_name,
        split=split_name,
        trust_remote_code=trust_remote_code,
    )
    problems: list[dict] = []

    for idx, sample in enumerate(dataset):
        if max_examples is not None and idx >= max_examples:
            break
        if not _filter_language(sample.get("language"), language):
            continue

        public_tests = _parse_test_blob(
            sample.get("public_test_cases"),
            max_tests=max_public_tests,
        )
        private_tests: list[dict] = []
        if include_private_tests:
            private_tests = _parse_test_blob(sample.get("private_test_cases"), max_tests=None)

        tests = public_tests + private_tests
        if not tests:
            continue

        starter = sample.get("starter_code") or sample.get("prompt")
        entry_point = sample.get("entry_point") or _guess_entry_point(starter)
        dataset_label = split_name
        if config_name:
            dataset_label = f"{config_name}/{split_name}"

        problem = {
            "dataset": dataset_label,
            "domain": DOMAIN,
            "id": sample.get("problem_id") or sample.get("question_id") or f"{split_name}-{idx}",
            "problem_id": sample.get("problem_id") or sample.get("question_id") or f"{split_name}-{idx}",
            "source": sample.get("source"),
            "language": sample.get("language"),
            "title": sample.get("question_title"),
            "question": sample.get("question_content") or sample.get("prompt"),
            "difficulty": sample.get("difficulty"),
            "starter_code": starter,
            "entry_point": entry_point,
            "tests": tests,
            "release": config_name,
        }
        problems.append(problem)
    return problems


def _load_livecodebench(selector: DatasetSelector, cfg: Dict[str, Any], default_split: str) -> list[dict]:
    dataset_id = cfg.get("dataset_id") or DEFAULT_DATASET_ID
    if selector.alias == "livecodebench" and "/" in selector.dataset_key:
        dataset_id = selector.dataset_key
    split_name = selector.split_name or cfg.get("split", default_split)
    config_name = selector.config_name
    return _load_livecodebench_split(
        dataset_id,
        split_name,
        config_name=config_name,
        language=cfg.get("language"),
        max_public_tests=cfg.get("max_public_tests", DEFAULT_MAX_PUBLIC_TESTS),
        include_private_tests=cfg.get("include_private_tests", False),
        max_examples=cfg.get("max_examples_per_split"),
        trust_remote_code=cfg.get("trust_remote_code", False),
    )


def _load_apps(selector: DatasetSelector, cfg: Dict[str, Any], default_split: str) -> list[dict]:
    dataset_id = cfg.get("dataset_id") or DEFAULT_APPS_DATASET_ID
    split_name = selector.split_name or cfg.get("split", "train") or default_split
    difficulty = selector.config_name or cfg.get("difficulty")
    trust_remote_code = cfg.get("trust_remote_code", False)
    dataset = load_dataset(dataset_id, split=split_name, trust_remote_code=trust_remote_code)

    max_examples = cfg.get("max_examples_per_split")
    max_tests = cfg.get("max_public_tests", DEFAULT_MAX_PUBLIC_TESTS)
    problems: list[dict] = []
    for idx, sample in enumerate(dataset):
        if max_examples is not None and len(problems) >= max_examples:
            break
        sample_difficulty = str(sample.get("difficulty", "")).lower()
        if difficulty and sample_difficulty != difficulty.lower():
            continue
        tests = _build_io_tests(sample.get("input_output"), max_tests=max_tests)
        if not tests:
            continue
        starter = sample.get("starter_code") or sample.get("prompt")
        entry_point = sample.get("entry_point") or _guess_entry_point(starter)
        dataset_label = f"apps/{difficulty or 'all'}/{split_name}"
        problem_id = sample.get("problem_id") or sample.get("question_id") or f"apps-{split_name}-{idx}"
        problems.append(
            {
                "dataset": dataset_label,
                "domain": DOMAIN,
                "id": problem_id,
                "problem_id": problem_id,
                "source": "apps",
                "language": "python",
                "title": sample.get("question_id"),
                "question": sample.get("question") or sample.get("description"),
                "difficulty": sample.get("difficulty"),
                "starter_code": starter,
                "entry_point": entry_point,
                "tests": tests,
            }
        )
    return problems


def _maybe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _maybe_int(value: Any, default: int | None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _match_codecontests_bucket(sample: Dict[str, Any], bucket_cfg: Dict[str, Any]) -> bool:
    if not bucket_cfg:
        return True
    rating = sample.get("difficulty_rating") or sample.get("rating")
    if rating is not None:
        rating = _maybe_int(rating, None) if not isinstance(rating, int) else rating
        if rating is not None:
            min_rating = bucket_cfg.get("min_rating")
            max_rating = bucket_cfg.get("max_rating")
            if min_rating is not None and rating < min_rating:
                return False
            if max_rating is not None and rating >= max_rating:
                return False
    label = str(sample.get("difficulty", sample.get("category", ""))).lower()
    allowed_labels = bucket_cfg.get("labels")
    if allowed_labels and label not in {str(item).lower() for item in allowed_labels}:
        return False
    return True


def _load_codecontests(selector: DatasetSelector, cfg: Dict[str, Any], default_split: str) -> list[dict]:
    dataset_id = cfg.get("dataset_id") or DEFAULT_CODECONTESTS_DATASET_ID
    split_name = selector.split_name or cfg.get("split", "train") or default_split
    buckets = cfg.get("difficulty_buckets") or DEFAULT_CODECONTESTS_BUCKETS
    bucket_key = selector.config_name
    bucket_cfg = buckets.get(bucket_key, {}) if bucket_key else {}
    dataset = load_dataset(dataset_id, split=split_name, trust_remote_code=cfg.get("trust_remote_code", False))

    include_private = cfg.get("include_private_tests", False)
    max_examples = cfg.get("max_examples_per_split")
    max_tests = cfg.get("max_public_tests", DEFAULT_MAX_PUBLIC_TESTS)
    problems: list[dict] = []
    for sample in dataset:
        if max_examples is not None and len(problems) >= max_examples:
            break
        if bucket_cfg and not _match_codecontests_bucket(sample, bucket_cfg):
            continue
        public_tests = _build_io_tests(sample.get("public_tests"), max_tests=max_tests, case_type="stdin")
        private_tests: list[dict] = []
        if include_private:
            private_tests = _build_io_tests(
                sample.get("private_tests"),
                max_tests=None,
                case_type="stdin",
                start_id=len(public_tests),
            )
        tests = public_tests + private_tests
        if not tests:
            continue
        time_limit_ms = sample.get("time_limit") or sample.get("time_limit_ms")
        memory_limit = sample.get("memory_limit") or sample.get("memory_limit_mb")
        dataset_label = f"codecontests/{bucket_key or 'all'}/{split_name}"
        problem_id = sample.get("problem_id") or sample.get("id") or sample.get("name")
        question = sample.get("problem_statement") or sample.get("description") or sample.get("statement")
        starter = sample.get("starter_code") or sample.get("default_code")
        entry_point = sample.get("entry_point") or _guess_entry_point(starter)
        problems.append(
            {
                "dataset": dataset_label,
                "domain": DOMAIN,
                "id": problem_id,
                "problem_id": problem_id,
                "source": sample.get("source", "codecontests"),
                "language": "python",
                "title": sample.get("name"),
                "question": question,
                "difficulty": sample.get("difficulty"),
                "starter_code": starter,
                "entry_point": entry_point,
                "tests": tests,
                "time_limit_s": _maybe_float(time_limit_ms, 0.0) / 1000 if time_limit_ms else None,
                "memory_limit_bytes": _maybe_int(memory_limit, 0) * 1024 * 1024 if memory_limit else None,
            }
        )
    return problems


def _extract_ds1000_tests(sample: Dict[str, Any], max_tests: int | None) -> list[dict]:
    raw_tests = sample.get("tests") or sample.get("unit_tests") or sample.get("public_tests")
    if isinstance(raw_tests, str):
        try:
            raw_tests = json.loads(raw_tests)
        except json.JSONDecodeError:
            logger.debug("Skipping DS-1000 sample with non-JSON tests")
            return []
    return _build_io_tests(raw_tests, max_tests=max_tests)


def _load_ds1000(selector: DatasetSelector, cfg: Dict[str, Any], default_split: str) -> list[dict]:
    dataset_id = cfg.get("dataset_id") or DEFAULT_DS1000_DATASET_ID
    split_name = selector.split_name or cfg.get("split", "train") or default_split
    category = selector.config_name or cfg.get("category")
    dataset = load_dataset(dataset_id, split=split_name, trust_remote_code=cfg.get("trust_remote_code", False))

    max_examples = cfg.get("max_examples_per_split")
    max_tests = cfg.get("max_public_tests", DEFAULT_MAX_PUBLIC_TESTS)
    problems: list[dict] = []
    for idx, sample in enumerate(dataset):
        if max_examples is not None and len(problems) >= max_examples:
            break
        sample_category = str(sample.get("task_category", sample.get("category", ""))).lower()
        if category and sample_category != category.lower():
            continue
        tests = _extract_ds1000_tests(sample, max_tests)
        if not tests:
            continue
        starter = sample.get("starter_code") or sample.get("prompt") or sample.get("starter")
        entry_point = sample.get("entry_point") or _guess_entry_point(starter)
        dataset_label = f"ds1000/{category or 'all'}/{split_name}"
        problem_id = sample.get("task_id") or sample.get("id") or f"ds1000-{split_name}-{idx}"
        problems.append(
            {
                "dataset": dataset_label,
                "domain": DOMAIN,
                "id": problem_id,
                "problem_id": problem_id,
                "source": "ds1000",
                "language": "python",
                "title": sample.get("title"),
                "question": sample.get("question") or sample.get("prompt"),
                "difficulty": sample.get("difficulty"),
                "starter_code": starter,
                "entry_point": entry_point,
                "tests": tests,
            }
        )
    return problems


def _load_evalplus_payload(path: str) -> dict[str, list[Any]]:
    content = Path(path).expanduser().read_text(encoding="utf-8")
    data = json.loads(content)
    if isinstance(data, list):
        mapping: dict[str, list[Any]] = {}
        for entry in data:
            if not isinstance(entry, dict):
                continue
            key = entry.get("task_id") or entry.get("problem_id")
            tests = entry.get("tests") or entry.get("public_tests") or entry.get("extra_tests")
            if key and isinstance(tests, list):
                mapping[str(key)] = tests
        return mapping
    if isinstance(data, dict):
        mapping: dict[str, list[Any]] = {}
        for key, value in data.items():
            if isinstance(value, dict) and "tests" in value:
                mapping[str(key)] = value.get("tests") or []
            elif isinstance(value, list):
                mapping[str(key)] = value
        return mapping
    return {}


def _apply_evalplus_augmentations(problems: list[dict], augments: Dict[str, Dict[str, Any]] | None) -> None:
    if not augments:
        return
    for label, spec in augments.items():
        path = spec.get("path")
        if not path:
            continue
        try:
            payload = _load_evalplus_payload(path)
        except FileNotFoundError:
            logger.warning("EvalPlus augment '%s' missing file %s", label, path)
            continue
        dataset_filters = {str(name).lower() for name in spec.get("datasets", []) if name}
        max_tests = spec.get("max_tests")
        augmented = 0
        for problem in problems:
            dataset_name = str(problem.get("dataset", "")).lower()
            if dataset_filters and dataset_name not in dataset_filters:
                continue
            problem_id = str(problem.get("problem_id") or problem.get("id"))
            extra_cases = payload.get(problem_id)
            if not extra_cases:
                continue
            start_id = len(problem.get("tests", []))
            normalized = _build_io_tests(extra_cases, max_tests=max_tests, start_id=start_id)
            if not normalized:
                continue
            problem.setdefault("tests", []).extend(normalized)
            augmented += len(normalized)
        logger.info("EvalPlus augment '%s' added %d tests", label, augmented)


ADAPTERS = {
    "livecodebench": _load_livecodebench,
    "apps": _load_apps,
    "codecontests": _load_codecontests,
    "ds1000": _load_ds1000,
}


def load_problems(
    dataset_names: Sequence[str] | str | None,
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    language: str | None = DEFAULT_LANGUAGE,
    max_public_tests: int | None = DEFAULT_MAX_PUBLIC_TESTS,
    include_private_tests: bool = False,
    max_examples_per_split: int | None = None,
    trust_remote_code: bool = False,
    default_split: str = "test",
    apps_dataset_id: str = DEFAULT_APPS_DATASET_ID,
    codecontests_dataset_id: str = DEFAULT_CODECONTESTS_DATASET_ID,
    ds1000_dataset_id: str = DEFAULT_DS1000_DATASET_ID,
    adapter_overrides: Dict[str, Dict[str, Any]] | None = None,
    evalplus_augments: Dict[str, Dict[str, Any]] | None = None,
    **_: dict,
) -> List[Dict]:
    """Load coding problems from the configured adapters."""
    if dataset_names is None:
        return []
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    adapter_configs: Dict[str, Dict[str, Any]] = {
        "livecodebench": {
            "dataset_id": dataset_id,
            "language": language,
            "max_public_tests": max_public_tests,
            "include_private_tests": include_private_tests,
            "max_examples_per_split": max_examples_per_split,
            "trust_remote_code": trust_remote_code,
        },
        "apps": {
            "dataset_id": apps_dataset_id,
            "max_public_tests": max_public_tests,
            "max_examples_per_split": max_examples_per_split,
        },
        "codecontests": {
            "dataset_id": codecontests_dataset_id,
            "max_public_tests": max_public_tests,
            "max_examples_per_split": max_examples_per_split,
            "difficulty_buckets": DEFAULT_CODECONTESTS_BUCKETS,
        },
        "ds1000": {
            "dataset_id": ds1000_dataset_id,
            "max_public_tests": max_public_tests,
            "max_examples_per_split": max_examples_per_split,
        },
    }

    if adapter_overrides:
        for alias, override in adapter_overrides.items():
            base = adapter_configs.setdefault(alias, {})
            base.update(dict(override or {}))

    problems: list[dict] = []
    for name in dataset_names:
        selector = _parse_selector(str(name), default_split=default_split)
        alias = selector.alias
        adapter = ADAPTERS.get(alias)
        if adapter is None:
            raise ValueError(f"Unsupported coding dataset alias '{alias}' in selector '{name}'")
        adapter_cfg = adapter_configs.get(alias, {})
        loaded = adapter(selector, adapter_cfg, default_split)
        problems.extend(loaded)

    _apply_evalplus_augmentations(problems, evalplus_augments)
    return problems
