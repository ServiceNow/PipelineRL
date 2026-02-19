from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

from pipelinerl.domains.math.load_datasets import load_datasets as load_math_datasets
from pipelinerl.domains.guessing.guessing import load_problems as load_guessing_problems
from pipelinerl.domains.counting.counting import load_problems as load_counting_problems
from pipelinerl.domains.chartqa.load_datasets import load_problems as load_chartqa_problems
from pipelinerl.domains.coding.dataset import load_problems as load_coding_problems
from pipelinerl.domains.coding.livecodebench import load_problems as load_livecodebench_problems
from pipelinerl.domains.fn_calling.dataset import load_problems as load_fn_calling_problems
from pipelinerl.domains.logic.dataset import load_problems as load_logic_problems
from pipelinerl.domains.miniwob.load_tasks import load_tasks as load_miniwob_tasks
from pipelinerl.domains.ifeval.dataset import load_problems as load_ifeval_problems
from pipelinerl.domains.ifeval.google_ifeval import load_problems as load_google_ifeval_problems


def _load_math(dataset_names: Sequence[str], *, seed=None, **_: dict) -> List[Dict]:
    return load_math_datasets(list(dataset_names), seed=seed)


def _load_guessing(dataset_names: Sequence[str], **_: dict) -> List[Dict]:
    return load_guessing_problems(list(dataset_names))


def _load_coding(dataset_names: Sequence[str], **loader_kwargs: dict) -> List[Dict]:
    return load_coding_problems(list(dataset_names), **loader_kwargs)


def _load_livecodebench(dataset_names: Sequence[str], **loader_kwargs: dict) -> List[Dict]:
    return load_livecodebench_problems(list(dataset_names), **loader_kwargs)


def _load_fn_calling(dataset_names: Sequence[str], **loader_kwargs: dict) -> List[Dict]:
    return load_fn_calling_problems(list(dataset_names), **loader_kwargs)


def _load_logic(dataset_names: Sequence[str], **loader_kwargs: dict) -> List[Dict]:
    return load_logic_problems(list(dataset_names), **loader_kwargs)


def _load_counting(dataset_names: Sequence[str], **_: dict) -> List[Dict]:
    return load_counting_problems(list(dataset_names))


def _load_chartqa(dataset_names: Sequence[str], **_: dict) -> List[Dict]:
    return load_chartqa_problems(list(dataset_names))


def _load_miniwob(dataset_names: Sequence[str], **loader_kwargs: dict) -> List[Dict]:
    return load_miniwob_tasks(list(dataset_names), **loader_kwargs)


def _load_ifeval(dataset_names: Sequence[str], **loader_kwargs: dict) -> List[Dict]:
    return load_ifeval_problems(list(dataset_names), **loader_kwargs)


def _load_google_ifeval(dataset_names: Sequence[str], **loader_kwargs: dict) -> List[Dict]:
    return load_google_ifeval_problems(list(dataset_names), **loader_kwargs)


DOMAIN_LOADERS = {
    "math": _load_math,
    "guessing": _load_guessing,
    "coding": _load_coding,
    "livecodebench": _load_livecodebench,
    "counting": _load_counting,
    "chartqa": _load_chartqa,
    "miniwob": _load_miniwob,
    "fn_calling": _load_fn_calling,
    "logic": _load_logic,
    "ifeval": _load_ifeval,
    "google_ifeval": _load_google_ifeval,
}


def _parse_entry(entry: str) -> tuple[str, str, str | None]:
    """Parse a dataset entry into (domain, dataset_name, subset).

    Format: '<domain>::<dataset_name>[@subset]'
    Examples:
        'coding::coding' -> ('coding', 'coding', None)
        'coding::coding@train' -> ('coding', 'coding', 'train')
        'coding::coding@test' -> ('coding', 'coding', 'test')
    """
    if "::" not in entry:
        raise ValueError(
            f"Dataset entry '{entry}' is missing a domain prefix. "
            "Expected format '<domain>::<dataset_name>[@subset]'."
        )
    domain, dataset = entry.split("::", 1)
    domain = domain.strip()
    dataset = dataset.strip()

    # Parse optional @subset suffix
    subset = None
    if "@" in dataset:
        dataset, subset = dataset.rsplit("@", 1)
        subset = subset.strip()
        if subset not in ("train", "test", "all"):
            # Not a subset suffix, restore original
            dataset = f"{dataset}@{subset}"
            subset = None

    return domain, dataset, subset


def load_datasets(
    dataset_names: Iterable[str] | str | None,
    *,
    seed: int | None = None,
    per_domain_params: dict[str, dict] | None = None,
    **kwargs,
) -> List[Dict]:
    if dataset_names is None:
        return []
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    # Group by (domain, subset) to handle train/test separately
    grouped: dict[tuple[str, str | None], list[str]] = defaultdict(list)
    for entry in dataset_names:
        domain, name, subset = _parse_entry(str(entry))
        grouped[(domain, subset)].append(name)

    counters: dict[tuple[str, str], int] = defaultdict(int)
    problems: List[Dict] = []
    per_domain_params = dict(per_domain_params or {})

    for (domain, subset), names in grouped.items():
        loader = DOMAIN_LOADERS.get(domain)
        if loader is None:
            raise ValueError(f"No loader registered for domain '{domain}'")

        # Build kwargs: start with global, then domain-specific, then subset override
        domain_kwargs = dict(kwargs)
        if domain in per_domain_params:
            domain_kwargs.update(dict(per_domain_params[domain] or {}))

        # Override subset if specified in dataset name (e.g., @train, @test)
        if subset is not None:
            domain_kwargs["subset"] = subset

        # Use explicit seed if provided, otherwise use one from domain_kwargs
        effective_seed = seed if seed is not None else domain_kwargs.pop("seed", None)

        loaded = loader(names, seed=effective_seed, **domain_kwargs)
        for sample in loaded:
            dataset_name = str(sample.get("dataset", names[0] if names else domain))
            sample.setdefault("domain", domain)
            if "id" not in sample:
                key = (sample["domain"], dataset_name)
                sample["id"] = counters[key]
                counters[key] += 1
            if "dataset" not in sample:
                sample["dataset"] = dataset_name
            problems.append(sample)
    return problems
