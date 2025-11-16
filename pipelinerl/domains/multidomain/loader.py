from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

from pipelinerl.domains.math.load_datasets import load_datasets as load_math_datasets
from pipelinerl.domains.guessing.guessing import load_problems as load_guessing_problems
from pipelinerl.domains.counting.counting import load_problems as load_counting_problems
from pipelinerl.domains.chartqa.load_datasets import load_problems as load_chartqa_problems
from pipelinerl.domains.coding.dataset import load_problems as load_coding_problems
from pipelinerl.domains.symbolic.dataset import load_problems as load_symbolic_problems
from pipelinerl.miniwob.load_tasks import load_tasks as load_miniwob_tasks


def _load_math(dataset_names: Sequence[str], *, seed=None, **_: dict) -> List[Dict]:
    return load_math_datasets(list(dataset_names), seed=seed)


def _load_guessing(dataset_names: Sequence[str], **_: dict) -> List[Dict]:
    return load_guessing_problems(list(dataset_names))


def _load_coding(dataset_names: Sequence[str], **loader_kwargs: dict) -> List[Dict]:
    return load_coding_problems(list(dataset_names), **loader_kwargs)


def _load_counting(dataset_names: Sequence[str], **_: dict) -> List[Dict]:
    return load_counting_problems(list(dataset_names))


def _load_chartqa(dataset_names: Sequence[str], **_: dict) -> List[Dict]:
    return load_chartqa_problems(list(dataset_names))


def _load_miniwob(dataset_names: Sequence[str], **loader_kwargs: dict) -> List[Dict]:
    return load_miniwob_tasks(list(dataset_names), **loader_kwargs)


def _load_symbolic(dataset_names: Sequence[str], **loader_kwargs: dict) -> List[Dict]:
    return load_symbolic_problems(list(dataset_names), **loader_kwargs)


DOMAIN_LOADERS = {
    "math": _load_math,
    "guessing": _load_guessing,
    "coding": _load_coding,
    "counting": _load_counting,
    "chartqa": _load_chartqa,
    "miniwob": _load_miniwob,
    "symbolic": _load_symbolic,
}


def _parse_entry(entry: str) -> tuple[str, str]:
    if "::" not in entry:
        raise ValueError(
            f"Dataset entry '{entry}' is missing a domain prefix. "
            "Expected format '<domain>::<dataset_name>'."
        )
    domain, dataset = entry.split("::", 1)
    return domain.strip(), dataset.strip()


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

    grouped: dict[str, list[str]] = defaultdict(list)
    for entry in dataset_names:
        domain, name = _parse_entry(str(entry))
        grouped[domain].append(name)

    counters: dict[tuple[str, str], int] = defaultdict(int)
    problems: List[Dict] = []
    per_domain_params = dict(per_domain_params or {})
    for domain, names in grouped.items():
        loader = DOMAIN_LOADERS.get(domain)
        if loader is None:
            raise ValueError(f"No loader registered for domain '{domain}'")
        domain_kwargs = dict(kwargs)
        if domain in per_domain_params:
            domain_kwargs.update(dict(per_domain_params[domain] or {}))
        loaded = loader(names, seed=seed, **domain_kwargs)
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
