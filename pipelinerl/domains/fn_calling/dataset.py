"""Dataset loader for the fn_calling domain."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from pipelinerl.domains.coding import dataset as mixed_dataset

logger = logging.getLogger(__name__)

DOMAIN_NAME = "fn_calling"
ALIAS_NAMES = frozenset({DOMAIN_NAME, "fn-calling", "fncalling"})
BASE_DATASET_NAME = "mixed-training-text-datasets"
DEFAULT_DATASET_CONFIG = mixed_dataset.DEFAULT_DATASET_CONFIG
ABILITY_FILTER = "agentic_fn_calling"


@dataclass(frozen=True)
class DatasetResolution:
    requested: list[str]
    resolved: list[str]
    alias_map: dict[str, str]


def _resolve_dataset_requests(dataset_names: List[str] | str | None) -> DatasetResolution:
    if dataset_names is None:
        requested = [DOMAIN_NAME]
    elif isinstance(dataset_names, str):
        requested = [dataset_names]
    else:
        requested = [str(entry) for entry in dataset_names]

    resolved: list[str] = []
    alias_map: dict[str, str] = {}
    for entry in requested:
        spec = mixed_dataset.parse_dataset_name(entry)
        base_name = BASE_DATASET_NAME if spec.name in ALIAS_NAMES else spec.name
        resolved_entry = f"{base_name}@{spec.split}"
        resolved.append(resolved_entry)
        alias_map.setdefault(resolved_entry, f"{spec.name}@{spec.split}")
    return DatasetResolution(requested=requested, resolved=resolved, alias_map=alias_map)


def load_datasets(
    dataset_names: List[str] | str | None,
    seed: int | None = None,
    **loader_kwargs: Any,
) -> List[Dict]:
    resolution = _resolve_dataset_requests(dataset_names)
    defaults = {
        "dataset_id": mixed_dataset.DEFAULT_DATASET_ID,
        "dataset_config": DEFAULT_DATASET_CONFIG,
        # fn_calling loads all call_types (no filtering), unlike coding which filters to assert/std
        "allowed_call_types": (),
        "ability_filter": ABILITY_FILTER,
    }
    options = {**defaults, **loader_kwargs}

    # Validate that allowed_call_types is empty for fn_calling domain
    final_call_types = options.get("allowed_call_types")
    if final_call_types:
        logger.warning(
            "fn_calling domain received non-empty allowed_call_types=%s. "
            "This may filter out valid fn_calling samples. "
            "Consider using allowed_call_types=[] for this domain.",
            final_call_types,
        )

    samples = mixed_dataset.load_datasets(resolution.resolved, seed=seed, **options)

    for sample in samples:
        dataset_label = str(sample.get("dataset", ""))
        alias_label = resolution.alias_map.get(dataset_label)
        if alias_label:
            sample["dataset"] = alias_label
        elif dataset_label.startswith(BASE_DATASET_NAME):
            sample["dataset"] = dataset_label.replace(BASE_DATASET_NAME, DOMAIN_NAME, 1)
        sample["domain"] = DOMAIN_NAME
    if not samples:
        logger.warning("fn_calling loader returned zero samples for entries: %s", resolution.requested)
    return samples


def load_problems(dataset_names: List[str] | str | None, **loader_kwargs: Any) -> List[Dict]:
    seed = loader_kwargs.pop("seed", None)
    return load_datasets(dataset_names, seed=seed, **loader_kwargs)


__all__ = ["load_datasets", "load_problems", "DEFAULT_DATASET_CONFIG", "DOMAIN_NAME"]
