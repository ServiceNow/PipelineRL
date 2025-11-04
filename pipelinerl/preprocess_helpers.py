from __future__ import annotations

from typing import Iterable


def group_rollout_idx(group: Iterable[dict]) -> set[int] | None:
    """Extract rollout idx from a rollout group."""
    rollout_indices: set[int] = set()
    for text in group:
        metadata = text.get("metadata")
        if not isinstance(metadata, dict):
            return None
        rollout_index = metadata.get("rollout_index")
        if rollout_index is None:
            return None
        rollout_indices.add(rollout_index)
    return rollout_indices


def validate_rollout_group(group: Iterable[dict], group_size: int) -> tuple[bool, list[int], list[int]]:
    """Return whether a group is complete and any missing or extra rollout indices."""
    rollout_indices = group_rollout_idx(group)
    if rollout_indices is None:
        return False, [], []
    if len(rollout_indices) != group_size:
        expected_indices = set(range(group_size))
        if rollout_indices.issubset(expected_indices):
            missing = sorted(expected_indices - rollout_indices)
            extra: list[int] = []
        else:
            missing = sorted(expected_indices - rollout_indices)
            extra = sorted(rollout_indices - expected_indices)
        return False, missing, extra
    return True, [], []
