"""Offline mining of same-task rollout twins with shared command prefixes."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

COMMAND_TRUNCATION_SUFFIX = "...[truncated]"
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class TwinRecord:
    group_id: str
    task_id: str | None
    index_a: int
    index_b: int
    divergence_turn: int
    lcp_len: int
    len_a: int
    len_b: int
    reward_a: float | None
    reward_b: float | None


def normalize_command(command: str) -> str:
    return _WHITESPACE_RE.sub(" ", command.strip())


def common_prefix_len(a: list[str], b: list[str]) -> int:
    count = 0
    for command_a, command_b in zip(a, b):
        if normalize_command(command_a) != normalize_command(command_b):
            break
        count += 1
    return count


def _has_truncated_command(commands: list[str]) -> bool:
    return any(command.endswith(COMMAND_TRUNCATION_SUFFIX) for command in commands)


def _mineable_row(row: dict[str, Any]) -> bool:
    commands = row.get("commands")
    if row.get("group_id") is None:
        return False
    if bool(row.get("dropped")):
        return False
    if not isinstance(commands, list) or not all(isinstance(command, str) for command in commands):
        return False
    return not _has_truncated_command(commands)


def _row_index(row: dict[str, Any], fallback: int) -> int:
    value = row.get("rollout_index")
    return fallback if value is None else int(value)


def _reward(row: dict[str, Any]) -> float | None:
    value = row.get("reward")
    return None if value is None else float(value)


def mine_twins(rows: list[dict[str, Any]], min_prefix: int = 3) -> list[TwinRecord]:
    groups: dict[str, list[tuple[int, dict[str, Any], list[str]]]] = defaultdict(list)
    for input_index, row in enumerate(rows):
        if not _mineable_row(row):
            continue
        commands = [normalize_command(command) for command in row["commands"]]
        groups[str(row["group_id"])].append((input_index, row, commands))

    twins: list[TwinRecord] = []
    for group_id, group_rows in groups.items():
        for i, (input_index_a, row_a, commands_a) in enumerate(group_rows):
            for input_index_b, row_b, commands_b in group_rows[i + 1:]:
                if bool(row_a.get("verifier_pass")) == bool(row_b.get("verifier_pass")):
                    continue
                lcp_len = common_prefix_len(commands_a, commands_b)
                if lcp_len < min_prefix:
                    continue
                twins.append(
                    TwinRecord(
                        group_id=group_id,
                        task_id=row_a.get("task_id"),
                        index_a=_row_index(row_a, input_index_a),
                        index_b=_row_index(row_b, input_index_b),
                        divergence_turn=lcp_len,
                        lcp_len=lcp_len,
                        len_a=len(commands_a),
                        len_b=len(commands_b),
                        reward_a=_reward(row_a),
                        reward_b=_reward(row_b),
                    )
                )
    return twins


def yield_report(rows: list[dict[str, Any]], twins: list[TwinRecord]) -> dict[str, Any]:
    groups: dict[str, set[bool]] = defaultdict(set)
    eligible_rollouts = 0
    for row in rows:
        if not _mineable_row(row):
            continue
        eligible_rollouts += 1
        groups[str(row["group_id"])].add(bool(row.get("verifier_pass")))

    lcp_histogram = Counter(twin.lcp_len for twin in twins)
    return {
        "rollouts_seen": len(rows),
        "eligible_rollouts": eligible_rollouts,
        "groups_seen": len(groups),
        "mixed_outcome_groups": sum(len(outcomes) > 1 for outcomes in groups.values()),
        "twin_pairs": len(twins),
        "pairs_per_1k_rollouts": (1000.0 * len(twins) / eligible_rollouts) if eligible_rollouts else 0.0,
        "lcp_histogram": {str(key): lcp_histogram[key] for key in sorted(lcp_histogram)},
    }


def read_jsonl(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open() as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def twin_to_dict(twin: TwinRecord) -> dict[str, Any]:
    return asdict(twin)


def write_twins_jsonl(twins: list[TwinRecord], path: str | Path) -> None:
    with Path(path).open("w") as f:
        for twin in twins:
            f.write(json.dumps(twin_to_dict(twin), sort_keys=True) + "\n")
