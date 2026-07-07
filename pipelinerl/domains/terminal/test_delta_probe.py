"""Offline replay prober for hindsight verifier-test deltas."""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

COMMAND_TRUNCATION_SUFFIX = "...[truncated]"
PostJson = Callable[[str, dict[str, Any], float], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class CheckpointResult:
    turn: int
    passed: bool
    passed_tests: int
    total_tests: int
    abort_kind: str | None
    replay_successes_agree: bool | None


@dataclass(frozen=True)
class ProbeResult:
    task_id: str
    dataset_name: str | None
    group_id: str | None
    rollout_index: int | None
    original_reward: float | None
    original_verifier_pass: bool
    original_passed_tests: int
    original_total_tests: int
    checkpoints: list[CheckpointResult]


@dataclass(frozen=True)
class ProbeSummary:
    n_rollouts: int
    n_final_compared: int
    replay_fidelity: float | None
    mean_abs_test_delta_per_turn: float | None
    fraction_any_intermediate_progress: float | None


def read_jsonl(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open() as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def has_truncated_command(commands: list[str]) -> bool:
    return any(command.endswith(COMMAND_TRUNCATION_SUFFIX) for command in commands)


def is_probeable_row(row: dict[str, Any], max_turns: int) -> bool:
    commands = row.get("commands")
    if not bool(row.get("submitted")):
        return False
    if row.get("abort_kind") is not None:
        return False
    if int(row.get("total_tests") or 0) <= 0:
        return False
    if bool(row.get("dropped")):
        return False
    if not isinstance(commands, list) or not commands:
        return False
    if not all(isinstance(command, str) for command in commands):
        return False
    n_turns = int(row.get("n_turns") or len(commands))
    if n_turns > max_turns or len(commands) > max_turns:
        return False
    return not has_truncated_command(commands)


def checkpoint_turns(n_commands: int) -> list[int]:
    if n_commands <= 0:
        return []
    turns = {max(1, math.ceil(n_commands * fraction)) for fraction in (0.25, 0.5, 0.75, 1.0)}
    turns.add(n_commands)
    return sorted(turns)


def task_index(problems: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for problem in problems:
        task_id = str(problem["task_id"])
        if task_id in index:
            duplicates.add(task_id)
        index[task_id] = problem
    if duplicates:
        raise ValueError(f"task_id is not unique for {len(duplicates)} tasks; sample={sorted(duplicates)[:5]}")
    return index


def select_probe_rows(
    rows: list[dict[str, Any]],
    task_by_id: dict[str, dict[str, Any]],
    *,
    sample: int,
    seed: int,
    max_turns: int,
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if is_probeable_row(row, max_turns) and row.get("task_id") in task_by_id]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:sample] if sample > 0 else candidates


def replay_successes_agree(row: dict[str, Any], turn: int, successes: list[Any]) -> bool | None:
    command_errors = row.get("command_errors")
    if not isinstance(command_errors, list) or len(command_errors) < turn or len(successes) != turn:
        return None
    expected = [not bool(error) for error in command_errors[:turn]]
    return [bool(success) for success in successes] == expected


async def probe_checkpoint(
    row: dict[str, Any],
    task: dict[str, Any],
    turn: int,
    post_json: PostJson,
    request_timeout: float,
) -> CheckpointResult:
    session_id = None
    try:
        start = await post_json("/start_task", {"task_data": task}, request_timeout)
        session_id = start.get("session_id")
        if not session_id or not start.get("started") or not start.get("init_ok"):
            return CheckpointResult(
                turn=turn,
                passed=False,
                passed_tests=0,
                total_tests=0,
                abort_kind="start_failed",
                replay_successes_agree=None,
            )

        commands = list(row["commands"][:turn])
        replay = await post_json("/replay", {"session_id": session_id, "commands": commands}, request_timeout)
        agree = replay_successes_agree(row, turn, replay.get("successes", []))
        replay_abort = replay.get("abort_kind")
        if replay_abort is not None:
            return CheckpointResult(
                turn=turn,
                passed=False,
                passed_tests=0,
                total_tests=0,
                abort_kind=str(replay_abort),
                replay_successes_agree=agree,
            )
        if int(replay.get("n_executed", 0)) != turn:
            return CheckpointResult(
                turn=turn,
                passed=False,
                passed_tests=0,
                total_tests=0,
                abort_kind="replay_incomplete",
                replay_successes_agree=agree,
            )

        finish = await post_json("/finish", {"session_id": session_id}, request_timeout)
        session_id = None
        return CheckpointResult(
            turn=turn,
            passed=bool(finish.get("passed", False)),
            passed_tests=int(finish.get("passed_tests", 0)),
            total_tests=int(finish.get("total_tests", 0)),
            abort_kind=finish.get("abort_kind"),
            replay_successes_agree=agree,
        )
    except Exception as exc:
        return CheckpointResult(
            turn=turn,
            passed=False,
            passed_tests=0,
            total_tests=0,
            abort_kind=f"error:{type(exc).__name__}",
            replay_successes_agree=None,
        )
    finally:
        if session_id:
            try:
                await post_json("/close", {"session_id": session_id}, request_timeout)
            except Exception:
                pass


async def probe_rollout(
    row: dict[str, Any],
    task: dict[str, Any],
    post_json: PostJson,
    request_timeout: float,
) -> ProbeResult:
    checkpoints = []
    for turn in checkpoint_turns(len(row["commands"])):
        checkpoints.append(await probe_checkpoint(row, task, turn, post_json, request_timeout))
    return ProbeResult(
        task_id=str(row.get("task_id")),
        dataset_name=row.get("dataset_name") or row.get("dataset"),
        group_id=row.get("group_id"),
        rollout_index=row.get("rollout_index"),
        original_reward=row.get("reward"),
        original_verifier_pass=bool(row.get("verifier_pass")),
        original_passed_tests=int(row.get("passed_tests") or 0),
        original_total_tests=int(row.get("total_tests") or 0),
        checkpoints=checkpoints,
    )


def checkpoint_passed(checkpoint: CheckpointResult) -> bool:
    return checkpoint.passed


def compute_summary(results: list[ProbeResult]) -> ProbeSummary:
    final_compared = 0
    final_matches = 0
    deltas: list[float] = []
    progress_count = 0

    for result in results:
        if not result.checkpoints:
            continue
        final = result.checkpoints[-1]
        final_compared += 1
        if checkpoint_passed(final) == result.original_verifier_pass:
            final_matches += 1

        previous_turn = 0
        previous_passed = 0
        for checkpoint in result.checkpoints:
            turn_delta = max(1, checkpoint.turn - previous_turn)
            deltas.append(abs(checkpoint.passed_tests - previous_passed) / turn_delta)
            previous_turn = checkpoint.turn
            previous_passed = checkpoint.passed_tests

        if any(cp.turn < final.turn and cp.passed_tests > 0 for cp in result.checkpoints):
            progress_count += 1

    return ProbeSummary(
        n_rollouts=len(results),
        n_final_compared=final_compared,
        replay_fidelity=(final_matches / final_compared) if final_compared else None,
        mean_abs_test_delta_per_turn=(sum(deltas) / len(deltas)) if deltas else None,
        fraction_any_intermediate_progress=(progress_count / len(results)) if results else None,
    )


def result_to_dict(result: ProbeResult) -> dict[str, Any]:
    return asdict(result)


def summary_to_dict(summary: ProbeSummary) -> dict[str, Any]:
    return asdict(summary)


def write_results_jsonl(results: list[ProbeResult], path: str | Path) -> None:
    with Path(path).open("w") as f:
        for result in results:
            f.write(json.dumps(result_to_dict(result), sort_keys=True) + "\n")
