import asyncio

from pipelinerl.domains.terminal.test_delta_probe import (
    CheckpointResult,
    checkpoint_passed,
    checkpoint_turns,
    compute_summary,
    is_probeable_row,
    probe_rollout,
    result_to_dict,
    select_probe_rows,
    task_index,
)


def _row(**overrides):
    row = {
        "task_id": "task-1",
        "dataset_name": "tmax-15k@debugging",
        "group_id": "group-1",
        "rollout_index": 3,
        "reward": 1.0,
        "verifier_pass": True,
        "passed_tests": 2,
        "total_tests": 2,
        "submitted": True,
        "abort_kind": None,
        "dropped": False,
        "commands": ["a", "b", "c", "d"],
        "command_errors": [False, True, False, False],
    }
    row.update(overrides)
    return row


def test_checkpoint_turns_short_long_and_dedup():
    assert checkpoint_turns(0) == []
    assert checkpoint_turns(1) == [1]
    assert checkpoint_turns(2) == [1, 2]
    assert checkpoint_turns(3) == [1, 2, 3]
    assert checkpoint_turns(8) == [2, 4, 6, 8]


def test_filter_predicate_rejects_unprobeable_rows():
    assert is_probeable_row(_row(), max_turns=4)
    assert not is_probeable_row(_row(submitted=False), max_turns=4)
    assert not is_probeable_row(_row(abort_kind="timeout"), max_turns=4)
    assert not is_probeable_row(_row(total_tests=0), max_turns=4)
    assert not is_probeable_row(_row(dropped=True), max_turns=4)
    assert not is_probeable_row(_row(commands=[]), max_turns=4)
    assert not is_probeable_row(_row(commands=["x...[truncated]"]), max_turns=4)
    assert not is_probeable_row(_row(commands=["a", "b", "c", "d", "e"]), max_turns=4)


def test_task_index_requires_unique_task_id():
    assert task_index([{"task_id": "a"}]) == {"a": {"task_id": "a"}}
    try:
        task_index([{"task_id": "a"}, {"task_id": "a"}])
    except ValueError as exc:
        assert "not unique" in str(exc)
    else:
        raise AssertionError("expected duplicate task_id failure")


def test_select_probe_rows_filters_missing_tasks_and_samples_deterministically():
    rows = [_row(task_id="task-1"), _row(task_id="missing"), _row(task_id="task-2")]
    tasks = {"task-1": {}, "task-2": {}}

    first = select_probe_rows(rows, tasks, sample=1, seed=7, max_turns=4)
    second = select_probe_rows(rows, tasks, sample=1, seed=7, max_turns=4)

    assert first == second
    assert len(first) == 1
    assert first[0]["task_id"] in {"task-1", "task-2"}


def test_probe_rollout_output_schema_with_stubbed_http():
    row = _row()
    task = {"task_id": "task-1", "task": "fix it"}
    calls = []
    replay_lengths_by_session = {}
    session_counter = 0

    async def post_json(route, payload, timeout):
        nonlocal session_counter
        calls.append((route, dict(payload)))
        if route == "/start_task":
            session_counter += 1
            return {"session_id": f"session-{session_counter}", "started": True, "init_ok": True, "build_ok": True}
        if route == "/replay":
            commands = payload["commands"]
            replay_lengths_by_session[payload["session_id"]] = len(commands)
            successes = [command != "b" for command in commands]
            return {"n_executed": len(commands), "successes": successes, "abort_kind": None, "last_output": "ok"}
        if route == "/finish":
            turn = replay_lengths_by_session[payload["session_id"]]
            passed = {1: 0, 2: 1, 3: 1, 4: 2}[turn]
            return {"passed": passed == 2, "passed_tests": passed, "total_tests": 2, "abort_kind": None, "output": ""}
        raise AssertionError(route)

    result = asyncio.run(probe_rollout(row, task, post_json, request_timeout=10.0))
    data = result_to_dict(result)

    assert data["task_id"] == "task-1"
    assert data["group_id"] == "group-1"
    assert data["rollout_index"] == 3
    assert data["original_verifier_pass"] is True
    assert [checkpoint["turn"] for checkpoint in data["checkpoints"]] == [1, 2, 3, 4]
    assert [checkpoint["passed"] for checkpoint in data["checkpoints"]] == [False, False, False, True]
    assert [checkpoint["passed_tests"] for checkpoint in data["checkpoints"]] == [0, 1, 1, 2]
    assert [checkpoint["replay_successes_agree"] for checkpoint in data["checkpoints"]] == [True, True, True, True]
    assert [route for route, _ in calls] == [
        "/start_task", "/replay", "/finish",
        "/start_task", "/replay", "/finish",
        "/start_task", "/replay", "/finish",
        "/start_task", "/replay", "/finish",
    ]


def test_probe_rollout_closes_session_when_replay_aborts():
    row = _row(commands=["a", "b"])
    task = {"task_id": "task-1", "task": "fix it"}
    calls = []

    async def post_json(route, payload, timeout):
        calls.append((route, dict(payload)))
        if route == "/start_task":
            return {"session_id": f"session-{len(calls)}", "started": True, "init_ok": True, "build_ok": True}
        if route == "/replay":
            return {"n_executed": len(payload["commands"]), "successes": [True] * len(payload["commands"]), "abort_kind": "timeout", "last_output": ""}
        if route == "/close":
            return {"status": "ok"}
        raise AssertionError(route)

    result = asyncio.run(probe_rollout(row, task, post_json, request_timeout=10.0))

    assert all(checkpoint.abort_kind == "timeout" for checkpoint in result.checkpoints)
    assert [route for route, _ in calls] == ["/start_task", "/replay", "/close", "/start_task", "/replay", "/close"]


def test_checkpoint_passed_uses_verifier_verdict_not_counts():
    checkpoint = CheckpointResult(
        turn=4,
        passed=True,
        passed_tests=1,
        total_tests=2,
        abort_kind=None,
        replay_successes_agree=True,
    )

    assert checkpoint_passed(checkpoint)


def test_compute_summary_fidelity_and_progress():
    result = asyncio.run(
        probe_rollout(
            _row(),
            {"task_id": "task-1"},
            _summary_post_json(),
            request_timeout=10.0,
        )
    )

    summary = compute_summary([result])

    assert summary.n_rollouts == 1
    assert summary.n_final_compared == 1
    assert summary.replay_fidelity == 1.0
    assert summary.fraction_any_intermediate_progress == 1.0
    assert summary.mean_abs_test_delta_per_turn is not None


def _summary_post_json():
    replay_lengths_by_session = {}
    counter = 0

    async def post_json(route, payload, timeout):
        nonlocal counter
        if route == "/start_task":
            counter += 1
            return {"session_id": f"session-{counter}", "started": True, "init_ok": True, "build_ok": True}
        if route == "/replay":
            replay_lengths_by_session[payload["session_id"]] = len(payload["commands"])
            return {"n_executed": len(payload["commands"]), "successes": [True] * len(payload["commands"]), "abort_kind": None}
        if route == "/finish":
            turn = replay_lengths_by_session[payload["session_id"]]
            passed_tests = 1
            return {
                "passed": turn == 4,
                "passed_tests": passed_tests,
                "total_tests": 2,
                "abort_kind": None,
            }
        raise AssertionError(route)

    return post_json
