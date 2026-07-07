from pipelinerl.domains.terminal.twin_mining import (
    COMMAND_TRUNCATION_SUFFIX,
    common_prefix_len,
    mine_twins,
    normalize_command,
    twin_to_dict,
    yield_report,
)


def _row(index, commands, passed, **overrides):
    row = {
        "group_id": "group-1",
        "task_id": "task-1",
        "rollout_index": index,
        "commands": commands,
        "verifier_pass": passed,
        "reward": 1.0 if passed else -1.0,
        "dropped": False,
    }
    row.update(overrides)
    return row


def test_normalize_command_and_common_prefix_len():
    assert normalize_command("  pytest   -q\n tests  ") == "pytest -q tests"
    assert common_prefix_len(
        [" echo   hi ", "pytest -q", "cat file"],
        ["echo hi", " pytest   -q ", "cat other"],
    ) == 2


def test_mine_twins_requires_mixed_outcome_and_min_prefix_boundary():
    rows = [
        _row(0, ["setup", "inspect", "patch", "pytest"], True),
        _row(1, [" setup ", "inspect", "patch", "pytest -q"], False),
        _row(2, ["setup", "inspect", "different"], False),
        _row(3, ["setup", "inspect", "patch", "extra"], True),
    ]

    assert mine_twins(rows, min_prefix=4) == []
    twins = mine_twins(rows, min_prefix=3)

    assert [twin_to_dict(twin) for twin in twins] == [
        {
            "group_id": "group-1",
            "task_id": "task-1",
            "index_a": 0,
            "index_b": 1,
            "divergence_turn": 3,
            "lcp_len": 3,
            "len_a": 4,
            "len_b": 4,
            "reward_a": 1.0,
            "reward_b": -1.0,
        },
        {
            "group_id": "group-1",
            "task_id": "task-1",
            "index_a": 1,
            "index_b": 3,
            "divergence_turn": 3,
            "lcp_len": 3,
            "len_a": 4,
            "len_b": 4,
            "reward_a": -1.0,
            "reward_b": 1.0,
        },
    ]


def test_mine_twins_skips_dropped_missing_and_truncated_rows():
    rows = [
        _row(0, ["a", "b", "c"], True),
        _row(1, ["a", "b", "c", "d" + COMMAND_TRUNCATION_SUFFIX], False),
        _row(2, ["a", "b", "c"], False, dropped=True),
        _row(3, ["a", "b", "c"], False, group_id=None),
    ]

    assert mine_twins(rows, min_prefix=3) == []


def test_yield_report_counts_groups_pairs_and_histogram():
    rows = [
        _row(0, ["a", "b", "c", "d"], True),
        _row(1, ["a", "b", "c", "x"], False),
        _row(2, ["z"], True, group_id="group-2"),
    ]
    twins = mine_twins(rows, min_prefix=3)

    assert yield_report(rows, twins) == {
        "rollouts_seen": 3,
        "eligible_rollouts": 3,
        "groups_seen": 2,
        "mixed_outcome_groups": 1,
        "twin_pairs": 1,
        "pairs_per_1k_rollouts": 1000.0 / 3,
        "lcp_histogram": {"3": 1},
    }
