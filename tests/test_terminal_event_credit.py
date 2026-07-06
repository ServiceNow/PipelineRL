import pytest

from pipelinerl.domains.terminal.rollouts import stamp_event_credit
from pipelinerl.finetune.rl import RLConfig, populate_rl_data
from pipelinerl.rollouts import TrainingText


def _text() -> TrainingText:
    return TrainingText(text="x", n_predicted=1)


def test_stamp_event_credit_sets_flags_and_rollout_mean() -> None:
    texts = [_text(), _text(), _text(), _text()]
    stamp_event_credit(texts, [False, True, False, False])
    assert [t.metadata["event_error"] for t in texts] == [0.0, 1.0, 0.0, 0.0]
    assert all(t.metadata["rollout_error_mean"] == 0.25 for t in texts)


def test_stamp_event_credit_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        stamp_event_credit([_text()], [True, False])


def test_stamp_event_credit_empty_is_noop() -> None:
    stamp_event_credit([], [])


def _make_entry(
    *,
    group_id: str = "g0",
    rollout_index: int,
    step_index: int,
    reward: float,
    event_error: float,
    rollout_error_mean: float,
    n_tokens: int = 4,
) -> dict:
    return {
        "group_id": group_id,
        "rollout_index": rollout_index,
        "step_index": step_index,
        "input_ids": list(range(n_tokens)),
        "labels": [1] * n_tokens,
        "rewards": [reward] * n_tokens,
        "advantages": [0.0] * n_tokens,
        "group_tokens": [0.0] * n_tokens,
        "overflow": [0.0] * n_tokens,
        "finished": True,
        "event_error": event_error,
        "rollout_error_mean": rollout_error_mean,
    }


def _config(coef: float) -> RLConfig:
    return RLConfig(rollout_level_loo=True, divide_advantage_by_std=False, event_credit_coef=coef)


def test_event_credit_redistributes_within_rollout() -> None:
    # rollout 0 wins (z = +1 vs rollout 1's reward 0): its error turn gives up
    # credit, its clean turn gains it; rollout 1 has uniform errors -> delta 0.
    dataset = [
        _make_entry(rollout_index=0, step_index=0, reward=1.0, event_error=0.0, rollout_error_mean=0.5),
        _make_entry(rollout_index=0, step_index=1, reward=1.0, event_error=1.0, rollout_error_mean=0.5),
        _make_entry(rollout_index=1, step_index=0, reward=0.0, event_error=1.0, rollout_error_mean=1.0),
    ]
    processed = populate_rl_data(dataset=dataset, eos_token_id=999, config=_config(0.5))
    # A_t = z - coef*|z|*(e_t - mean_e); z=+1: clean 1.25, error 0.75; z=-1 delta=0: -1
    assert processed[0]["advantages"] == [1.25] * 4
    assert processed[1]["advantages"] == [0.75] * 4
    assert processed[2]["advantages"] == [-1.0] * 4
    # rollout total preserved: (1.25 + 0.75) / 2 == z
    assert (processed[0]["advantages"][0] + processed[1]["advantages"][0]) / 2 == 1.0


def test_event_credit_coef_zero_is_noop() -> None:
    dataset = [
        _make_entry(rollout_index=0, step_index=0, reward=1.0, event_error=1.0, rollout_error_mean=0.5),
        _make_entry(rollout_index=0, step_index=1, reward=1.0, event_error=0.0, rollout_error_mean=0.5),
        _make_entry(rollout_index=1, step_index=0, reward=0.0, event_error=0.0, rollout_error_mean=0.0),
    ]
    processed = populate_rl_data(dataset=dataset, eos_token_id=999, config=_config(0.0))
    assert processed[0]["advantages"] == [1.0] * 4
    assert processed[1]["advantages"] == [1.0] * 4
    assert processed[2]["advantages"] == [-1.0] * 4


def test_event_credit_requires_rollout_level_loo() -> None:
    with pytest.raises(ValueError, match="rollout_level_loo"):
        RLConfig(rollout_level_loo=False, event_credit_coef=0.5)


def test_event_credit_coef_must_stay_below_one() -> None:
    with pytest.raises(ValueError):
        RLConfig(rollout_level_loo=True, event_credit_coef=1.0)


def test_event_credit_without_metadata_columns_is_noop() -> None:
    dataset = [
        _make_entry(rollout_index=0, step_index=0, reward=1.0, event_error=0.0, rollout_error_mean=0.0),
        _make_entry(rollout_index=1, step_index=0, reward=0.0, event_error=0.0, rollout_error_mean=0.0),
    ]
    for entry in dataset:
        del entry["event_error"]
        del entry["rollout_error_mean"]
    processed = populate_rl_data(dataset=dataset, eos_token_id=999, config=_config(0.5))
    assert processed[0]["advantages"] == [1.0] * 4
    assert processed[1]["advantages"] == [-1.0] * 4
