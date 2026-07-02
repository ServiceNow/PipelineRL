import pytest

from pipelinerl.async_llm import make_training_texts_from_llm_calls
from pipelinerl.finetune.rl import RLConfig, populate_rl_data
from pipelinerl.rollouts import (
    TrainingText,
    apply_rollout_reward,
    rollout_has_overflow,
    summarize_training_texts,
)


def _make_entry(
    *,
    group_id: str,
    rollout_index: int,
    step_index: int,
    n_tokens: int,
    reward: float,
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
    }


def test_populate_rl_data_uses_total_rollout_tokens_for_multiturn_groups() -> None:
    dataset = [
        _make_entry(group_id="g0", rollout_index=0, step_index=0, n_tokens=3, reward=1.0),
        _make_entry(group_id="g0", rollout_index=0, step_index=1, n_tokens=5, reward=1.0),
        _make_entry(group_id="g0", rollout_index=1, step_index=0, n_tokens=4, reward=0.0),
    ]

    processed = populate_rl_data(dataset=dataset, eos_token_id=999, config=RLConfig())

    expected_group_tokens = 6.0  # mean of rollout totals: (3 + 5) and 4
    assert processed[0]["group_tokens"] == [expected_group_tokens] * 3
    assert processed[1]["group_tokens"] == [expected_group_tokens] * 5
    assert processed[2]["group_tokens"] == [expected_group_tokens] * 4


def test_populate_rl_data_supports_step_specific_rewards_within_a_rollout() -> None:
    dataset = [
        _make_entry(group_id="g0", rollout_index=0, step_index=0, n_tokens=3, reward=1.0),
        _make_entry(group_id="g0", rollout_index=0, step_index=1, n_tokens=5, reward=0.5),
        _make_entry(group_id="g0", rollout_index=1, step_index=0, n_tokens=4, reward=0.0),
        _make_entry(group_id="g0", rollout_index=1, step_index=1, n_tokens=2, reward=0.25),
    ]

    processed = populate_rl_data(
        dataset=dataset,
        eos_token_id=999,
        config=RLConfig(divide_advantage_by_std=False),
    )

    assert processed[0]["advantages"] == [1.0] * 3
    assert processed[1]["advantages"] == [0.25] * 5
    assert processed[2]["advantages"] == [-1.0] * 4
    assert processed[3]["advantages"] == [-0.25] * 2


def test_populate_rl_data_step_index_loo_zeroes_unpaired_late_turns() -> None:
    dataset = [
        _make_entry(group_id="g0", rollout_index=0, step_index=0, n_tokens=2, reward=1.0),
        _make_entry(group_id="g0", rollout_index=0, step_index=1, n_tokens=2, reward=1.0),
        _make_entry(group_id="g0", rollout_index=1, step_index=0, n_tokens=2, reward=-1.0),
        _make_entry(group_id="g0", rollout_index=1, step_index=1, n_tokens=2, reward=-1.0),
        _make_entry(group_id="g0", rollout_index=1, step_index=2, n_tokens=2, reward=-1.0),
        _make_entry(group_id="g0", rollout_index=1, step_index=3, n_tokens=2, reward=-1.0),
    ]

    processed = populate_rl_data(
        dataset=dataset,
        eos_token_id=999,
        config=RLConfig(divide_advantage_by_std=False),
    )

    assert processed[0]["advantages"] == [2.0] * 2
    assert processed[1]["advantages"] == [2.0] * 2
    assert processed[2]["advantages"] == [-2.0] * 2
    assert processed[3]["advantages"] == [-2.0] * 2
    assert processed[4]["advantages"] == [0.0] * 2
    assert processed[5]["advantages"] == [0.0] * 2


def test_populate_rl_data_rollout_level_loo_credits_unpaired_late_turns() -> None:
    dataset = [
        _make_entry(group_id="g0", rollout_index=0, step_index=0, n_tokens=2, reward=1.0),
        _make_entry(group_id="g0", rollout_index=0, step_index=1, n_tokens=2, reward=1.0),
        _make_entry(group_id="g0", rollout_index=1, step_index=0, n_tokens=2, reward=-1.0),
        _make_entry(group_id="g0", rollout_index=1, step_index=1, n_tokens=2, reward=-1.0),
        _make_entry(group_id="g0", rollout_index=1, step_index=2, n_tokens=2, reward=-1.0),
        _make_entry(group_id="g0", rollout_index=1, step_index=3, n_tokens=2, reward=-1.0),
    ]

    processed = populate_rl_data(
        dataset=dataset,
        eos_token_id=999,
        config=RLConfig(divide_advantage_by_std=False, rollout_level_loo=True),
    )

    assert processed[0]["advantages"] == [2.0] * 2
    assert processed[1]["advantages"] == [2.0] * 2
    assert processed[2]["advantages"] == [-2.0] * 2
    assert processed[3]["advantages"] == [-2.0] * 2
    assert processed[4]["advantages"] == [-2.0] * 2
    assert processed[5]["advantages"] == [-2.0] * 2


def test_apply_rollout_reward_broadcasts_reward_to_each_turn() -> None:
    training_texts = [
        TrainingText(text="a", n_predicted=1),
        TrainingText(text="b", n_predicted=1),
    ]

    updated = apply_rollout_reward(training_texts, reward=0.75)

    assert [text.reward for text in updated] == [0.75, 0.75]


def test_summarize_training_texts_reports_turn_level_summary() -> None:
    training_texts = [
        TrainingText(text="userassistant", n_predicted=9, finished=True, prompt_tokens=3, output_tokens=4),
        TrainingText(text="userassistant", n_predicted=9, finished=False, prompt_tokens=5, output_tokens=6),
    ]

    summary = summarize_training_texts(training_texts)

    assert summary.prompt_tokens == [3, 5]
    assert summary.output_tokens == [4, 6]
    assert summary.num_turns == 2
    assert summary.overflow is True


def test_make_training_texts_from_llm_calls_can_broadcast_rollout_reward(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_training_texts = [
        TrainingText(text="first", n_predicted=1, reward=0.0),
        TrainingText(text="second", n_predicted=1, reward=0.0),
    ]

    def fake_make_training_text(llm, llm_call):
        return mock_training_texts.pop(0)

    monkeypatch.setattr("pipelinerl.async_llm.make_training_text", fake_make_training_text)

    training_texts = make_training_texts_from_llm_calls(object(), [object(), object()], reward=1.25)

    assert [text.reward for text in training_texts] == [1.25, 1.25]


def test_rollout_has_overflow_is_true_if_any_turn_is_unfinished() -> None:
    training_texts = [
        TrainingText(text="userassistant", n_predicted=9, finished=True, prompt_tokens=3, output_tokens=4),
        TrainingText(text="userassistant", n_predicted=9, finished=False, prompt_tokens=5, output_tokens=6),
    ]

    assert rollout_has_overflow(training_texts) is True
