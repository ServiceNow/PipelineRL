import copy
from types import SimpleNamespace

import pytest

import pipelinerl.preprocess as preprocess
from pipelinerl.domains.terminal.rollouts import stamp_turn_mean_logprob
from pipelinerl.finetune.rl import RLConfig, populate_rl_data
from pipelinerl.llm import LLMCall, LLMOutput, Prompt, TokenLogprob
from pipelinerl.rollouts import TrainingText


def _llm_call(logprobs: list[float]) -> LLMCall:
    call = LLMCall(
        prompt=Prompt.from_user_message("task"),
        output=LLMOutput(content=""),
        cached=False,
    )
    call.logprobs = [
        TokenLogprob(logprob=logprob, token_id=i)
        for i, logprob in enumerate(logprobs)
    ]
    return call


def _text() -> TrainingText:
    return TrainingText(text="x", n_predicted=1)


def test_stamp_turn_mean_logprob_sets_mean_and_skips_absent_logprobs() -> None:
    with_logprobs = _text()
    stamp_turn_mean_logprob(with_logprobs, _llm_call([-1.0, -2.0, -3.0]))

    without_logprobs = _text()
    stamp_turn_mean_logprob(without_logprobs, _llm_call([]))

    assert with_logprobs.metadata["turn_mean_logprob"] == -2.0
    assert "turn_mean_logprob" not in without_logprobs.metadata


def test_preprocess_dataset_flattens_turn_mean_logprob_with_default(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_preprocess_fn(entry, tokenizer, seq_length, is_rl=False):
        return {
            "input_ids": [1, 2],
            "labels": [-100, 2],
            "attention_mask": [1, 1],
        }

    def fake_populate_rl_data(dataset, eos_token_id, config):
        return dataset

    monkeypatch.setattr(preprocess, "replace_oov_tokens_with_the", lambda data, tokenizer: data)
    monkeypatch.setattr(preprocess, "preprocess_fn", fake_preprocess_fn)
    monkeypatch.setattr(preprocess, "populate_rl_data", fake_populate_rl_data)

    data = [
        {
            "text": "a",
            "n_predicted": 1,
            "reward": 1.0,
            "logprobs": [-0.1],
            "metadata": {"model_version": 3, "rollout_index": 0, "step_index": 0},
        },
        {
            "text": "b",
            "n_predicted": 1,
            "reward": 0.0,
            "logprobs": [-0.2],
            "metadata": {
                "model_version": 3,
                "rollout_index": 1,
                "step_index": 0,
                "turn_mean_logprob": -2.5,
            },
        },
    ]

    processed = preprocess.preprocess_dataset(
        llm=None,
        data=data,
        tokenizer=SimpleNamespace(eos_token_id=999),
        seq_length=8,
        rl_config=RLConfig(),
    )

    assert [entry["turn_mean_logprob"] for entry in processed] == [0.0, -2.5]


def _rl_entry(*, rollout_index: int, reward: float, turn_mean_logprob: float | None = None) -> dict:
    entry = {
        "group_id": "g0",
        "rollout_index": rollout_index,
        "step_index": 0,
        "input_ids": [1, 2, 3],
        "labels": [1, 1, 1],
        "rewards": [reward, reward, reward],
        "advantages": [0.0, 0.0, 0.0],
        "group_tokens": [0.0, 0.0, 0.0],
        "overflow": [0.0, 0.0, 0.0],
        "finished": True,
    }
    if turn_mean_logprob is not None:
        entry["turn_mean_logprob"] = turn_mean_logprob
    return entry


def test_populate_rl_data_keeps_turn_mean_logprob_measurement_only() -> None:
    base_dataset = [
        _rl_entry(rollout_index=0, reward=1.0),
        _rl_entry(rollout_index=1, reward=0.0),
    ]
    measurement_dataset = [
        _rl_entry(rollout_index=0, reward=1.0, turn_mean_logprob=-1.5),
        _rl_entry(rollout_index=1, reward=0.0, turn_mean_logprob=-2.5),
    ]
    config = RLConfig(rollout_level_loo=True, divide_advantage_by_std=False)

    without_measurement = populate_rl_data(dataset=copy.deepcopy(base_dataset), eos_token_id=999, config=config)
    with_measurement = populate_rl_data(dataset=copy.deepcopy(measurement_dataset), eos_token_id=999, config=config)

    assert [entry["advantages"] for entry in with_measurement] == [
        entry["advantages"] for entry in without_measurement
    ]
    assert [entry["turn_mean_logprob"] for entry in with_measurement] == [-1.5, -2.5]
