import copy
import logging
from pathlib import Path

from omegaconf import OmegaConf

from pipelinerl.finetune.rl import RLConfig
from pipelinerl.preprocess import drop_oov_groups, preprocess_dataset


class TinyTokenizer:
    eos_token_id = 2

    def get_vocab(self):
        return {"the": 1, "eos": 2}


def _entry(group_id: str, input_ids: list[int]) -> dict:
    return {
        "group_id": group_id,
        "input_ids": input_ids,
        "labels": [-100] * (len(input_ids) - 1) + [input_ids[-1]],
        "logprobs": [-0.1],
        "metadata": {"model_version": 1, "rollout_index": 0, "step_index": 0},
        "n_predicted": 1,
        "reward": 0.0,
        "text": "x",
    }


def test_drop_oov_groups_drops_complete_group_without_mutation(caplog) -> None:
    data = [
        _entry("bad", [1, 99]),
        _entry("bad", [1, 2]),
        _entry("good", [1, 2]),
    ]
    original = copy.deepcopy(data)
    caplog.set_level(logging.WARNING, logger="pipelinerl.preprocess")

    filtered, dropped_groups, dropped_entries = drop_oov_groups(data, TinyTokenizer())

    assert [entry["group_id"] for entry in filtered] == ["good"]
    assert (dropped_groups, dropped_entries) == (1, 2)
    assert data == original
    assert "Strict TITO dropped 2 entries from 1 group(s)" in caplog.text


def test_strict_tito_preprocessing_returns_empty_after_oov_group_drop() -> None:
    data = [_entry("bad", [1, 99]), _entry("bad", [1, 2])]
    original = copy.deepcopy(data)

    result = preprocess_dataset(
        llm=None,
        data=data,
        tokenizer=TinyTokenizer(),
        seq_length=16,
        rl_config=RLConfig(),
        strict_tito=True,
    )

    assert result == []
    assert data == original


def test_terminal_enables_strict_tito() -> None:
    config_dir = Path(__file__).parents[1] / "conf"
    base = OmegaConf.load(config_dir / "base.yaml")
    terminal = OmegaConf.load(config_dir / "terminal.yaml")

    assert base.preprocess.strict_tito is False
    assert terminal.preprocess.strict_tito is True
