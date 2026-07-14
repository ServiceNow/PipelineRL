import copy
import logging
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from pipelinerl.finetune.data import collate_packed
from pipelinerl.finetune.rl import RLConfig, rl_step
from pipelinerl.finetune.utils import create_sentinel_example
from pipelinerl.finetune_loop import get_batch_sequence_count
from pipelinerl.preprocess import (
    UniformOneGroupBuffer,
    preprocess_dataset,
    select_uniform_one_turns,
    select_uniform_one_turns_and_positions,
    validate_rollout_turn_sampling,
)


class TinyTokenizer:
    eos_token_id = 2

    def get_vocab(self):
        return {"the": 1, "eos": 2}


def _entry(
    group_id: str,
    rollout_index: int,
    step_index: int,
    *,
    advantage: float = 1.0,
    model_version: int = 7,
) -> dict:
    input_ids = [0, 1, 1]
    return {
        "entry_id": f"{group_id}/{rollout_index}/{step_index}",
        "group_id": group_id,
        "rollout_index": rollout_index,
        "step_index": step_index,
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100, 1, 1],
        "rewards": [0.0, advantage, advantage],
        "advantages": [0.0, advantage, advantage],
        "ref_logprobs": [0.0, -0.7, -0.7],
        "old_logprobs": [0.0, -0.7, -0.7],
        "group_tokens": [1.0] * len(input_ids),
        "num_labels": [2.0] * len(input_ids),
        "overflow": [0.0] * len(input_ids),
        "model_version": model_version,
    }


def _group(group_id: str, size: int, *, advantage: float = 1.0) -> list[dict]:
    return [
        _entry(group_id, rollout_index, 0, advantage=advantage)
        for rollout_index in range(size)
    ]


def _selected_ids(groups: list[list[dict]]) -> dict[tuple[str, int], str]:
    return {
        (entry["group_id"], entry["rollout_index"]): entry["entry_id"]
        for group in groups
        for entry in group
    }


def test_config_defaults_to_legacy_all_and_rejects_unknown_mode() -> None:
    cfg = OmegaConf.load(Path(__file__).parents[1] / "conf" / "base.yaml")

    assert cfg.preprocess.rollout_turn_sampling == "all"
    validate_rollout_turn_sampling("all")
    validate_rollout_turn_sampling("uniform_one")
    with pytest.raises(ValueError, match="rollout_turn_sampling='latest'"):
        validate_rollout_turn_sampling("latest")


def test_uniform_one_selects_deterministic_original_entries_per_rollout() -> None:
    dataset = [
        _entry("g0", 0, 4),
        _entry("g0", 0, 0),
        _entry("g0", 0, 2),
        _entry("g0", 1, 7),
        _entry("g1", 0, 1),
        _entry("g1", 0, 5),
    ]
    original = copy.deepcopy(dataset)

    selected = select_uniform_one_turns(dataset)
    permuted = select_uniform_one_turns(list(reversed(dataset)))

    assert [len(group) for group in selected] == [2, 1]
    assert _selected_ids(selected) == _selected_ids(permuted)
    for group in selected:
        for entry in group:
            rollout_entries = [
                candidate
                for candidate in dataset
                if candidate["group_id"] == entry["group_id"]
                and candidate["rollout_index"] == entry["rollout_index"]
            ]
            ordered = sorted(
                rollout_entries, key=lambda candidate: candidate["step_index"]
            )
            index = zlib.crc32(
                f"{entry['group_id']}/{entry['rollout_index']}".encode()
            ) % len(ordered)
            assert entry is ordered[index]
    assert dataset == original

    _, positions = select_uniform_one_turns_and_positions(dataset)
    expected_positions = []
    for group in selected:
        for entry in group:
            rollout_steps = sorted(
                candidate["step_index"]
                for candidate in dataset
                if candidate["group_id"] == entry["group_id"]
                and candidate["rollout_index"] == entry["rollout_index"]
            )
            expected_positions.append(
                rollout_steps.index(entry["step_index"]) / (len(rollout_steps) - 1)
                if len(rollout_steps) > 1
                else 0.0
            )
    assert positions == expected_positions


def test_uniform_one_is_one_rollout_one_opportunity_without_turn_weighting() -> None:
    dataset = [_entry("g", 0, step, advantage=-2.0) for step in range(9)]
    dataset += [_entry("g", 1, 0, advantage=3.0)]

    [selected] = select_uniform_one_turns(dataset)

    assert [entry["rollout_index"] for entry in selected] == [0, 1]
    assert [entry["advantages"][1] for entry in selected] == [-2.0, 3.0]
    assert all("n_turns" not in entry for entry in selected)


def test_group_buffer_evicts_only_complete_ready_groups() -> None:
    buffer = UniformOneGroupBuffer(capacity=5, samples_per_step=8)
    assert buffer.enqueue(_group("old-a", 3), pop_old_data=True)
    assert buffer.enqueue(_group("old-b", 2), pop_old_data=True)
    assert buffer.enqueue(_group("new", 4), pop_old_data=True)

    assert [group[0]["group_id"] for group in buffer.ready_groups] == ["new"]
    assert buffer.ready_entries == 4
    assert buffer.evicted_groups == 2
    assert buffer.evicted_entries == 5


def test_group_buffer_backpressures_without_loss_when_eviction_is_disabled() -> None:
    first = _group("first", 3)
    blocked = _group("blocked", 2)
    buffer = UniformOneGroupBuffer(capacity=4, samples_per_step=8)

    assert buffer.enqueue(first, pop_old_data=False)
    assert not buffer.enqueue(blocked, pop_old_data=False)
    assert list(buffer.ready_groups) == [first]
    assert buffer.ready_entries == 3
    assert buffer.evicted_groups == 0


def test_staged_update_is_outside_queue_capacity_and_protected_from_eviction() -> None:
    staged = _group("staged", 3)
    evicted = _group("evicted", 4)
    crossing = _group("crossing", 2)
    buffer = UniformOneGroupBuffer(capacity=4, samples_per_step=6)

    assert buffer.enqueue(staged, pop_old_data=True)
    assert buffer.compose_update() is None
    assert buffer.staged_entries == 3
    assert buffer.ready_entries == 0

    assert buffer.enqueue(evicted, pop_old_data=True)
    assert buffer.enqueue(crossing, pop_old_data=True)
    update = buffer.compose_update()

    assert update is None
    assert [entry["group_id"] for group in buffer.staged_groups for entry in group] == [
        "staged",
        "staged",
        "staged",
        "crossing",
        "crossing",
    ]
    assert buffer.evicted_groups == 1
    assert buffer.evicted_entries == 4


def test_update_composer_keeps_groups_atomic_and_reports_padding() -> None:
    buffer = UniformOneGroupBuffer(capacity=20, samples_per_step=8)
    for group_id in ("a", "b", "c"):
        assert buffer.enqueue(_group(group_id, 3), pop_old_data=True)

    first = buffer.compose_update()

    assert first is not None
    assert first.n_groups == 2
    assert first.padding == 2
    assert [entry["group_id"] for entry in first.entries] == ["a"] * 3 + ["b"] * 3
    assert [group[0]["group_id"] for group in buffer.ready_groups] == ["c"]

    assert buffer.enqueue(_group("d", 5), pop_old_data=True)
    second = buffer.compose_update()
    assert second is not None
    assert second.padding == 0
    assert [entry["group_id"] for entry in second.entries] == ["c"] * 3 + ["d"] * 5


def test_group_larger_than_update_is_dropped_not_crashed(caplog) -> None:
    buffer = UniformOneGroupBuffer(capacity=20, samples_per_step=8)
    caplog.set_level(logging.WARNING, logger="pipelinerl.preprocess")

    assert buffer.enqueue(_group("too-big", 9), pop_old_data=True)

    assert buffer.ready_entries == 0
    assert buffer.dropped_oversized_groups == 1
    assert buffer.dropped_oversized_entries == 9
    assert "exceeds samples_per_step=8" in caplog.text


def test_padding_examples_count_as_zero_label_sample_slots() -> None:
    buffer = UniformOneGroupBuffer(capacity=20, samples_per_step=8)
    for group_id in ("a", "b", "c"):
        assert buffer.enqueue(_group(group_id, 3), pop_old_data=True)
    update = buffer.compose_update()
    assert update is not None

    examples = list(update.entries)
    examples.extend(
        create_sentinel_example(8, tokenizer=TinyTokenizer(), model_version=7)
        for _ in range(update.padding)
    )
    batch = collate_packed(examples, TinyTokenizer(), seq_parallel=1)

    assert get_batch_sequence_count(batch) == 8
    for segment in range(6, 8):
        assert torch.all(batch.labels[batch.segment_ids == segment] == -100)


class _FixedPolicy(nn.Module):
    def __init__(self, sequence_length: int):
        super().__init__()
        logits = torch.zeros(sequence_length, 2)
        logits[:, 1] = 0.2
        self.logits = nn.Parameter(logits)

    def forward(self, input_ids, attention_mask=None, labels=None, position_ids=None):
        return SimpleNamespace(logits=self.logits.unsqueeze(0))


def test_composed_entries_preserve_dppo_loss_gradients_and_boundaries() -> None:
    direct_entries = _group("a", 2, advantage=1.0) + _group("b", 2, advantage=-1.0)
    buffer = UniformOneGroupBuffer(capacity=8, samples_per_step=4)
    assert buffer.enqueue(direct_entries[:2], pop_old_data=True)
    assert buffer.enqueue(direct_entries[2:], pop_old_data=True)
    update = buffer.compose_update()
    assert update is not None and update.padding == 0

    direct_batch = collate_packed(direct_entries, TinyTokenizer(), seq_parallel=1)
    composed_batch = collate_packed(update.entries, TinyTokenizer(), seq_parallel=1)
    for field in (
        "input_ids",
        "labels",
        "segment_ids",
        "seq_boundaries",
        "advantages",
        "old_logprobs",
        "ref_logprobs",
        "overflow",
    ):
        torch.testing.assert_close(
            getattr(composed_batch, field), getattr(direct_batch, field)
        )

    direct_model = _FixedPolicy(direct_batch.input_ids.shape[1])
    composed_model = copy.deepcopy(direct_model)
    config = RLConfig(
        policy_loss="dppo",
        batch_size=4,
        kl_coef=0.0,
        final_kl_coef=0.0,
    )
    direct_loss, _ = rl_step(direct_model, direct_batch, 0, 1, config)
    composed_loss, _ = rl_step(composed_model, composed_batch, 0, 1, config)
    direct_loss.backward()
    composed_loss.backward()

    torch.testing.assert_close(composed_loss, direct_loss)
    torch.testing.assert_close(composed_model.logits.grad, direct_model.logits.grad)


def test_strict_tito_processing_stats_count_complete_oov_group_drop() -> None:
    data = [
        {
            "group_id": "bad",
            "input_ids": [1, 99],
            "labels": [-100, 99],
            "logprobs": [-0.1],
            "metadata": {"model_version": 1, "rollout_index": 0, "step_index": 0},
            "n_predicted": 1,
            "reward": 0.0,
            "text": "x",
        }
    ]
    stats = {}

    assert (
        preprocess_dataset(
            llm=None,
            data=data,
            tokenizer=TinyTokenizer(),
            seq_length=16,
            rl_config=RLConfig(),
            strict_tito=True,
            processing_stats=stats,
        )
        == []
    )
    assert stats == {"strict_tito_dropped_groups": 1, "strict_tito_dropped_entries": 1}
