from collections import defaultdict
from contextlib import contextmanager
from multiprocessing import Queue
from multiprocessing.managers import SharedMemoryManager
from types import SimpleNamespace

import pytest

import pipelinerl.preprocess as preprocess_module
from pipelinerl.finetune.data import collate
from pipelinerl.finetune.rl import RLConfig
from pipelinerl.finetune_loop import samples_per_optimizer_step
from pipelinerl.preprocess import (
    AtomicGroupBuffer,
    atomic_envelope_oov_token_ids,
    count_atomic_group_rejection,
    materialize_atomic_update,
    preprocess_atomic_envelope,
    put_atomic_input_queue,
    put_atomic_output_queue,
    replace_oov_tokens_with_the,
    run_dataset_loader,
)
from pipelinerl.rollouts import TrainingGroupEnvelope
from pipelinerl.shared_memory_array import EntrySizeExceeded, SharedMemoryQueue


class _Tokenizer:
    eos_token_id = 2
    padding_side = "right"

    def __init__(self):
        self.vocab_calls = 0

    def get_vocab(self):
        self.vocab_calls += 1
        return {"the": 1, "ok": 2}


def _entry(
    group_id: str,
    rollout_index: int,
    model_version: int,
    *,
    token_id: int = 2,
) -> dict:
    return {
        "input_ids": [token_id],
        "attention_mask": [1],
        "labels": [token_id],
        "rewards": [1.0],
        "advantages": [1.0],
        "ref_logprobs": [-0.2],
        "old_logprobs": [-0.1],
        "group_tokens": [0.0],
        "num_labels": [1.0],
        "overflow": [0.0],
        "model_version": model_version,
        "group_id": group_id,
        "metadata": {
            "model_version": model_version,
            "rollout_index": rollout_index,
            "step_index": 0,
        },
    }


def _envelope(group_id: str, versions: list[int]) -> TrainingGroupEnvelope:
    return TrainingGroupEnvelope(
        group_id=group_id,
        domain="tau2",
        expected_rollouts=len(versions),
        entries=[
            _entry(group_id, rollout_index, version)
            for rollout_index, version in enumerate(versions)
        ],
    )


def test_atomic_composition_preserves_arrival_order_and_never_splits_groups():
    buffer = AtomicGroupBuffer(capacity=10, samples_per_step=5)
    first = _envelope("first", [8, 10])
    crossing = _envelope("crossing", [11, 11, 11, 11])

    assert buffer.enqueue(first, pop_old_data=False)
    assert buffer.compose_update() is None
    assert [envelope.group_id for envelope in buffer.staged_groups] == ["first"]
    assert buffer.staged_entries == 2

    assert buffer.enqueue(crossing, pop_old_data=False)
    update = buffer.compose_update()

    assert [entry["group_id"] for entry in update.entries] == ["first", "first"]
    assert update.n_groups == 1
    assert update.padding == 3
    assert [envelope.group_id for envelope in buffer.ready_groups] == ["crossing"]
    assert buffer.padding == 3

    assert buffer.compose_update() is None
    assert [envelope.group_id for envelope in buffer.staged_groups] == ["crossing"]
    assert buffer.staged_entries == 4


def test_atomic_composition_is_deterministic_for_multiple_complete_groups():
    buffer = AtomicGroupBuffer(capacity=8, samples_per_step=4)
    first = _envelope("first", [8, 8])
    second = _envelope("second", [9, 9])

    assert buffer.enqueue(first, pop_old_data=False)
    assert buffer.enqueue(second, pop_old_data=False)
    update = buffer.compose_update()

    assert [entry["group_id"] for entry in update.entries] == [
        "first",
        "first",
        "second",
        "second",
    ]
    assert update.padding == 0


def test_atomic_ready_queue_evicts_whole_oldest_envelope_and_backpressures():
    buffer = AtomicGroupBuffer(capacity=4, samples_per_step=4)
    first = _envelope("first", [8, 8])
    second = _envelope("second", [9, 9])
    third = _envelope("third", [10, 10])

    assert buffer.enqueue(first, pop_old_data=True)
    assert buffer.enqueue(second, pop_old_data=True)
    assert buffer.enqueue(third, pop_old_data=True)

    assert [envelope.group_id for envelope in buffer.ready_groups] == [
        "second",
        "third",
    ]
    assert buffer.evicted_groups == 1
    assert buffer.evicted_entries == 2

    protected = AtomicGroupBuffer(capacity=2, samples_per_step=4)
    assert protected.enqueue(first, pop_old_data=False)
    assert not protected.enqueue(second, pop_old_data=False)
    assert [envelope.group_id for envelope in protected.ready_groups] == ["first"]


def test_atomic_buffer_rejection_counter_is_intentionally_nonzero():
    buffer = AtomicGroupBuffer(capacity=3, samples_per_step=4)
    oversized = _envelope("oversized", [8, 8, 8, 8])

    assert buffer.enqueue(oversized, pop_old_data=True)
    assert buffer.rejected_groups == 1
    assert buffer.rejected_entries == 4
    assert list(buffer.ready_groups) == []


def test_materialized_padding_uses_newest_real_version_and_collate_stays_conservative():
    buffer = AtomicGroupBuffer(capacity=5, samples_per_step=5)
    group = _envelope("group", [8, 10])
    assert buffer.enqueue(group, pop_old_data=False)
    assert buffer.compose_update() is None
    crossing = _envelope("crossing", [11, 11, 11, 11])
    assert buffer.enqueue(crossing, pop_old_data=False)
    update = buffer.compose_update()

    entries, newest_version = materialize_atomic_update(update, _Tokenizer())

    assert newest_version == 10
    assert len(entries) == 5
    assert [entry["model_version"] for entry in entries[:2]] == [8, 10]
    assert all(entry["model_version"] == 10 for entry in entries[2:])
    assert all(entry["labels"] == [-100] * 8 for entry in entries[2:])
    batch = collate(entries, tokenizer=_Tokenizer(), pad_to_multiple_of=1)
    assert batch.model_version == 8


def test_shared_samples_per_optimizer_step_is_the_capacity_source_of_truth():
    args = SimpleNamespace(
        train_batch_size=4,
        gradient_accumulation_passes=8,
    )

    assert samples_per_optimizer_step(args) == 32


def test_preprocess_input_and_output_oversize_counters_fire_on_real_queues():
    envelope = _envelope("large", [8, 8])
    envelope.entries[0]["payload"] = "x" * 10_000
    input_counts = defaultdict(int)
    output_counts = defaultdict(int)

    with SharedMemoryManager() as smm:
        input_queue = SharedMemoryQueue(smm, max_size=1, max_entry_size=1_000)
        output_queue = SharedMemoryQueue(smm, max_size=1, max_entry_size=1_000)

        assert not put_atomic_input_queue(input_queue, envelope, input_counts)
        put_atomic_output_queue(output_queue, envelope, envelope)
        rejection = output_queue.get()

    assert input_counts["preprocess_input/queue_oversize"] == 1
    assert rejection["group_id"] == "large"
    assert rejection["queue_hop"] == "preprocess_output"
    assert rejection["reason"] == "queue_oversize"
    assert rejection["serialized_size"] > rejection["max_size"] == 1_000
    count_atomic_group_rejection(rejection, output_counts)
    assert output_counts["preprocess_output/queue_oversize"] == 1


def test_preprocess_output_compact_rejection_that_cannot_fit_stays_loud():
    envelope = _envelope("large", [8])
    envelope.entries[0]["payload"] = "x" * 10_000

    with SharedMemoryManager() as smm:
        output_queue = SharedMemoryQueue(smm, max_size=1, max_entry_size=64)
        with pytest.raises(EntrySizeExceeded):
            put_atomic_output_queue(output_queue, envelope, envelope)


def test_atomic_oov_rejection_is_non_mutating_and_legacy_rewrite_is_unchanged():
    tokenizer = _Tokenizer()
    envelope = _envelope("oov", [8])
    envelope.entries[0]["input_ids"] = [2, 99]
    before = envelope.model_dump()

    assert atomic_envelope_oov_token_ids(envelope, tokenizer) == [99]
    rejection = preprocess_atomic_envelope(
        llm=None,
        envelope=envelope,
        tokenizer=tokenizer,
        seq_length=128,
        rl_config=RLConfig(
            policy_loss="reinforce",
            batch_size=1,
        ),
    )

    assert rejection["reason"] == "oov_token_ids"
    assert rejection["invalid_token_ids"] == [99]
    assert envelope.model_dump() == before
    assert tokenizer.vocab_calls == 1
    counts = defaultdict(int)
    count_atomic_group_rejection(rejection, counts)
    assert counts["preprocess_validation/oov_token_ids"] == 1

    legacy_entry = {"input_ids": [2, 99], "logprobs": []}
    legacy_result = replace_oov_tokens_with_the([legacy_entry], tokenizer)
    assert legacy_result[0] is legacy_entry
    assert legacy_result[0]["input_ids"] == [2, 1]


def test_legacy_dataset_loader_keeps_flattened_chunk_contract(monkeypatch):
    first_group = [
        _entry("first", 0, 8),
        _entry("first", 1, 8),
    ]
    second_group = [
        _entry("second", 0, 9),
        _entry("second", 1, 9),
    ]

    class _Reader:
        def __init__(self):
            self.calls = 0

        def read(self):
            self.calls += 1
            if self.calls == 1:
                return iter([first_group, second_group])
            raise RuntimeError("stop loader")

    @contextmanager
    def fake_read_stream(_):
        yield _Reader()

    monkeypatch.setattr(preprocess_module, "read_stream", fake_read_stream)
    raw_queue = Queue(maxsize=2)

    run_dataset_loader(
        raw_queue,
        data_stream=None,
        check_group_size=2,
        chunk_n_groups=2,
        pop_old_data=False,
    )

    assert raw_queue.get(timeout=2) == first_group + second_group
    error = raw_queue.get(timeout=2)
    assert isinstance(error, RuntimeError)
    assert str(error) == "stop loader"


def test_atomic_dataset_loader_preserves_one_complete_envelope(monkeypatch):
    envelope = _envelope("atomic", [8, 9])

    class _Reader:
        def __init__(self):
            self.calls = 0

        def read(self):
            self.calls += 1
            if self.calls == 1:
                return iter([envelope.model_dump()])
            raise RuntimeError("stop loader")

    @contextmanager
    def fake_read_stream(_):
        yield _Reader()

    monkeypatch.setattr(preprocess_module, "read_stream", fake_read_stream)
    raw_queue = Queue(maxsize=2)

    run_dataset_loader(
        raw_queue,
        data_stream=None,
        check_group_size=2,
        chunk_n_groups=2,
        pop_old_data=False,
    )
    loaded = raw_queue.get(timeout=2)
    assert isinstance(loaded, TrainingGroupEnvelope)
    assert loaded == envelope
    error = raw_queue.get(timeout=2)
    assert isinstance(error, RuntimeError)


def test_dataset_loader_fails_before_discarding_a_mixed_chunk(monkeypatch):
    legacy_group = [
        _entry("legacy", 0, 8),
        _entry("legacy", 1, 8),
    ]
    envelope = _envelope("atomic", [9, 9])

    class _Reader:
        def __init__(self):
            self.calls = 0

        def read(self):
            self.calls += 1
            if self.calls == 1:
                return iter([legacy_group, envelope.model_dump()])
            raise AssertionError("mixed stream must fail in its first read")

    @contextmanager
    def fake_read_stream(_):
        yield _Reader()

    monkeypatch.setattr(preprocess_module, "read_stream", fake_read_stream)
    raw_queue = Queue(maxsize=1)

    run_dataset_loader(
        raw_queue,
        data_stream=None,
        check_group_size=2,
        chunk_n_groups=2,
        pop_old_data=False,
    )

    error = raw_queue.get(timeout=2)
    assert isinstance(error, ValueError)
    assert str(error) == "stream mixes atomic envelopes and legacy groups"
