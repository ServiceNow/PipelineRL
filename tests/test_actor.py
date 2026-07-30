from multiprocessing.managers import SharedMemoryManager
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from pipelinerl.actor import (
    ActorLoop,
    SlidingWindowAggregator,
    apply_group_boundary_policy,
    make_rollout_audit_record,
    make_training_group_envelope,
    put_rollout_group,
    stamp_rollout_metadata,
    validate_atomic_group_admission,
    write_rollout_audit_records,
)
from pipelinerl.finetune.data import collate
from pipelinerl.rollouts import BaseMetrics, RolloutResult, TrainingText
from pipelinerl.shared_memory_array import SharedMemoryQueue


class _Writer:
    def __init__(self) -> None:
        self.records = []

    def write(self, record) -> None:
        self.records.append(record)


def _metrics(reward: float) -> BaseMetrics:
    return BaseMetrics(
        reward=reward,
        success=reward > 0,
        no_error=True,
        no_answer=False,
    )


def _result(
    rollout_index: int,
    *,
    reward: float,
    model_version: int | None = 8,
    boundary_reason: str | None = None,
    metrics_available: bool = True,
    atomic_group: bool = False,
    token_count: int = 1,
) -> RolloutResult:
    audit = {
        "rollout_index": rollout_index,
        "reward": reward,
        "model_calls": [
            {
                "call_index": 0,
                "version_start": 8,
                "version_end": 10,
                "endpoint": "http://actor:8000/v1",
                "token_start": 0,
                "token_end": 1,
            }
        ],
    }
    if boundary_reason is not None:
        audit["boundary_failure"] = {
            "reason": boundary_reason,
            "detail": f"failure {boundary_reason}",
        }
    if not metrics_available:
        audit["reward"] = None
        audit["reward_available"] = False
    texts = (
        []
        if boundary_reason is not None
        else [
            TrainingText(
                text="x" * token_count,
                n_predicted=token_count,
                reward=reward,
                input_ids=[1] * token_count,
                labels=[1] * token_count,
                prompt_tokens=0,
                output_tokens=token_count,
                metadata={
                    "model_version": model_version,
                    "rollout_index": rollout_index,
                    "step_index": 0,
                },
            )
        ]
    )
    return RolloutResult(
        training_texts=texts,
        metrics=_metrics(reward) if metrics_available else None,
        latency=0.1,
        model_version=model_version,
        dataset_name="telecom",
        group_id="group-1",
        domain="tau2",
        audit=audit,
        atomic_group=atomic_group,
    )


def test_stamp_rollout_metadata_preserves_exact_tau2_version_and_records_admission_version():
    result = _result(3, reward=1.0, model_version=8)

    stamp_rollout_metadata(
        result,
        scheduler_name="actor",
        group_id=12,
        rollout_index=3,
        admission_model_version=11,
    )

    assert result.model_version == 8
    assert result.group_id == "actor_12"
    assert result.audit["model_version"] == 8
    assert result.audit["admission_model_version"] == 11
    assert result.audit["rollout_index"] == 3
    assert result.training_texts[0].metadata["model_version"] == 8
    assert result.training_texts[0].metadata["rollout_index"] == 3
    assert result.training_texts[0].metadata["step_index"] == 0
    assert result.training_texts[0].group_id == "actor_12"


def test_lowest_rollout_index_boundary_failure_drops_whole_group_once_and_audits_all_members():
    higher_failure = _result(
        2,
        reward=0.25,
        model_version=None,
        boundary_reason="policy_endpoint_mismatch",
    )
    valid_sibling = _result(1, reward=0.75, model_version=9)
    trigger = _result(
        0,
        reward=0.5,
        model_version=None,
        boundary_reason="malformed_gym_response",
        metrics_available=False,
    )
    results = [higher_failure, valid_sibling, trigger]
    counters = {}

    reason = apply_group_boundary_policy(results, counters)

    assert reason == "malformed_gym_response"
    assert counters == {"malformed_gym_response": 1}
    assert higher_failure.metrics is not None
    assert higher_failure.metrics.reward == 0.25
    assert valid_sibling.metrics is not None
    assert valid_sibling.metrics.reward == 0.75
    assert trigger.metrics is None
    assert all(result.audit["published"] is False for result in results)
    assert all(result.audit["entered_training"] is False for result in results)
    assert all(result.audit["drop_reason"] == reason for result in results)
    assert all(result.audit["group_drop_trigger_rollout_index"] == 0 for result in results)
    assert [result.audit["group_drop_triggered_here"] for result in results] == [
        False,
        False,
        True,
    ]
    assert len(valid_sibling.training_texts) == 1

    writer = _Writer()
    write_rollout_audit_records(writer, results)

    assert len(writer.records) == 3
    assert [record["reward"] for record in writer.records] == [0.25, 0.75, None]
    assert all(record["entered_training"] is False for record in writer.records)
    assert [record["n_training_texts"] for record in writer.records] == [0, 1, 0]


def test_metrics_less_boundary_drops_whole_group_once_and_keeps_sibling_reward():
    trigger = _result(
        0,
        reward=0.0,
        model_version=None,
        boundary_reason="termination_user_error",
        metrics_available=False,
    )
    sibling = _result(1, reward=0.75, model_version=9)
    counters = {}

    reason = apply_group_boundary_policy([sibling, trigger], counters)

    assert reason == "termination_user_error"
    assert counters == {"termination_user_error": 1}
    assert trigger.metrics is None
    assert trigger.audit["group_drop_triggered_here"] is True
    assert sibling.audit["group_drop_triggered_here"] is False
    assert sibling.metrics is not None
    assert sibling.metrics.reward == 0.75
    assert all(
        result.audit["entered_training"] is False
        for result in (trigger, sibling)
    )


def test_metrics_less_boundary_stats_omit_reward_and_success_metrics():
    result = _result(
        0,
        reward=0.0,
        model_version=None,
        boundary_reason="judge_reward_invalid",
        metrics_available=False,
    )
    apply_group_boundary_policy([result], {})
    actor = ActorLoop.__new__(ActorLoop)
    actor.is_training = True
    actor.cfg = OmegaConf.create(
        {
            "llm": {"parameters": {"max_tokens": 1}},
            "actor": {},
            "wandb": {"use_wandb": False},
        }
    )
    actor.sliding_aggregator = SlidingWindowAggregator(window_size=2)
    actor.init_stats()

    actor.update_stats([result])

    stats = actor.stats
    assert "reward" not in stats
    assert "success" not in stats
    assert "overlong_success" not in stats
    assert stats["overlong"]["telecom"]["group-1"] == [False]
    assert stats["num_turns"]["telecom"]["group-1"] == [0]
    assert actor.latency_list == [0.1]
    assert make_rollout_audit_record(result)["reward"] is None

    writer = _Writer()
    actor.publish_stats(writer, {})

    published = writer.records[0]
    assert "reward_mean" not in published
    assert "always_success" not in published
    assert "never_success" not in published
    assert "sometimes_success" not in published


def test_generic_metrics_stats_are_unchanged():
    result = _result(0, reward=0.75, model_version=8)
    apply_group_boundary_policy([result], {})
    actor = ActorLoop.__new__(ActorLoop)
    actor.is_training = True
    actor.cfg = OmegaConf.create(
        {
            "llm": {"parameters": {"max_tokens": 1}},
            "actor": {},
            "wandb": {"use_wandb": False},
        }
    )
    actor.sliding_aggregator = SlidingWindowAggregator(window_size=2)
    actor.init_stats()

    actor.update_stats([result])

    stats = actor.stats
    assert stats["reward"]["telecom"]["group-1"] == [0.75]
    assert stats["success"]["telecom"]["group-1"] == [True]
    assert stats["overlong_success"]["telecom"]["group-1"] == [True]

    writer = _Writer()
    actor.publish_stats(writer, {})

    published = writer.records[0]
    assert published["reward_mean"] == 0.75
    assert published["always_success"] == 1.0
    assert published["never_success"] == 0.0
    assert published["sometimes_success"] == 0.0


def test_metrics_less_non_boundary_rollout_is_rejected():
    result = _result(
        0,
        reward=0.0,
        metrics_available=False,
    )

    with pytest.raises(ValueError, match="metrics-less rollout"):
        apply_group_boundary_policy([result], {})


def test_evaluation_group_is_published_without_being_marked_as_training_data():
    result = _result(0, reward=1.0, model_version=8)
    counters = {}

    reason = apply_group_boundary_policy(
        [result],
        counters,
        enters_training=False,
    )

    assert reason is None
    assert counters == {}
    assert result.audit["published"] is True
    assert result.audit["entered_training"] is False


def test_generic_audit_record_preserves_full_per_call_provenance():
    result = _result(0, reward=1.0, model_version=8)

    record = make_rollout_audit_record(result)

    assert record["model_calls"] == result.audit["model_calls"]
    assert record["model_version"] == 8
    assert record["policy_endpoint"] is None
    assert record["prompt_token_lengths"] == [0]
    assert record["output_token_lengths"] == [1]
    assert record["n_training_texts"] == 1


def test_existing_collate_uses_oldest_training_text_model_version():
    examples = [
        {
            "input_ids": [1, 2],
            "attention_mask": [1, 1],
            "labels": [-100, 2],
            "rewards": [0.0, 1.0],
            "advantages": [0.0, 1.0],
            "ref_logprobs": [0.0, -0.2],
            "old_logprobs": [0.0, -0.1],
            "group_tokens": [0.0, 0.0],
            "num_labels": [0.0, 1.0],
            "overflow": [0.0, 0.0],
            "model_version": 10,
        },
        {
            "input_ids": [3],
            "attention_mask": [1],
            "labels": [3],
            "rewards": [1.0],
            "advantages": [1.0],
            "ref_logprobs": [-0.3],
            "old_logprobs": [-0.2],
            "group_tokens": [0.0],
            "num_labels": [1.0],
            "overflow": [0.0],
            "model_version": 8,
        },
    ]

    batch = collate(
        examples,
        tokenizer=SimpleNamespace(padding_side="right"),
        pad_to_multiple_of=1,
    )

    assert batch.model_version == 8


def test_atomic_actor_queue_oversize_drops_whole_group_and_retains_audits():
    results = [
        _result(
            1,
            reward=0.25,
            atomic_group=True,
            token_count=10_000,
        ),
        _result(
            0,
            reward=0.75,
            atomic_group=True,
            token_count=10_000,
        ),
    ]

    with SharedMemoryManager() as smm:
        queue = SharedMemoryQueue(smm, max_size=1, max_entry_size=5_000)
        put_rollout_group(queue, results)
        compact_results = queue.get()

    assert [result.metrics.reward for result in compact_results] == [0.25, 0.75]
    assert all(result.training_texts == [] for result in compact_results)
    assert all(result.audit["model_calls"] for result in compact_results)
    assert all(result.audit["n_training_texts_before_drop"] == 1 for result in compact_results)
    trigger = next(
        result
        for result in compact_results
        if "boundary_failure" in result.audit
    )
    failure = trigger.audit["boundary_failure"]
    assert trigger.audit["rollout_index"] == 0
    assert failure["reason"] == "actor_queue_oversize"
    assert failure["queue_hop"] == "actor_result"
    assert failure["serialized_size"] > failure["max_size"] == 5_000

    counters = {}
    reason = apply_group_boundary_policy(compact_results, counters)

    assert reason == "actor_queue_oversize"
    assert counters == {"actor_queue_oversize": 1}
    assert all(result.audit["entered_training"] is False for result in compact_results)


def test_atomic_envelope_admission_rejection_fires_group_counter():
    results = [
        _result(1, reward=0.25, atomic_group=True),
        _result(0, reward=0.75, atomic_group=True),
    ]

    assert validate_atomic_group_admission(
        results,
        attempts=2,
        samples_per_update=1,
        ready_capacity=2,
    )

    counters = {}
    reason = apply_group_boundary_policy(results, counters)

    assert reason == "atomic_envelope_rejected"
    assert counters == {"atomic_envelope_rejected": 1}
    trigger = next(
        result
        for result in results
        if "boundary_failure" in result.audit
    )
    assert trigger.audit["rollout_index"] == 0
    assert trigger.audit["boundary_failure"]["entry_count"] == 2


def test_atomic_envelope_is_complete_and_deterministically_ordered():
    higher = _result(1, reward=0.25, atomic_group=True)
    lower = _result(0, reward=0.75, atomic_group=True)
    stamp_rollout_metadata(higher, "actor", 7, 1, 10)
    stamp_rollout_metadata(lower, "actor", 7, 0, 10)

    assert validate_atomic_group_admission(
        [higher, lower],
        attempts=2,
        samples_per_update=4,
        ready_capacity=4,
    )
    envelope = make_training_group_envelope([higher, lower], attempts=2)

    assert envelope.kind == "atomic_training_group"
    assert envelope.group_id == "actor_7"
    assert envelope.domain == "tau2"
    assert envelope.expected_rollouts == 2
    assert [
        (
            entry["metadata"]["rollout_index"],
            entry["metadata"]["step_index"],
        )
        for entry in envelope.entries
    ] == [(0, 0), (1, 0)]


def test_legacy_group_admission_is_an_exact_noop():
    results = [
        _result(0, reward=0.25),
        _result(1, reward=0.75),
    ]
    before = [result.model_dump() for result in results]

    assert not validate_atomic_group_admission(
        results,
        attempts=2,
        samples_per_update=1,
        ready_capacity=1,
    )
    assert [result.model_dump() for result in results] == before
