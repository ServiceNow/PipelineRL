from types import SimpleNamespace

from pipelinerl.actor import (
    apply_group_boundary_policy,
    make_rollout_audit_record,
    stamp_rollout_metadata,
    write_rollout_audit_records,
)
from pipelinerl.finetune.data import collate
from pipelinerl.rollouts import BaseMetrics, RolloutResult, TrainingText


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
    texts = (
        []
        if boundary_reason is not None
        else [
            TrainingText(
                text="sample",
                n_predicted=1,
                reward=reward,
                input_ids=[1],
                labels=[1],
                prompt_tokens=0,
                output_tokens=1,
                metadata={"model_version": model_version},
            )
        ]
    )
    return RolloutResult(
        training_texts=texts,
        metrics=_metrics(reward),
        latency=0.1,
        model_version=model_version,
        dataset_name="telecom",
        group_id="group-1",
        domain="tau2",
        audit=audit,
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
    )
    results = [higher_failure, valid_sibling, trigger]
    counters = {}

    reason = apply_group_boundary_policy(results, counters)

    assert reason == "malformed_gym_response"
    assert counters == {"malformed_gym_response": 1}
    assert [result.metrics.reward for result in results] == [0.25, 0.75, 0.5]
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
    assert [record["reward"] for record in writer.records] == [0.25, 0.75, 0.5]
    assert all(record["entered_training"] is False for record in writer.records)
    assert [record["n_training_texts"] for record in writer.records] == [0, 1, 0]


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
