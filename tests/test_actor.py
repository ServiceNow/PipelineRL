from pipelinerl.actor import make_rollout_audit_record, write_rollout_audit_records
from pipelinerl.domains.terminal.rollouts import TerminalMetrics
from pipelinerl.rollouts import RolloutResult, TrainingText


class CaptureWriter:
    def __init__(self):
        self.records = []

    def write(self, data):
        self.records.append(data)


def _metrics(reward=-1.0):
    return TerminalMetrics(
        reward=reward,
        success=reward > 0,
        no_error=True,
        no_answer=False,
        verifier_pass=reward > 0,
        passed_tests=1 if reward > 0 else 0,
        total_tests=1,
        pass_fraction=1.0 if reward > 0 else 0.0,
    )


def test_make_rollout_audit_record_preserves_terminal_fields():
    result = RolloutResult(
        training_texts=[],
        metrics=_metrics(),
        latency=1.2,
        model_version=7,
        dataset_name="terminal@train",
        group_id="train_12",
        domain="terminal",
        audit={
            "task_id": "task-1",
            "complexity": "hard",
            "rollout_index": 3,
            "abort_kind": "timeout",
            "abort_phase": "finish",
            "dropped": True,
            "drop_reason": "finish_abort",
            "contamination_result": {"sampled": True, "count": 5},
            "finish_stdout_tail": "pytest timed out",
        },
    )

    record = make_rollout_audit_record(result)

    assert record["task_id"] == "task-1"
    assert record["domain"] == "terminal"
    assert record["complexity"] == "hard"
    assert record["group_id"] == "train_12"
    assert record["rollout_index"] == 3
    assert record["model_version"] == 7
    assert record["dataset_name"] == "terminal@train"
    assert record["reward"] == -1.0
    assert record["verifier_pass"] is False
    assert record["passed_tests"] == 0
    assert record["total_tests"] == 1
    assert record["pass_fraction"] == 0.0
    assert record["abort_kind"] == "timeout"
    assert record["abort_phase"] == "finish"
    assert record["build_ok"] is True
    assert record["init_ok"] is True
    assert record["submitted"] is False
    assert record["format_retry_exceeded"] is False
    assert record["context_exhausted"] is False
    assert record["contamination_result"] == {"sampled": True, "count": 5}
    assert record["finish_stdout_tail"] == "pytest timed out"
    assert record["dropped"] is True
    assert record["drop_reason"] == "finish_abort"
    assert record["n_training_texts"] == 0


def test_write_rollout_audit_records_writes_zero_sample_rollouts():
    writer = CaptureWriter()
    results = [
        RolloutResult(
            training_texts=[],
            metrics=_metrics(),
            latency=0.0,
            model_version=1,
            group_id="g",
            domain="terminal",
            audit={"task_id": "empty", "rollout_index": 0},
        ),
        RolloutResult(
            training_texts=[TrainingText(text="x", n_predicted=1, reward=1.0)],
            metrics=_metrics(reward=1.0),
            latency=0.0,
            model_version=1,
            group_id="g",
            domain="terminal",
            audit={"task_id": "kept", "rollout_index": 1},
        ),
    ]

    write_rollout_audit_records(writer, results)

    assert [record["task_id"] for record in writer.records] == ["empty", "kept"]
    assert [record["n_training_texts"] for record in writer.records] == [0, 1]
