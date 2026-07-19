import json
import pickle
from pathlib import Path

import pytest
import torch

import pipelinerl.prerun_evidence as prerun_evidence
from pipelinerl.actor import (
    actor_result_payload_size,
    make_calibration_group_evidence,
)
from pipelinerl.domains.tau2.prerun import (
    CALIBRATION_CAVEAT,
    GIB,
    CalibrationGroupEvidence,
    PINNED_SOURCES,
    PreRunSpec,
    ServiceIdentity,
    TrainerMemoryCandidate,
    build_trainer_memory_budgets,
    finalize_prerun_manifest,
    require_ready_manifest,
    select_calibration_problems,
    summarize_calibration,
)
from pipelinerl.prerun_evidence import (
    EndpointTransferReceipt,
    ParameterTransferReceipt,
    ParityEvidence,
    TensorFingerprint,
    TransferEvidence,
    WorkerTransferReceipt,
    append_evidence_record,
    hash_model_snapshot,
    inspect_gemma_snapshot,
    make_parity_evidence,
    parameter_categories,
    tensor_fingerprint,
)
from pipelinerl.rollouts import (
    BaseMetrics,
    RolloutResult,
    TrainingText,
)


POLICY_ENDPOINTS = [
    "http://actor-0:8000",
    "http://actor-1:8000",
]


def _memory_candidates() -> list[TrainerMemoryCandidate]:
    return [
        TrainerMemoryCandidate(
            name="4x8-sp1",
            node_count=4,
            gpus_per_node=8,
            actor_gpus=8,
            trainer_gpus=24,
            seq_parallel=1,
            gpu_memory_bytes=80 * GIB,
            reserve_bytes_per_gpu=8 * GIB,
        ),
        TrainerMemoryCandidate(
            name="6x8-sp2",
            node_count=6,
            gpus_per_node=8,
            actor_gpus=8,
            trainer_gpus=40,
            seq_parallel=2,
            gpu_memory_bytes=80 * GIB,
            reserve_bytes_per_gpu=8 * GIB,
        ),
        TrainerMemoryCandidate(
            name="8x8-sp8-cpu-offload",
            node_count=8,
            gpus_per_node=8,
            actor_gpus=8,
            trainer_gpus=56,
            seq_parallel=8,
            gpu_memory_bytes=80 * GIB,
            reserve_bytes_per_gpu=8 * GIB,
            optimizer_cpu_offload=True,
        ),
    ]


def _snapshot(tmp_path: Path) -> Path:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    config = {
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "num_hidden_layers": 30,
            "hidden_size": 2816,
            "num_experts": 128,
            "top_k_experts": 8,
            "moe_intermediate_size": 704,
            "max_position_embeddings": 262_144,
            "vocab_size": 262_144,
            "tie_word_embeddings": True,
        },
    }
    (snapshot / "config.json").write_text(
        json.dumps(config)
    )
    weight_map = {
        "model.language_model.embed_tokens.weight": (
            "model-00001-of-00002.safetensors"
        ),
        "model.language_model.layers.0.experts.gate_up_proj": (
            "model-00001-of-00002.safetensors"
        ),
        "model.language_model.layers.0.experts.down_proj": (
            "model-00001-of-00002.safetensors"
        ),
        "model.language_model.layers.0.router.proj.weight": (
            "model-00002-of-00002.safetensors"
        ),
        "model.language_model.layers.0.router.scale": (
            "model-00002-of-00002.safetensors"
        ),
        "model.vision_tower.encoder.layers.0.weight": (
            "model-00002-of-00002.safetensors"
        ),
    }
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(
        b"text-shard-one"
    )
    (snapshot / "model-00002-of-00002.safetensors").write_bytes(
        b"text-and-vision-shard-two"
    )
    (snapshot / "tokenizer.json").write_text('{"version":"1"}')
    return snapshot


def _fingerprint(value: float) -> TensorFingerprint:
    return TensorFingerprint(
        shape=[2],
        dtype="torch.float32",
        numel=2,
        finite_count=2,
        value_sum=value,
        squared_sum=value * value,
    )


def _source_fingerprints() -> dict[str, TensorFingerprint]:
    return {
        "model.embed_tokens.weight": _fingerprint(1.0),
        "model.layers.0.experts.gate_up_proj": _fingerprint(2.0),
        "model.layers.0.router.proj.weight": _fingerprint(3.0),
        "model.layers.0.self_attn.qkv_proj.weight": (
            _fingerprint(4.0)
        ),
    }


def _loaded_names(source_name: str) -> list[str]:
    if ".experts." in source_name:
        return [
            "model.layers.0.moe.experts.0.w13_weight",
            "model.layers.0.moe.experts.1.w13_weight",
        ]
    if "qkv_proj" in source_name:
        return [
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
        ]
    return [source_name]


def _worker_receipt(
    rank: int,
    sources: dict[str, TensorFingerprint],
) -> WorkerTransferReceipt:
    receipts = []
    for name, fingerprint in sources.items():
        loaded_names = _loaded_names(name)
        categories = {
            category
            for parameter_name in [name, *loaded_names]
            for category in parameter_categories(parameter_name)
        }
        receipts.append(
            ParameterTransferReceipt(
                source_name=name,
                loaded_names=loaded_names,
                categories=sorted(categories),
                received_fingerprint=fingerprint,
            )
        )
    return WorkerTransferReceipt(
        rank=rank,
        receipts=receipts,
    )


def _transfer_evidence() -> TransferEvidence:
    sources = _source_fingerprints()
    return TransferEvidence(
        phase="after_optimizer",
        version=192,
        source_fingerprints=sources,
        endpoints=[
            EndpointTransferReceipt(
                endpoint=endpoint,
                workers=[
                    _worker_receipt(0, sources),
                    _worker_receipt(1, sources),
                ],
            )
            for endpoint in POLICY_ENDPOINTS
        ],
    )


def _calibration_groups() -> list[CalibrationGroupEvidence]:
    groups = []
    index = 0
    for dataset in ("airline", "retail", "telecom"):
        for task_index in range(4):
            groups.append(
                CalibrationGroupEvidence(
                    group_id=f"train_{index}",
                    dataset=dataset,
                    task_id=f"{dataset}-{task_index}",
                    attempted_rollouts=16,
                    published_rollouts=16,
                    actor_result_bytes=32_000_000 + index,
                    training_envelope_bytes=30_000_000 + index,
                    call_prefix_tokens=[1000 + index] * 16,
                    generation_tokens=[200 + index] * 16,
                    merged_sequence_tokens=[1200 + 2 * index] * 16,
                )
            )
            index += 1
    return groups


def _parity_events() -> list[ParityEvidence]:
    events = []
    for phase, version in (
        ("before_update", 0),
        ("after_update", 192),
    ):
        for endpoint in POLICY_ENDPOINTS:
            events.append(
                make_parity_evidence(
                    phase=phase,
                    version=version,
                    endpoint=endpoint,
                    prompt_token_ids=[2, 10],
                    completion_token_ids=[20, 30],
                    trainer_logprobs=[-0.5, -0.7],
                    vllm_logprobs=[-0.51, -0.69],
                )
            )
    return events


def test_snapshot_identity_and_gemma_text_topology_are_immutable(
    tmp_path: Path,
):
    snapshot = _snapshot(tmp_path)

    first = hash_model_snapshot(snapshot, "model", "revision")
    second = hash_model_snapshot(snapshot, "model", "revision")
    topology = inspect_gemma_snapshot(snapshot)

    assert first == second
    assert first.snapshot_digest
    assert {
        artifact.path for artifact in first.artifacts
    } >= {
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
    }
    assert topology.num_hidden_layers == 30
    assert topology.hidden_size == 2816
    assert topology.num_experts == 128
    assert topology.top_k_experts == 8
    assert topology.transfer_categories == [
        "embedding",
        "expert",
        "output_head",
        "router",
    ]
    assert topology.vision_tensor_count == 1

    (snapshot / "tokenizer.json").write_text('{"version":"2"}')
    changed = hash_model_snapshot(snapshot, "model", "revision")
    assert changed.snapshot_digest != first.snapshot_digest


def test_calibration_selection_and_caps_record_exact_small_sample():
    problems = [
        {
            "dataset": dataset,
            "task_id": f"{dataset}-{task_index:02d}",
            "task": {"id": f"{dataset}-{task_index:02d}"},
        }
        for dataset in ("telecom", "airline", "retail")
        for task_index in reversed(range(6))
    ]

    selected = select_calibration_problems(problems)
    assert [problem["dataset"] for problem in selected] == (
        ["airline"] * 4 + ["retail"] * 4 + ["telecom"] * 4
    )
    assert [problem["task_id"] for problem in selected[:4]] == [
        "airline-00",
        "airline-01",
        "airline-02",
        "airline-03",
    ]

    summary = summarize_calibration(_calibration_groups())
    assert summary.groups == 12
    assert summary.attempted_rollouts == 192
    assert summary.published_rollouts == 192
    assert summary.domain_groups == {
        "airline": 4,
        "retail": 4,
        "telecom": 4,
    }
    for limit in summary.derived_limits.values():
        assert limit.observed_values
        assert limit.derived_value >= limit.observed_max
        assert limit.calibration_rollouts == 192
        assert limit.calibration_groups == 12
        assert limit.caveat == CALIBRATION_CAVEAT


def test_memory_budget_escalates_from_tight_4x8_candidate():
    groups = _calibration_groups()
    for index, group in enumerate(groups):
        group.merged_sequence_tokens = [120_000 + index] * 16
    calibration = summarize_calibration(groups)

    remaining = 26_000_000_000
    fingerprints = {}
    tensor_index = 0
    while remaining:
        numel = min(4_000_000_000, remaining)
        fingerprints[f"model.tensor_{tensor_index}"] = (
            TensorFingerprint(
                shape=[numel],
                dtype="torch.bfloat16",
                numel=numel,
                finite_count=numel,
                value_sum=0.0,
                squared_sum=0.0,
            )
        )
        remaining -= numel
        tensor_index += 1
    transfer = TransferEvidence(
        phase="after_optimizer",
        version=192,
        source_fingerprints=fingerprints,
        endpoints=[],
    )

    budgets, selected = build_trainer_memory_budgets(
        calibration,
        [transfer],
        _memory_candidates(),
    )

    assert budgets[0].candidate.name == "4x8-sp1"
    assert budgets[0].measured_p99_merged_tokens == 120_011
    assert budgets[0].fits is False
    assert budgets[1].candidate.name == "6x8-sp2"
    assert budgets[1].fits is True
    assert selected is not None
    assert selected.name == "6x8-sp2"
    assert budgets[1].activation_bytes_per_gpu < (
        budgets[0].activation_bytes_per_gpu
    )
    assert budgets[0].all_gather_temp_bytes_per_gpu == (
        4_000_000_000 * 2
    )
    assert budgets[2].zero3_gpu_bytes_per_parameter == 4
    assert budgets[2].optimizer_cpu_bytes_per_rank > 0
    assert all(
        budget.sample_size_caveat == CALIBRATION_CAVEAT
        for budget in budgets
    )


def test_tensor_fingerprint_covers_all_values_and_finiteness():
    tensor = torch.tensor([1.0, -2.0, 3.0])
    fingerprint = tensor_fingerprint(tensor)

    assert fingerprint.numel == 3
    assert fingerprint.finite_count == 3
    assert fingerprint.value_sum == pytest.approx(2.0)
    assert fingerprint.squared_sum == pytest.approx(14.0)
    assert fingerprint.matches(
        tensor_fingerprint(tensor.clone())
    )
    assert not fingerprint.matches(
        tensor_fingerprint(tensor + 1)
    )


def test_finalizer_requires_all_nine_gates_and_records_evidence(
    tmp_path: Path,
):
    evidence_dir = tmp_path / "evidence"
    append_evidence_record(
        evidence_dir,
        "transfer",
        _transfer_evidence(),
    )
    for event in _parity_events():
        append_evidence_record(
            evidence_dir,
            "parity",
            event,
        )
    for group in _calibration_groups():
        append_evidence_record(
            evidence_dir,
            "calibration_groups",
            group,
        )
    spec = PreRunSpec(
        model_snapshot=str(_snapshot(tmp_path)),
        source_pins=PINNED_SOURCES,
        user_simulator=ServiceIdentity(
            model="gpt-4.1-2025-04-14",
            endpoint="https://api.openai.com/v1",
        ),
        policy_model=(
            "google/gemma-4-26B-A4B-it@"
            "01e5b3ee840d3a9e0b0b493c593e85398a30ef75"
        ),
        policy_endpoints=POLICY_ENDPOINTS,
        expected_tp_size=2,
        fixed_prompt_token_ids=[2, 10],
        fixed_completion_token_ids=[20, 30],
        user_separation_asserted=True,
        packing_enabled=False,
        mixed_version_loss_gate_passed=True,
        production_memory_candidates=_memory_candidates(),
        job_spec_sha256="a" * 64,
    )

    manifest = finalize_prerun_manifest(
        spec,
        evidence_dir,
    )

    require_ready_manifest(manifest)
    assert manifest.ready is True
    assert len(manifest.gates) == 9
    assert all(gate.passed for gate in manifest.gates.values())
    assert manifest.parity_tolerance == 0.05
    assert "6.4 BF16 epsilons" in (
        manifest.parity_tolerance_basis
    )
    assert "actor publication only" in manifest.audit_semantics
    assert len(manifest.trainer_memory_budgets) == 3
    assert manifest.trainer_memory_budgets[0].sample_size_caveat == (
        CALIBRATION_CAVEAT
    )
    assert manifest.recommended_production_topology is not None
    assert manifest.recommended_production_topology.name == "4x8-sp1"
    assert "n_predicted=0" in manifest.tau2_text_views
    assert "Verified 2026-07-19 by Claude" in (
        manifest.model_revision_provenance
    )
    assert "refs API reports" in manifest.model_revision_provenance
    assert "/mnt/llmd/base_models/gemma-4-26B-A4B-it" in (
        manifest.topology_provenance
    )
    assert "1127684971bbca40465435a5cad69d67" in (
        manifest.topology_provenance
    )
    assert "developers.openai.com" in manifest.user_simulator_provenance


def test_actor_calibration_record_uses_attempted_oversize_and_real_spans():
    metrics = BaseMetrics(
        reward=1.0,
        success=True,
        no_error=True,
        no_answer=False,
    )
    results = []
    for rollout_index in range(2):
        text = TrainingText(
            text="trajectory",
            n_predicted=0,
            input_ids=[1, 2, 3],
            labels=[-100, 2, 3],
            prompt_tokens=1,
            output_tokens=2,
            metadata={},
        )
        results.append(
            RolloutResult(
                training_texts=[text],
                metrics=metrics,
                latency=0.1,
                dataset_name="airline",
                group_id="train_0",
                domain="tau2",
                atomic_group=True,
                audit={
                    "task_id": "airline-0",
                    "sequence_tokens": 3,
                    "model_calls": [
                        {
                            "token_start": 1,
                            "token_end": 3,
                        }
                    ],
                    "rollout_index": rollout_index,
                },
            )
        )

    measured = actor_result_payload_size(results)
    assert measured == len(pickle.dumps(results))

    results[0].audit["boundary_failure"] = {
        "reason": "actor_queue_oversize",
        "queue_hop": "actor_result",
        "serialized_size": 35_000_000,
    }
    assert actor_result_payload_size(results) == 35_000_000

    record = make_calibration_group_evidence(
        results,
        attempts=2,
        actor_result_bytes=35_000_000,
        training_envelope_bytes=30_000_000,
        drop_reason=None,
    )
    assert record.published_rollouts == 2
    assert record.call_prefix_tokens == [1, 1]
    assert record.generation_tokens == [2, 2]
    assert record.merged_sequence_tokens == [3, 3]
