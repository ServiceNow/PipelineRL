import json
import pickle
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import pipelinerl.domains.tau2.prerun as tau2_prerun
from pipelinerl.actor import (
    actor_result_payload_size,
    make_calibration_group_evidence,
)
from pipelinerl.domains.tau2.dataset import (
    NEMO_GYM_REPOSITORY,
    TAU2_DATA_REPOSITORY,
    TAU2_NORMALIZATION_CONTRACT,
    Tau2PreparedDataManifest,
    Tau2PreparedFileIdentity,
    ValidatedTau2PreparedData,
)
from pipelinerl.domains.tau2.prerun import (
    CALIBRATION_CAVEAT,
    GIB,
    AUXILIARY_JUDGE_MODEL,
    AUXILIARY_MODEL_ID,
    AUXILIARY_MODEL_REVISION,
    AUXILIARY_USER_MODEL,
    AuxiliaryModelDeployment,
    ServiceIdentity,
    CalibrationGroupEvidence,
    PINNED_SOURCES,
    PreRunSpec,
    validate_auxiliary_model_deployment,
    validate_auxiliary_model_endpoint,
    TrainerMemoryCandidate,
    build_trainer_memory_budgets,
    finalize_prerun_manifest,
    require_ready_manifest,
    select_calibration_problems,
    summarize_calibration,
)
from pipelinerl.prerun_evidence import (
    EndpointTransferReceipt,
    GEMMA_MODEL_DESCRIPTOR,
    QWEN35_27B_MODEL_DESCRIPTOR,
    ParameterTransferReceipt,
    ParityEvidence,
    TensorFingerprint,
    TextModelTopology,
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

USER_SIMULATOR_ENDPOINT = (
    "http://dns-test-account-tau2-user:8000/v1"
)


def _prepared_data_fixture(tmp_path, monkeypatch):
    files = [
        Tau2PreparedFileIdentity(
            dataset=dataset,
            filename=f"tau2_{dataset}.jsonl",
            sha256=str(index) * 64,
            row_count=count,
            task_split_name="base",
            evaluation_type="all",
            timeout=None,
            nl_reward_basis_rows=112 if dataset == "retail" else 0,
            nonempty_nl_assertion_rows={
                "airline": 50,
                "retail": 40,
                "telecom": 0,
            }[dataset],
            task_identity_sha256=str(index + 3) * 64,
        )
        for index, (dataset, count) in enumerate(
            (("airline", 50), ("retail", 114), ("telecom", 114)),
            start=1,
        )
    ]
    manifest = Tau2PreparedDataManifest(
        schema_version=1,
        source_repository=TAU2_DATA_REPOSITORY,
        source_revision=PINNED_SOURCES.tau2_data_sha,
        reference_normalizer_repository=NEMO_GYM_REPOSITORY,
        reference_normalizer_revision=PINNED_SOURCES.nemo_gym_sha,
        normalization_contract=TAU2_NORMALIZATION_CONTRACT,
        files=files,
        total_row_count=278,
        composite_identity_sha256="f" * 64,
    )
    manifest_path = tmp_path / "tau2-prepared-manifest.json"
    manifest_path.write_text("{}\n")
    data_files = {
        dataset: tmp_path / f"tau2_{dataset}.jsonl"
        for dataset in ("airline", "retail", "telecom")
    }
    validated = ValidatedTau2PreparedData(
        manifest=manifest,
        data_files=data_files,
        rows_by_dataset={dataset: [] for dataset in data_files},
    )
    problems = [
        {
            "dataset": dataset,
            "task_id": (
                str(task_index)
                if dataset in ("airline", "retail")
                else f"telecom-{task_index}"
            ),
            "task": {"id": str(task_index)},
        }
        for dataset in ("airline", "retail", "telecom")
        for task_index in range(6)
    ]
    monkeypatch.setattr(
        tau2_prerun,
        "validate_tau2_prepared_data",
        lambda _path: validated,
    )
    monkeypatch.setattr(
        tau2_prerun,
        "load_tau2_problems",
        lambda *_args, **_kwargs: problems,
    )
    return manifest_path, manifest


def _memory_candidates() -> list[TrainerMemoryCandidate]:
    return [
        TrainerMemoryCandidate(
            name="2x8-sp1",
            node_count=2,
            gpus_per_node=8,
            actor_gpus=8,
            trainer_gpus=8,
            seq_parallel=1,
            gpu_memory_bytes=80 * GIB,
            reserve_bytes_per_gpu=8 * GIB,
        ),
        TrainerMemoryCandidate(
            name="3x8-sp1",
            node_count=3,
            gpus_per_node=8,
            actor_gpus=8,
            trainer_gpus=16,
            seq_parallel=1,
            gpu_memory_bytes=80 * GIB,
            reserve_bytes_per_gpu=8 * GIB,
        ),
        TrainerMemoryCandidate(
            name="4x8-sp2",
            node_count=4,
            gpus_per_node=8,
            actor_gpus=8,
            trainer_gpus=24,
            seq_parallel=2,
            gpu_memory_bytes=80 * GIB,
            reserve_bytes_per_gpu=8 * GIB,
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


def _write_auxiliary_snapshot(snapshot: Path) -> None:
    descriptor = QWEN35_27B_MODEL_DESCRIPTOR
    config = {
        "model_type": descriptor.composite_model_type,
        "tie_word_embeddings": descriptor.tie_word_embeddings,
        "text_config": {
            "model_type": descriptor.text_model_type,
            "num_hidden_layers": descriptor.num_hidden_layers,
            "hidden_size": descriptor.hidden_size,
            "intermediate_size": descriptor.intermediate_size,
            "max_position_embeddings": descriptor.max_position_embeddings,
            "vocab_size": descriptor.vocab_size,
            "tie_word_embeddings": descriptor.tie_word_embeddings,
        },
    }
    (snapshot / "config.json").write_text(json.dumps(config))
    shard = "model.safetensors"
    weight_map = {
        "model.language_model.embed_tokens.weight": shard,
        "model.language_model.norm.weight": shard,
        "model.visual.blocks.0.weight": shard,
        "mtp.layers.0.weight": shard,
        "lm_head.weight": shard,
    }
    for layer in range(descriptor.num_hidden_layers):
        weight_map[
            f"model.language_model.layers.{layer}.self_attn.q_proj.weight"
        ] = shard
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    (snapshot / shard).write_bytes(b"auxiliary-weights")
    (snapshot / "tokenizer.json").write_text("{\"version\":\"qwen\"}")


def _auxiliary_model_deployment(
    tmp_path: Path,
    monkeypatch,
) -> AuxiliaryModelDeployment:
    snapshot = tmp_path / "auxiliary-model"
    snapshot.mkdir()
    _write_auxiliary_snapshot(snapshot)
    model = hash_model_snapshot(
        snapshot,
        AUXILIARY_MODEL_ID,
        AUXILIARY_MODEL_REVISION,
    )
    descriptor = replace(
        QWEN35_27B_MODEL_DESCRIPTOR,
        artifact_sha256=tuple(
            (artifact.path, artifact.sha256) for artifact in model.artifacts
        ),
    )
    monkeypatch.setattr(
        tau2_prerun,
        "AUXILIARY_MODEL_DESCRIPTOR",
        descriptor,
    )
    monkeypatch.setattr(
        tau2_prerun,
        "AUXILIARY_MODEL_SNAPSHOT",
        str(snapshot),
    )
    endpoint = USER_SIMULATOR_ENDPOINT
    return AuxiliaryModelDeployment(
        user_service=ServiceIdentity(
            model=AUXILIARY_USER_MODEL,
            endpoint=endpoint,
        ),
        judge_service=ServiceIdentity(
            model=AUXILIARY_JUDGE_MODEL,
            endpoint=endpoint,
        ),
        model=model,
        snapshot_path=str(snapshot),
        job_spec_sha256="b" * 64,
        submission_mode="restartable",
        batch_invariant=False,
        gpu_type="test-h100-80gb",
        gpu_count=2,
        tensor_parallel_size=2,
        max_model_len=20_000,
        max_num_seqs=8,
        measured_peak_in_flight=4,
        observed_request_latencies_s={
            "user_simulator": [0.2, 0.3],
            "judge": [0.5, 0.7],
        },
        observed_prompt_tokens={
            "user_simulator": [4_000, 5_000],
            "judge": [8_000, 9_000],
        },
        prompt_headroom_factors={
            "user_simulator": 1.25,
            "judge": 1.5,
        },
        generation_reserve_tokens={
            "user_simulator": 2_048,
            "judge": 4_096,
        },
        snapshot_hash_bytes=sum(artifact.size for artifact in model.artifacts),
    )


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
    sources = {
        "model.embed_tokens.weight": _fingerprint(1.0),
        "model.layers.0.experts.gate_up_proj": _fingerprint(2.0),
        "model.layers.0.router.proj.weight": _fingerprint(3.0),
    }
    sources.update(
        {
            f"model.layers.{index}.self_attn.q_proj.weight": _fingerprint(
                4.0 + index
            )
            for index in range(30)
        }
    )
    return sources


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
    *,
    tied_output_head: bool = True,
) -> WorkerTransferReceipt:
    receipts = []
    for name, fingerprint in sources.items():
        loaded_names = _loaded_names(name)
        categories = {
            category
            for parameter_name in [name, *loaded_names]
            for category in parameter_categories(
                parameter_name,
                tied_output_head=tied_output_head,
            )
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


def _gemma_text_topology() -> TextModelTopology:
    return TextModelTopology(
        model_type="gemma4",
        text_model_type="gemma4_text",
        num_hidden_layers=30,
        hidden_size=2816,
        intermediate_size=2112,
        num_experts=128,
        top_k_experts=8,
        moe_intermediate_size=704,
        max_position_embeddings=262_144,
        vocab_size=262_144,
        tie_word_embeddings=True,
        layer_indices=list(range(30)),
        text_tensor_count=len(_source_fingerprints()),
        vision_tensor_count=1,
        nontransferred_tensor_count=0,
        transfer_categories=[
            "backbone",
            "embedding",
            "expert",
            "output_head",
            "router",
        ],
    )


def _dense_text_topology() -> TextModelTopology:
    return TextModelTopology(
        model_type="qwen3_5",
        text_model_type="qwen3_5_text",
        num_hidden_layers=2,
        hidden_size=8,
        intermediate_size=16,
        num_experts=0,
        top_k_experts=0,
        moe_intermediate_size=0,
        max_position_embeddings=64,
        vocab_size=32,
        tie_word_embeddings=False,
        layer_indices=[0, 1],
        text_tensor_count=4,
        vision_tensor_count=0,
        nontransferred_tensor_count=0,
        transfer_categories=[
            "backbone",
            "embedding",
            "output_head",
        ],
    )


def _dense_transfer_evidence() -> TransferEvidence:
    sources = {
        "model.embed_tokens.weight": _fingerprint(1.0),
        "model.layers.0.self_attn.q_proj.weight": _fingerprint(2.0),
        "model.layers.1.self_attn.q_proj.weight": _fingerprint(3.0),
        "lm_head.weight": _fingerprint(4.0),
    }
    return TransferEvidence(
        phase="after_optimizer",
        version=192,
        source_fingerprints=sources,
        endpoints=[
            EndpointTransferReceipt(
                endpoint=endpoint,
                workers=[
                    _worker_receipt(
                        rank,
                        sources,
                        tied_output_head=False,
                    )
                    for rank in range(2)
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


def test_auxiliary_deployment_requires_two_role_evidence_and_pins(
    tmp_path,
    monkeypatch,
):
    deployment = _auxiliary_model_deployment(tmp_path, monkeypatch)
    validate_auxiliary_model_deployment(
        deployment,
        deployment.user_service,
        deployment.judge_service,
    )
    extra_gpu = deployment.model_copy(update={"gpu_count": 4})
    with pytest.raises(ValueError, match="GPU/TP"):
        validate_auxiliary_model_deployment(
            extra_gpu,
            extra_gpu.user_service,
            extra_gpu.judge_service,
        )
    too_short = deployment.model_copy(update={"max_model_len": 16_000})
    with pytest.raises(ValueError, match="larger role-specific"):
        validate_auxiliary_model_deployment(
            too_short,
            too_short.user_service,
            too_short.judge_service,
        )
    nonfinite = deployment.model_copy(
        update={
            "observed_request_latencies_s": {
                "user_simulator": [0.2],
                "judge": [float("nan")],
            }
        }
    )
    with pytest.raises(ValueError, match="request-latency evidence"):
        validate_auxiliary_model_deployment(
            nonfinite,
            nonfinite.user_service,
            nonfinite.judge_service,
        )
    incomplete = deployment.model_copy(
        update={
            "observed_prompt_tokens": {
                "user_simulator": [4_000],
            }
        }
    )
    with pytest.raises(ValueError, match="cover exactly"):
        validate_auxiliary_model_deployment(
            incomplete,
            incomplete.user_service,
            incomplete.judge_service,
        )
    reordered = deployment.model_copy(
        update={
            "user_service": deployment.judge_service,
            "judge_service": deployment.user_service,
        }
    )
    with pytest.raises(ValueError, match="ordered model aliases"):
        validate_auxiliary_model_deployment(
            reordered,
            reordered.user_service,
            reordered.judge_service,
        )
    monkeypatch.setattr(
        tau2_prerun,
        "AUXILIARY_MODEL_DESCRIPTOR",
        QWEN35_27B_MODEL_DESCRIPTOR,
    )
    with pytest.raises(ValueError, match="artifact SHA256"):
        validate_auxiliary_model_deployment(
            deployment,
            deployment.user_service,
            deployment.judge_service,
        )


def test_auxiliary_endpoint_requires_resolved_internal_dns():
    assert (
        validate_auxiliary_model_endpoint(USER_SIMULATOR_ENDPOINT)
        == USER_SIMULATOR_ENDPOINT
    )
    with pytest.raises(ValueError, match="resolved account-scoped"):
        validate_auxiliary_model_endpoint("http://dns-<account>-tau2-user:8000/v1")


def test_calibration_selection_and_caps_record_exact_small_sample():
    problems = [
        {
            "dataset": dataset,
            "task_id": (
                str(task_index)
                if dataset in ("airline", "retail")
                else f"telecom-{task_index}"
            ),
            "task": {"id": str(task_index)},
        }
        for dataset in ("telecom", "airline", "retail")
        for task_index in reversed(range(6))
    ]

    selected = select_calibration_problems(problems)
    assert [problem["dataset"] for problem in selected] == (
        ["airline"] * 4 + ["retail"] * 4 + ["telecom"] * 4
    )
    assert [problem["task_id"] for problem in selected[:4]] == [
        "0",
        "1",
        "2",
        "3",
    ]
    assert len(
        {(problem["dataset"], problem["task_id"]) for problem in selected}
    ) == 12
    assert len({problem["task_id"] for problem in selected}) == 8

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


def test_dense_qwen_memory_budget_uses_exact_transfer_surface():
    groups = _calibration_groups()
    for index, group in enumerate(groups):
        group.merged_sequence_tokens = [120_000 + index] * 16
    calibration = summarize_calibration(groups)

    parameter_count = 8_953_803_264
    largest_numel = 1_017_118_720
    remaining = parameter_count - largest_numel
    tensor_count = 426
    per_tensor, remainder = divmod(remaining, tensor_count)
    fingerprints = {
        "lm_head.weight": TensorFingerprint(
            shape=[248_320, 4_096],
            dtype="torch.bfloat16",
            numel=largest_numel,
            finite_count=largest_numel,
            value_sum=0.0,
            squared_sum=0.0,
        )
    }
    for index in range(tensor_count):
        numel = per_tensor + (index < remainder)
        fingerprints[f"model.layers.{index}.weight"] = TensorFingerprint(
            shape=[numel],
            dtype="torch.bfloat16",
            numel=numel,
            finite_count=numel,
            value_sum=0.0,
            squared_sum=0.0,
        )
    assert len(fingerprints) == 427
    assert sum(item.numel for item in fingerprints.values()) == parameter_count
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

    assert budgets[0].candidate.name == "2x8-sp1"
    assert budgets[0].measured_p99_merged_tokens == 120_011
    assert budgets[0].estimated_total_bytes_per_gpu / GIB == pytest.approx(
        86.5778,
        abs=1e-4,
    )
    assert budgets[0].fits is False
    assert budgets[1].candidate.name == "3x8-sp1"
    assert budgets[1].estimated_total_bytes_per_gpu / GIB == pytest.approx(
        78.2389,
        abs=1e-4,
    )
    assert budgets[1].fits is True
    assert selected is not None
    assert selected.name == "3x8-sp1"
    assert budgets[2].candidate.name == "4x8-sp2"
    assert budgets[2].estimated_total_bytes_per_gpu / GIB == pytest.approx(
        45.4565,
        abs=1e-4,
    )
    assert budgets[2].fits is True
    assert budgets[2].candidate.seq_parallel == 2
    assert budgets[0].gpu_memory_bytes == 80 * GIB
    assert budgets[0].all_gather_temp_bytes_per_gpu == largest_numel * 2
    assert all(budget.text_parameter_count == parameter_count for budget in budgets)
    assert all(budget.sample_size_caveat == CALIBRATION_CAVEAT for budget in budgets)


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


def test_dense_untied_transfer_covers_every_layer_and_required_category():
    gate = tau2_prerun._validate_transfer(
        [_dense_transfer_evidence()],
        endpoints=POLICY_ENDPOINTS,
        tp_size=2,
        topology=_dense_text_topology(),
    )

    assert gate.passed is True


def test_transfer_rejects_missing_trainer_source_layer():
    transfer = _dense_transfer_evidence()
    layer_one = "model.layers.1.self_attn.q_proj.weight"
    del transfer.source_fingerprints[layer_one]
    for endpoint in transfer.endpoints:
        for worker in endpoint.workers:
            worker.receipts = [
                receipt
                for receipt in worker.receipts
                if receipt.source_name != layer_one
            ]

    gate = tau2_prerun._validate_transfer(
        [transfer],
        endpoints=POLICY_ENDPOINTS,
        tp_size=2,
        topology=_dense_text_topology(),
    )

    assert gate.passed is False
    assert "trainer source layer coverage mismatch" in gate.detail


def test_transfer_rejects_missing_loaded_layer_on_one_worker():
    transfer = _dense_transfer_evidence()
    worker = transfer.endpoints[0].workers[0]
    layer_one = next(
        receipt
        for receipt in worker.receipts
        if receipt.source_name.startswith("model.layers.1.")
    )
    layer_one.loaded_names = [
        "model.layers.0.self_attn.q_proj.weight"
    ]

    gate = tau2_prerun._validate_transfer(
        [transfer],
        endpoints=POLICY_ENDPOINTS,
        tp_size=2,
        topology=_dense_text_topology(),
    )

    assert gate.passed is False
    assert "rank 0 loaded layer coverage mismatch" in gate.detail


def test_transfer_rejects_pipeline_parallel_worker_topology():
    transfer = _dense_transfer_evidence()
    for endpoint in transfer.endpoints:
        endpoint.workers.extend(
            _worker_receipt(
                rank,
                transfer.source_fingerprints,
                tied_output_head=False,
            )
            for rank in (2, 3)
        )

    gate = tau2_prerun._validate_transfer(
        [transfer],
        endpoints=POLICY_ENDPOINTS,
        tp_size=2,
        topology=_dense_text_topology(),
    )

    assert gate.passed is False
    assert "requires vLLM pipeline parallel size 1" in gate.detail


def test_dense_untied_transfer_requires_output_head():
    transfer = _dense_transfer_evidence()
    del transfer.source_fingerprints["lm_head.weight"]
    for endpoint in transfer.endpoints:
        for worker in endpoint.workers:
            worker.receipts = [
                receipt
                for receipt in worker.receipts
                if receipt.source_name != "lm_head.weight"
            ]

    gate = tau2_prerun._validate_transfer(
        [transfer],
        endpoints=POLICY_ENDPOINTS,
        tp_size=2,
        topology=_dense_text_topology(),
    )

    assert gate.passed is False
    assert "missing categories ['output_head']" in gate.detail


def test_moe_transfer_still_requires_router_category():
    transfer = _transfer_evidence()
    router = "model.layers.0.router.proj.weight"
    del transfer.source_fingerprints[router]
    for endpoint in transfer.endpoints:
        for worker in endpoint.workers:
            worker.receipts = [
                receipt
                for receipt in worker.receipts
                if receipt.source_name != router
            ]

    gate = tau2_prerun._validate_transfer(
        [transfer],
        endpoints=POLICY_ENDPOINTS,
        tp_size=2,
        topology=_gemma_text_topology(),
    )

    assert gate.passed is False
    assert "missing categories ['router']" in gate.detail


def test_finalizer_requires_all_nine_gates_and_records_evidence(
    tmp_path: Path,
    monkeypatch,
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
    auxiliary_model_deployment = _auxiliary_model_deployment(
        tmp_path,
        monkeypatch,
    )
    real_inspect = tau2_prerun.inspect_text_model_snapshot
    auxiliary_topologies = []

    def inspect_snapshot(snapshot, descriptor, identity):
        if descriptor is GEMMA_MODEL_DESCRIPTOR:
            assert identity.model_id == descriptor.model_id
            assert identity.revision == descriptor.revision
            return _gemma_text_topology()
        topology = real_inspect(snapshot, descriptor, identity)
        auxiliary_topologies.append(topology)
        return topology

    monkeypatch.setattr(
        tau2_prerun,
        "inspect_text_model_snapshot",
        inspect_snapshot,
    )
    prepared_data_path, prepared_data = _prepared_data_fixture(
        tmp_path,
        monkeypatch,
    )
    spec = PreRunSpec(
        prepared_data_manifest_path=str(prepared_data_path),
        model_snapshot=str(_snapshot(tmp_path)),
        source_pins=PINNED_SOURCES,
        auxiliary_model_deployment=auxiliary_model_deployment,
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
    assert manifest.schema_version == 3
    assert manifest.prepared_data == prepared_data
    assert len(auxiliary_topologies) == 1
    assert auxiliary_topologies[0].layer_indices == list(range(64))
    assert auxiliary_topologies[0].vision_tensor_count == 1
    assert auxiliary_topologies[0].nontransferred_tensor_count == 1
    assert "judge model=" in manifest.gates["1_user_separation"].detail
    assert len(manifest.gates) == 9
    assert all(gate.passed for gate in manifest.gates.values())
    assert manifest.gates["3_model_transfer"].passed is True
    assert manifest.topology.layer_indices == list(range(30))
    assert "backbone" in manifest.topology.transfer_categories
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
    assert manifest.recommended_production_topology.name == "2x8-sp1"
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
    assert "fc05daec18b0a78c" in manifest.auxiliary_model_provenance
    assert (
        manifest.auxiliary_model_deployment.snapshot_hash_bytes
        == sum(
            artifact.size
            for artifact in auxiliary_model_deployment.model.artifacts
        )
    )
    assert "every non-cache" in (
        manifest.auxiliary_model_snapshot_hash_io
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("config", "text topology"),
        ("index", "shard map"),
    ),
)
def test_finalizer_rejects_auxiliary_load_surface_drift(
    tmp_path,
    monkeypatch,
    mutation,
    message,
):
    deployment = _auxiliary_model_deployment(tmp_path, monkeypatch)
    snapshot = Path(deployment.snapshot_path)
    if mutation == "config":
        config_path = snapshot / "config.json"
        config = json.loads(config_path.read_text())
        config["text_config"]["hidden_size"] += 1
        config_path.write_text(json.dumps(config))
    else:
        index_path = snapshot / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        index["weight_map"][
            "model.language_model.layers.0.self_attn.q_proj.weight"
        ] = "unreviewed.safetensors"
        index_path.write_text(json.dumps(index))
    model = hash_model_snapshot(
        snapshot,
        AUXILIARY_MODEL_ID,
        AUXILIARY_MODEL_REVISION,
    )
    descriptor = replace(
        tau2_prerun.AUXILIARY_MODEL_DESCRIPTOR,
        artifact_sha256=tuple(
            (artifact.path, artifact.sha256)
            for artifact in model.artifacts
        ),
    )
    monkeypatch.setattr(
        tau2_prerun,
        "AUXILIARY_MODEL_DESCRIPTOR",
        descriptor,
    )
    deployment = deployment.model_copy(
        update={
            "model": model,
            "snapshot_hash_bytes": sum(
                artifact.size for artifact in model.artifacts
            ),
        }
    )
    prepared_data_path, prepared_data = _prepared_data_fixture(
        tmp_path,
        monkeypatch,
    )
    spec = PreRunSpec(
        prepared_data_manifest_path=str(prepared_data_path),
        model_snapshot=str(_snapshot(tmp_path)),
        source_pins=PINNED_SOURCES,
        auxiliary_model_deployment=deployment,
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

    with pytest.raises(ValueError, match=message):
        finalize_prerun_manifest(spec, tmp_path / "evidence")


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
