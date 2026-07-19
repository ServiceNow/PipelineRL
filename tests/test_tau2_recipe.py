import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir

import pipelinerl.launch as launch
import pipelinerl.prerun_evidence as prerun_evidence
from pipelinerl.domains.tau2.prerun import (
    CALIBRATION_GROUPS,
    CALIBRATION_ROLLOUTS,
    GIB,
    GSPO_TOKEN_UPGRADE_TRIGGER,
    PINNED_SOURCES,
    POLICY_LOSS_FALLBACK,
    POLICY_LOSS_FALLBACK_TRIGGER,
    RUN1_POLICY_LOSS,
    USER_SIMULATOR_ENDPOINT,
    USER_SIMULATOR_MODEL,
    CalibrationSummary,
    DerivedLimit,
    GateResult,
    PreRunManifest,
    PreRunSpec,
    ServiceIdentity,
    TrainerMemoryBudget,
    TrainerMemoryCandidate,
)
from pipelinerl.prerun_evidence import (
    GEMMA_MODEL_ID,
    GEMMA_MODEL_REVISION,
    GEMMA_POLICY_IDENTITY,
    REQUIRED_TRANSFER_CATEGORIES,
    ArtifactDigest,
    GemmaTopology,
    ModelArtifactIdentity,
    hash_model_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_JOB = ROOT / "tau2_gemma_calibration.yaml"
POLICY_ENDPOINTS = [
    "http://tau2-gemma-calibration-3:8080",
    "http://tau2-gemma-calibration-3:8082",
    "http://tau2-gemma-calibration-3:8084",
    "http://tau2-gemma-calibration-3:8086",
]


def _compose_recipe():
    with initialize_config_dir(
        config_dir=str(ROOT / "conf"),
        version_base="1.3.2",
    ):
        return compose(config_name="tau2_gemma")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate() -> TrainerMemoryCandidate:
    return TrainerMemoryCandidate(
        name="4x8-sp1",
        node_count=4,
        gpus_per_node=8,
        actor_gpus=8,
        trainer_gpus=24,
        seq_parallel=1,
        gpu_memory_bytes=80 * GIB,
        reserve_bytes_per_gpu=8 * GIB,
    )


def _limit(value: int) -> DerivedLimit:
    return DerivedLimit(
        observed_values=[value],
        observed_max=value,
        headroom_factor=1.25,
        rounding_quantum=1024,
        derived_value=value,
    )


def _ready_manifest(job_digest: str) -> PreRunManifest:
    candidate = _candidate()
    limits = {
        "actor_queue_entry_bytes": _limit(48 * (1 << 20)),
        "training_envelope_entry_bytes": _limit(48 * (1 << 20)),
        "model_context_tokens": _limit(65536),
        "training_sequence_tokens": _limit(65536),
        "generation_margin_tokens": _limit(16384),
    }
    calibration = CalibrationSummary(
        groups=CALIBRATION_GROUPS,
        attempted_rollouts=CALIBRATION_ROLLOUTS,
        published_rollouts=CALIBRATION_ROLLOUTS,
        drop_counts={},
        domain_groups={"airline": 4, "retail": 4, "telecom": 4},
        call_prefix_tokens=[1024],
        generation_tokens=[16384],
        merged_sequence_tokens=[65536],
        actor_result_bytes=[40 * (1 << 20)],
        training_envelope_bytes=[40 * (1 << 20)],
        derived_limits=limits,
    )
    budget = TrainerMemoryBudget(
        candidate=candidate,
        measured_p99_merged_tokens=65536,
        text_parameter_count=26_000_000_000,
        zero3_gpu_bytes_per_parameter=16,
        sharded_model_optimizer_bytes_per_gpu=17 * GIB,
        optimizer_cpu_bytes_per_rank=0,
        activation_bytes_per_gpu=32 * GIB,
        all_gather_temp_bytes_per_gpu=4 * GIB,
        reserve_bytes_per_gpu=8 * GIB,
        estimated_total_bytes_per_gpu=61 * GIB,
        gpu_memory_bytes=80 * GIB,
        fits=True,
    )
    gates = {
        f"{index}_{name}": GateResult(passed=True, detail="test")
        for index, name in enumerate(
            (
                "user_separation",
                "source_pins",
                "weight_transfer",
                "logprob_parity",
                "packing_isolation",
                "mixed_version_loss",
                "calibration",
                "memory_budget",
                "context_fit",
            ),
            start=1,
        )
    }
    return PreRunManifest(
        job_spec_sha256=job_digest,
        model=ModelArtifactIdentity(
            model_id=GEMMA_MODEL_ID,
            revision=GEMMA_MODEL_REVISION,
            snapshot_digest="1" * 64,
            artifacts=[
                ArtifactDigest(
                    path="model.safetensors",
                    size=1,
                    sha256="2" * 64,
                )
            ],
        ),
        topology=GemmaTopology(
            model_type="gemma4",
            text_model_type="gemma4_text",
            num_hidden_layers=30,
            hidden_size=2816,
            num_experts=128,
            top_k_experts=8,
            moe_intermediate_size=704,
            max_position_embeddings=262144,
            vocab_size=262144,
            tie_word_embeddings=True,
            text_tensor_count=1,
            vision_tensor_count=0,
            transfer_categories=sorted(
                REQUIRED_TRANSFER_CATEGORIES
            ),
        ),
        source_pins=PINNED_SOURCES,
        user_simulator=ServiceIdentity(
            model=USER_SIMULATOR_MODEL,
            endpoint=USER_SIMULATOR_ENDPOINT,
        ),
        policy_model=GEMMA_POLICY_IDENTITY,
        policy_endpoints=POLICY_ENDPOINTS,
        expected_tp_size=2,
        fixed_prompt_token_ids=[1, 2],
        fixed_completion_token_ids=[3],
        policy_loss=RUN1_POLICY_LOSS,
        policy_loss_fallback=POLICY_LOSS_FALLBACK,
        policy_loss_fallback_trigger=(
            POLICY_LOSS_FALLBACK_TRIGGER
        ),
        gspo_token_upgrade_trigger=(
            GSPO_TOKEN_UPGRADE_TRIGGER
        ),
        transfer_evidence=[],
        parity_evidence=[],
        calibration=calibration,
        trainer_memory_budgets=[budget],
        recommended_production_topology=candidate,
        gates=gates,
        ready=True,
    )


def _production_cfg(tmp_path: Path, manifest: PreRunManifest):
    snapshot_path = tmp_path / "snapshot"
    snapshot_path.mkdir()
    (snapshot_path / "model.safetensors").write_bytes(
        b"weights"
    )
    model_identity = hash_model_snapshot(
        snapshot_path,
        GEMMA_MODEL_ID,
        GEMMA_MODEL_REVISION,
    )
    manifest = manifest.model_copy(update={"model": model_identity})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        manifest.model_dump_json(indent=2) + "\n"
    )
    cfg = _compose_recipe()
    cfg.output_dir = str(tmp_path / "output")
    cfg.tau2_prerun.manifest_path = str(manifest_path)
    cfg.tau2_prerun.job_spec_path = str(CALIBRATION_JOB)
    cfg.tau2_prerun.model_snapshot = str(snapshot_path)
    return cfg


def _assert_main_fails_before_process(
    cfg,
    monkeypatch,
    pattern: str,
) -> None:
    started = []
    monkeypatch.setattr(
        launch,
        "_popen",
        lambda *args, **kwargs: started.append((args, kwargs)),
    )
    monkeypatch.setenv("PIPELINERL_PREFLIGHT_ONLY", "1")
    with pytest.raises(ValueError, match=pattern):
        launch.main.__wrapped__(cfg)
    assert started == []


def test_recipe_records_loss_identity_and_pending_measured_caps():
    cfg = _compose_recipe()
    assert cfg.finetune.rl.policy_loss == "gspo"
    assert cfg.tau2_prerun.policy_loss_fallback == "dppo"
    assert (
        cfg.tau2_prerun.policy_loss_fallback_trigger
        == POLICY_LOSS_FALLBACK_TRIGGER
    )
    assert (
        cfg.tau2_prerun.gspo_token_upgrade_trigger
        == GSPO_TOKEN_UPGRADE_TRIGGER
    )
    assert cfg.finetune.seq_packing is False
    assert cfg.finetune.seq_parallel == 1
    assert cfg.actor.shared_memory_entry_size is None
    assert cfg.preprocess.shared_memory_entry_size is None
    assert cfg.finetune.seq_length is None
    assert cfg.vllm_config.vllm_kwargs.max_model_len is None
    assert cfg.tau2_gym.user_model_name == USER_SIMULATOR_MODEL
    assert cfg.tau2_gym.user_model_url == USER_SIMULATOR_ENDPOINT
    assert len(cfg.tau2_prerun.production_memory_candidates) == 3


def test_calibration_job_records_s1_s2_and_pending_s5_table():
    job = yaml.safe_load(CALIBRATION_JOB.read_text())
    assert job["resources"]["replicas"] == 4
    assert job["resources"]["gpu"] == 8
    assert job["command"][-2:] == [
        "tau2_gemma.sh",
        "calibration",
    ]
    env = {
        entry.split("=", 1)[0]: entry.split("=", 1)[1]
        for entry in job["environmentVars"]
    }
    assert env["TAU2_USER_SIMULATOR_MODEL"] == USER_SIMULATOR_MODEL
    assert env["TAU2_USER_SIMULATOR_ENDPOINT"] == USER_SIMULATOR_ENDPOINT
    assert "zero EAI GPUs" in env["TAU2_USER_SIMULATOR_PLACEMENT"]
    assert env["TAU2_MODEL_REVISION_STATUS"] == (
        "UNVERIFIED_PLACEHOLDER_FAIL_CLOSED"
    )
    assert env["TAU2_POLICY_LOSS"] == "gspo"
    assert env["TAU2_CALIBRATION_PACKING"] == "false"
    assert env["TAU2_CALIBRATION_SEQ_PARALLEL"] == "1"
    assert "CPU offload" in env["TAU2_CALIBRATION_MEMORY_LEVER"]
    assert env["TAU2_PARITY_MAX_ABS_TOLERANCE"] == "0.05"
    table = json.loads(env["TAU2_S5_MEMORY_BUDGET_TABLE_JSON"])
    assert [row["nodes"] for row in table] == [4, 6, 8]
    assert all(
        row["activation_bytes_at_measured_p99"]
        == "pending_calibration"
        for row in table
    )
    assert all(row["fits"] == "pending_calibration" for row in table)


def test_cpu_offload_profile_only_offloads_optimizer():
    profile = json.loads(
        (
            ROOT
            / "conf/deepspeed/deepspeed_stage3_bf16_cpu_offload.json"
        ).read_text()
    )
    zero = profile["zero_optimization"]
    assert zero["offload_optimizer"] == {
        "device": "cpu",
        "pin_memory": True,
    }
    assert zero["offload_param"] == {"device": "none"}


def test_wrapper_preflights_before_gym_and_is_valid_bash():
    wrapper = ROOT / "tau2_gemma.sh"
    text = wrapper.read_text()
    assert text.index("PIPELINERL_PREFLIGHT_ONLY=1") < text.index(
        "git clone"
    )
    assert "run_tau2_prerun finalize" in text
    subprocess.run(
        ["bash", "-n", str(wrapper)],
        check=True,
    )


def test_production_manifest_applies_measured_caps(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        prerun_evidence,
        "GEMMA_MODEL_REVISION_VERIFIED",
        True,
    )
    monkeypatch.setenv("WORLD_SIZE", "4")
    manifest = _ready_manifest(_sha256(CALIBRATION_JOB))
    cfg = _production_cfg(tmp_path, manifest)
    launch.validate_config(cfg)
    assert cfg.actor.shared_memory_entry_size == 48 * (1 << 20)
    assert cfg.preprocess.shared_memory_entry_size == 48 * (1 << 20)
    assert cfg.finetune.seq_length == 65536
    assert cfg.vllm_config.vllm_kwargs.max_model_len == 65536
    assert cfg.llm.parameters.max_tokens == 16384
    assert cfg.world.actor_fraction == 8
    assert cfg.world.finetune_fraction == 24
    assert cfg.deepspeed_config == "deepspeed_stage3_bf16"


def test_absent_manifest_fails_before_process(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        prerun_evidence,
        "GEMMA_MODEL_REVISION_VERIFIED",
        True,
    )
    cfg = _compose_recipe()
    cfg.output_dir = str(tmp_path / "output")
    cfg.tau2_prerun.manifest_path = str(tmp_path / "missing.json")
    cfg.tau2_prerun.job_spec_path = str(CALIBRATION_JOB)
    _assert_main_fails_before_process(
        cfg,
        monkeypatch,
        "manifest_path does not exist",
    )


def test_unready_manifest_fails_before_process(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        prerun_evidence,
        "GEMMA_MODEL_REVISION_VERIFIED",
        True,
    )
    manifest = _ready_manifest(_sha256(CALIBRATION_JOB))
    gates = dict(manifest.gates)
    gates["9_context_fit"] = GateResult(
        passed=False,
        detail="test rejection",
    )
    manifest = manifest.model_copy(
        update={"gates": gates, "ready": False}
    )
    cfg = _production_cfg(tmp_path, manifest)
    _assert_main_fails_before_process(
        cfg,
        monkeypatch,
        "pre-run gates have not passed",
    )


def test_manifest_digest_mismatch_fails_before_process(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        prerun_evidence,
        "GEMMA_MODEL_REVISION_VERIFIED",
        True,
    )
    manifest = _ready_manifest("0" * 64)
    cfg = _production_cfg(tmp_path, manifest)
    _assert_main_fails_before_process(
        cfg,
        monkeypatch,
        "manifest job digest",
    )


def test_sp_topology_without_gate5_proof_fails_before_process(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        prerun_evidence,
        "GEMMA_MODEL_REVISION_VERIFIED",
        True,
    )
    monkeypatch.setenv("WORLD_SIZE", "6")
    manifest = _ready_manifest(_sha256(CALIBRATION_JOB))
    candidate = TrainerMemoryCandidate(
        name="6x8-sp2",
        node_count=6,
        gpus_per_node=8,
        actor_gpus=8,
        trainer_gpus=40,
        seq_parallel=2,
        gpu_memory_bytes=80 * GIB,
        reserve_bytes_per_gpu=8 * GIB,
    )
    budget = manifest.trainer_memory_budgets[0].model_copy(
        update={"candidate": candidate}
    )
    manifest = manifest.model_copy(
        update={
            "trainer_memory_budgets": [budget],
            "recommended_production_topology": candidate,
        }
    )
    cfg = _production_cfg(tmp_path, manifest)
    _assert_main_fails_before_process(
        cfg,
        monkeypatch,
        "gate 5 requires a packed-vs-unpacked equivalence proof",
    )


def test_unverified_revision_fails_before_process(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        prerun_evidence,
        "GEMMA_MODEL_REVISION_VERIFIED",
        False,
    )
    manifest = _ready_manifest(_sha256(CALIBRATION_JOB))
    cfg = _production_cfg(tmp_path, manifest)
    _assert_main_fails_before_process(
        cfg,
        monkeypatch,
        "unverified placeholder",
    )


def test_calibration_spec_enforces_safe_profile(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        prerun_evidence,
        "GEMMA_MODEL_REVISION_VERIFIED",
        True,
    )
    monkeypatch.setenv("WORLD_SIZE", "4")
    cfg = _compose_recipe()
    cfg.output_dir = str(tmp_path / "output")
    cfg.tau2_prerun.phase = "calibration"
    cfg.tau2_prerun.enabled = True
    cfg.finetune.max_train_steps = 1
    cfg.finetune.interrupt_train_steps = 1
    cfg.tau2_prerun.job_spec_path = str(CALIBRATION_JOB)
    snapshot_path = tmp_path / "snapshot"
    snapshot_path.mkdir()
    cfg.tau2_prerun.model_snapshot = str(snapshot_path)
    cfg.tau2_prerun.policy_endpoints = POLICY_ENDPOINTS
    cfg.tau2_prerun.fixed_prompt_token_ids = [1, 2]
    cfg.tau2_prerun.fixed_completion_token_ids = [3]
    candidates = [
        TrainerMemoryCandidate.model_validate(candidate)
        for candidate in cfg.tau2_prerun.production_memory_candidates
    ]
    spec = PreRunSpec(
        model_snapshot=str(snapshot_path),
        source_pins=PINNED_SOURCES,
        user_simulator=ServiceIdentity(
            model=USER_SIMULATOR_MODEL,
            endpoint=USER_SIMULATOR_ENDPOINT,
        ),
        policy_model=GEMMA_POLICY_IDENTITY,
        policy_endpoints=POLICY_ENDPOINTS,
        expected_tp_size=2,
        fixed_prompt_token_ids=[1, 2],
        fixed_completion_token_ids=[3],
        user_separation_asserted=True,
        packing_enabled=False,
        mixed_version_loss_gate_passed=True,
        production_memory_candidates=candidates,
        job_spec_sha256=_sha256(CALIBRATION_JOB),
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(spec.model_dump_json(indent=2) + "\n")
    cfg.tau2_prerun.spec_path = str(spec_path)

    launch.validate_config(cfg)
    assert cfg.finetune.seq_packing is False
    assert cfg.finetune.seq_parallel == 1
    assert cfg.finetune.seq_length == 262144
    assert cfg.actor.shared_memory_entry_size == 134217728
    assert cfg.preprocess.shared_memory_entry_size == 134217728
    assert cfg.deepspeed_config == (
        "deepspeed_stage3_bf16_cpu_offload"
    )


def test_vllm_cli_list_values_preserve_all_served_names():
    kwargs = {
        "served-model-name": [
            GEMMA_MODEL_ID,
            GEMMA_POLICY_IDENTITY,
        ]
    }
    launch._add_actor_model_alias(
        kwargs,
        "/tmp/checkpoint",
    )
    assert kwargs["served-model-name"][-1] == "/tmp/checkpoint"
    generic_kwargs = {"dtype": "bfloat16"}
    launch._add_actor_model_alias(generic_kwargs, "/tmp/checkpoint")
    assert generic_kwargs == {"dtype": "bfloat16"}

    command = ["vllm"]
    launch._append_vllm_kwargs(command, kwargs)
    assert command == [
        "vllm",
        "--served-model-name",
        GEMMA_MODEL_ID,
        GEMMA_POLICY_IDENTITY,
        "/tmp/checkpoint",
    ]
