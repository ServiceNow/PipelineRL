import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

import pipelinerl.domains.tau2.client as tau2_client
import pipelinerl.domains.tau2.prerun as tau2_prerun
import pipelinerl.launch as launch
import pipelinerl.prerun_evidence as prerun_evidence
from pipelinerl.domains.tau2.dataset import (
    NEMO_GYM_REPOSITORY,
    TAU2_DATA_REPOSITORY,
    TAU2_NORMALIZATION_CONTRACT,
    Tau2PreparedDataManifest,
    Tau2PreparedFileIdentity,
)
from pipelinerl.domains.tau2.prerun import (
    CALIBRATION_GROUPS,
    CALIBRATION_ROLLOUTS,
    GIB,
    GSPO_TOKEN_UPGRADE_TRIGGER,
    PINNED_SOURCES,
    POLICY_LOSS_FALLBACK,
    POLICY_LOSS_FALLBACK_TRIGGER,
    RUN1_POLICY_LOSS,
    AUXILIARY_JUDGE_MODEL,
    AUXILIARY_MODEL_ID,
    AUXILIARY_MODEL_REVISION,
    AUXILIARY_MODEL_SNAPSHOT,
    AUXILIARY_USER_MODEL,
    AuxiliaryModelDeployment,
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
    QWEN35_27B_MODEL_DESCRIPTOR,
    ArtifactDigest,
    ModelArtifactIdentity,
    TextModelTopology,
    hash_model_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_JOB = ROOT / "tau2_gemma_calibration.yaml"
AUXILIARY_MODEL_JOB = ROOT / "tau2_qwen_auxiliary.yaml"
USER_SIMULATOR_ENDPOINT = "http://dns-test-account-tau2-user:8000/v1"
POLICY_ENDPOINTS = [
    "http://tau2-gemma-calibration-3:8080",
    "http://tau2-gemma-calibration-3:8082",
    "http://tau2-gemma-calibration-3:8084",
    "http://tau2-gemma-calibration-3:8086",
]


_REAL_VALIDATE_PREPARED_DATA = launch._validated_tau2_prepared_data


def _prepared_data_manifest() -> Tau2PreparedDataManifest:
    return Tau2PreparedDataManifest(
        schema_version=1,
        source_repository=TAU2_DATA_REPOSITORY,
        source_revision=PINNED_SOURCES.tau2_data_sha,
        reference_normalizer_repository=NEMO_GYM_REPOSITORY,
        reference_normalizer_revision=PINNED_SOURCES.nemo_gym_sha,
        normalization_contract=TAU2_NORMALIZATION_CONTRACT,
        files=[
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
        ],
        total_row_count=278,
        composite_identity_sha256="f" * 64,
    )


@pytest.fixture(autouse=True)
def _stub_prepared_data_validation(monkeypatch):
    def validate(cfg):
        path = Path(str(cfg.dataset_loader_params.prepared_data_manifest))
        return path, SimpleNamespace(manifest=_prepared_data_manifest())

    monkeypatch.setattr(launch, "_validated_tau2_prepared_data", validate)


def _compose_recipe():
    with initialize_config_dir(
        config_dir=str(ROOT / "conf"),
        version_base="1.3.2",
    ):
        return compose(config_name="tau2_gemma")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _auxiliary_model_fixture(
    tmp_path: Path,
    monkeypatch,
) -> tuple[Path, AuxiliaryModelDeployment]:
    snapshot = tmp_path / "auxiliary-model-snapshot"
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
    observed_latencies = {
        "user_simulator": [0.2, 0.3],
        "judge": [0.5, 0.7],
    }
    observed_prompts = {
        "user_simulator": [4_000, 5_000],
        "judge": [8_000, 9_000],
    }
    headroom = {"user_simulator": 1.25, "judge": 1.5}
    generation_reserve = {"user_simulator": 2_048, "judge": 4_096}
    snapshot_hash_bytes = sum(artifact.size for artifact in model.artifacts)
    job_path = tmp_path / "auxiliary-model.yaml"
    job = {
        "name": "tau2-qwen-auxiliary",
        "bid": 9999,
        "data": [
            "snow.research.tapes.base_models:/mnt/llmd/base_models:ro",
        ],
        "command": [
            "python -m vllm.entrypoints.openai.api_server "
            '--model "${TAU2_AUX_MODEL_SNAPSHOT}" '
            "--served-model-name "
            '"${TAU2_USER_MODEL_ALIAS}" '
            '"${TAU2_JUDGE_MODEL_ALIAS}" '
            "--dtype bfloat16 --host 0.0.0.0 --port 8000 "
            '--tensor-parallel-size "${TAU2_AUX_TP_SIZE}" '
            '--max-model-len "${TAU2_AUX_MAX_MODEL_LEN}" '
            '--max-num-seqs "${TAU2_AUX_MAX_NUM_SEQS}"'
        ],
        "resources": {
            "cpu": 16,
            "gpu": 2,
            "gpuModel": "test-h100-80gb",
            "mem": 128,
            "replicas": 1,
        },
        "options": {
            "internal-dns": {
                "name": "tau2-user",
                "ports": [
                    {
                        "port": 8000,
                        "target-port": 8000,
                        "protocol": "TCP",
                    }
                ],
            }
        },
        "preemptable": True,
        "restartable": True,
        "environmentVars": [
            f"TAU2_AUX_MODEL_ID={AUXILIARY_MODEL_ID}",
            f"TAU2_AUX_MODEL_REVISION={AUXILIARY_MODEL_REVISION}",
            f"TAU2_USER_MODEL_ALIAS={AUXILIARY_USER_MODEL}",
            f"TAU2_JUDGE_MODEL_ALIAS={AUXILIARY_JUDGE_MODEL}",
            f"TAU2_AUX_MODEL_SNAPSHOT={snapshot}",
            "TAU2_AUX_GPU_TYPE=test-h100-80gb",
            "TAU2_AUX_GPU_COUNT=2",
            "TAU2_AUX_TP_SIZE=2",
            "TAU2_AUX_MAX_MODEL_LEN=20000",
            "TAU2_AUX_MAX_NUM_SEQS=8",
            "TAU2_AUX_MEASURED_PEAK_IN_FLIGHT=4",
            "TAU2_AUX_OBSERVED_REQUEST_LATENCIES_S_JSON="
            + json.dumps(observed_latencies),
            "TAU2_AUX_OBSERVED_PROMPT_TOKENS_JSON=" + json.dumps(observed_prompts),
            "TAU2_AUX_PROMPT_HEADROOM_FACTORS_JSON=" + json.dumps(headroom),
            "TAU2_AUX_GENERATION_RESERVE_TOKENS_JSON=" + json.dumps(generation_reserve),
            f"TAU2_AUX_SNAPSHOT_HASH_BYTES={snapshot_hash_bytes}",
            "TAU2_AUX_SUBMISSION_MODE=restartable",
            "VLLM_BATCH_INVARIANT=0",
        ],
    }
    job_path.write_text(yaml.safe_dump(job, sort_keys=False))
    endpoint = USER_SIMULATOR_ENDPOINT
    deployment = AuxiliaryModelDeployment(
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
        job_spec_sha256=_sha256(job_path),
        submission_mode="restartable",
        batch_invariant=False,
        gpu_type="test-h100-80gb",
        gpu_count=2,
        tensor_parallel_size=2,
        max_model_len=20_000,
        max_num_seqs=8,
        measured_peak_in_flight=4,
        observed_request_latencies_s=observed_latencies,
        observed_prompt_tokens=observed_prompts,
        prompt_headroom_factors=headroom,
        generation_reserve_tokens=generation_reserve,
        snapshot_hash_bytes=snapshot_hash_bytes,
    )
    return job_path, deployment


def _set_auxiliary_identity(cfg) -> None:
    with open_dict(cfg.tau2_gym):
        cfg.tau2_gym.user_model_url = USER_SIMULATOR_ENDPOINT
        cfg.tau2_gym.user_model_name = AUXILIARY_USER_MODEL
        cfg.tau2_gym.judge_model_url = USER_SIMULATOR_ENDPOINT
        cfg.tau2_gym.judge_model_name = AUXILIARY_JUDGE_MODEL


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


def _ready_manifest(
    job_digest: str,
    auxiliary_model_deployment: AuxiliaryModelDeployment,
) -> PreRunManifest:
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
        schema_version=3,
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
        topology=TextModelTopology(
            model_type="gemma4",
            text_model_type="gemma4_text",
            num_hidden_layers=30,
            hidden_size=2816,
            intermediate_size=2112,
            num_experts=128,
            top_k_experts=8,
            moe_intermediate_size=704,
            max_position_embeddings=262144,
            vocab_size=262144,
            tie_word_embeddings=True,
            layer_indices=list(range(30)),
            text_tensor_count=1,
            vision_tensor_count=0,
            nontransferred_tensor_count=0,
            transfer_categories=sorted(
                {
                    "backbone", "embedding", "expert", "output_head", "router"
                }
            ),
        ),
        source_pins=PINNED_SOURCES,
        prepared_data=_prepared_data_manifest(),
        auxiliary_model_deployment=auxiliary_model_deployment,
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


def _production_cfg(
    tmp_path: Path,
    manifest: PreRunManifest,
    auxiliary_model_job_path: Path,
    auxiliary_model_deployment: AuxiliaryModelDeployment,
):
    snapshot_path = tmp_path / "snapshot"
    snapshot_path.mkdir()
    (snapshot_path / "model.safetensors").write_bytes(b"weights")
    model_identity = hash_model_snapshot(
        snapshot_path,
        GEMMA_MODEL_ID,
        GEMMA_MODEL_REVISION,
    )
    manifest = manifest.model_copy(update={"model": model_identity})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    cfg = _compose_recipe()
    cfg.output_dir = str(tmp_path / "output")
    cfg.tau2_prerun.manifest_path = str(manifest_path)
    cfg.tau2_prerun.job_spec_path = str(CALIBRATION_JOB)
    cfg.tau2_prerun.model_snapshot = str(snapshot_path)
    _set_auxiliary_identity(cfg)
    cfg.tau2_prerun.user_simulator_job_spec_path = str(auxiliary_model_job_path)
    cfg.tau2_prerun.user_simulator_snapshot = auxiliary_model_deployment.snapshot_path
    return cfg


def _production_case(
    tmp_path: Path,
    monkeypatch,
    manifest_job_digest: str,
) -> tuple[PreRunManifest, Path, AuxiliaryModelDeployment]:
    auxiliary_model_job_path, deployment = _auxiliary_model_fixture(
        tmp_path, monkeypatch
    )
    manifest = _ready_manifest(
        manifest_job_digest,
        deployment,
    )
    return manifest, auxiliary_model_job_path, deployment


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


@pytest.mark.parametrize(
    ("manifest_path", "message"),
    (
        ("PENDING_TAU2_PREPARED_DATA_MANIFEST", "unresolved"),
        ("/tmp/missing-tau2-prepared-manifest.json", "does not exist"),
    ),
)
def test_prepared_data_manifest_fails_before_process(
    tmp_path,
    monkeypatch,
    manifest_path,
    message,
):
    monkeypatch.setattr(
        launch,
        "_validated_tau2_prepared_data",
        _REAL_VALIDATE_PREPARED_DATA,
    )
    cfg = _compose_recipe()
    cfg.output_dir = str(tmp_path / "output")
    cfg.dataset_loader_params.prepared_data_manifest = manifest_path
    _assert_main_fails_before_process(cfg, monkeypatch, message)


def test_recipe_records_transitional_policy_boundary_and_pending_caps():
    cfg = _compose_recipe()
    assert cfg.finetune.rl.policy_loss == "gspo"
    assert cfg.tau2_prerun.policy_loss_fallback == "dppo"
    assert cfg.tau2_prerun.policy_loss_fallback_trigger == POLICY_LOSS_FALLBACK_TRIGGER
    assert cfg.tau2_prerun.gspo_token_upgrade_trigger == GSPO_TOKEN_UPGRADE_TRIGGER
    assert cfg.finetune.seq_packing is False
    assert cfg.finetune.seq_parallel == 1
    assert cfg.actor.shared_memory_entry_size is None
    assert cfg.preprocess.shared_memory_entry_size is None
    assert cfg.finetune.seq_length is None
    assert cfg.vllm_config.vllm_kwargs.max_model_len is None
    # W12 changes only the auxiliary contract. W14 atomically repoints these
    # remaining executable recipe fields and removes the Gemma descriptor.
    assert str(cfg.tau2_gym.user_model_name).startswith("google/gemma-")
    assert str(cfg.tau2_gym.user_model_url).startswith("PENDING_")
    assert str(cfg.tau2_prerun.user_simulator_job_spec_path).startswith("PENDING_")
    assert str(
        cfg.dataset_loader_params.prepared_data_manifest
    ).startswith("PENDING_")
    assert str(cfg.tau2_prerun.user_simulator_snapshot).endswith("gemma-4-31B-it")
    assert len(cfg.tau2_prerun.production_memory_candidates) == 3


def test_calibration_job_records_shared_auxiliary_provenance():
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
    assert env["TAU2_USER_SIMULATOR_MODEL"] == AUXILIARY_USER_MODEL
    assert env["TAU2_JUDGE_MODEL"] == AUXILIARY_JUDGE_MODEL
    assert env["TAU2_USER_SIMULATOR_ENDPOINT"].startswith("PENDING_")
    assert env["TAU2_JUDGE_ENDPOINT"] == env["TAU2_USER_SIMULATOR_ENDPOINT"]
    assert "separately managed" in env["TAU2_AUXILIARY_PLACEMENT"]
    assert env["TAU2_USER_API_KEY"] == "keyless-internal-dummy"
    assert env["TAU2_USER_SIM_JOB_SPEC_PATH"] == str(AUXILIARY_MODEL_JOB)
    assert env["TAU2_USER_SIM_SNAPSHOT"] == AUXILIARY_MODEL_SNAPSHOT
    assert env["TAU2_AUXILIARY_BATCH_INVARIANT"].startswith("0 ")
    assert env["TAU2_AUXILIARY_MAX_MODEL_LEN"].startswith("PENDING_")
    assert env["TAU2_AUXILIARY_ROLE_EVIDENCE"].startswith("PENDING_")
    assert env["TAU2_AUXILIARY_SNAPSHOT_HASH_IO"].startswith("PENDING_")
    assert env["TAU2_STRICT_TITO_PATCH_SHA256"] == (
        PINNED_SOURCES.strict_tito_patch_sha256
    )
    assert env["TAU2_POLICY_LOSS"] == "gspo"
    assert env["TAU2_CALIBRATION_PACKING"] == "false"
    assert env["TAU2_CALIBRATION_SEQ_PARALLEL"] == "1"
    assert "CPU offload" in env["TAU2_CALIBRATION_MEMORY_LEVER"]
    assert env["TAU2_PARITY_MAX_ABS_TOLERANCE"] == "0.05"
    table = json.loads(env["TAU2_S5_MEMORY_BUDGET_TABLE_JSON"])
    assert [row["nodes"] for row in table] == [4, 6, 8]
    assert all(
        row["activation_bytes_at_measured_p99"] == "pending_calibration"
        for row in table
    )
    assert all(row["fits"] == "pending_calibration" for row in table)


def test_calibration_source_pins_match_executed_contract():
    job = yaml.safe_load(CALIBRATION_JOB.read_text())
    env = {
        entry.split("=", 1)[0]: entry.split("=", 1)[1]
        for entry in job["environmentVars"]
    }
    expected = {
        "nemo_gym_sha": tau2_client.NEMO_GYM_SHA,
        "strict_tito_patch_sha256": tau2_client.NEMO_GYM_TITO_PATCH_SHA,
        "tau2_runtime_sha": tau2_client.TAU2_RUNTIME_SHA,
        "tau2_data_sha": tau2_client.TAU2_DATA_SHA,
    }
    assert PINNED_SOURCES.model_dump() == expected
    assert env["TAU2_NEMO_GYM_SHA"] == expected["nemo_gym_sha"]
    assert (
        env["TAU2_STRICT_TITO_PATCH_SHA256"]
        == expected["strict_tito_patch_sha256"]
    )
    assert env["TAU2_RUNTIME_SHA"] == expected["tau2_runtime_sha"]
    assert env["TAU2_DATA_SHA"] == expected["tau2_data_sha"]


def test_auxiliary_job_is_two_alias_restartable_and_measurement_gated():
    job = yaml.safe_load(AUXILIARY_MODEL_JOB.read_text())
    calibration_job = yaml.safe_load(CALIBRATION_JOB.read_text())
    environment = {
        entry.split("=", 1)[0]: entry.split("=", 1)[1]
        for entry in job["environmentVars"]
    }
    assert job["name"] == "tau2-qwen-auxiliary"
    assert job["restartable"] is True
    assert job["preemptable"] is True
    assert job["bid"] == calibration_job["bid"] == 9999
    assert "snow.research.tapes.base_models:/mnt/llmd/base_models:ro" in job["data"]
    assert job["resources"]["replicas"] == 1
    assert str(job["resources"]["gpu"]) == environment["TAU2_AUX_GPU_COUNT"]
    assert str(job["resources"]["gpu"]).startswith("PENDING_")
    assert str(job["resources"]["gpuModel"]).startswith("PENDING_")
    assert environment["TAU2_AUX_MODEL_ID"] == AUXILIARY_MODEL_ID
    assert environment["TAU2_AUX_MODEL_REVISION"] == AUXILIARY_MODEL_REVISION
    assert environment["TAU2_USER_MODEL_ALIAS"] == AUXILIARY_USER_MODEL
    assert environment["TAU2_JUDGE_MODEL_ALIAS"] == AUXILIARY_JUDGE_MODEL
    assert environment["TAU2_AUX_MODEL_SNAPSHOT"] == AUXILIARY_MODEL_SNAPSHOT
    assert environment["TAU2_AUX_MAX_MODEL_LEN"].startswith("PENDING_")
    assert environment["TAU2_AUX_MAX_NUM_SEQS"].startswith("PENDING_")
    assert environment["TAU2_AUX_SNAPSHOT_HASH_BYTES"].startswith("PENDING_")
    assert environment["TAU2_AUX_OBSERVED_PROMPT_TOKENS_JSON"].startswith("PENDING_")
    assert environment["VLLM_BATCH_INVARIANT"] == "0"
    assert environment["TAU2_AUX_SUBMISSION_MODE"] == "restartable"
    command = " ".join(str(part) for part in job["command"])
    assert "vllm.entrypoints.openai.api_server" in command
    assert '--model "${TAU2_AUX_MODEL_SNAPSHOT}"' in command
    assert (
        '--served-model-name "${TAU2_USER_MODEL_ALIAS}" "${TAU2_JUDGE_MODEL_ALIAS}"'
    ) in command
    assert '--tensor-parallel-size "${TAU2_AUX_TP_SIZE}"' in command
    assert '--max-model-len "${TAU2_AUX_MAX_MODEL_LEN}"' in command
    assert '--max-num-seqs "${TAU2_AUX_MAX_NUM_SEQS}"' in command
    assert "--default-chat-template-kwargs" not in command


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
    manifest, user_job, deployment = _production_case(
        tmp_path,
        monkeypatch,
        _sha256(CALIBRATION_JOB),
    )
    cfg = _production_cfg(
        tmp_path,
        manifest,
        user_job,
        deployment,
    )
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
    user_job, deployment = _auxiliary_model_fixture(
        tmp_path,
        monkeypatch,
    )
    _set_auxiliary_identity(cfg)
    cfg.tau2_prerun.user_simulator_job_spec_path = str(user_job)
    cfg.tau2_prerun.user_simulator_snapshot = (
        deployment.snapshot_path
    )
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
    manifest, user_job, deployment = _production_case(
        tmp_path,
        monkeypatch,
        _sha256(CALIBRATION_JOB),
    )
    gates = dict(manifest.gates)
    gates["9_context_fit"] = GateResult(
        passed=False,
        detail="test rejection",
    )
    manifest = manifest.model_copy(
        update={"gates": gates, "ready": False}
    )
    cfg = _production_cfg(
        tmp_path,
        manifest,
        user_job,
        deployment,
    )
    _assert_main_fails_before_process(
        cfg,
        monkeypatch,
        "pre-run gates have not passed",
    )


def test_missing_manifest_schema_fails_before_process(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        prerun_evidence,
        "GEMMA_MODEL_REVISION_VERIFIED",
        True,
    )
    manifest, user_job, deployment = _production_case(
        tmp_path,
        monkeypatch,
        _sha256(CALIBRATION_JOB),
    )
    cfg = _production_cfg(
        tmp_path,
        manifest,
        user_job,
        deployment,
    )
    manifest_path = Path(str(cfg.tau2_prerun.manifest_path))
    payload = json.loads(manifest_path.read_text())
    del payload["schema_version"]
    manifest_path.write_text(json.dumps(payload))

    _assert_main_fails_before_process(
        cfg,
        monkeypatch,
        "schema_version",
    )


def test_stale_manifest_schema_fails_before_process(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        prerun_evidence,
        "GEMMA_MODEL_REVISION_VERIFIED",
        True,
    )
    manifest, user_job, deployment = _production_case(
        tmp_path,
        monkeypatch,
        _sha256(CALIBRATION_JOB),
    )
    manifest = manifest.model_copy(update={"schema_version": 1})
    cfg = _production_cfg(
        tmp_path,
        manifest,
        user_job,
        deployment,
    )
    _assert_main_fails_before_process(
        cfg,
        monkeypatch,
        "manifest schema version 1 is not supported",
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
    manifest, user_job, deployment = _production_case(
        tmp_path,
        monkeypatch,
        "0" * 64,
    )
    cfg = _production_cfg(
        tmp_path,
        manifest,
        user_job,
        deployment,
    )
    _assert_main_fails_before_process(
        cfg,
        monkeypatch,
        "manifest job digest",
    )


def test_auxiliary_model_job_digest_mismatch_fails_before_process(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        prerun_evidence,
        "GEMMA_MODEL_REVISION_VERIFIED",
        True,
    )
    monkeypatch.setenv("WORLD_SIZE", "4")
    manifest, user_job, deployment = _production_case(
        tmp_path,
        monkeypatch,
        _sha256(CALIBRATION_JOB),
    )
    bad_deployment = deployment.model_copy(
        update={"job_spec_sha256": "0" * 64}
    )
    manifest = manifest.model_copy(
        update={"auxiliary_model_deployment": bad_deployment}
    )
    cfg = _production_cfg(
        tmp_path,
        manifest,
        user_job,
        bad_deployment,
    )
    _assert_main_fails_before_process(
        cfg,
        monkeypatch,
        "auxiliary model job digest",
    )


def test_auxiliary_model_profile_mismatch_fails_before_process(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        prerun_evidence,
        "GEMMA_MODEL_REVISION_VERIFIED",
        True,
    )
    monkeypatch.setenv("WORLD_SIZE", "4")
    manifest, user_job, deployment = _production_case(
        tmp_path,
        monkeypatch,
        _sha256(CALIBRATION_JOB),
    )
    job = yaml.safe_load(user_job.read_text())
    job["environmentVars"] = [
        (
            "TAU2_AUX_MAX_MODEL_LEN=8192"
            if entry.startswith("TAU2_AUX_MAX_MODEL_LEN=")
            else entry
        )
        for entry in job["environmentVars"]
    ]
    user_job.write_text(yaml.safe_dump(job, sort_keys=False))
    mismatched_deployment = deployment.model_copy(
        update={"job_spec_sha256": _sha256(user_job)}
    )
    manifest = manifest.model_copy(
        update={
            "auxiliary_model_deployment": mismatched_deployment
        }
    )
    cfg = _production_cfg(
        tmp_path,
        manifest,
        user_job,
        mismatched_deployment,
    )
    _assert_main_fails_before_process(
        cfg,
        monkeypatch,
        "TAU2_AUX_MAX_MODEL_LEN",
    )


@pytest.mark.parametrize(
    ("tamper", "pattern"),
    [
        ("gpu_model", "auxiliary model job GPU type"),
        ("snapshot_hash", "TAU2_AUX_SNAPSHOT_HASH_BYTES"),
        ("pending", "unresolved PENDING_"),
        ("cpu", "cpu resource is unresolved"),
        ("name", "auxiliary model job name"),
        ("bid", "auxiliary model job bid"),
        ("command", "auxiliary model job command is missing"),
        ("alias_reorder", "auxiliary ordered served aliases"),
        ("alias_missing", "auxiliary ordered served aliases"),
        ("alias_extra", "auxiliary ordered served aliases"),
        ("default_thinking", "must not set a server thinking default"),
        ("batch_invariant", "batch-invariance calibration setting"),
        ("role_evidence", "TAU2_AUX_OBSERVED_PROMPT_TOKENS_JSON"),
        ("mount", "snapshot mount must be read-only"),
    ],
)
def test_auxiliary_execution_profile_mismatch_fails_before_process(
    tmp_path,
    monkeypatch,
    tamper,
    pattern,
):
    monkeypatch.setattr(
        prerun_evidence,
        "GEMMA_MODEL_REVISION_VERIFIED",
        True,
    )
    monkeypatch.setenv("WORLD_SIZE", "4")
    manifest, user_job, deployment = _production_case(
        tmp_path,
        monkeypatch,
        _sha256(CALIBRATION_JOB),
    )
    job = yaml.safe_load(user_job.read_text())
    if tamper == "gpu_model":
        job["resources"]["gpuModel"] = "wrong-gpu"
    elif tamper == "snapshot_hash":
        job["environmentVars"] = [
            (
                "TAU2_AUX_SNAPSHOT_HASH_BYTES=1"
                if entry.startswith("TAU2_AUX_SNAPSHOT_HASH_BYTES=")
                else entry
            )
            for entry in job["environmentVars"]
        ]
    elif tamper == "pending":
        job["resources"]["mem"] = "PENDING_TEST_MEMORY"
    elif tamper == "cpu":
        job["resources"]["cpu"] = 0
    elif tamper == "name":
        job["name"] = "pending-auxiliary"
    elif tamper == "bid":
        job["bid"] = 9998
    elif tamper == "command":
        job["command"] = [
            part.replace(
                '--max-model-len "${TAU2_AUX_MAX_MODEL_LEN}" ',
                "",
            )
            for part in job["command"]
        ]
    elif tamper == "alias_reorder":
        job["command"] = [
            part.replace(
                '"${TAU2_USER_MODEL_ALIAS}" "${TAU2_JUDGE_MODEL_ALIAS}"',
                '"${TAU2_JUDGE_MODEL_ALIAS}" "${TAU2_USER_MODEL_ALIAS}"',
            )
            for part in job["command"]
        ]
    elif tamper == "alias_missing":
        job["command"] = [
            part.replace(' "${TAU2_JUDGE_MODEL_ALIAS}"', "") for part in job["command"]
        ]
    elif tamper == "alias_extra":
        job["command"] = [
            part.replace(
                '"${TAU2_JUDGE_MODEL_ALIAS}"',
                '"${TAU2_JUDGE_MODEL_ALIAS}" unexpected-alias',
            )
            for part in job["command"]
        ]
    elif tamper == "default_thinking":
        job["command"] = [
            part + " --default-chat-template-kwargs {}" for part in job["command"]
        ]
    elif tamper == "batch_invariant":
        job["environmentVars"] = [
            (
                "VLLM_BATCH_INVARIANT=1"
                if entry.startswith("VLLM_BATCH_INVARIANT=")
                else entry
            )
            for entry in job["environmentVars"]
        ]
    elif tamper == "role_evidence":
        job["environmentVars"] = [
            (
                'TAU2_AUX_OBSERVED_PROMPT_TOKENS_JSON={"user_simulator":[1]}'
                if entry.startswith("TAU2_AUX_OBSERVED_PROMPT_TOKENS_JSON=")
                else entry
            )
            for entry in job["environmentVars"]
        ]
    else:
        job["data"] = ["snow.research.tapes.base_models:/mnt/llmd/base_models:rw"]
    user_job.write_text(yaml.safe_dump(job, sort_keys=False))
    mismatched_deployment = deployment.model_copy(
        update={"job_spec_sha256": _sha256(user_job)}
    )
    manifest = manifest.model_copy(
        update={"auxiliary_model_deployment": mismatched_deployment}
    )
    cfg = _production_cfg(
        tmp_path,
        manifest,
        user_job,
        mismatched_deployment,
    )
    _assert_main_fails_before_process(
        cfg,
        monkeypatch,
        pattern,
    )


def test_prepared_data_identity_mismatch_fails_before_process(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        prerun_evidence,
        "GEMMA_MODEL_REVISION_VERIFIED",
        True,
    )
    monkeypatch.setenv("WORLD_SIZE", "4")
    manifest, user_job, deployment = _production_case(
        tmp_path,
        monkeypatch,
        _sha256(CALIBRATION_JOB),
    )
    cfg = _production_cfg(
        tmp_path,
        manifest,
        user_job,
        deployment,
    )
    mismatched = _prepared_data_manifest().model_copy(
        update={"total_row_count": 277}
    )
    monkeypatch.setattr(
        launch,
        "_validated_tau2_prepared_data",
        lambda current_cfg: (
            Path(
                str(
                    current_cfg.dataset_loader_params.prepared_data_manifest
                )
            ),
            SimpleNamespace(manifest=mismatched),
        ),
    )
    _assert_main_fails_before_process(
        cfg,
        monkeypatch,
        "prepared-data identity",
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
    manifest, user_job, deployment = _production_case(
        tmp_path,
        monkeypatch,
        _sha256(CALIBRATION_JOB),
    )
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
    cfg = _production_cfg(
        tmp_path,
        manifest,
        user_job,
        deployment,
    )
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
    manifest, user_job, deployment = _production_case(
        tmp_path,
        monkeypatch,
        _sha256(CALIBRATION_JOB),
    )
    cfg = _production_cfg(
        tmp_path,
        manifest,
        user_job,
        deployment,
    )
    _assert_main_fails_before_process(
        cfg,
        monkeypatch,
        "unverified placeholder",
    )


def test_pipeline_parallel_policy_is_rejected_before_process(
    tmp_path,
    monkeypatch,
):
    cfg = _compose_recipe()
    cfg.output_dir = str(tmp_path / "output")
    _set_auxiliary_identity(cfg)
    cfg.vllm_config.vllm_kwargs["pipeline-parallel-size"] = 2
    _assert_main_fails_before_process(
        cfg,
        monkeypatch,
        "vLLM pipeline parallel size for gate 3",
    )


def test_qwen_27b_simulator_is_rejected_as_run1_policy_before_process(
    tmp_path,
    monkeypatch,
):
    cfg = _compose_recipe()
    cfg.output_dir = str(tmp_path / "output")
    cfg.finetune.config_name = QWEN35_27B_MODEL_DESCRIPTOR.model_id
    cfg.finetune.model_revision = QWEN35_27B_MODEL_DESCRIPTOR.revision
    _assert_main_fails_before_process(
        cfg,
        monkeypatch,
        "not eligible for the run-1 policy role",
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
    user_job, deployment = _auxiliary_model_fixture(
        tmp_path,
        monkeypatch,
    )
    _set_auxiliary_identity(cfg)
    cfg.tau2_prerun.user_simulator_job_spec_path = str(user_job)
    cfg.tau2_prerun.user_simulator_snapshot = (
        deployment.snapshot_path
    )
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
        prepared_data_manifest_path=str(
            cfg.dataset_loader_params.prepared_data_manifest
        ),
        model_snapshot=str(snapshot_path),
        source_pins=PINNED_SOURCES,
        auxiliary_model_deployment=deployment,
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

    cfg.dataset_loader_params.prepared_data_manifest = str(
        tmp_path / "different-prepared-manifest.json"
    )
    with pytest.raises(ValueError, match="prepared-data manifest path"):
        launch.validate_config(cfg)


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
