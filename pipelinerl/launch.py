import hashlib
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, TextIO

import hydra
from omegaconf import DictConfig, OmegaConf

from pipelinerl.state import TrainerState
from pipelinerl.streams import SingleStreamSpec, connect_to_redis, read_stream, set_streams_backend, write_to_streams
from pipelinerl.utils import terminate_with_children
from pipelinerl.world import Job, WorldMap

logger = logging.getLogger(__name__)

# All the launch commands in this file pass the environment to child processes
os.environ["PYTHONPATH"] = f"/home/toolkit/TapeAgents"
os.environ["NCCL_CUMEM_ENABLE"] = "0"
os.environ["TORCH_DISABLE_SHARE_RDZV_TCP_STORE"] = "1"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


@dataclass
class LaunchedProcess:
    kind: str
    handle: subprocess.Popen


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _required_prerun_path(prerun: DictConfig, key: str) -> Path:
    value = str(prerun.get(key, ""))
    if not value or value.startswith("PENDING_"):
        raise ValueError(f"Tau2/Gemma {key} is unresolved")
    path = Path(value)
    if not path.is_file():
        raise ValueError(f"Tau2/Gemma {key} does not exist: {path}")
    return path


def _require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise ValueError(
            f"Tau2/Gemma {label}={actual!r}, expected {expected!r}"
        )


def _validate_tau2_recipe_identity(cfg: DictConfig) -> None:
    from pipelinerl.domains.tau2.prerun import (
        GSPO_TOKEN_UPGRADE_TRIGGER,
        POLICY_LOSS_FALLBACK,
        POLICY_LOSS_FALLBACK_TRIGGER,
        RUN1_POLICY_LOSS,
        USER_SIMULATOR_ENDPOINT,
        USER_SIMULATOR_MODEL,
    )
    from pipelinerl.prerun_evidence import (
        GEMMA_MODEL_ID,
        GEMMA_MODEL_REVISION,
        GEMMA_POLICY_IDENTITY,
    )

    prerun = cfg.tau2_prerun
    _require_equal(str(cfg.model_path), GEMMA_MODEL_ID, "model_path")
    _require_equal(
        str(cfg.finetune.config_name),
        GEMMA_MODEL_ID,
        "finetune.config_name",
    )
    _require_equal(
        str(cfg.finetune.get("model_revision")),
        GEMMA_MODEL_REVISION,
        "finetune.model_revision",
    )
    _require_equal(
        bool(cfg.finetune.get("text_only_gemma4", False)),
        True,
        "finetune.text_only_gemma4",
    )
    _require_equal(
        str(cfg.finetune.rl.policy_loss),
        RUN1_POLICY_LOSS,
        "policy_loss",
    )
    _require_equal(
        str(prerun.policy_model),
        GEMMA_POLICY_IDENTITY,
        "policy_model",
    )
    _require_equal(
        str(prerun.policy_loss_fallback),
        POLICY_LOSS_FALLBACK,
        "policy_loss_fallback",
    )
    _require_equal(
        str(prerun.policy_loss_fallback_trigger),
        POLICY_LOSS_FALLBACK_TRIGGER,
        "policy_loss_fallback_trigger",
    )
    _require_equal(
        str(prerun.gspo_token_upgrade_trigger),
        GSPO_TOKEN_UPGRADE_TRIGGER,
        "gspo_token_upgrade_trigger",
    )
    _require_equal(
        str(cfg.tau2_gym.policy_model_name),
        GEMMA_POLICY_IDENTITY,
        "Gym policy model",
    )
    _require_equal(
        str(cfg.tau2_gym.user_model_url),
        USER_SIMULATOR_ENDPOINT,
        "user endpoint",
    )
    _require_equal(
        str(cfg.tau2_gym.user_model_name),
        USER_SIMULATOR_MODEL,
        "user model",
    )
    _require_equal(
        int(cfg.vllm_config.vllm_kwargs["tensor-parallel-size"]),
        2,
        "vLLM tensor parallel size",
    )
    _require_equal(
        bool(cfg.use_deepspeed),
        True,
        "DeepSpeed enablement",
    )


def _validate_tau2_calibration(
    cfg: DictConfig,
    job_spec_path: Path,
) -> None:
    from pipelinerl.domains.tau2.prerun import (
        CALIBRATION_ROLLOUTS,
        GSPO_TOKEN_UPGRADE_TRIGGER,
        PINNED_SOURCES,
        POLICY_LOSS_FALLBACK,
        POLICY_LOSS_FALLBACK_TRIGGER,
        RUN1_POLICY_LOSS,
        USER_SIMULATOR_ENDPOINT,
        USER_SIMULATOR_MODEL,
        PreRunSpec,
        ServiceIdentity,
        TrainerMemoryCandidate,
    )
    from pipelinerl.prerun_evidence import (
        GEMMA_MODEL_ID,
        GEMMA_MODEL_REVISION,
        GEMMA_POLICY_IDENTITY,
    )

    prerun = cfg.tau2_prerun
    if not prerun.enabled:
        raise ValueError(
            "Tau2/Gemma calibration requires evidence collection"
        )
    spec_path = _required_prerun_path(prerun, "spec_path")
    spec = PreRunSpec.model_validate_json(spec_path.read_text())
    candidates_payload = OmegaConf.to_container(
        prerun.production_memory_candidates,
        resolve=True,
    )
    candidates = [
        TrainerMemoryCandidate.model_validate(candidate)
        for candidate in candidates_payload
    ]

    _require_equal(
        spec.job_spec_sha256,
        _sha256_file(job_spec_path),
        "calibration job digest",
    )
    _require_equal(spec.model_id, GEMMA_MODEL_ID, "spec model")
    _require_equal(
        spec.model_revision,
        GEMMA_MODEL_REVISION,
        "spec revision",
    )
    _require_equal(
        Path(spec.model_snapshot).resolve(),
        Path(str(prerun.model_snapshot)).resolve(),
        "snapshot",
    )
    _require_equal(spec.source_pins, PINNED_SOURCES, "source pins")
    _require_equal(
        spec.user_simulator,
        ServiceIdentity(
            model=USER_SIMULATOR_MODEL,
            endpoint=USER_SIMULATOR_ENDPOINT,
        ),
        "user simulator",
    )
    _require_equal(
        spec.policy_model,
        GEMMA_POLICY_IDENTITY,
        "spec policy model",
    )
    _require_equal(
        spec.policy_endpoints,
        list(prerun.policy_endpoints),
        "policy endpoints",
    )
    _require_equal(spec.expected_tp_size, 2, "spec TP size")
    _require_equal(
        spec.fixed_prompt_token_ids,
        list(prerun.fixed_prompt_token_ids),
        "prompt token IDs",
    )
    _require_equal(
        spec.fixed_completion_token_ids,
        list(prerun.fixed_completion_token_ids),
        "completion token IDs",
    )
    _require_equal(
        spec.user_separation_asserted,
        True,
        "user separation assertion",
    )
    _require_equal(spec.packing_enabled, False, "packing gate")
    _require_equal(
        spec.mixed_version_loss_gate_passed,
        True,
        "mixed-version loss gate",
    )
    _require_equal(
        spec.policy_loss,
        RUN1_POLICY_LOSS,
        "spec policy loss",
    )
    _require_equal(
        spec.policy_loss_fallback,
        POLICY_LOSS_FALLBACK,
        "spec fallback",
    )
    _require_equal(
        spec.policy_loss_fallback_trigger,
        POLICY_LOSS_FALLBACK_TRIGGER,
        "spec fallback trigger",
    )
    _require_equal(
        spec.gspo_token_upgrade_trigger,
        GSPO_TOKEN_UPGRADE_TRIGGER,
        "spec GSPO-token trigger",
    )
    _require_equal(
        spec.production_memory_candidates,
        candidates,
        "memory candidates",
    )

    snapshot_path = Path(str(prerun.model_snapshot))
    if not snapshot_path.is_dir():
        raise ValueError(
            "Tau2/Gemma calibration model snapshot does not exist: "
            f"{snapshot_path}"
        )
    _require_equal(
        int(os.environ.get("WORLD_SIZE", "1")),
        4,
        "calibration node count",
    )
    _require_equal(
        bool(cfg.finetune.seq_packing),
        False,
        "calibration packing",
    )
    _require_equal(
        int(cfg.finetune.seq_parallel),
        1,
        "calibration seq_parallel",
    )
    _require_equal(
        str(cfg.deepspeed_config),
        "deepspeed_stage3_bf16_cpu_offload",
        "calibration optimizer profile",
    )
    _require_equal(
        int(cfg.finetune.max_train_steps),
        1,
        "calibration max train steps",
    )
    _require_equal(
        int(cfg.finetune.interrupt_train_steps),
        1,
        "calibration interrupt step",
    )
    _require_equal(
        int(cfg.finetune.gradient_accumulation_passes),
        CALIBRATION_ROLLOUTS,
        "calibration samples",
    )
    _require_equal(
        int(cfg.finetune.attempts),
        16,
        "calibration group size",
    )
    _require_equal(
        int(cfg.actor.rollout_workers),
        4,
        "calibration rollout workers",
    )
    _require_equal(
        int(cfg.actor.llm_max_rollouts),
        4,
        "calibration vLLM concurrency",
    )
    _require_equal(
        int(cfg.world.replicas),
        1,
        "calibration actor replicas",
    )
    _require_equal(
        int(cfg.world.actor_fraction),
        1,
        "calibration actor fraction",
    )
    _require_equal(
        int(cfg.world.preprocessor_fraction),
        0,
        "calibration preprocessor fraction",
    )
    _require_equal(
        int(cfg.world.finetune_fraction),
        3,
        "calibration trainer fraction",
    )

    context_tokens = int(prerun.calibration_context_tokens)
    queue_entry_bytes = int(prerun.calibration_queue_entry_bytes)
    _require_equal(
        context_tokens,
        262144,
        "calibration context tokens",
    )
    _require_equal(
        queue_entry_bytes,
        128 * (1 << 20),
        "calibration queue entry bytes",
    )
    _require_equal(
        int(cfg.actor.result_queue_size),
        2,
        "calibration result queue size",
    )
    _require_equal(
        int(cfg.preprocess.input_queue_size),
        2,
        "calibration preprocess input queue size",
    )
    _require_equal(
        int(cfg.preprocess.output_queue_size),
        2,
        "calibration preprocess output queue size",
    )
    cfg.finetune.seq_length = context_tokens
    cfg.vllm_config.vllm_kwargs.max_model_len = context_tokens
    cfg.actor.shared_memory_entry_size = queue_entry_bytes
    cfg.preprocess.shared_memory_entry_size = queue_entry_bytes


def _validate_tau2_production(
    cfg: DictConfig,
    job_spec_path: Path,
) -> None:
    from pipelinerl.domains.tau2.prerun import (
        CALIBRATION_CAVEAT,
        CALIBRATION_GROUPS,
        CALIBRATION_ROLLOUTS,
        GSPO_TOKEN_UPGRADE_TRIGGER,
        PINNED_SOURCES,
        POLICY_LOSS_FALLBACK,
        POLICY_LOSS_FALLBACK_TRIGGER,
        RUN1_POLICY_LOSS,
        USER_SIMULATOR_ENDPOINT,
        USER_SIMULATOR_MODEL,
        PreRunManifest,
        ServiceIdentity,
        require_ready_manifest,
    )
    from pipelinerl.prerun_evidence import (
        GEMMA_MODEL_ID,
        GEMMA_MODEL_REVISION,
        GEMMA_POLICY_IDENTITY,
        hash_model_snapshot,
    )

    prerun = cfg.tau2_prerun
    if prerun.enabled:
        raise ValueError(
            "Tau2/Gemma production must not collect calibration evidence"
        )
    manifest_path = _required_prerun_path(
        prerun,
        "manifest_path",
    )
    manifest = PreRunManifest.model_validate_json(
        manifest_path.read_text()
    )
    require_ready_manifest(manifest)
    _require_equal(
        manifest.job_spec_sha256,
        _sha256_file(job_spec_path),
        "manifest job digest",
    )
    _require_equal(
        manifest.model.model_id,
        GEMMA_MODEL_ID,
        "manifest model",
    )
    _require_equal(
        manifest.model.revision,
        GEMMA_MODEL_REVISION,
        "manifest revision",
    )
    snapshot_path = Path(str(prerun.model_snapshot))
    if not snapshot_path.is_dir():
        raise ValueError(
            "Tau2/Gemma production model snapshot does not exist: "
            f"{snapshot_path}"
        )
    model_identity = hash_model_snapshot(
        snapshot_path,
        GEMMA_MODEL_ID,
        GEMMA_MODEL_REVISION,
    )
    _require_equal(model_identity, manifest.model, "model artifact identity")
    _require_equal(
        manifest.source_pins,
        PINNED_SOURCES,
        "manifest source pins",
    )
    _require_equal(
        manifest.user_simulator,
        ServiceIdentity(
            model=USER_SIMULATOR_MODEL,
            endpoint=USER_SIMULATOR_ENDPOINT,
        ),
        "manifest user simulator",
    )
    _require_equal(
        manifest.policy_model,
        GEMMA_POLICY_IDENTITY,
        "manifest policy model",
    )
    _require_equal(
        manifest.expected_tp_size,
        2,
        "manifest TP size",
    )
    _require_equal(
        manifest.policy_loss,
        RUN1_POLICY_LOSS,
        "manifest policy loss",
    )
    _require_equal(
        manifest.policy_loss_fallback,
        POLICY_LOSS_FALLBACK,
        "manifest fallback",
    )
    _require_equal(
        manifest.policy_loss_fallback_trigger,
        POLICY_LOSS_FALLBACK_TRIGGER,
        "manifest fallback trigger",
    )
    _require_equal(
        manifest.gspo_token_upgrade_trigger,
        GSPO_TOKEN_UPGRADE_TRIGGER,
        "manifest GSPO-token trigger",
    )
    _require_equal(
        manifest.calibration.groups,
        CALIBRATION_GROUPS,
        "calibration groups",
    )
    _require_equal(
        manifest.calibration.attempted_rollouts,
        CALIBRATION_ROLLOUTS,
        "calibration rollouts",
    )
    _require_equal(
        manifest.calibration.sample_size_caveat,
        CALIBRATION_CAVEAT,
        "calibration caveat",
    )

    required_limits = {
        "actor_queue_entry_bytes",
        "training_envelope_entry_bytes",
        "model_context_tokens",
        "training_sequence_tokens",
        "generation_margin_tokens",
    }
    missing_limits = (
        required_limits
        - manifest.calibration.derived_limits.keys()
    )
    if missing_limits:
        raise ValueError(
            "Tau2/Gemma manifest is missing derived limits "
            f"{sorted(missing_limits)}"
        )
    for name in required_limits:
        limit = manifest.calibration.derived_limits[name]
        _require_equal(
            limit.calibration_rollouts,
            CALIBRATION_ROLLOUTS,
            f"{name} sample count",
        )
        _require_equal(
            limit.calibration_groups,
            CALIBRATION_GROUPS,
            f"{name} group count",
        )
        _require_equal(
            limit.caveat,
            CALIBRATION_CAVEAT,
            f"{name} caveat",
        )

    topology = manifest.recommended_production_topology
    if topology is None:
        raise ValueError(
            "Tau2/Gemma manifest has no fitting production topology"
        )
    if not any(
        budget.candidate == topology and budget.fits
        for budget in manifest.trainer_memory_budgets
    ):
        raise ValueError(
            "Tau2/Gemma recommended topology has no fitting memory budget"
        )
    if (
        topology.seq_parallel > 1
        and not cfg.finetune.seq_packing
    ):
        raise ValueError(
            "Tau2/Gemma gate 5 requires a packed-vs-unpacked "
            "equivalence proof before packing with seq_parallel > 1"
        )
    _require_equal(
        int(os.environ.get("WORLD_SIZE", "1")),
        topology.node_count,
        "production node count",
    )
    _require_equal(
        int(cfg.world.replicas),
        1,
        "production actor replicas",
    )

    limits = manifest.calibration.derived_limits
    cfg.actor.shared_memory_entry_size = (
        limits["actor_queue_entry_bytes"].derived_value
    )
    cfg.preprocess.shared_memory_entry_size = (
        limits["training_envelope_entry_bytes"].derived_value
    )
    cfg.vllm_config.vllm_kwargs.max_model_len = (
        limits["model_context_tokens"].derived_value
    )
    cfg.finetune.seq_length = max(
        limits["model_context_tokens"].derived_value,
        limits["training_sequence_tokens"].derived_value,
    )
    cfg.llm.parameters.max_tokens = (
        limits["generation_margin_tokens"].derived_value
    )
    cfg.test_llm.parameters.max_tokens = (
        limits["generation_margin_tokens"].derived_value
    )
    cfg.finetune.seq_parallel = topology.seq_parallel
    cfg.world.actor_fraction = topology.actor_gpus
    cfg.world.preprocessor_fraction = 0
    cfg.world.finetune_fraction = topology.trainer_gpus
    cfg.deepspeed_config = (
        "deepspeed_stage3_bf16_cpu_offload"
        if topology.optimizer_cpu_offload
        else "deepspeed_stage3_bf16"
    )


def validate_tau2_prerun(cfg: DictConfig) -> None:
    prerun = cfg.get("tau2_prerun")
    if not prerun:
        return
    from pipelinerl.prerun_evidence import (
        require_verified_gemma_revision,
    )

    require_verified_gemma_revision()
    _validate_tau2_recipe_identity(cfg)
    job_spec_path = _required_prerun_path(
        prerun,
        "job_spec_path",
    )
    phase = str(prerun.get("phase", ""))
    if phase == "calibration":
        _validate_tau2_calibration(cfg, job_spec_path)
    elif phase == "production":
        _validate_tau2_production(cfg, job_spec_path)
    else:
        raise ValueError(
            f"Unknown Tau2/Gemma pre-run phase {phase!r}"
        )


def _popen(
    cmd: list[str],
    env: dict | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> subprocess.Popen:
    """Wrapper around subprocess.Popen that allows for easier debugging."""
    if os.environ.get("DRY_RUN", "0") == "1":
        return  # type: ignore
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=stdout,
        stderr=stderr,
    )


def validate_config(cfg: DictConfig):
    validate_tau2_prerun(cfg)

    if "fp32_lm_head" in cfg:
        raise ValueError(
            "fp32_lm_head is no longer configurable; PipelineRL always uses FP32 output-head logits"
        )

    if cfg.world.preprocessor_fraction == 0 and cfg.finetune.rl.kl_coef > 0.0:
        raise ValueError("Preprocessor fraction must be > 0 if KL is used")
    
    # Check for vision language model constraints
    if cfg.finetune.model_class == "vision2seq-language-modeling":
        if "Qwen2.5-VL" not in cfg.model_path:
            raise ValueError("Only Qwen2.5-VL models are supported for vision language modeling")
        if cfg.finetune.seq_packing:
            raise ValueError("Vision language models cannot use sequence packing (seq_packing must be false)")
        if cfg.finetune.train_batch_size > 1:
            raise ValueError("Vision language models cannot use batch size > 1 (train_batch_size must be 1)")
    
    if cfg.finetune.seq_parallel > 1:
        if not cfg.finetune.seq_packing:
            raise ValueError("seq_parallel > 1 requires seq_packing to be true")
    
    if cfg.preprocess.dataset_buffer_size > 0:
        if cfg.preprocess.dataset_buffer_size != cfg.preprocess.ring_buffer_size:
            raise ValueError("dataset_buffer_size must be equal to ring_buffer_size")
        if cfg.pop_old_data:
            raise ValueError("Cannot use pop_old_data with preprocessor dataset_buffer_size > 0")

    # Check for value loss coefficient constraints
    if cfg.finetune.model_class == "causal-language-modeling-with-value-head":
        if not hasattr(cfg.finetune.rl, "value_loss_coef") or cfg.finetune.rl.value_loss_coef <= 0.0:
            raise ValueError("value_loss_coef must be greater than 0 when using causal-language-modeling-with-value-head")

    # Check that model being tuned to the max length accepted by inference
    if cfg.finetune.seq_length < cfg.vllm_config.vllm_kwargs.max_model_len:
        raise ValueError(
            f"seq_length {cfg.finetune.seq_length} must be greater than or equal to "
            f"vllm_kwargs.max_model_len {cfg.vllm_config.vllm_kwargs.max_model_len}"
        )

    # Check for asymmetric PPO clipping
    if cfg.finetune.rl.policy_loss == "ppo" and cfg.finetune.rl.epsilon_low != cfg.finetune.rl.epsilon_high:
        if cfg.finetune.model_class == "causal-language-modeling-with-value-head":
            logger.warning(
                "Asymmetric clipping with value head has not been tested and it may lead to unexpected behavior. "
                "It was recommended in DAPO (https://arxiv.org/abs/2503.14476) for GRPO (PPO without value head and group_size > 1)."
            )
        else:
            logger.warning(
                "Using asymmetric clipping. Note: this was recommended in DAPO (https://arxiv.org/abs/2503.14476) for GRPO."
            )


def _get_quantization_args(cfg: DictConfig) -> list[str]:
    """Build quantization CLI args for vLLM."""
    quantization = cfg.vllm_config.get("quantization")
    if quantization and quantization != "bf16_last_layer_fp32":
        raise ValueError(
            f"vllm_config.quantization='{quantization}' is incompatible with PipelineRL's "
            "required FP32 lm_head inference path"
        )
    return ["--quantization", "bf16_last_layer_fp32"]


def _get_quantization_env(cfg: DictConfig) -> dict[str, str]:
    """Get environment variables for quantization config."""
    prefix = cfg.get("fp32_layer_prefix", "lm_head")
    return {"PIPELINERL_FP32_LAYER_PREFIX": prefix}


def _get_vllm_cache_env(exp_dir: Path, tag: str) -> dict[str, str]:
    """Per-process compile cache dirs for a vLLM server.

    vLLM engines must not share one torch_compile_cache directory, because
    concurrent inductor writes can race. The compile artifacts are also large
    enough that putting all per-engine caches under /tmp can exceed EAI's
    ephemeral storage limit, so keep them isolated under the mounted experiment
    directory.
    """
    base = str(exp_dir / "vllm_cache" / tag)
    return {
        "VLLM_CACHE_ROOT": base,
        "TORCHINDUCTOR_CACHE_DIR": f"{base}/inductor",
        "TRITON_CACHE_DIR": f"{base}/triton",
    }


def _get_vllm_kwargs(cfg: DictConfig) -> dict:
    """Return launchable vLLM CLI kwargs for the supported V1 server path."""
    kwargs = OmegaConf.to_container(cfg.vllm_config.vllm_kwargs, resolve=True)
    if kwargs is None:
        return {}
    if not isinstance(kwargs, dict):
        raise TypeError(f"vllm_kwargs must resolve to a mapping, got {type(kwargs)}")

    for legacy_flag in ("disable-log-requests", "disable-frontend-multiprocessing"):
        if legacy_flag in kwargs:
            kwargs.pop(legacy_flag)
            logger.info(f"Dropping legacy vLLM flag '--{legacy_flag}' for V1 launch")

    return kwargs


def _append_vllm_kwargs(cmd: list[str], kwargs: dict) -> None:
    for k, v in kwargs.items():
        if isinstance(v, bool):
            cmd.append(f"--{k}" if v else f"--no-{k}")
            continue
        cmd.append(f"--{k}")
        if isinstance(v, (list, tuple)):
            cmd.extend(str(item) for item in v)
            continue
        if v not in [None, ""]:
            cmd.append(str(v))


def _add_actor_model_alias(
    kwargs: dict,
    actor_model_path: Path | str,
) -> None:
    if "served-model-name" not in kwargs:
        return
    served_names = kwargs["served-model-name"]
    if not isinstance(served_names, list):
        served_names = [served_names]
    served_names = [str(name) for name in served_names]
    actor_model_name = str(actor_model_path)
    if actor_model_name not in served_names:
        served_names.append(actor_model_name)
    kwargs["served-model-name"] = served_names


def run_ref_llm(cfg: DictConfig, preprocessor_llm_idx: int, local_idx: int, gpus: list[int], exp_dir: Path):
    kwargs = _get_vllm_kwargs(cfg)
    if kwargs.get("num-scheduler-steps", 1) > 1:
        kwargs["num-scheduler-steps"] = 1
        logger.warning("Set num-scheduler-steps to 1 for reference vLLM")
    log_dir = exp_dir / f"ref_vllm_{preprocessor_llm_idx}"
    os.makedirs(log_dir, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(cfg.model_path),
        "--port",
        str(8180 + local_idx),
        "--host",
        "0.0.0.0",
        "--seed",
        str(cfg.seed + preprocessor_llm_idx),
    ]

    cmd.extend(_get_quantization_args(cfg))

    _append_vllm_kwargs(cmd, kwargs)

    gpu_str = ",".join([str(gpu) for gpu in gpus])
    logger.info(f"Running reference LLM with command: {' '.join(cmd)} with gpus: {gpu_str}")
    log_file_path = os.path.join(log_dir, "stdout.log")
    err_file_path = os.path.join(log_dir, "stderr.log")
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": gpu_str,
        **_get_quantization_env(cfg),
        **_get_vllm_cache_env(exp_dir, f"ref_{preprocessor_llm_idx}"),
    }
    with open(log_file_path, "a") as log_file, open(err_file_path, "a") as err_file:
        proc = _popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=err_file,
        )
    if proc is not None:
        yield LaunchedProcess(kind="preprocessor_llm", handle=proc)


def run_actor_llm(
    cfg: DictConfig, world_map: WorldMap, actor_llm_idx: int, local_idx: int, gpus: list[int], exp_dir: Path
):
    finetune_model_path = exp_dir / "finetune" / "current"
    if os.path.exists(finetune_model_path):
        actor_model_path = finetune_model_path
    else:
        actor_model_path = cfg.model_path

    # TODO: add support for tensor and process parallelism
    log_dir = exp_dir / f"actor_vllm_{actor_llm_idx}"
    os.makedirs(log_dir, exist_ok=True)
    entrypoint = "pipelinerl.entrypoints.run_vllm1"
    cmd = [
        "python",
        "-m",
        entrypoint,
        "--model",
        str(actor_model_path),
        "--host",
        "0.0.0.0",
        "--port",
        str(8080 + local_idx),
        "--seed",
        str(cfg.seed + actor_llm_idx),
        "--actor-llm-idx",
        str(actor_llm_idx),
        "--weight-update-group-init-method",
        f"tcp://{world_map.master_addr}:{cfg.world.actor_group_port}",
        "--weight-update-group-world-size",
        str(world_map.weight_update_group_size),
    ]

    cmd.extend(_get_quantization_args(cfg))

    kwargs = _get_vllm_kwargs(cfg)
    _add_actor_model_alias(kwargs, actor_model_path)
    if kwargs:
        _append_vllm_kwargs(cmd, kwargs)

    if cfg.debug.mode:
        cmd.append("--disable-weight-updates")

    gpu_str = ",".join([str(gpu) for gpu in gpus])
    logger.info(f"Running actor_llm with command: {' '.join(cmd)} on gpus: {gpu_str}")
    save_command(log_dir, cmd)
    log_file_path = os.path.join(log_dir, "stdout.log")
    err_file_path = os.path.join(log_dir, "stderr.log")
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": gpu_str,
        **_get_quantization_env(cfg),
        **_get_vllm_cache_env(exp_dir, f"actor_{actor_llm_idx}"),
    }
    with open(log_file_path, "a") as log_file, open(err_file_path, "a") as err_file:
        proc = _popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=err_file,
        )
    if proc is not None:
        yield LaunchedProcess(kind="actor_llm", handle=proc)


def run_actor(world_map: WorldMap, actor_idx: int, exp_dir: Path):
    if actor_idx != 0:
        raise NotImplementedError("Can only do 1 actor yet")
    llm_urls = "+".join(world_map.get_actor_urls())
    cmd = [
        "python",
        "-m",
        "pipelinerl.entrypoints.run_actor",
        "--config-dir",
        f"{exp_dir}/conf",
        "--config-name",
        "exp_config",
        f"output_dir={exp_dir}",
        f"hydra.run.dir={exp_dir}/actor",
        f"+me.llm_urls={llm_urls}",
    ]
    logger.info(f"Running actor with command: {' '.join(cmd)}")
    save_command(exp_dir / "actor", cmd)
    proc = _popen(
        cmd,
        env=dict(os.environ),
    )
    if proc is not None:
        yield LaunchedProcess(kind="actor", handle=proc)

def run_environment(cfg: DictConfig, job: Job):
    # run in a subprocess like in the rest of the code
    run_dir = Path(cfg.output_dir) / f"environment_{job.replica_idx}"
    cmd = [
        "python",
        "-m",
        "pipelinerl.entrypoints.run_environment",
        "--config-dir",
        f"{cfg.output_dir}/conf",
        "--config-name",
        "exp_config",
        f"output_dir={cfg.output_dir}",
        f"hydra.run.dir={str(run_dir)}",
        f"me.job_idx={job.idx}",
    ]
    logger.info(f"Running environment with command: {' '.join(cmd)}")
    os.makedirs(run_dir, exist_ok=True)    
    save_command(run_dir, cmd)
    log_file_path = str(run_dir / "stdout.log")
    err_file_path = str(run_dir / "stderr.log")
    with open(log_file_path, "a") as log_file, open(err_file_path, "a") as err_file:
        proc = _popen(
            cmd,
            env=dict(os.environ),
            stdout=log_file,
            stderr=err_file,
        )
    if proc is not None:
        yield LaunchedProcess(kind="environment", handle=proc)


def run_finetune(cfg: DictConfig, world_map: WorldMap, gpus: list[int], exp_dir: Path):
    if cfg.use_fsdp and cfg.use_deepspeed:
        raise ValueError("Cannot use both FSDP and DeepSpeed")
    cmd = [
        "python",
        "-m",
        "accelerate.commands.launch",
    ]
    if world_map.world_size > 1:
        # DeepSpeed multi-node args
        assert cfg.use_deepspeed
        assert world_map.master_addr.startswith("dns-") and world_map.master_addr.endswith("-0")
        hosts = [world_map.master_addr[:-2] + f"-{i}" for i in range(world_map.world_size)]
        filter_parts = []
        for rank, job_list in world_map.job_map.items():
            for job in job_list:
                if job.kind == "finetune":
                    filter_parts.append(f"{hosts[rank]}:{','.join(map(str, job.gpus))}")
        deepspeed_include_filter = "@".join(filter_parts)
        logger.info(f"Deepspeed include filter: {deepspeed_include_filter}")
        # Orchestrator rank must have already created hostfile.txt
        hostfile_path = str(exp_dir / "hostfile.txt")
        cmd += [
            "--num_machines",
            str(len(world_map.nodes_with_finetuning())),
            "--machine_rank",
            str(world_map.my_finetuning_rank()),
            "--main_process_ip",
            str(os.environ.get("MASTER_ADDR")),
            "--main_process_port",
            str(os.environ.get("MASTER_PORT")),
            "--deepspeed_hostfile",
            hostfile_path,
            "--deepspeed_inclusion_filter",
            deepspeed_include_filter,
            "--deepspeed_multinode_launcher",
            "nossh"
        ]
    # get path to this file
    this_file_path = Path(os.path.dirname(os.path.abspath(__file__)))
    if cfg.use_deepspeed:
        # DeepSpeed single-node args
        cmd += [
            "--use_deepspeed",
            "--deepspeed_config_file",
            str(this_file_path / f"../conf/deepspeed/{cfg.deepspeed_config}.json"),
        ]
    # DeepSpeed and non-DeepSpeed args
    accelerate_config = cfg.accelerate_config
    if accelerate_config is None:
        if cfg.use_deepspeed:
            accelerate_config = "deepspeed"
        elif cfg.use_fsdp:
            accelerate_config = "fsdp_mp"
        else:
            accelerate_config = "base_mp"
    cmd += [
        "--config_file",
        str(this_file_path / f"../conf/accelerate/{accelerate_config}.yaml"),
        "--rdzv_backend",
        "c10d",
    ]
    if gpus:
        gpus_str = str(",".join([str(gpu) for gpu in gpus])) if len(gpus) < world_map.node_size else "all"
        cmd += [
            "--gpu-ids",
            gpus_str,
        ]
    cmd += [
        "--num_processes",
        str(world_map.total_finetune_gpus),
        "pipelinerl/entrypoints/run_finetune.py",
        "--config-dir",
        f"{exp_dir}/conf",
        "--config-name",
        "exp_config",
        f"output_dir={exp_dir}",
        f"hydra.run.dir={exp_dir}/finetune",
        # TODO: figure out why we can't build WorldMap in run_finetune.py
        # Current workaround: pass the essential information as follows:
        f"+me.weight_update_group_init_method=tcp://{world_map.master_addr}:{cfg.world.actor_group_port}",
        f"+me.weight_update_group_world_size={world_map.weight_update_group_size}",
        f"+me.llm_urls={'+'.join(world_map.get_actor_urls())}",
    ]
    if cfg.debug.mode in ["finetune", "open_loop", "finetune+preprocessor"]:
        cmd.append("finetune.send_weight_updates=False")

    logger.info(f"Running finetune with command: {' '.join(cmd)}")
    save_command(exp_dir / "finetune", cmd)
    env = dict(os.environ)
    env["DS_ENV_FILE"] = str(exp_dir / ".deepspeed_env")
    proc = _popen(cmd, env=env)
    if proc is not None:
        yield LaunchedProcess(kind="finetune", handle=proc)


def run_preprocess(world_map: WorldMap, preprocessor_idx: int, exp_dir: Path):
    if preprocessor_idx != 0:
        raise NotImplementedError("Can only do 1 preprocessor yet")
    llm_urls = "+".join(world_map.get_preprocessor_urls())
    cmd = [
        "python",
        "-m",
        "pipelinerl.entrypoints.run_preprocess",
        "--config-dir",
        f"{exp_dir}/conf",
        "--config-name",
        "exp_config",
        f"output_dir={exp_dir}",
        f"hydra.run.dir={exp_dir}/preprocess",
        f"+me.llm_urls={llm_urls}",
    ]
    logger.info(f"Running preprocess with command: {' '.join(cmd)}")
    save_command(exp_dir / "preprocess", cmd)
    proc = _popen(
        cmd,
        env=dict(os.environ),
    )
    if proc is not None:
        yield LaunchedProcess(kind="preprocessor", handle=proc)


def run_redis(cfg: DictConfig):
    # Launch redis-server
    cmd = [
        "redis-server",
        "--bind",
        "0.0.0.0",
        "--port",
        str(cfg.streams.port),
        "--dir",
        str(cfg.output_dir),
        "--protected-mode",
        "no",
        "--save",
        cfg.streams.save,
    ]
    logger.info(f"Running redis with command: {' '.join(cmd)}")
    save_command(Path(cfg.output_dir) / "redis", cmd)
    proc = _popen(cmd, env=dict(os.environ))
    if proc is not None:
        yield LaunchedProcess(kind="redis", handle=proc)


def save_command(script_dir: Path, cmd):
    os.makedirs(script_dir, exist_ok=True)
    script_path = script_dir / "start.sh"
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        # Properly quote arguments for the shell script
        quoted_cmd = [f"'{arg}'" if " " in arg or "$" in arg else arg for arg in cmd]
        f.write(" ".join(quoted_cmd) + "\n")
    os.chmod(script_path, 0o755)
    logger.info(f"Saved start script to {script_path}")


def clean_up(exp_dir, force_restart):
    logger.info("Cleaning up streams directory")
    if os.path.exists(f"{exp_dir}/streams"):
        if os.path.isdir(f"{exp_dir}/streams") and not os.path.islink(f"{exp_dir}/streams"):
            shutil.rmtree(f"{exp_dir}/streams")
        else:
            os.remove(f"{exp_dir}/streams")
    if os.path.exists(f"{exp_dir}/dump.rdb"):
        os.remove(f"{exp_dir}/dump.rdb")

    if force_restart:
        if os.path.exists(f"{exp_dir}/finetune"):
            logger.info("Cleaning up finetune directory")
            shutil.rmtree(f"{exp_dir}/finetune")
        if os.path.exists(f"{exp_dir}/vllm_cache"):
            logger.info("Cleaning up vLLM cache directory")
            shutil.rmtree(f"{exp_dir}/vllm_cache")

        # erase all the logs
        log_files = list(exp_dir.glob("**/*.log"))
        for log_file in log_files:
            logger.info(f"Erasing {log_file}")
            with open(log_file, "r"):
                pass


def is_inference_process(proc: LaunchedProcess) -> bool:
    return proc.kind in {"actor_llm", "preprocessor_llm"}


def watch_processes_running(exp_path: Path, processes: List[LaunchedProcess], debug_mode: bool = False):
    if not debug_mode:
        trainer_state = TrainerState(exp_path)
        trainer_state.start_listening()
    else:
        trainer_state = None

    # Wait for all processes to complete
    def gently_stop_all_processes():
        logger.info("\nShutting down processes...")
        # Terminate all running processes
        for proc in processes:
            logger.info(f"Terminating {proc.handle.args}")
            terminate_with_children(proc.handle.pid)

    logger.info("I have launched everyone, waiting for them to finish...")

    # last_trainer_version = -1
    # last_time_new_version = time.time()

    try:
        # Wait for all processes to complete
        # if just one dies non-zero, stop all
        alive = list(processes)
        logger.info(f"Starting process monitoring with {len(alive)} processes: {[proc.kind for proc in alive]}")
        while alive:
            for proc in list(alive):
                return_code = proc.handle.poll()
                if return_code is None:
                    continue
                if return_code != 0:
                    logger.error(f"Process {proc.handle.args} terminated with code {return_code}")
                    gently_stop_all_processes()
                    sys.exit(1)
                logger.info(f"Process {proc.handle.args} finished cleanly")
                alive.remove(proc)
            if alive and all(is_inference_process(proc) for proc in alive):
                # shut down inference servers after training is complete
                if trainer_state is not None and not trainer_state.training_done:
                    # check if training is completed
                    logger.info(f"Waiting for training completion signal (training_done={trainer_state.training_done})")
                    trainer_state.wait_for_training_done(timeout=5.0)
                    continue
                logger.info(f"Trainer completion detected; stopping remaining {len(alive)} inference server(s)")
                for proc in list(alive):
                    logger.info(f"Terminating inference server {proc.handle.args}")
                    terminate_with_children(proc.handle.pid)
                for proc in list(alive):
                    proc.handle.wait()
                    logger.info(f"Inference server {proc.handle.args} stopped")
                    alive.remove(proc)
            # TODO: make the watcdog code below more stable
            # if (trainer_state is not None
            #     and (version := trainer_state.propagated_weight_version is not None)
            #     and version > last_trainer_version):
            #     last_trainer_version = version
            #     last_time_new_version = time.time()
            # if not debug_mode and time.time() - last_time_new_version > 1800:
            #     logger.error("No new weight update in 30 minutes, exiting")
            #     sys.exit(1)
            time.sleep(1.0)
    except KeyboardInterrupt:
        gently_stop_all_processes()


def debug_link_streams(cfg: DictConfig, topics: list[str]):
    if not cfg.debug.streams_from:
        raise ValueError("Need to specify streams_from for debug mode")
    stream_dir = Path(cfg.output_dir) / "streams"
    for topic in topics:
        source_topic_dir = Path(cfg.debug.streams_from) / "streams" / topic
        target_topic_dir = stream_dir / topic
        if not os.path.exists(source_topic_dir):
            raise ValueError(f"Source topic {source_topic_dir} does not exist")
        os.symlink(source_topic_dir, target_topic_dir)
        logger.info(f"Linked {source_topic_dir} to {target_topic_dir}")


def launch_jobs(cfg: DictConfig, world_map: WorldMap, job_kind_filter: list | None = None):
    exp_dir = Path(cfg.output_dir)
    processes = []
    all_job_kinds = ["actor", "environment", "actor_llm", "preprocessor", "preprocessor_llm", "finetune"]
    if job_kind_filter is None:
        job_kind_filter = all_job_kinds
    for job in world_map.my_jobs():
        if job.kind not in all_job_kinds:
            raise ValueError(f"Unknown job kind {job.kind}")
        if job.kind not in job_kind_filter:
            continue
        if job.kind == "actor":
            processes.extend(run_actor(world_map, job.replica_idx, exp_dir))
        elif job.kind == "environment":
            processes.extend(run_environment(cfg, job))
        elif job.kind == "actor_llm":
            if cfg.debug.use_existing_llms:
                continue
            processes.extend(run_actor_llm(cfg, world_map, job.replica_idx, job.local_idx, job.gpus, exp_dir))
        elif job.kind == "preprocessor":
            processes.extend(run_preprocess(world_map, job.replica_idx, exp_dir))
        elif job.kind == "preprocessor_llm":
            if cfg.debug.use_existing_llms:
                continue            
            processes.extend(run_ref_llm(cfg, job.replica_idx, job.local_idx, job.gpus, exp_dir))
        elif job.kind == "finetune":
            processes.extend(run_finetune(cfg, world_map, job.gpus, exp_dir))
        else:
            raise ValueError(f"Unknown job kind {job.kind}")
    return processes


def setup_logging(log_file: Path):
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    logger.info("Logging setup complete")


def validate_external_services(cfg: DictConfig, world_map: WorldMap) -> None:
    tau2_gym = cfg.get("tau2_gym")
    if not tau2_gym:
        return
    from pipelinerl.domains.tau2.client import validate_tau2_gym_sync

    settings = OmegaConf.to_container(tau2_gym, resolve=True)
    validate_tau2_gym_sync(settings, world_map.get_actor_urls())


@hydra.main(
    config_path="../conf/",
    config_name="base",
    version_base="1.3.2",
)
def main(cfg: DictConfig):
    validate_config(cfg)
    if os.environ.get("PIPELINERL_PREFLIGHT_ONLY", "0") == "1":
        return

    exp_dir = Path(cfg.output_dir)
    config_dir = exp_dir / "conf"

    os.makedirs(exp_dir / "launcher", exist_ok=True)
    log_file = exp_dir / "launcher" / f"launcher_{os.environ.get('RANK', 0)}.log"
    setup_logging(log_file)
    world_map = WorldMap(cfg, verbose=True)
    cfg.jobs = [job.model_dump() for job in world_map.get_all_jobs()]

    if world_map.my_rank == 0:
        validate_external_services(cfg, world_map)
    group = str(exp_dir)
    root = cfg.wandb.wandb_workspace_root
    if root:
        if not group.startswith(root + "/"):
            raise ValueError(f"run_dir {exp_dir} does not start with root {root}")
        cfg.wandb.wandb_group = group[len(root) + 1 :]
    if world_map.total_finetune_gpus:
        accum_passes = cfg.finetune.gradient_accumulation_passes
        n_gpus = world_map.total_finetune_gpus
        if accum_passes % n_gpus != 0:
            new_accum_passes = math.ceil(accum_passes / n_gpus) * n_gpus
            logger.warning(
                f"Adjusting gradient_accumulation_passes from {accum_passes} to {new_accum_passes} "
                f"to make it divisible by {n_gpus} processes"
            )
            cfg.finetune.gradient_accumulation_passes = new_accum_passes
    if cfg.streams.backend == "redis":
        cfg.streams.host = world_map.master_addr
    set_streams_backend(**cfg.streams)

    processes = []

    lead_launcher_stream = SingleStreamSpec(exp_path=exp_dir, topic="launcher_0")
    init_msg = {"exp_init": "true"}
    if world_map.my_rank == 0:
        clean_up(exp_dir, cfg.force_restart)
        os.makedirs(config_dir, exist_ok=True)
        OmegaConf.save(cfg, config_dir / "exp_config.yaml")
        logger.info("Orchestrator 0 created the exp folder")
        if cfg.streams.backend == "redis":
            processes.extend(run_redis(cfg))
            redis = connect_to_redis(cfg.streams)
            redis.flushall()

        if world_map.world_size > 1:
            assert world_map.master_addr.startswith("dns-") and world_map.master_addr.endswith("-0")
            hosts = [world_map.master_addr[:-2] + f"-{i}" for i in range(world_map.world_size)]
            hostfile_lines = [f"{host} slots=8" for host in hosts]
            deepspeed_hostfile_content = "\n".join(hostfile_lines)
            hostfile_path = str(exp_dir / "hostfile.txt")
            with open(hostfile_path, "w") as f:
                f.write(deepspeed_hostfile_content)
            logger.info(f"Deepspeed hostfile content:\n{deepspeed_hostfile_content}")
            logger.info(f"Orchestrator 0 created hostfile at {hostfile_path}")

        with write_to_streams(lead_launcher_stream) as stream:
            stream.write(init_msg)
        if cfg.debug.mode == "finetune":
            debug_link_streams(cfg, [cfg.finetune.input])
        elif cfg.debug.mode == "preprocessor":
            debug_link_streams(cfg, [cfg.preprocess.input])
        elif cfg.debug.mode == "finetune+preprocessor":
            debug_link_streams(cfg, [cfg.preprocess.input])
    else:
        with read_stream(lead_launcher_stream) as stream:
            if (msg := next(stream.read())) != init_msg:
                raise ValueError(f"Expected {init_msg}, got {msg}")
        logger.info(f"Orchestrator {world_map.my_rank} heard that the exp folder is ready.")

    if cfg.debug.mode == "finetune":
        processes.extend(launch_jobs(cfg, world_map, ["finetune"]))
    elif cfg.debug.mode == "actor":
        processes.extend(launch_jobs(cfg, world_map, ["actor", "environment", "actor_llm"]))
    elif cfg.debug.mode == "preprocessor":
        processes.extend(launch_jobs(cfg, world_map, ["preprocessor", "preprocessor_llm"]))
    elif cfg.debug.mode == "actor+preprocessor":
        processes.extend(launch_jobs(cfg, world_map, ["actor", "environment", "actor_llm", "preprocessor", "preprocessor_llm"]))       
    elif cfg.debug.mode == "finetune+preprocessor":
        processes.extend(launch_jobs(cfg, world_map, ["finetune", "preprocessor", "preprocessor_llm"]))
    elif cfg.debug.mode in ["", "open_loop"]:
        processes.extend(launch_jobs(cfg, world_map))
    else:
        raise NotImplementedError(f"Unknown debug mode {cfg.debug.mode}")

    if os.environ.get("DRY_RUN", "0") == "1":
        assert not processes
        return
    watch_processes_running(exp_dir, processes, bool(cfg.debug.mode))


if __name__ == "__main__":
    main()
