import logging
import math
import os
import shutil
import socket
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
    if cfg.use_fast_llm:
        max_seq_length = cfg.fast_llm.data.micro_batch_size
        seq_length_label = "fast_llm.data.micro_batch_size"
    else:
        max_seq_length = cfg.finetune.seq_length
        seq_length_label = "finetune.seq_length"
    if max_seq_length < cfg.vllm_config.vllm_kwargs.max_model_len:
        raise ValueError(
            f"{seq_length_label} {max_seq_length} must be greater than or equal to "
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
    if cfg.get("fp32_lm_head", False):
        explicit_quant = cfg.vllm_config.get("quantization")
        if explicit_quant and explicit_quant != "bf16_last_layer_fp32":
            logger.warning(
                f"fp32_lm_head=true overrides explicit vllm_config.quantization='{explicit_quant}' "
                f"with 'bf16_last_layer_fp32'"
            )
        return ["--quantization", "bf16_last_layer_fp32"]
    elif cfg.vllm_config.get("quantization"):
        return ["--quantization", cfg.vllm_config.quantization]
    return []


def _get_quantization_env(cfg: DictConfig) -> dict[str, str]:
    """Get environment variables for quantization config."""
    env = {}
    if cfg.get("fp32_lm_head", False):
        # Pass the layer prefix to the quantization config via environment variable
        prefix = cfg.get("fp32_layer_prefix", "lm_head")
        env["PIPELINERL_FP32_LAYER_PREFIX"] = prefix
    return env


def _get_vllm_kwargs(cfg: DictConfig, *, use_v1: bool) -> dict:
    """Return launchable vLLM CLI kwargs, dropping legacy flags for V1."""
    kwargs = OmegaConf.to_container(cfg.vllm_config.vllm_kwargs, resolve=True)
    if kwargs is None:
        return {}
    if not isinstance(kwargs, dict):
        raise TypeError(f"vllm_kwargs must resolve to a mapping, got {type(kwargs)}")

    if use_v1:
        # Keep V1 actor/reference serving closer to the legacy V0 path by default.
        kwargs.setdefault("enable-prefix-caching", False)
        kwargs.setdefault("async-scheduling", False)
        # processed_logprobs returns log-probs computed during the forward pass,
        # avoiding stale values if weights change between generation and scoring.
        kwargs.setdefault("logprobs-mode", "processed_logprobs")
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
        if v not in [None, ""]:
            cmd.append(str(v))


def run_ref_llm(cfg: DictConfig, preprocessor_llm_idx: int, local_idx: int, gpus: list[int], exp_dir: Path):
    kwargs = _get_vllm_kwargs(cfg, use_v1=cfg.vllm_config.use_v1)
    if kwargs.get("num-scheduler-steps", 1) > 1:
        kwargs["num-scheduler-steps"] = 1
        logger.warning("Set num-scheduler-steps to 1 for reference vLLM")
    log_dir = exp_dir / f"ref_vllm_{preprocessor_llm_idx}"
    os.makedirs(log_dir, exist_ok=True)

    cmd = [
        sys.executable,
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
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_str, **_get_quantization_env(cfg)}
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
    entrypoint = (
        "pipelinerl.entrypoints.run_vllm1" 
        if cfg.vllm_config.use_v1 else 
        "pipelinerl.entrypoints.run_vllm0"
    )
    broadcast_port = cfg.world.actor_group_port
    cmd = [
        sys.executable,
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
        f"tcp://{world_map.master_addr}:{broadcast_port}",
        "--weight-update-group-world-size",
        str(world_map.weight_update_group_size),
    ]

    cmd.extend(_get_quantization_args(cfg))

    kwargs = _get_vllm_kwargs(cfg, use_v1=cfg.vllm_config.use_v1)
    # vLLM v1 rejects num-scheduler-steps; defensively drop it for v1 launches.
    if cfg.vllm_config.use_v1 and "num-scheduler-steps" in kwargs:
        kwargs.pop("num-scheduler-steps")
    if kwargs:
        _append_vllm_kwargs(cmd, kwargs)

    if cfg.debug.mode or not cfg.weight_broadcast:
        cmd.append("--disable-weight-updates")

    # Always tell the vLLM actor server which weight-update protocol to use,
    # so its conditional init takes the right branch (HTTP vs fast-llm broadcast).
    if cfg.use_fast_llm:
        cmd += [
            "--weight-update-mode", "fast-llm",
            "--redis-host", cfg.streams.host,
            "--redis-port", str(cfg.streams.port),
        ]
    else:
        cmd += ["--weight-update-mode", "http"]

    gpu_str = ",".join([str(gpu) for gpu in gpus])
    logger.info(f"Running actor_llm with command: {' '.join(cmd)} on gpus: {gpu_str}")
    save_command(log_dir, cmd)
    log_file_path = os.path.join(log_dir, "stdout.log")
    err_file_path = os.path.join(log_dir, "stderr.log")
    # Give each actor a distinct base port so vLLM's get_open_port() race condition
    # (TOCTOU: find-free-port then bind) doesn't cause EADDRINUSE when multiple servers start simultaneously.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_str, "VLLM_PORT": str(30000 + actor_llm_idx * 20), **_get_quantization_env(cfg)}
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
        sys.executable,
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
        sys.executable,
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
    if cfg.use_fast_llm:
        yield from _run_finetune_fast_llm(cfg, world_map, gpus, exp_dir)
    else:
        yield from _run_finetune_deepspeed(cfg, world_map, gpus, exp_dir)


def _run_finetune_deepspeed(cfg: DictConfig, world_map: WorldMap, gpus: list[int], exp_dir: Path):
    if cfg.use_fsdp and cfg.use_deepspeed:
        raise ValueError("Cannot use both FSDP and DeepSpeed")
    cmd = [
        "python",
        "-m",
        "accelerate.commands.launch",
    ]
    if world_map.world_size > 1:
        assert cfg.use_deepspeed
        # Use original DNS names (pod IP exchange may have replaced address_map with IPs).
        dns_map = getattr(world_map, "dns_address_map", world_map.address_map)
        hosts = [dns_map[i] for i in range(world_map.world_size)]
        filter_parts = []
        for rank, job_list in world_map.job_map.items():
            for job in job_list:
                if job.kind == "finetune":
                    filter_parts.append(f"{hosts[rank]}:{','.join(map(str, job.gpus))}")
        deepspeed_include_filter = "@".join(filter_parts)
        logger.info(f"Deepspeed include filter: {deepspeed_include_filter}")
        hostfile_path = str(exp_dir / "hostfile.txt")
        cmd += [
            "--num_machines", str(len(world_map.nodes_with_finetuning())),
            "--machine_rank", str(world_map.my_finetuning_rank()),
            "--main_process_ip", str(os.environ.get("MASTER_ADDR")),
            "--main_process_port", str(os.environ.get("MASTER_PORT")),
            "--deepspeed_hostfile", hostfile_path,
            "--deepspeed_inclusion_filter", deepspeed_include_filter,
            "--deepspeed_multinode_launcher", "nossh",
        ]
    this_file_path = Path(os.path.dirname(os.path.abspath(__file__)))
    if cfg.use_deepspeed:
        cmd += [
            "--use_deepspeed",
            "--deepspeed_config_file",
            str(this_file_path / f"../conf/deepspeed/{cfg.deepspeed_config}.json"),
        ]
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
        "--rdzv_backend", "c10d",
    ]
    if gpus:
        gpus_str = str(",".join([str(gpu) for gpu in gpus])) if len(gpus) < world_map.node_size else "all"
        cmd += ["--gpu-ids", gpus_str]
    cmd += [
        "--num_processes", str(world_map.total_finetune_gpus),
        str(this_file_path / "entrypoints/run_finetune.py"),
        "--config-dir", f"{exp_dir}/conf",
        "--config-name", "exp_config",
        f"output_dir={exp_dir}",
        f"hydra.run.dir={exp_dir}/finetune",
        f"+me.weight_update_group_init_method=tcp://{world_map.master_addr}:{cfg.world.actor_group_port}",
        f"+me.weight_update_group_world_size={world_map.weight_update_group_size}",
        f"+me.llm_urls={'+'.join(world_map.get_actor_urls())}",
    ]
    if cfg.debug.mode in ["finetune", "open_loop", "finetune+preprocessor"]:
        cmd.append("finetune.send_weight_updates=False")

    finetune_nodes = world_map.nodes_with_finetuning()
    finetune_rank = world_map.my_finetuning_rank()
    node_suffix = f"_node{finetune_rank}" if len(finetune_nodes) > 1 else ""

    logger.info(f"Running DeepSpeed finetune with command: {' '.join(cmd)}")
    save_command(exp_dir / "finetune", cmd, suffix=node_suffix)
    env = dict(os.environ)
    env["DS_ENV_FILE"] = str(exp_dir / ".deepspeed_env")
    save_dir = exp_dir / "finetune"
    os.makedirs(save_dir, exist_ok=True)
    log_file_path = save_dir / f"stdout{node_suffix}.log"
    err_file_path = save_dir / f"stderr{node_suffix}.log"
    with open(log_file_path, "a") as log_file, open(err_file_path, "a") as err_file:
        proc = _popen(cmd, env=env, stdout=log_file, stderr=err_file)
    if proc is not None:
        yield LaunchedProcess(kind="finetune", handle=proc)


def _run_finetune_fast_llm(cfg: DictConfig, world_map: WorldMap, gpus: list[int], exp_dir: Path):
    save_dir = exp_dir / "finetune"
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.isdir(cfg.model_path):
        raise ValueError(
            f"fast-llm requires a local model path but got: {cfg.model_path!r}. "
            "Download the model first and set model_path to its local directory."
        )

    # Build fast-llm config, stripping callbacks when weight broadcast is disabled or in debug mode.
    fast_llm_cfg = OmegaConf.to_container(cfg.fast_llm, resolve=True, throw_on_missing=False)
    if not cfg.weight_broadcast or bool(cfg.debug.mode):
        fast_llm_cfg.pop("callbacks", None)

    # Derive experiment name for wandb from save_dir relative to workspace root.
    root = cfg.wandb.wandb_workspace_root
    save_dir_str = str(save_dir)
    experiment_name = save_dir_str[len(root) + 1:] if root and save_dir_str.startswith(root + "/") else save_dir.name

    # Fill in all dynamic values so the saved config is fully functional.
    fast_llm_cfg["pretrained"]["path"] = cfg.model_path
    fast_llm_cfg["run"]["experiment_dir"] = str(save_dir)
    fast_llm_cfg["run"]["experiment_name"] = experiment_name
    fast_llm_cfg["data"]["datasets"]["training"]["host"] = cfg.streams.host
    fast_llm_cfg["data"]["datasets"]["training"]["port"] = cfg.streams.port
    if cfg.debug.log_data_pipeline:
        fast_llm_cfg["data"]["datasets"]["training"]["log_data_pipeline"] = True
        fast_llm_cfg.setdefault("schedule", {})["log_data_pipeline"] = True
    fast_llm_cfg["training"]["wandb"]["entity_name"] = cfg.wandb.wandb_entity_name
    fast_llm_cfg["training"]["wandb"]["project_name"] = cfg.wandb.wandb_project_name
    fast_llm_cfg["training"]["wandb"]["group_name"] = cfg.wandb.wandb_group
    if cfg.weight_broadcast and not bool(cfg.debug.mode):
        fast_llm_cfg["callbacks"]["streaming"]["host"] = cfg.streams.host
        fast_llm_cfg["callbacks"]["streaming"]["port"] = cfg.streams.port
        # fast-llm runs on node 0 (same node as the TCPStore server); use localhost
        # to avoid DNS self-resolution issues.  vLLM (on node 1) uses master_addr.
        fast_llm_cfg["callbacks"]["streaming"]["broadcast"]["host"] = "localhost"
        fast_llm_cfg["callbacks"]["streaming"]["broadcast"]["port"] = cfg.world.actor_group_port
        fast_llm_cfg["callbacks"]["streaming"]["broadcast"]["external_world_size"] = world_map.weight_update_group_size - 1

    # Use per-node suffixes for all output files to avoid NFS write races when multiple
    # finetune nodes share the same experiment directory.
    model_type = cfg.fast_llm_finetune.model_type
    torchrun_port = cfg.fast_llm_finetune.torchrun_port
    finetune_nodes = world_map.nodes_with_finetuning()
    finetune_rank = world_map.my_finetuning_rank()
    node_suffix = f"_node{finetune_rank}" if len(finetune_nodes) > 1 else ""

    config_path = save_dir / f"fast_llm_config{node_suffix}.yaml"
    OmegaConf.save(OmegaConf.create(fast_llm_cfg), config_path)

    if len(finetune_nodes) > 1:
        finetune_master = world_map.address_map[finetune_nodes[0]]
        cmd = [
            "torchrun",
            f"--nproc_per_node={len(gpus)}",
            f"--nnodes={len(finetune_nodes)}",
            f"--node_rank={finetune_rank}",
            "--rdzv_backend=static",
            "--rdzv_id=0",
            f"--rdzv_endpoint={finetune_master}:{torchrun_port}",
            "--rdzv_conf=timeout=3600",
            "--max_restarts=0",
            "--no_python",
            str(Path(sys.executable).parent / "fast-llm"),
            "train",
            model_type,
            "--config",
            str(config_path),
        ]
    else:
        cmd = [
            "torchrun",
            f"--nproc_per_node={len(gpus)}",
            f"--master_port={torchrun_port}",
            "--no_python",
            str(Path(sys.executable).parent / "fast-llm"),
            "train",
            model_type,
            "--config",
            str(config_path),
        ]

    logger.info(f"Running finetune with command: {' '.join(cmd)}")
    save_command(save_dir, cmd, suffix=node_suffix)
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "42"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpus)
    log_file_path = save_dir / f"stdout{node_suffix}.log"
    err_file_path = save_dir / f"stderr{node_suffix}.log"
    with open(log_file_path, "a") as log_file, open(err_file_path, "a") as err_file:
        proc = _popen(cmd, env=env, stdout=log_file, stderr=err_file)
    if proc is not None:
        yield LaunchedProcess(kind="finetune", handle=proc)

# def run_finetune(cfg: DictConfig, world_map: WorldMap, gpus: list[int], exp_dir: Path):
#     if cfg.use_fsdp and cfg.use_deepspeed:
#         raise ValueError("Cannot use both FSDP and DeepSpeed")
#     cmd = [
#         "python",
#         "-m",
#         "accelerate.commands.launch",
#     ]
#     if world_map.world_size > 1:
#         # DeepSpeed multi-node args
#         assert cfg.use_deepspeed
#         assert world_map.master_addr.startswith("dns-") and world_map.master_addr.endswith("-0")
#         hosts = [world_map.master_addr[:-2] + f"-{i}" for i in range(world_map.world_size)]
#         filter_parts = []
#         for rank, job_list in world_map.job_map.items():
#             for job in job_list:
#                 if job.kind == "finetune":
#                     filter_parts.append(f"{hosts[rank]}:{','.join(map(str, job.gpus))}")
#         deepspeed_include_filter = "@".join(filter_parts)
#         logger.info(f"Deepspeed include filter: {deepspeed_include_filter}")
#         # Orchestrator rank must have already created hostfile.txt
#         hostfile_path = str(exp_dir / "hostfile.txt")
#         cmd += [
#             "--num_machines",
#             str(len(world_map.nodes_with_finetuning())),
#             "--machine_rank",
#             str(world_map.my_finetuning_rank()),
#             "--main_process_ip",
#             str(os.environ.get("MASTER_ADDR")),
#             "--main_process_port",
#             str(os.environ.get("MASTER_PORT")),
#             "--deepspeed_hostfile",
#             hostfile_path,
#             "--deepspeed_inclusion_filter",
#             deepspeed_include_filter,
#             "--deepspeed_multinode_launcher",
#             "nossh"
#         ]
#     # get path to this file
#     this_file_path = Path(os.path.dirname(os.path.abspath(__file__)))
#     if cfg.use_deepspeed:
#         # DeepSpeed single-node args
#         cmd += [
#             "--use_deepspeed",
#             "--deepspeed_config_file",
#             str(this_file_path / f"../conf/deepspeed/{cfg.deepspeed_config}.json"),
#         ]
#     # DeepSpeed and non-DeepSpeed args
#     accelerate_config = cfg.accelerate_config
#     if accelerate_config is None:
#         if cfg.use_deepspeed:
#             accelerate_config = "deepspeed"
#         elif cfg.use_fsdp:
#             accelerate_config = "fsdp_mp"
#         else:
#             accelerate_config = "base_mp"
#     cmd += [
#         "--config_file",
#         str(this_file_path / f"../conf/accelerate/{accelerate_config}.yaml"),
#         "--rdzv_backend",
#         "c10d",
#     ]
#     if gpus:
#         gpus_str = str(",".join([str(gpu) for gpu in gpus])) if len(gpus) < world_map.node_size else "all"
#         cmd += [
#             "--gpu-ids",
#             gpus_str,
#         ]
#     cmd += [
#         "--num_processes",
#         str(world_map.total_finetune_gpus),
#         "pipelinerl/entrypoints/run_finetune.py",
#         "--config-dir",
#         f"{exp_dir}/conf",
#         "--config-name",
#         "exp_config",
#         f"output_dir={exp_dir}",
#         f"hydra.run.dir={exp_dir}/finetune",
#         # TODO: figure out why we can't build WorldMap in run_finetune.py
#         # Current workaround: pass the essential information as follows:
#         f"+me.weight_update_group_init_method=tcp://{world_map.master_addr}:{cfg.world.actor_group_port}",
#         f"+me.weight_update_group_world_size={world_map.weight_update_group_size}",
#         f"+me.llm_urls={'+'.join(world_map.get_actor_urls())}",
#     ]
#     if cfg.debug.mode in ["finetune", "open_loop", "finetune+preprocessor"]:
#         cmd.append("finetune.send_weight_updates=False")

#     logger.info(f"Running finetune with command: {' '.join(cmd)}")
#     save_command(exp_dir / "finetune", cmd)
#     env = dict(os.environ)
#     env["DS_ENV_FILE"] = str(exp_dir / ".deepspeed_env")
#     proc = _popen(cmd, env=env)
#     if proc is not None:
#         yield LaunchedProcess(kind="finetune", handle=proc)


def run_preprocess(world_map: WorldMap, preprocessor_idx: int, exp_dir: Path):
    if preprocessor_idx != 0:
        raise NotImplementedError("Can only do 1 preprocessor yet")
    llm_urls = "+".join(world_map.get_preprocessor_urls())
    cmd = [
        sys.executable,
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
    # Launch redis-server. Resolve paths to absolutes because redis-server
    # chdir's to --dir before opening --logfile, which breaks relative paths.
    output_dir = Path(cfg.output_dir).resolve()
    redis_dir = output_dir / "redis"
    os.makedirs(redis_dir, exist_ok=True)
    cmd = [
        "redis-server",
        "--bind",
        "0.0.0.0",
        "--port",
        str(cfg.streams.port),
        "--dir",
        str(output_dir),
        "--protected-mode",
        "no",
        "--save",
        cfg.streams.save,
        "--logfile",
        str(redis_dir / "redis.log"),
        "--loglevel",
        "verbose",
    ]
    logger.info(f"Running redis with command: {' '.join(cmd)}")
    save_command(Path(cfg.output_dir) / "redis", cmd)
    proc = _popen(cmd, env=dict(os.environ))
    if proc is not None:
        yield LaunchedProcess(kind="redis", handle=proc)


def save_command(script_dir: Path, cmd, suffix: str = ""):
    os.makedirs(script_dir, exist_ok=True)
    script_path = script_dir / f"start{suffix}.sh"
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
    # Remove stale pod IP files so the exchange waits for all live ranks.
    pod_ips_dir = Path(exp_dir) / ".pod_ips"
    if pod_ips_dir.exists():
        shutil.rmtree(pod_ips_dir)
        logger.info("Removed stale .pod_ips directory")

    if force_restart:
        if os.path.exists(f"{exp_dir}/finetune"):
            logger.info("Cleaning up finetune directory")
            shutil.rmtree(f"{exp_dir}/finetune")

        # erase all the logs
        log_files = list(exp_dir.glob("**/*.log"))
        for log_file in log_files:
            logger.info(f"Erasing {log_file}")
            with open(log_file, "r"):
                pass


def is_inference_process(proc: LaunchedProcess) -> bool:
    return proc.kind in {"actor_llm", "preprocessor_llm"}


def watch_processes_running(exp_path: Path, processes: List[LaunchedProcess], debug_mode: bool = False, use_fast_llm: bool = False, weight_broadcast: bool = True):
    if not debug_mode:
        trainer_state = TrainerState(exp_path, use_fast_llm=use_fast_llm, weight_broadcast=weight_broadcast)
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


def _get_pod_ip() -> str:
    """Return this pod's primary IP (bypasses Kubernetes Service kube-proxy)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def _exchange_pod_ips(world_map: "WorldMap", exp_dir: Path) -> None:
    """Exchange pod IPs across replicas via the shared NFS mount.

    Kubernetes Services only expose the declared master port; all other ports
    (Redis, vLLM HTTP, TCPStore) are silently dropped for Service ClusterIPs.
    Using pod IPs bypasses kube-proxy and gives full port access.

    After the exchange, all Job.url and Job.hostname fields are updated to use
    pod IPs so every cross-node HTTP/TCP connection bypasses the Service.
    """
    # Save DNS names before overwriting so DeepSpeed hostfile can use them.
    world_map.dns_address_map = dict(world_map.address_map)

    ip_dir = exp_dir / ".pod_ips"
    ip_dir.mkdir(parents=True, exist_ok=True)
    my_ip = _get_pod_ip()
    ip_file = ip_dir / f"rank_{world_map.my_rank}.txt"
    ip_file.write_text(my_ip)
    logger.info(f"Pod IP exchange: rank {world_map.my_rank} pod IP = {my_ip}")

    pod_ips = {}
    for rank in range(world_map.world_size):
        peer_file = ip_dir / f"rank_{rank}.txt"
        waited = 0
        while not peer_file.exists():
            time.sleep(0.5)
            waited += 0.5
            if waited % 10 == 0:
                logger.info(f"Waiting for pod IP from rank {rank} ({waited:.0f}s)...")
        pod_ip = peer_file.read_text().strip()
        pod_ips[rank] = pod_ip
        world_map.address_map[rank] = pod_ip
        logger.info(f"Pod IP exchange: rank {rank} → {pod_ip}")

    world_map.master_addr = pod_ips[0]
    logger.info(f"Updated master_addr to pod IP: {world_map.master_addr}")

    # Update all Job URLs and hostnames to pod IPs so cross-node connections
    # bypass the Kubernetes Service (which only exposes declared ports).
    for node, jobs in world_map.job_map.items():
        pod_ip = pod_ips[node]
        dns_name = world_map.dns_address_map[node]
        for job in jobs:
            job.hostname = pod_ip
            if job.url:
                job.url = job.url.replace(dns_name, pod_ip)
    logger.info("Updated all job URLs to pod IPs for direct pod-to-pod connectivity.")


@hydra.main(
    config_path="../conf/",
    config_name="base",
    version_base="1.3.2",
)
def main(cfg: DictConfig):
    validate_config(cfg)

    exp_dir = Path(cfg.output_dir)
    config_dir = exp_dir / "conf"

    os.makedirs(exp_dir / "launcher", exist_ok=True)
    log_file = exp_dir / "launcher" / f"launcher_{os.environ.get('RANK', 0)}.log"
    setup_logging(log_file)
    world_map = WorldMap(cfg, verbose=True)

    # In multi-node EAI jobs the `dns-<uuid>-<rank>` names are Kubernetes Services
    # that expose only the declared master port.  Connecting to those Service IPs
    # on any other port (Redis, vLLM HTTP, TCPStore) gets SYN-dropped by kube-proxy.
    # Pod IPs bypass kube-proxy and have all ports open, so we exchange pod IPs via
    # a shared NFS file and update address_map before any TCP connections are made.
    if world_map.world_size > 1:
        _exchange_pod_ips(world_map, exp_dir)

    cfg.jobs = [job.model_dump() for job in world_map.get_all_jobs()]

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
        if world_map.world_size > 1:
            # Multi-node: use the pod IP of rank 0 (world_map.master_addr after pod IP
            # exchange).  Pod-to-pod connections are unrestricted on all ports, so rank 0
            # can reach its own Redis via its pod IP, and rank 1 via the cross-node pod IP.
            # Using the pod IP (not localhost or a DNS name) also ensures the saved
            # exp_config.yaml has a reachable address for DeepSpeed workers on node 1.
            cfg.streams.host = world_map.master_addr
        else:
            cfg.streams.host = "localhost"
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
            # Use original DNS names (pod IP exchange may have replaced address_map with IPs).
            dns_map = getattr(world_map, "dns_address_map", world_map.address_map)
            hosts = [dns_map[i] for i in range(world_map.world_size)]
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

    # Pre-create the broadcast rendezvous TCPStore on actor_group_port so that
    # fast-llm (launched via torchrun) can connect as a client.  Torchrun sets
    # TORCHELASTIC_USE_AGENT_STORE=True which makes PyTorch treat ALL ranks as
    # clients in _create_c10d_store; without a pre-existing server the port is
    # never opened and both fast-llm and vLLM hang forever.  Only the master
    # node (my_rank == 0) hosts the server; vLLM workers connect via master_addr.
    broadcast_store = None
    if cfg.use_fast_llm and cfg.weight_broadcast and world_map.my_rank == 0:
        from torch.distributed import TCPStore
        broadcast_store = TCPStore(
            host_name="0.0.0.0",
            port=cfg.world.actor_group_port,
            world_size=world_map.weight_update_group_size,
            is_master=True,
            wait_for_workers=False,
        )
        logger.info(
            f"Broadcast TCPStore server started on "
            f"{world_map.master_addr}:{cfg.world.actor_group_port} "
            f"(world_size={world_map.weight_update_group_size})"
        )

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
    watch_processes_running(exp_dir, processes, bool(cfg.debug.mode), cfg.use_fast_llm, cfg.weight_broadcast)


if __name__ == "__main__":
    main()
