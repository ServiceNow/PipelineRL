import logging
import math
import os
import shutil
import subprocess
import sys
import time
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
os.environ["PYTHONPATH"] = f"{os.getcwd()}"
os.environ["NCCL_CUMEM_ENABLE"] = "0"
os.environ["TORCH_DISABLE_SHARE_RDZV_TCP_STORE"] = "1"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
os.environ["VLLM_USE_V1"] = "0"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

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


def run_ref_llm(cfg: DictConfig, preprocessor_llm_idx: int, local_idx: int, gpus: list[int], exp_dir: Path):
    kwargs = cfg.vllm_config.vllm_kwargs
    if kwargs["num-scheduler-steps"] > 1:
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
        str(preprocessor_llm_idx),
    ]

    # Add vLLM kwargs as separate arguments
    for k, v in kwargs.items():
        cmd.append(f"--{k}")
        if v not in [None, ""]:
            cmd.append(str(v))

    gpu_str = ",".join([str(gpu) for gpu in gpus])
    logger.info(f"Running reference LLM with command: {' '.join(cmd)} with gpus: {gpu_str}")
    log_file_path = os.path.join(log_dir, "stdout.log")
    err_file_path = os.path.join(log_dir, "stderr.log")
    with open(log_file_path, "a") as log_file, open(err_file_path, "a") as err_file:
        yield _popen(
            cmd,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_str},
            stdout=log_file,
            stderr=err_file,
        )


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
    cmd = [
        "python",
        "-m",
        "pipelinerl.entrypoints.llm",
        "--model",
        str(actor_model_path),
        "--host",
        "0.0.0.0",
        "--port",
        str(8080 + local_idx),
        "--seed",
        str(actor_llm_idx),
        "--exp-root-dir",
        str(exp_dir),
        "--actor-llm-idx",
        str(actor_llm_idx),
        "--weight-update-group-init-method",
        f"tcp://{world_map.master_addr}:{cfg.world.actor_group_port}",
        "--weight-update-group-world-size",
        str(world_map.weight_update_group_size),
    ]

    # Add vLLM kwargs as separate arguments
    if cfg.vllm_config.vllm_kwargs:
        for k, v in cfg.vllm_config.vllm_kwargs.items():
            cmd.append(f"--{k}")
            if v not in [None, ""]:
                cmd.append(str(v))

    if cfg.debug.mode in ["actor", "open_loop"]:
        cmd.append("--disable-weight-updates")

    gpu_str = ",".join([str(gpu) for gpu in gpus])
    logger.info(f"Running actor_llm with command: {' '.join(cmd)} on gpus: {gpu_str}")
    save_command(log_dir, cmd)
    log_file_path = os.path.join(log_dir, "stdout.log")
    err_file_path = os.path.join(log_dir, "stderr.log")
    with open(log_file_path, "a") as log_file, open(err_file_path, "a") as err_file:
        yield _popen(
            cmd,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_str},
            stdout=log_file,
            stderr=err_file,
        )


def run_actor(world_map: WorldMap, actor_idx: int, exp_dir: Path):
    if actor_idx != 0:
        raise NotImplementedError("Can only do 1 actor yet")
    llm_urls = "+".join(world_map.get_actor_urls())
    cmd = [
        "python",
        "-m",
        "pipelinerl.entrypoints.actor",
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
    yield _popen(
        cmd,
        env=dict(os.environ),
    )


def run_environment(cfg: DictConfig, job: Job):
    # run in a subprocess like in the rest of the code
    run_dir = Path(cfg.output_dir) / f"environment_{job.replica_idx}"
    cmd = [
        "python",
        "-m",
        "pipelinerl.entrypoints.environment",
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
        yield _popen(
            cmd,
            env=dict(os.environ),
            stdout=log_file,
            stderr=err_file,
        )


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
        "pipelinerl/entrypoints/finetune.py",
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
    if cfg.debug.mode in ["finetune", "open_loop"]:
        cmd.append("finetune.send_weight_updates=False")

    logger.info(f"Running finetune with command: {' '.join(cmd)}")
    save_command(exp_dir / "finetune", cmd)
    env = dict(os.environ)
    env["DS_ENV_FILE"] = str(exp_dir / ".deepspeed_env")
    yield _popen(cmd, env=env)


def run_preprocess(world_map: WorldMap, preprocessor_idx: int, exp_dir: Path):
    if preprocessor_idx != 0:
        raise NotImplementedError("Can only do 1 preprocessor yet")
    llm_urls = "+".join(world_map.get_preprocessor_urls())
    cmd = [
        "python",
        "-m",
        "pipelinerl.entrypoints.preprocess",
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
    yield _popen(
        cmd,
        env=dict(os.environ),
    )


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
    yield _popen(cmd, env=dict(os.environ))


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

        # erase all the logs
        log_files = list(exp_dir.glob("**/*.log"))
        for log_file in log_files:
            logger.info(f"Erasing {log_file}")
            with open(log_file, "r"):
                pass


def watch_processes_running(exp_path: Path, processes: List[subprocess.Popen], debug_mode: bool = False):
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
            logger.info(f"Terminating {proc.args}")
            terminate_with_children(proc.pid)

    logger.info("I have launched everyone, waiting for them to finish...")

    # last_trainer_version = -1
    # last_time_new_version = time.time()

    try:
        # Wait for all processes to complete
        # if just one dies, stop all
        while True:
            for proc in processes:
                if (return_code := proc.poll()) is not None:
                    # print which process terminate and with what code
                    logger.error(f"Process {proc.args} terminated with code {proc.returncode}")
                    gently_stop_all_processes()
                    sys.exit(1)
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
            processes.extend(run_actor_llm(cfg, world_map, job.replica_idx, job.local_idx, job.gpus, exp_dir))
        elif job.kind == "preprocessor":
            processes.extend(run_preprocess(world_map, job.replica_idx, exp_dir))
        elif job.kind == "preprocessor_llm":
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
    cfg.jobs = [job.model_dump() for job in world_map.get_all_jobs()]

    group = str(exp_dir)
    root = cfg.finetune.wandb_workspace_root
    if root:
        if not group.startswith(root + "/"):
            raise ValueError(f"run_dir {exp_dir} does not start with root {root}")
        cfg.finetune.wandb_group = group[len(root) + 1 :]
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
