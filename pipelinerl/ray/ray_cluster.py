import logging
import math
import os
import subprocess
import time
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from pipelinerl.streams import SingleStreamSpec, read_stream, write_to_streams
from pipelinerl.world import WorldMap


logger = logging.getLogger(__name__)


def uses_ray_actor(cfg: DictConfig) -> bool:
    return str(getattr(cfg.actor, "launcher", "asyncio")).strip().lower() == "ray"


def should_launch_ray_cluster(cfg: DictConfig, world_map: WorldMap) -> bool:
    return bool(getattr(cfg.actor, "launch_ray_cluster", False)) and uses_ray_actor(cfg) and world_map.world_size > 1


def configure_actor_ray_address(cfg: DictConfig, world_map: WorldMap):
    if not should_launch_ray_cluster(cfg, world_map):
        return
    head_port = int(getattr(cfg.actor, "ray_head_port", 6379))
    ray_address = f"{world_map.address_map[0]}:{head_port}"
    OmegaConf.update(cfg, "actor.ray_address", ray_address, merge=True)
    logger.info("Configured actor.ray_address=%s for launcher-managed Ray cluster", ray_address)


def _ray_cpus_per_node(cfg: DictConfig, world_map: WorldMap) -> int:
    configured = getattr(cfg.actor, "ray_num_cpus_per_node", None)
    if configured is not None:
        return int(configured)

    cube_workers = int(getattr(cfg.actor, "cube_workers", 1))
    worker_num_cpus = float(getattr(cfg.actor, "cube_workers_num_cpus", 1.0))
    total_required = max(1, int(math.ceil(cube_workers * worker_num_cpus)))
    extra_cpus = int(getattr(cfg.actor, "ray_extra_cpus_per_node", 1))
    return max(1, int(math.ceil(total_required / world_map.world_size)) + extra_cpus)


def _ray_port_args(cfg: DictConfig) -> list[str]:
    worker_port_start = int(getattr(cfg.actor, "ray_worker_port_start", 20000))
    configured_port_count = getattr(cfg.actor, "ray_worker_port_count", None)
    if configured_port_count is None:
        cube_workers = int(getattr(cfg.actor, "cube_workers", 1))
        worker_port_count = max(128, cube_workers + 32)
    else:
        worker_port_count = int(configured_port_count)
    if worker_port_count < 1:
        raise ValueError("actor.ray_worker_port_count must be >= 1")

    return [
        "--node-manager-port",
        str(int(getattr(cfg.actor, "ray_node_manager_port", 6380))),
        "--object-manager-port",
        str(int(getattr(cfg.actor, "ray_object_manager_port", 6381))),
        "--min-worker-port",
        str(worker_port_start),
        "--max-worker-port",
        str(worker_port_start + worker_port_count - 1),
    ]


def _save_command(script_dir: Path, cmd: list[str]):
    os.makedirs(script_dir, exist_ok=True)
    script_path = script_dir / "start.sh"
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        quoted_cmd = [f"'{arg}'" if " " in arg or "$" in arg else arg for arg in cmd]
        f.write(" ".join(quoted_cmd) + "\n")
    os.chmod(script_path, 0o755)


def _popen(cmd: list[str], env: dict, stdout, stderr) -> subprocess.Popen | None:
    if os.environ.get("DRY_RUN", "0") == "1":
        return None
    return subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr)


def _start_ray_node(cfg: DictConfig, world_map: WorldMap, exp_dir: Path) -> subprocess.Popen | None:
    rank = world_map.my_rank
    head_port = int(getattr(cfg.actor, "ray_head_port", 6379))
    cpus = _ray_cpus_per_node(cfg, world_map)
    node_host = world_map.address_map[rank]
    head_address = f"{world_map.address_map[0]}:{head_port}"

    if rank == 0:
        cmd = [
            "ray",
            "start",
            "--head",
            "--node-ip-address",
            node_host,
            "--port",
            str(head_port),
            "--num-cpus",
            str(cpus),
            "--num-gpus",
            "0",
            "--include-dashboard=false",
            "--block",
        ]
    else:
        cmd = [
            "ray",
            "start",
            "--address",
            head_address,
            "--node-ip-address",
            node_host,
            "--num-cpus",
            str(cpus),
            "--num-gpus",
            "0",
            "--block",
        ]
    cmd.extend(_ray_port_args(cfg))

    rank_log_dir = exp_dir / "ray" / f"rank_{rank}"
    os.makedirs(rank_log_dir, exist_ok=True)
    _save_command(rank_log_dir, cmd)

    log_file_path = rank_log_dir / "stdout.log"
    err_file_path = rank_log_dir / "stderr.log"
    logger.info("Starting launcher-managed Ray node with command: %s", " ".join(cmd))
    env = {**os.environ, "RAY_USAGE_STATS_ENABLED": "0"}
    with open(log_file_path, "a") as log_file, open(err_file_path, "a") as err_file:
        proc = _popen(cmd, env=env, stdout=log_file, stderr=err_file)

    if proc is None:
        return None

    time.sleep(2.0)
    if proc.poll() is not None:
        raise RuntimeError(f"Ray start process exited early with code {proc.returncode}; see {err_file_path}")
    return proc


def _write_ready(exp_dir: Path, rank: int):
    ready_stream = SingleStreamSpec(exp_path=exp_dir, topic=f"ray_node_{rank}_ready")
    with write_to_streams(ready_stream) as stream:
        stream.write({"ray_node_ready": str(rank)})


def _wait_ready(exp_dir: Path, rank: int):
    ready_stream = SingleStreamSpec(exp_path=exp_dir, topic=f"ray_node_{rank}_ready")
    with read_stream(ready_stream) as stream:
        msg = next(stream.read())
    expected = {"ray_node_ready": str(rank)}
    if msg != expected:
        raise ValueError(f"Expected {expected}, got {msg}")


def launch_ray_cluster_node(cfg: DictConfig, world_map: WorldMap, exp_dir: Path) -> list[subprocess.Popen]:
    if not should_launch_ray_cluster(cfg, world_map):
        return []

    rank = world_map.my_rank
    if rank == 0:
        proc = _start_ray_node(cfg, world_map, exp_dir)
        _write_ready(exp_dir, 0)
        for worker_rank in range(1, world_map.world_size):
            _wait_ready(exp_dir, worker_rank)
        logger.info("All launcher-managed Ray nodes reported ready")
    else:
        _wait_ready(exp_dir, 0)
        proc = _start_ray_node(cfg, world_map, exp_dir)
        _write_ready(exp_dir, rank)

    return [proc] if proc is not None else []