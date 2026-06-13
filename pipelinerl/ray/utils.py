from __future__ import annotations

from pathlib import Path
from typing import Any

import psutil
import ray


def is_expected_ray_shutdown(exc: BaseException) -> bool:
    if not isinstance(exc, (ray.exceptions.ActorDiedError, ray.exceptions.RayActorError)):
        return False
    message = str(exc)
    return "terminated expectedly" in message and "received SIGTERM" in message


def close_ray_actor_best_effort(actor: Any, logger: Any, actor_name: str, *, timeout: float = 2.0) -> None:
    try:
        ray.get(actor.close.remote(), timeout=timeout)
        logger.info("Closed %s: %s", actor_name, actor)
    except Exception as exc:
        if is_expected_ray_shutdown(exc):
            logger.info("%s already gone during shutdown: %s", actor_name, actor)
        else:
            logger.exception("Failed to close %s: %s", actor_name, actor)


def kill_ray_actor_best_effort(actor: Any, logger: Any, actor_name: str) -> None:
    try:
        ray.kill(actor, no_restart=True)
        logger.info("Killed %s: %s", actor_name, actor)
    except Exception:
        logger.exception("Failed to kill %s: %s", actor_name, actor)


def get_rollout_resource_guard(cfg: Any) -> Any:
    actor_cfg = cfg.get("actor", None)
    if actor_cfg is not None and getattr(actor_cfg, "ray_resource_guard", None) is not None:
        return getattr(actor_cfg, "ray_resource_guard")
    cube_params = cfg.get("cube_params", None)
    if cube_params is not None:
        return getattr(cube_params, "resource_guard", None)
    return None


def read_int_file(path: str) -> int | None:
    try:
        value = Path(path).read_text().strip()
    except OSError:
        return None
    if not value or value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def effective_memory_bytes() -> tuple[int, int]:
    vm = psutil.virtual_memory()
    total_bytes = int(vm.total)
    available_bytes = int(vm.available)

    cgroup_limit = read_int_file("/sys/fs/cgroup/memory.max")
    cgroup_current = read_int_file("/sys/fs/cgroup/memory.current")
    if cgroup_limit is None:
        cgroup_limit = read_int_file("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        cgroup_current = read_int_file("/sys/fs/cgroup/memory/memory.usage_in_bytes")

    # Some runtimes expose a huge sentinel instead of a real limit.
    if cgroup_limit is not None and cgroup_limit > 0 and cgroup_limit < total_bytes:
        total_bytes = cgroup_limit
        if cgroup_current is not None and cgroup_current >= 0:
            available_bytes = max(0, cgroup_limit - cgroup_current)

    return total_bytes, available_bytes


def effective_logical_cpus() -> float:
    logical_cpus = float(psutil.cpu_count(logical=True) or 1)

    cpu_max = None
    try:
        cpu_max = Path("/sys/fs/cgroup/cpu.max").read_text().strip().split()
    except OSError:
        pass
    if cpu_max and len(cpu_max) == 2 and cpu_max[0] != "max":
        try:
            quota = float(cpu_max[0])
            period = float(cpu_max[1])
            if quota > 0 and period > 0:
                logical_cpus = min(logical_cpus, quota / period)
        except ValueError:
            pass
    else:
        quota = read_int_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period = read_int_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota is None:
            quota = read_int_file("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us")
            period = read_int_file("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us")
        if quota is not None and period is not None and quota > 0 and period > 0:
            logical_cpus = min(logical_cpus, quota / period)

    return max(1.0, logical_cpus)


def check_local_ray_worker_resources(
    cfg: Any,
    *,
    instances: int,
    worker_num_cpus: float,
    required_ray_cpus: int,
) -> None:
    guard = get_rollout_resource_guard(cfg)
    if guard is None:
        return

    logical_cpus = effective_logical_cpus()
    if float(required_ray_cpus) > logical_cpus:
        raise ValueError(
            "Ray rollout worker configuration requires more CPU slots than this node appears to have: "
            f"required_ray_cpus={required_ray_cpus}, effective_logical_cpus={logical_cpus:.2f}, "
            f"instances={instances}, worker_num_cpus={worker_num_cpus}. "
            "Reduce actor.ray_workers or actor.ray_worker_num_cpus, or run on a node with more CPUs."
        )

    actor_memory_gb = float(getattr(guard, "actor_memory_gb", 1.25))
    memory_overhead_gb = float(getattr(guard, "memory_overhead_gb", 8.0))
    memory_usage_threshold = float(getattr(guard, "memory_usage_threshold", 0.90))
    if actor_memory_gb <= 0:
        raise ValueError("rollout resource guard actor_memory_gb must be positive")
    if memory_overhead_gb < 0:
        raise ValueError("rollout resource guard memory_overhead_gb must be non-negative")
    if not (0 < memory_usage_threshold <= 1.0):
        raise ValueError("rollout resource guard memory_usage_threshold must be in (0, 1]")

    total_memory_bytes, available_memory_bytes = effective_memory_bytes()
    total_memory_gb = total_memory_bytes / 2**30
    available_memory_gb = available_memory_bytes / 2**30
    estimated_memory_gb = instances * actor_memory_gb + memory_overhead_gb
    allowed_total_gb = total_memory_gb * memory_usage_threshold

    if estimated_memory_gb > allowed_total_gb:
        raise ValueError(
            "Ray rollout worker configuration is likely to exceed safe node memory. "
            f"Estimated requirement is {estimated_memory_gb:.2f} GiB "
            f"({instances} workers * {actor_memory_gb:.2f} GiB + {memory_overhead_gb:.2f} GiB overhead), "
            f"but this node has {total_memory_gb:.2f} GiB total and the configured threshold allows "
            f"{allowed_total_gb:.2f} GiB. Reduce actor.ray_workers or actor.eval.workers, reduce the number of actor vLLMs, "
            "or increase the resource guard memory_usage_threshold only if you know the estimate is conservative."
        )

    if estimated_memory_gb > available_memory_gb:
        raise ValueError(
            "Ray rollout worker configuration is likely to exceed currently available node memory. "
            f"Estimated requirement is {estimated_memory_gb:.2f} GiB, but only "
            f"{available_memory_gb:.2f} GiB is currently available. "
            "Free memory, reduce actor.ray_workers or actor.eval.workers, or lower the resource guard actor_memory_gb "
            "if measured actors are smaller on this workload."
        )
