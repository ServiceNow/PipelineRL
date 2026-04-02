from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import psutil
import torch
from omegaconf import DictConfig

from .types import PipelineBatchEncoding, TrainingMetrics


def _read_int_file(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, PermissionError, ValueError, OSError):
        return None


def _parse_stat_kv_file(path: Path) -> dict[str, int]:
    stats: dict[str, int] = {}
    try:
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            key, value = parts
            stats[key] = int(value)
    except (FileNotFoundError, PermissionError, ValueError, OSError):
        return {}
    return stats


def _parse_smaps_rollup(path: Path) -> dict[str, int]:
    stats: dict[str, int] = {}
    try:
        for line in path.read_text().splitlines():
            if ":" not in line:
                continue
            key, raw_value = line.split(":", maxsplit=1)
            parts = raw_value.strip().split()
            if not parts:
                continue
            value = int(parts[0])
            if len(parts) > 1 and parts[1].lower() == "kb":
                value *= 1024
            stat_name = key.strip().lower().replace(" ", "_")
            stats[f"smaps_{stat_name}_bytes"] = value
    except (FileNotFoundError, PermissionError, ValueError, OSError):
        return {}
    return stats


def _read_cgroup_stats() -> dict[str, int | str]:
    root_v2 = Path("/sys/fs/cgroup")
    root_v1 = root_v2 / "memory"

    if (root_v2 / "memory.current").exists():
        memory_stat = _parse_stat_kv_file(root_v2 / "memory.stat")
        stats: dict[str, int | str] = {"cgroup_version": "v2"}
        for output_key, stat_key in {
            "cgroup_memory_current_bytes": "memory.current",
            "cgroup_memory_peak_bytes": "memory.peak",
            "cgroup_memory_max_bytes": "memory.max",
        }.items():
            value = _read_int_file(root_v2 / stat_key)
            if value is not None:
                stats[output_key] = value
        for output_key, stat_key in {
            "cgroup_memory_anon_bytes": "anon",
            "cgroup_memory_file_bytes": "file",
            "cgroup_memory_shmem_bytes": "shmem",
            "cgroup_memory_slab_bytes": "slab",
            "cgroup_memory_kernel_stack_bytes": "kernel_stack",
        }.items():
            if stat_key in memory_stat:
                stats[output_key] = memory_stat[stat_key]
        return stats

    if (root_v1 / "memory.usage_in_bytes").exists():
        memory_stat = _parse_stat_kv_file(root_v1 / "memory.stat")
        stats = {"cgroup_version": "v1"}
        for output_key, stat_key in {
            "cgroup_memory_current_bytes": "memory.usage_in_bytes",
            "cgroup_memory_max_bytes": "memory.limit_in_bytes",
        }.items():
            value = _read_int_file(root_v1 / stat_key)
            if value is not None:
                stats[output_key] = value
        for output_key, stat_key in {
            "cgroup_memory_cache_bytes": "cache",
            "cgroup_memory_rss_bytes": "rss",
            "cgroup_memory_shmem_bytes": "shmem",
            "cgroup_memory_swap_bytes": "swap",
            "cgroup_memory_mapped_file_bytes": "mapped_file",
        }.items():
            if stat_key in memory_stat:
                stats[output_key] = memory_stat[stat_key]
        return stats

    return {}


def _safe_queue_size(queue_size: int | None) -> int | None:
    if queue_size is None:
        return None
    try:
        return int(queue_size)
    except (TypeError, ValueError):
        return None


def _batch_metadata(batch: PipelineBatchEncoding | None) -> dict[str, Any]:
    if batch is None:
        return {}

    metadata: dict[str, Any] = {
        "batch_model_version": int(batch.model_version),
        "batch_sentinel": bool(batch.sentinel),
        "batch_is_packed": bool(batch.is_packed),
        "batch_padding": int(batch.padding),
        "batch_input_ids_numel": int(batch.input_ids.numel()),
    }

    if batch.position_ids is not None and batch.seq_boundaries is not None:
        sequence_count = len(batch.seq_boundaries) - 1
        if batch.padding:
            sequence_count -= 1
        metadata["batch_sequence_count"] = max(sequence_count, 0)
    else:
        metadata["batch_sequence_count"] = int(batch.input_ids.shape[0])

    try:
        metadata["batch_token_count"] = int(batch.attention_mask.sum().item())
    except RuntimeError:
        pass

    return metadata


def create_memory_debugger(cfg: DictConfig | Mapping[str, Any] | None, output_dir: Path, rank: int) -> "MemoryDebugger | None":
    if cfg is None:
        return None

    memory_cfg = cfg.get("memory_debug") if hasattr(cfg, "get") else None
    if not memory_cfg:
        return None

    enabled = bool(memory_cfg.get("enabled", False))
    if not enabled:
        return None

    log_every_n_steps = int(memory_cfg.get("log_every_n_steps", 1))
    if log_every_n_steps <= 0:
        raise ValueError(f"memory_debug.log_every_n_steps must be positive, got {log_every_n_steps}")

    phase_granularity = str(memory_cfg.get("phase_granularity", "step"))
    if phase_granularity not in {"step", "micro_batch"}:
        raise ValueError(
            f"memory_debug.phase_granularity must be 'step' or 'micro_batch', got {phase_granularity!r}"
        )

    return MemoryDebugger(
        log_path=output_dir / "log" / f"memory_rank{rank}.jsonl",
        rank=rank,
        log_every_n_steps=log_every_n_steps,
        capture_smaps_rollup=bool(memory_cfg.get("capture_smaps_rollup", False)),
        capture_cgroup_stats=bool(memory_cfg.get("capture_cgroup_stats", True)),
        phase_granularity=phase_granularity,
    )


class MemoryDebugger:
    def __init__(
        self,
        log_path: Path,
        rank: int,
        log_every_n_steps: int,
        capture_smaps_rollup: bool,
        capture_cgroup_stats: bool,
        phase_granularity: str,
    ):
        self.log_path = log_path
        self.rank = rank
        self.log_every_n_steps = log_every_n_steps
        self.capture_smaps_rollup = capture_smaps_rollup
        self.capture_cgroup_stats = capture_cgroup_stats
        self.phase_granularity = phase_granularity
        self.pid = os.getpid()
        self.process = psutil.Process(self.pid)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = self.log_path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._closed = False

    def should_log_step(self, logical_step: int | None) -> bool:
        if logical_step is None:
            return True
        return logical_step == 1 or logical_step % self.log_every_n_steps == 0

    def should_log_micro_batch(self, logical_step: int | None) -> bool:
        return self.phase_granularity == "micro_batch" and self.should_log_step(logical_step)

    def should_log_loader_event(self) -> bool:
        return self.phase_granularity == "micro_batch"

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._writer.close()

    def log_snapshot(
        self,
        phase: str,
        *,
        logical_step: int | None = None,
        training_metrics: TrainingMetrics | None = None,
        batch: PipelineBatchEncoding | None = None,
        queue_size: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ):
        if self._closed:
            return

        record: dict[str, Any] = {
            "timestamp": time.time(),
            "phase": phase,
            "rank": self.rank,
            "pid": self.pid,
            "logical_step": logical_step,
            "queue_size": _safe_queue_size(queue_size),
        }

        if training_metrics is not None:
            record.update(
                {
                    "completed_steps": int(training_metrics.completed_steps),
                    "samples": int(training_metrics.samples),
                    "tokens": int(training_metrics.tokens),
                    "passes": int(training_metrics.passes),
                    "last_broadcasted_version": int(training_metrics.last_broadcasted_version),
                }
            )

        record.update(self._process_memory_stats())
        record.update(self._cuda_memory_stats())
        record.update(_batch_metadata(batch))

        if self.capture_smaps_rollup:
            record.update(_parse_smaps_rollup(Path("/proc/self/smaps_rollup")))
        if self.capture_cgroup_stats:
            record.update(_read_cgroup_stats())
        if extra:
            record.update(extra)

        with self._lock:
            if self._closed:
                return
            self._writer.write(json.dumps(record, sort_keys=True) + "\n")
            self._writer.flush()

    def _process_memory_stats(self) -> dict[str, Any]:
        try:
            memory_info = self.process.memory_info()
            memory_full_info = self.process.memory_full_info()
        except psutil.Error:
            return {}

        stats: dict[str, Any] = {
            "rss_bytes": int(memory_info.rss),
            "vms_bytes": int(memory_info.vms),
            "shared_bytes": int(getattr(memory_full_info, "shared", getattr(memory_info, "shared", 0))),
            "uss_bytes": int(getattr(memory_full_info, "uss", 0)),
            "pss_bytes": int(getattr(memory_full_info, "pss", 0)),
            "swap_bytes": int(getattr(memory_full_info, "swap", 0)),
        }

        try:
            stats["num_threads"] = int(self.process.num_threads())
        except psutil.Error:
            pass

        if hasattr(self.process, "num_fds"):
            try:
                stats["num_fds"] = int(self.process.num_fds())
            except psutil.Error:
                pass

        return stats

    def _cuda_memory_stats(self) -> dict[str, int]:
        if not torch.cuda.is_available():
            return {}

        try:
            device = torch.cuda.current_device()
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            return {
                "cuda_device_index": int(device),
                "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "cuda_mem_free_bytes": int(free_bytes),
                "cuda_mem_total_bytes": int(total_bytes),
            }
        except RuntimeError:
            return {}
