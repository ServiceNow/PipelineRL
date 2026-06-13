from __future__ import annotations

import contextvars
import json
import logging
import os
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ray

from pipelinerl.rollouts import RolloutRequest

_current_rollout_id: contextvars.ContextVar[str] = contextvars.ContextVar("rollout_id", default="")
_current_task_id: contextvars.ContextVar[str] = contextvars.ContextVar("task_id", default="")

_LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


@ray.remote(max_restarts=0, max_task_retries=0, num_cpus=0)
class RayWorkerLogCollector:
    def __init__(self, log_path: str):
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._log_path.open("a", buffering=1, encoding="utf-8")

    def write(self, event: dict[str, Any]) -> None:
        event.setdefault("written_at", datetime.now(timezone.utc).isoformat())
        self._file.write(json.dumps(event, default=str, sort_keys=True) + "\n")

    def close(self) -> None:
        self._file.flush()
        self._file.close()


def start_worker_rollout_log_context(request: RolloutRequest | str) -> tuple[contextvars.Token[str], contextvars.Token[str]]:
    if isinstance(request, RolloutRequest):
        rollout_id = request.request_id
        dataset_item = request.dataset_item
        task_id = str(dataset_item.get("task_id") or dataset_item.get("id") or rollout_id)
    else:
        rollout_id = uuid.uuid4().hex
        task_id = str(request)
    return (_current_rollout_id.set(rollout_id), _current_task_id.set(task_id))


def reset_worker_rollout_log_context(tokens: tuple[contextvars.Token[str], contextvars.Token[str]]) -> None:
    rollout_id_token, task_id_token = tokens
    _current_rollout_id.reset(rollout_id_token)
    _current_task_id.reset(task_id_token)


def _coerce_log_level(level: Any, default: int = logging.ERROR) -> int:
    if level is None:
        return default
    if isinstance(level, int):
        return level
    return _LOG_LEVELS.get(str(level).upper(), default)


class _RayWorkerLogForwardingHandler(logging.Handler):
    def __init__(self, *, worker_name: str, collector: Any, level: int):
        super().__init__(level=level)
        self._worker_name = worker_name
        self._collector = collector

    def emit(self, record: logging.LogRecord) -> None:
        event = {
            "timestamp": record.created,
            "entry_id": uuid.uuid4().hex,
            "rollout_id": _current_rollout_id.get(),
            "task_id": _current_task_id.get(),
            "worker": self._worker_name,
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if record.exc_info:
            event["exception"] = "".join(traceback.format_exception(*record.exc_info))
        if record.stack_info:
            event["stack"] = record.stack_info
        try:
            self._collector.write.remote(event)
        except Exception:
            self.handleError(record)


def configure_noisy_worker_loggers(litellm_log_level: Any) -> None:
    level = _coerce_log_level(litellm_log_level, default=logging.WARNING)
    os.environ["LITELLM_LOGGING_LEVEL"] = logging.getLevelName(level)
    for logger_name in ("litellm", "LiteLLM", "litellm.proxy"):
        logging.getLogger(logger_name).setLevel(level)


def configure_ray_worker_logging(
    *,
    worker_name: str,
    log_collector: Any | None,
    log_level: Any,
    litellm_log_level: Any = logging.WARNING,
) -> None:
    configure_noisy_worker_loggers(litellm_log_level)
    if log_collector is None:
        return

    level = _coerce_log_level(log_level, default=logging.ERROR)
    root_logger = logging.getLogger()
    if root_logger.level > level:
        root_logger.setLevel(level)

    collector_key = repr(log_collector)
    for handler in root_logger.handlers:
        if getattr(handler, "_ray_worker_log_collector_key", None) == collector_key:
            return

    handler = _RayWorkerLogForwardingHandler(worker_name=worker_name, collector=log_collector, level=level)
    handler._ray_worker_log_collector_key = collector_key  # type: ignore[attr-defined]
    root_logger.addHandler(handler)
