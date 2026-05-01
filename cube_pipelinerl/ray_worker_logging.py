from __future__ import annotations

import contextvars
import logging
import os
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import ray


_current_rollout_log_id: contextvars.ContextVar[str] = contextvars.ContextVar("cube_rollout_log_id", default="")
_current_task_id: contextvars.ContextVar[str] = contextvars.ContextVar("cube_task_id", default="")

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
class CubeRayWorkerLogCollector:
    def __init__(self, log_path: str):
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._log_path.open("a", buffering=1, encoding="utf-8")

    def write(self, event: dict[str, Any]) -> None:
        timestamp = datetime.fromtimestamp(float(event["timestamp"])).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        message = str(event["message"]).replace("\n", " | ")
        context_parts = [f"actor={event['actor']}", f"entry_id={event['entry_id']}"]
        if event.get("task_id"):
            context_parts.append(f"task_id={event['task_id']}")
        if event.get("rollout_id"):
            context_parts.append(f"rollout_id={event['rollout_id']}")
        line = (
            f"[ray_worker]: {timestamp} - {event['logger']} - {event['level']} - "
            f"{message} ({' '.join(context_parts)})"
        )
        exception = event.get("exception")
        if exception:
            line = f"{line}\n{exception.rstrip()}"
        self._file.write(line + "\n")

    def close(self) -> None:
        self._file.flush()
        self._file.close()


def start_worker_rollout_log_context(task_id: str) -> tuple[contextvars.Token[str], contextvars.Token[str]]:
    return (
        _current_rollout_log_id.set(uuid.uuid4().hex),
        _current_task_id.set(str(task_id)),
    )


def reset_worker_rollout_log_context(tokens: tuple[contextvars.Token[str], contextvars.Token[str]]) -> None:
    rollout_log_id_token, task_id_token = tokens
    _current_rollout_log_id.reset(rollout_log_id_token)
    _current_task_id.reset(task_id_token)


def _coerce_log_level(level: Any, default: int = logging.ERROR) -> int:
    if level is None:
        return default
    if isinstance(level, int):
        return level
    return _LOG_LEVELS.get(str(level).upper(), default)


class _CubeRayWorkerLogForwardingHandler(logging.Handler):
    def __init__(self, *, actor_name: str, collector: Any, level: int):
        super().__init__(level=level)
        self._actor_name = actor_name
        self._collector = collector

    def emit(self, record: logging.LogRecord) -> None:
        event = {
            "timestamp": record.created,
            "entry_id": uuid.uuid4().hex,
            "rollout_id": _current_rollout_log_id.get(),
            "task_id": _current_task_id.get(),
            "actor": self._actor_name,
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
    actor_name: str,
    log_collector: Any | None,
    log_level: Any,
    litellm_log_level: Any,
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
        if getattr(handler, "_cube_ray_worker_log_collector_key", None) == collector_key:
            return

    handler = _CubeRayWorkerLogForwardingHandler(
        actor_name=actor_name,
        collector=log_collector,
        level=level,
    )
    handler._cube_ray_worker_log_collector_key = collector_key  # type: ignore[attr-defined]
    root_logger.addHandler(handler)
