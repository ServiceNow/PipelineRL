"""Centralized logging for all agent LLM calls and tool executions."""

import threading
from pathlib import Path
from typing import Any

from .config import RunConfig

_lock = threading.Lock()
_counter = 0


def _next_filename(prefix: str) -> str:
    global _counter
    with _lock:
        _counter += 1
        return f"{_counter:04d}_{prefix}.txt"


def _ensure_prompts_dir(run_config: RunConfig) -> Path | None:
    if not run_config.log_prompts or not run_config.run_dir:
        return None
    d = Path(run_config.run_dir) / "prompts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_llm_call(run_config: RunConfig, prompt: str, response: str | Any, name: str = "llm") -> None:
    """Log an LLM call (prompt + response) to run_dir/prompts/."""
    d = _ensure_prompts_dir(run_config)
    if not d:
        return
    resp_str = str(response) if not isinstance(response, str) else response
    content = f"=== PROMPT ===\n{prompt}\n\n=== RESPONSE ===\n{resp_str}"
    path = d / _next_filename(name)
    path.write_text(content, encoding="utf-8")


def log_tool_call(
    run_config: RunConfig,
    tool_name: str,
    query: str,
    status: str = "",
    extra: str = "",
) -> None:
    """Log a tool execution (no LLM)."""
    d = _ensure_prompts_dir(run_config)
    if not d:
        return
    lines = [f"TOOL: {tool_name}", f"QUERY: {query}", f"STATUS: {status}"]
    if extra:
        lines.append(f"ERROR: {extra}")
    content = "\n".join(lines)
    safe_name = "".join(c if c.isalnum() else "_" for c in tool_name)[:30]
    path = d / _next_filename(f"tool_{safe_name}")
    path.write_text(content, encoding="utf-8")
