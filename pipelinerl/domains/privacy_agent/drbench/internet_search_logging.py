"""Web-search logging helpers for privacy_agent."""


import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .config import RunConfig


def _get_log_path(run_config: RunConfig) -> Path | None:
    if not run_config.log_searches:
        return None

    run_dir = run_config.run_dir
    if not run_dir:
        raise ValueError(
            "RunConfig.run_dir is required when search logging is enabled. "
            "Set a run directory or disable log_searches."
        )
    return Path(run_dir) / "internet_searches.jsonl"


def log_internet_search(
    run_config: RunConfig,
    tool: str,
    query: str,
    params: Dict[str, Any],
    result: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    log_path = _get_log_path(run_config)
    if log_path is None:
        return

    record: Dict[str, Any] = {
        "tool": tool,
        "query": query,
        "params": {key: value for key, value in params.items() if key != "query" and value is not None},
        "success": result.get("success"),
        "data_retrieved": result.get("data_retrieved"),
        "results_count": result.get("results_count"),
        "error": result.get("error"),
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if result.get("results"):
        result_urls = []
        for item in result["results"][:10]:
            if isinstance(item, dict) and item.get("url"):
                result_urls.append(item["url"])
        if result_urls:
            record["result_urls"] = result_urls

    if extra:
        record.update(extra)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
