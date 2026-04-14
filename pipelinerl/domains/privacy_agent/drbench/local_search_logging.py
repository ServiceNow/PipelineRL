"""Local-search logging helpers for privacy_agent."""


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
    return Path(run_dir) / "local_searches.jsonl"


def _to_list(value: Any) -> Optional[list]:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def log_local_document_search(
    run_config: RunConfig,
    query: str,
    params: Dict[str, Any],
    result: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    log_path = _get_log_path(run_config)
    if log_path is None:
        return

    local_docs = (result.get("results") or {}).get("local_documents") or []
    record: Dict[str, Any] = {
        "tool": "local_document_search",
        "query": query,
        "query_raw": params.get("raw_query"),
        "top_k": params.get("top_k"),
        "file_type_filter": _to_list(params.get("file_type_filter")),
        "folder_filter": _to_list(params.get("folder_filter")),
        "success": result.get("success"),
        "data_retrieved": result.get("data_retrieved"),
        "results_count": result.get("results_count"),
        "files_searched": result.get("files_searched"),
        "file_types_found": _to_list(result.get("file_types_found")),
        "folders_searched": _to_list(result.get("folders_searched")),
        "synthesis_length": len(str(result.get("synthesis", ""))) if result.get("synthesis") else 0,
        "error": result.get("error"),
        "result_files": [doc.get("file_path") for doc in local_docs if doc.get("file_path")][:10],
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if extra:
        record.update(extra)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
