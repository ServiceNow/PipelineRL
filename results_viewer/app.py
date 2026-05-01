from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


TOPICS = ("actor", "actor_test", "stats", "stats_test")
ACTOR_TOPICS = {"actor", "actor_test"}
STATS_TOPICS = {"stats", "stats_test"}
MAX_DETAIL_TEXT = 200_000


def create_app(results_root: Path) -> FastAPI:
    app = FastAPI(title="PipelineRL Results Viewer")
    static_root = Path(__file__).parent / "static"

    @lru_cache(maxsize=1)
    def root() -> Path:
        if not results_root.exists():
            raise RuntimeError(f"Results directory does not exist: {results_root}")
        return results_root

    def experiment_path(name: str) -> Path:
        path = (root() / name).resolve()
        if root() not in path.parents and path != root():
            raise HTTPException(status_code=400, detail="Invalid experiment name")
        if not path.exists() or not path.is_dir():
            raise HTTPException(status_code=404, detail="Experiment not found")
        return path

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_root / "index.html")

    @app.get("/api/experiments")
    def experiments() -> dict[str, Any]:
        items = []
        for exp in sorted(p for p in root().iterdir() if p.is_dir()):
            streams = exp / "streams"
            topics = {
                topic: topic_summary(streams / topic)
                for topic in TOPICS
            }
            config = load_yaml(exp / "conf" / "exp_config.yaml") or load_yaml(exp / ".hydra" / "config.yaml")
            items.append(
                {
                    "name": exp.name,
                    "path": str(exp),
                    "topics": topics,
                    "config": {
                        "has_config": bool(config),
                        "keys": sorted(config.keys())[:20] if isinstance(config, dict) else [],
                    },
                }
            )
        return {"results_root": str(root()), "experiments": items}

    @app.get("/api/experiments/{name}/summary")
    def summary(name: str) -> dict[str, Any]:
        exp = experiment_path(name)
        out = {"name": name, "path": str(exp), "topics": {}}
        for topic in TOPICS:
            files = stream_files(exp / "streams" / topic)
            if topic in ACTOR_TOPICS:
                stats_files = stream_files(exp / "streams" / stats_topic_for_actor(topic))
                out["topics"][topic] = summarize_actor_topic_from_stats(files, stats_files)
            else:
                out["topics"][topic] = summarize_stats_topic(files)
        return out

    @app.get("/api/experiments/{name}/stats/{topic}")
    def stats(name: str, topic: str, max_points: int = Query(1000, ge=50, le=10000)) -> dict[str, Any]:
        if topic not in STATS_TOPICS:
            raise HTTPException(status_code=400, detail="Topic must be stats or stats_test")
        files = stream_files(experiment_path(name) / "streams" / topic)
        total = total_lines(files)
        rows = read_jsonl_downsample(files, max_points=max_points, total=total)
        latest = read_jsonl_last(files)
        key_rows = rows + ([latest] if isinstance(latest, dict) else [])
        keys = sorted({k for row in key_rows if isinstance(row, dict) for k, v in row.items() if is_number(v)})
        return {
            "topic": topic,
            "total": total,
            "sampled": len(rows),
            "rows": rows,
            "latest": latest,
            "numeric_keys": keys,
            "files": [str(path) for path in files],
        }

    @app.get("/api/experiments/{name}/rollouts/{topic}")
    def rollouts(
        name: str,
        topic: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(10, ge=1, le=100),
        q: str = "",
        success: str = "all",
    ) -> dict[str, Any]:
        if topic not in ACTOR_TOPICS:
            raise HTTPException(status_code=400, detail="Topic must be actor or actor_test")
        exp = experiment_path(name)
        files = stream_files(exp / "streams" / topic)
        stats_files = stream_files(exp / "streams" / stats_topic_for_actor(topic))
        total = total_lines(files)
        actor_rows = list(read_jsonl_slice(files, offset=offset, limit=limit))
        stats_rows = list(read_jsonl_slice(stats_files, offset=offset, limit=limit)) if total_lines(stats_files) == total else []
        page = []
        for i, (file_index, row_index, row) in enumerate(actor_rows):
            stats_row = stats_rows[i][2] if i < len(stats_rows) else None
            page.append(summarize_prompt_group(row, file_index, row_index, stats_row=stats_row))

        filtered_page = filter_rollouts(page, q=q.strip().lower(), success=success)
        return {
            "topic": topic,
            "total": total,
            "filtered": total,
            "filter_scope": "page" if q.strip() or success != "all" else "all",
            "offset": offset,
            "limit": limit,
            "rows": filtered_page,
        }

    @app.get("/api/experiments/{name}/rollouts/{topic}/{row_index}")
    def rollout_detail(name: str, topic: str, row_index: int) -> dict[str, Any]:
        if topic not in ACTOR_TOPICS:
            raise HTTPException(status_code=400, detail="Topic must be actor or actor_test")
        exp = experiment_path(name)
        files = stream_files(exp / "streams" / topic)
        stats_files = stream_files(exp / "streams" / stats_topic_for_actor(topic))
        actor_row = read_jsonl_at(files, row_index)
        if actor_row is not None:
            file_index, current_index, row = actor_row
            stats_match = read_jsonl_at(stats_files, row_index) if total_lines(stats_files) == total_lines(files) else None
            stats_row = stats_match[2] if stats_match is not None else None
            return {
                "topic": topic,
                "stats_topic": stats_topic_for_actor(topic),
                "file_index": file_index,
                "row_index": current_index,
                "group": trim_large_arrays(normalize_prompt_group(row, stats_row=stats_row)),
            }
        raise HTTPException(status_code=404, detail="Rollout not found")

    app.mount("/static", StaticFiles(directory=static_root), name="static")
    return app


def load_yaml(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def stream_files(topic_dir: Path) -> list[Path]:
    if not topic_dir.exists():
        return []
    return sorted(topic_dir.glob("*/*/*.jsonl"))


def topic_summary(topic_dir: Path, row_count_dir: Path | None = None) -> dict[str, Any]:
    files = stream_files(topic_dir)
    row_files = stream_files(row_count_dir) if row_count_dir is not None else files
    return {
        "exists": topic_dir.exists(),
        "files": len(files),
        "rows": total_lines(row_files),
    }


def summarize_stats_topic(files: list[Path]) -> dict[str, Any]:
    rows = read_jsonl_files(files)
    base = {"files": len(files), "rows": len(rows)}
    numeric_keys = sorted({k for row in rows if isinstance(row, dict) for k, v in row.items() if is_number(v)})
    base["numeric_keys"] = numeric_keys
    base["last_stats"] = rows[-1] if rows else {}
    return base


def summarize_actor_topic_from_stats(actor_files: list[Path], stats_files: list[Path]) -> dict[str, Any]:
    stats_rows = read_jsonl_files(stats_files)
    base = {"files": len(actor_files), "rows": total_lines(actor_files), "stats_rows": len(stats_rows)}
    base["success_rate"] = mean([number_or_none(row.get("success_mean")) for row in stats_rows if isinstance(row, dict)])
    base["reward_mean"] = mean([number_or_none(row.get("reward_mean")) for row in stats_rows if isinstance(row, dict)])
    base["latency_mean"] = mean([number_or_none(row.get("latency_mean")) for row in stats_rows if isinstance(row, dict)])
    base["attempts"] = None
    base["trace_steps"] = None
    return base


def read_jsonl_files(files: list[Path]) -> list[Any]:
    return [row for _, _, row in iter_jsonl_rows(files)]


def read_jsonl_slice(files: list[Path], offset: int, limit: int):
    stop = offset + limit
    for file_index, row_index, row in iter_jsonl_rows(files, parse_start=offset, parse_stop=stop):
        yield file_index, row_index, row


def read_jsonl_at(files: list[Path], row_index: int) -> tuple[int, int, Any] | None:
    for item in read_jsonl_slice(files, offset=row_index, limit=1):
        return item
    return None


def read_jsonl_last(files: list[Path]) -> Any:
    for path in reversed(files):
        try:
            with path.open("rb") as f:
                f.seek(0, 2)
                end = f.tell()
                if end == 0:
                    continue
                pos = end - 1
                while pos > 0:
                    f.seek(pos)
                    if f.read(1) == b"\n" and pos != end - 1:
                        break
                    pos -= 1
                f.seek(pos + 1 if pos > 0 else 0)
                line = f.readline().decode("utf-8").strip()
                return json.loads(line) if line else None
        except OSError:
            continue
    return None


def read_jsonl_downsample(files: list[Path], max_points: int, total: int | None = None) -> list[Any]:
    total = total_lines(files) if total is None else total
    if total <= max_points:
        return read_jsonl_files(files)
    stride = max(1, math.ceil(total / max_points))
    rows = []
    for _, row_index, row in iter_jsonl_rows(files):
        if row_index % stride == 0:
            rows.append(row)
    return rows


def iter_jsonl_rows(files: list[Path], parse_start: int = 0, parse_stop: int | None = None):
    row_index = 0
    for file_index, path in enumerate(files):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if parse_stop is not None and row_index >= parse_stop:
                    return
                current_index = row_index
                row_index += 1
                if current_index < parse_start:
                    continue
                line = line.strip()
                if not line:
                    continue
                yield file_index, current_index, json.loads(line)


def total_lines(files: list[Path]) -> int:
    return sum(count_lines(path) for path in files)


def count_lines(path: Path) -> int:
    try:
        stat = path.stat()
        return count_lines_cached(path, stat.st_mtime_ns, stat.st_size)
    except OSError:
        return 0


@lru_cache(maxsize=512)
def count_lines_cached(path: Path, mtime_ns: int, size: int) -> int:
    del mtime_ns, size
    with path.open("rb") as f:
        return sum(1 for _ in f)


def stats_topic_for_actor(topic: str) -> str:
    return "stats_test" if topic == "actor_test" else "stats"


def summarize_prompt_group(row: Any, file_index: int, row_index: int, stats_row: dict[str, Any] | None = None) -> dict[str, Any]:
    group = normalize_prompt_group(row, stats_row=stats_row)
    attempts = group.get("attempts", [])
    calls = [call for attempt in attempts for call in attempt.get("calls", []) if isinstance(call, dict)]
    metrics = group.get("metrics", {}) if isinstance(group.get("metrics"), dict) else {}
    first = calls[0] if calls else {}
    rewards = [number_or_none(call.get("reward")) for call in calls]
    rewards = [reward for reward in rewards if reward is not None]
    prompt_stats = group.get("prompt_stats") if isinstance(group.get("prompt_stats"), dict) else {}
    stats_reward_mean = number_or_none(prompt_stats.get("reward_mean"))
    stats_success_mean = number_or_none(prompt_stats.get("success_mean"))
    explicit_success = bool_or_none(metrics.get("success"))
    successes = [reward > 0 for reward in rewards]
    success_rate = mean(successes) if successes else (1.0 if explicit_success is True else 0.0 if explicit_success is False else None)
    attempt_indices = [attempt["rollout_index"] for attempt in attempts]
    step_counts = [len(attempt.get("calls", [])) for attempt in attempts]
    return {
        "file_index": file_index,
        "row_index": row_index,
        "dataset_name": group.get("dataset_name"),
        "domain": group.get("domain"),
        "group_id": group.get("group_id") or first.get("group_id"),
        "model_version": group.get("model_version") or first.get("metadata", {}).get("model_version"),
        "rollout_indices": attempt_indices,
        "reward_mean": stats_reward_mean if stats_reward_mean is not None else mean(rewards),
        "reward_max": first_not_none(number_or_none(prompt_stats.get("reward_max")), max(rewards) if rewards else None, number_or_none(metrics.get("reward"))),
        "success": (stats_success_mean > 0) if stats_success_mean is not None else explicit_success if explicit_success is not None else (any(successes) if successes else None),
        "success_rate": stats_success_mean if stats_success_mean is not None else success_rate,
        "no_error": bool_or_none(metrics.get("no_error")),
        "no_answer": bool_or_none(metrics.get("no_answer")),
        "latency": number_or_none(group.get("latency")),
        "attempts": len(attempts),
        "trace_steps": len(calls),
        "steps_min": min(step_counts) if step_counts else None,
        "steps_max": max(step_counts) if step_counts else None,
        "prompt_tokens_mean": first_not_none(number_or_none(prompt_stats.get("prompt_tokens_mean")), mean([number_or_zero(call.get("prompt_tokens")) for call in calls])),
        "output_tokens_mean": first_not_none(number_or_none(prompt_stats.get("output_tokens_mean")), mean([number_or_zero(call.get("output_tokens")) for call in calls])),
        "stats_row": row_index if prompt_stats else None,
        "finished_steps": sum(1 for call in calls if bool(call.get("finished"))),
        "preview": compact_text(first.get("text") or "", 280),
        "metric_keys": sorted(metrics.keys()),
    }


def normalize_prompt_group(row: Any, stats_row: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(row, dict) and "training_texts" in row:
        group = {
            "attempts": group_calls_by_attempt(row.get("training_texts", [])),
            "metrics": row.get("metrics", {}),
            "prompt_stats": stats_row or {},
            "latency": row.get("latency"),
            "model_version": row.get("model_version"),
            "dataset_name": row.get("dataset_name"),
            "group_id": row.get("group_id"),
            "domain": row.get("domain"),
        }
        return group
    if isinstance(row, list):
        metrics = {}
        if row:
            rewards = [t.get("reward") for t in row if isinstance(t, dict) and is_number(t.get("reward"))]
            if rewards:
                metrics["reward"] = sum(rewards) / len(rewards)
        return {"attempts": group_calls_by_attempt(row), "metrics": metrics, "prompt_stats": stats_row or {}, "latency": None}
    return {"attempts": [], "metrics": {}, "prompt_stats": stats_row or {}, "latency": None, "raw": row}


def group_calls_by_attempt(training_texts: list[Any]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    unknown_index = 0
    for call in training_texts:
        if not isinstance(call, dict):
            continue
        metadata = call.get("metadata", {}) if isinstance(call.get("metadata"), dict) else {}
        rollout_index = metadata.get("rollout_index")
        if not isinstance(rollout_index, int):
            rollout_index = unknown_index
            unknown_index += 1
        grouped.setdefault(rollout_index, []).append(call)

    attempts = []
    for rollout_index, calls in sorted(grouped.items()):
        calls.sort(key=lambda item: item.get("metadata", {}).get("step_index", 0) if isinstance(item.get("metadata"), dict) else 0)
        rewards = [number_or_none(call.get("reward")) for call in calls]
        rewards = [reward for reward in rewards if reward is not None]
        attempts.append(
            {
                "rollout_index": rollout_index,
                "steps": len(calls),
                "reward_mean": mean(rewards),
                "reward_final": rewards[-1] if rewards else None,
                "success": any(reward > 0 for reward in rewards) if rewards else None,
                "calls": calls,
            }
        )
    return attempts


def filter_rollouts(rows: list[dict[str, Any]], q: str, success: str) -> list[dict[str, Any]]:
    out = rows
    if success in {"true", "false"}:
        expected = success == "true"
        out = [r for r in out if r["success"] is expected]
    if q:
        out = [
            r
            for r in out
            if q in " ".join(str(r.get(k) or "") for k in ("dataset_name", "domain", "group_id", "preview")).lower()
        ]
    return out


def trim_large_arrays(value: Any) -> Any:
    if isinstance(value, list):
        if len(value) > 80 and all(not isinstance(x, (dict, list)) for x in value):
            return {"type": "array", "length": len(value), "head": value[:12], "tail": value[-12:]}
        return [trim_large_arrays(v) for v in value]
    if isinstance(value, dict):
        return {k: trim_large_arrays(v) for k, v in value.items() if k != "visual_features"}
    if isinstance(value, str) and len(value) > MAX_DETAIL_TEXT:
        return value[:MAX_DETAIL_TEXT] + "\n\n[truncated]"
    return value


def compact_text(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "..."


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def number_or_none(value: Any) -> float | None:
    return float(value) if is_number(value) else None


def number_or_zero(value: Any) -> float:
    return float(value) if is_number(value) else 0.0


def bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def mean(values: list[Any]) -> float | None:
    nums = [float(v) for v in values if is_number(v) or isinstance(v, bool)]
    return sum(nums) / len(nums) if nums else None
