import json
import pickle
import re
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence


PLANNING_LOG_NAMES = {
    "query_planner",
    "action_plan_initial",
    "action_plan_fallback",
    "action_dependencies",
}

INLINE_SECTION_RE = re.compile(r"^\s*([A-Z][A-Za-z0-9_ /()'\"&-]{1,80}):\s*(.*)$")
HEADER_SECTION_RE = re.compile(r"^\s*([A-Z][A-Za-z0-9_ /()'\"&-]{1,80}):\s*$")


def pickle_size_bytes(value: Any) -> int:
    try:
        return len(pickle.dumps(value))
    except Exception:
        return -1


def _privacy_agent_metadata(sample: Mapping[str, Any]) -> dict[str, Any] | None:
    metadata = sample.get("metadata")
    if not isinstance(metadata, dict):
        return None
    privacy_agent = metadata.get("privacy_agent")
    if not isinstance(privacy_agent, dict):
        return None
    return privacy_agent


def privacy_agent_log_name(sample: Mapping[str, Any]) -> str:
    privacy_agent = _privacy_agent_metadata(sample)
    if not privacy_agent:
        return "unknown"
    return str(privacy_agent.get("log_name") or "unknown")


def privacy_agent_stage(log_name: str) -> str:
    if log_name in PLANNING_LOG_NAMES or log_name.startswith("adaptive_iter"):
        return "planning"
    if log_name.startswith("report_") or log_name == "report_to_insights":
        return "reporting"
    if log_name == "llm":
        return "execution"
    if any(token in log_name for token in ("analysis", "synthesis", "summary")):
        return "analysis"
    return "other"


def summarize_privacy_agent_payload(
    samples: Sequence[Mapping[str, Any]],
    *,
    container: Any | None = None,
    limit_bytes: int | None = None,
    top_n_fields: int = 10,
    top_n_log_names: int = 10,
    top_n_stage_fields: int = 5,
    top_n_log_name_fields: int = 5,
) -> dict[str, Any] | None:
    filtered_samples = [sample for sample in samples if _privacy_agent_metadata(sample)]
    if not filtered_samples:
        return None

    container_value = container if container is not None else filtered_samples
    container_pickle_bytes = pickle_size_bytes(container_value)
    sample_sizes = [max(0, pickle_size_bytes(sample)) for sample in filtered_samples]

    field_bytes: dict[str, int] = defaultdict(int)
    stage_field_bytes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    log_name_field_bytes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    stage_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"samples": 0, "sample_pickle_bytes": 0, "text_chars": 0, "prompt_tokens": 0, "output_tokens": 0}
    )
    log_name_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"samples": 0, "sample_pickle_bytes": 0, "text_chars": 0, "prompt_tokens": 0, "output_tokens": 0}
    )

    for sample, sample_pickle_bytes in zip(filtered_samples, sample_sizes):
        log_name = privacy_agent_log_name(sample)
        stage = privacy_agent_stage(log_name)
        text_chars = len(str(sample.get("text") or ""))
        prompt_tokens = int(sample.get("prompt_tokens") or 0)
        output_tokens = int(sample.get("output_tokens") or 0)

        stage_stats[stage]["samples"] += 1
        stage_stats[stage]["sample_pickle_bytes"] += sample_pickle_bytes
        stage_stats[stage]["text_chars"] += text_chars
        stage_stats[stage]["prompt_tokens"] += prompt_tokens
        stage_stats[stage]["output_tokens"] += output_tokens

        log_name_stats[log_name]["samples"] += 1
        log_name_stats[log_name]["sample_pickle_bytes"] += sample_pickle_bytes
        log_name_stats[log_name]["text_chars"] += text_chars
        log_name_stats[log_name]["prompt_tokens"] += prompt_tokens
        log_name_stats[log_name]["output_tokens"] += output_tokens

        for key, value in sample.items():
            value_pickle_bytes = max(0, pickle_size_bytes(value))
            field_bytes[key] += value_pickle_bytes
            stage_field_bytes[stage][key] += value_pickle_bytes
            log_name_field_bytes[log_name][key] += value_pickle_bytes

    sorted_fields = sorted(field_bytes.items(), key=lambda item: item[1], reverse=True)[:top_n_fields]
    sorted_stage_stats = dict(
        sorted(stage_stats.items(), key=lambda item: item[1]["sample_pickle_bytes"], reverse=True)
    )
    sorted_log_name_stats = dict(
        sorted(log_name_stats.items(), key=lambda item: item[1]["sample_pickle_bytes"], reverse=True)[:top_n_log_names]
    )
    sorted_stage_field_bytes = {}
    for stage in sorted_stage_stats:
        field_map = stage_field_bytes.get(stage)
        if field_map:
            sorted_stage_field_bytes[stage] = dict(
                sorted(field_map.items(), key=lambda item: item[1], reverse=True)[:top_n_stage_fields]
            )
    sorted_log_name_field_bytes = {
        log_name: dict(sorted(log_name_field_bytes[log_name].items(), key=lambda item: item[1], reverse=True)[:top_n_log_name_fields])
        for log_name in sorted_log_name_stats
        if log_name in log_name_field_bytes
    }

    summary: dict[str, Any] = {
        "sample_count": len(filtered_samples),
        "container_pickle_bytes": container_pickle_bytes,
        "sample_pickle_bytes_sum": sum(sample_sizes),
        "sample_pickle_bytes_max": max(sample_sizes) if sample_sizes else 0,
        "text_chars_sum": sum(len(str(sample.get("text") or "")) for sample in filtered_samples),
        "approx_field_pickle_bytes": dict(sorted_fields),
        "by_stage": sorted_stage_stats,
        "by_stage_top_fields": sorted_stage_field_bytes,
        "by_log_name": sorted_log_name_stats,
        "by_log_name_top_fields": sorted_log_name_field_bytes,
    }

    if limit_bytes:
        summary["limit_bytes"] = int(limit_bytes)
        summary["limit_utilization"] = round(container_pickle_bytes / limit_bytes, 4) if container_pickle_bytes >= 0 else -1

    return summary


def format_privacy_agent_payload_summary(summary: Mapping[str, Any]) -> str:
    return json.dumps(summary, sort_keys=True)


def should_log_privacy_agent_payload(
    summary: Mapping[str, Any],
    *,
    min_bytes: int = 5 * 1024 * 1024,
    min_utilization: float = 0.1,
) -> bool:
    container_pickle_bytes = int(summary.get("container_pickle_bytes") or 0)
    if container_pickle_bytes >= min_bytes:
        return True
    limit_utilization = summary.get("limit_utilization")
    if isinstance(limit_utilization, (int, float)) and limit_utilization >= min_utilization:
        return True
    return False


def _is_prompt_section_header(line: str) -> tuple[str, str | None] | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith(("-", "*", "{", "[", "(")):
        return None
    if re.match(r"^\d+\.\s", stripped):
        return None

    inline_match = INLINE_SECTION_RE.match(line)
    if inline_match:
        header = inline_match.group(1).strip()
        if len(header.split()) <= 12:
            inline_content = inline_match.group(2).strip()
            return header, inline_content or None

    header_match = HEADER_SECTION_RE.match(line)
    if header_match:
        header = header_match.group(1).strip()
        if len(header.split()) <= 12:
            return header, None

    return None


def _prompt_section_category(section_name: str) -> str:
    normalized = section_name.lower()
    if any(token in normalized for token in ("document", "evidence", "content", "source")):
        return "documents_or_evidence"
    if any(token in normalized for token in ("finding", "insight", "analysis", "synthesis", "summary")):
        return "findings_or_summary"
    if any(token in normalized for token in ("report", "executive")):
        return "reporting_context"
    if any(token in normalized for token in ("question", "query", "task", "research focus", "sub-questions")):
        return "task_definition"
    if any(token in normalized for token in ("tool", "action plan", "strategy", "status")):
        return "planning_context"
    if any(token in normalized for token in ("requirement", "rule", "guideline", "provide", "return", "format")):
        return "instructions"
    return "other"


def summarize_privacy_agent_prompt(
    prompt: str,
    *,
    log_name: str,
    token_counter: Callable[[str], int],
    top_n_sections: int = 8,
) -> dict[str, Any]:
    lines = prompt.splitlines()
    sections: list[dict[str, Any]] = []
    current = {"section_name": "preamble", "content_lines": []}

    for line in lines:
        header_info = _is_prompt_section_header(line)
        if header_info:
            if current["content_lines"]:
                sections.append(current)
            section_name, inline_content = header_info
            current = {"section_name": section_name, "content_lines": []}
            if inline_content:
                current["content_lines"].append(inline_content)
            continue
        current["content_lines"].append(line)

    if current["content_lines"]:
        sections.append(current)

    category_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"tokens": 0, "chars": 0, "sections": 0})
    section_summaries: list[dict[str, Any]] = []
    for section in sections:
        section_name = str(section["section_name"])
        content = "\n".join(section["content_lines"]).strip()
        if not content:
            continue
        tokens = max(0, token_counter(content))
        chars = len(content)
        category = _prompt_section_category(section_name)
        category_stats[category]["tokens"] += tokens
        category_stats[category]["chars"] += chars
        category_stats[category]["sections"] += 1
        section_summaries.append(
            {
                "section_name": section_name,
                "category": category,
                "tokens": tokens,
                "chars": chars,
            }
        )

    total_tokens = max(0, token_counter(prompt))
    sorted_categories = dict(
        sorted(category_stats.items(), key=lambda item: item[1]["tokens"], reverse=True)
    )
    top_sections = sorted(section_summaries, key=lambda item: item["tokens"], reverse=True)[:top_n_sections]
    return {
        "log_name": log_name,
        "prompt_chars": len(prompt),
        "prompt_tokens": total_tokens,
        "by_category": sorted_categories,
        "top_sections": top_sections,
    }


def should_log_privacy_agent_prompt_summary(
    summary: Mapping[str, Any],
    *,
    min_tokens: int = 4096,
) -> bool:
    return int(summary.get("prompt_tokens") or 0) >= min_tokens
