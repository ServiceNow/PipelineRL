"""Generic per-LLM-call reward shaping for Cube tool-calling agents.

Shaping is computed from the LLMCall payload that cube-harness emits on each
trainable `llm_call` event:
  - `call["prompt"]["tools"]`  — OpenAI-style tool schemas exposed for this call
  - `call["output"]`           — litellm `Message` (or its dumped dict) with
                                 `content` and `tool_calls`

Two independent components, each opt-in via config:
  - format: did the assistant output contain the required structural markers?
  - correctness: are emitted tool calls well-formed w.r.t. the exposed tools?

Neither component judges task success — that is owned by the terminal reward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class FormatRewardConfig:
    enabled: bool = False
    weight: float = 0.0
    required_tokens: list[str] = field(default_factory=list)
    forbidden_tokens: list[str] = field(default_factory=list)
    require_tool_call: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "FormatRewardConfig":
        if not data:
            return cls()
        return cls(
            enabled=bool(data.get("enabled", False)),
            weight=float(data.get("weight", 0.0)),
            required_tokens=list(data.get("required_tokens") or []),
            forbidden_tokens=list(data.get("forbidden_tokens") or []),
            require_tool_call=bool(data.get("require_tool_call", False)),
        )


@dataclass
class CorrectnessRewardConfig:
    enabled: bool = False
    weight: float = 0.0
    reward_valid: float = 1.0
    penalty_unknown_tool: float = -1.0
    penalty_bad_args: float = -1.0
    penalty_no_tool_call: float = 0.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "CorrectnessRewardConfig":
        if not data:
            return cls()
        return cls(
            enabled=bool(data.get("enabled", False)),
            weight=float(data.get("weight", 0.0)),
            reward_valid=float(data.get("reward_valid", 1.0)),
            penalty_unknown_tool=float(data.get("penalty_unknown_tool", -1.0)),
            penalty_bad_args=float(data.get("penalty_bad_args", -1.0)),
            penalty_no_tool_call=float(data.get("penalty_no_tool_call", 0.0)),
        )

@dataclass
class RewardShapingConfig:
    enabled: bool = False
    format: FormatRewardConfig = field(default_factory=FormatRewardConfig)
    correctness: CorrectnessRewardConfig = field(default_factory=CorrectnessRewardConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "RewardShapingConfig":
        if not data:
            return cls()
        return cls(
            enabled=bool(data.get("enabled", False)),
            format=FormatRewardConfig.from_mapping(data.get("format")),
            correctness=CorrectnessRewardConfig.from_mapping(data.get("correctness")),
        )

    @property
    def is_active(self) -> bool:
        return self.enabled and (self.format.enabled or self.correctness.enabled)


def compute_format_reward(
    output_text: str,
    has_tool_call: bool,
    cfg: FormatRewardConfig,
) -> tuple[float, dict[str, Any]]:
    if not cfg.enabled:
        return 0.0, {"enabled": False}

    text = output_text or ""
    missing = [tok for tok in cfg.required_tokens if tok not in text]
    forbidden_present = [tok for tok in cfg.forbidden_tokens if tok in text]
    tool_call_ok = has_tool_call or not cfg.require_tool_call

    passed = not missing and not forbidden_present and tool_call_ok
    reward = cfg.weight if passed else 0.0
    info = {
        "enabled": True,
        "passed": passed,
        "missing_required": missing,
        "forbidden_present": forbidden_present,
        "tool_call_required": cfg.require_tool_call,
        "tool_call_present": has_tool_call,
    }
    return reward, info


def _index_available_tools(tools: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if "function" in tool else tool
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if isinstance(name, str) and name:
            index[name] = fn.get("parameters") or {}
    return index


def _parse_arguments(arguments: Any) -> tuple[dict[str, Any] | None, bool]:
    """Return (parsed_dict, ok). `ok=False` only on malformed JSON; missing/empty -> {}."""
    if arguments is None or arguments == "":
        return {}, True
    if isinstance(arguments, dict):
        return arguments, True
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (ValueError, TypeError):
            return None, False
        return (parsed, True) if isinstance(parsed, dict) else ({}, True)
    return {}, True


def _validate_call(
    name: str,
    arguments: Any,
    tool_index: dict[str, dict[str, Any]],
    cfg: CorrectnessRewardConfig,
) -> tuple[float, dict[str, Any]]:
    if name not in tool_index:
        return cfg.penalty_unknown_tool, {"name": name, "status": "unknown_tool"}

    parsed, ok = _parse_arguments(arguments)
    if not ok:
        return cfg.penalty_bad_args, {"name": name, "status": "bad_args_json"}

    schema = tool_index[name] or {}
    properties = schema.get("properties") or {}
    required = schema.get("required") or []
    declared = set(properties.keys()) if isinstance(properties, dict) else set()
    provided = set(parsed.keys()) if isinstance(parsed, dict) else set()

    if declared and not provided.issubset(declared):
        return cfg.penalty_bad_args, {
            "name": name,
            "status": "unknown_args",
            "unexpected": sorted(provided - declared),
        }
    missing = [k for k in required if k not in provided]
    if missing:
        return cfg.penalty_bad_args, {
            "name": name,
            "status": "missing_required",
            "missing": missing,
        }
    return cfg.reward_valid, {"name": name, "status": "ok"}


def compute_correctness_reward(
    tool_calls: list[dict[str, Any]],
    available_tools: list[dict[str, Any]] | None,
    cfg: CorrectnessRewardConfig,
) -> tuple[float, dict[str, Any]]:
    if not cfg.enabled:
        return 0.0, {"enabled": False}

    tool_index = _index_available_tools(available_tools)
    if not tool_calls:
        return cfg.weight * cfg.penalty_no_tool_call, {
            "enabled": True,
            "n_calls": 0,
            "calls": [],
        }

    raw = 0.0
    per_call_info: list[dict[str, Any]] = []
    for tc in tool_calls:
        fn = (tc.get("function") if isinstance(tc, dict) else None) or {}
        name = fn.get("name") or (tc.get("name") if isinstance(tc, dict) else "") or ""
        arguments = fn.get("arguments") if "arguments" in fn else (
            tc.get("arguments") if isinstance(tc, dict) else None
        )
        score, info = _validate_call(str(name), arguments, tool_index, cfg)
        raw += score
        per_call_info.append(info)
    return cfg.weight * raw, {
        "enabled": True,
        "n_calls": len(tool_calls),
        "calls": per_call_info,
    }


def _extract_output_fields(output: Any) -> tuple[str, list[dict[str, Any]]]:
    """Extract `(content_text, tool_calls)` from a litellm Message or its dump."""
    if output is None:
        return "", []
    content = ""
    tool_calls: list[dict[str, Any]] = []
    if isinstance(output, dict):
        raw_content = output.get("content")
        raw_tool_calls = output.get("tool_calls")
    else:
        raw_content = getattr(output, "content", None)
        raw_tool_calls = getattr(output, "tool_calls", None)

    if isinstance(raw_content, str):
        content = raw_content
    elif isinstance(raw_content, list):
        # litellm content can be a list of blocks: {"type": "text", "text": "..."}
        parts = []
        for block in raw_content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        content = "".join(parts)

    if isinstance(raw_tool_calls, list):
        for tc in raw_tool_calls:
            if isinstance(tc, dict):
                tool_calls.append(tc)
            else:
                # Pydantic-style object: coerce to dict shallowly
                tool_calls.append(
                    {
                        "id": getattr(tc, "id", None),
                        "type": getattr(tc, "type", "function"),
                        "function": {
                            "name": getattr(getattr(tc, "function", None), "name", None),
                            "arguments": getattr(getattr(tc, "function", None), "arguments", None),
                        },
                    }
                )
    return content, tool_calls


def compute_call_shaping(call: Mapping[str, Any], cfg: RewardShapingConfig) -> dict[str, Any]:
    """Compute the per-LLM-call shaping bundle from a (dumped) LLMCall.

    Returns a dict shaped as:
        {"format": float, "correctness": float, "total": float, "info": {...}}
    Always returns numeric components (0.0 when disabled) so callers can rely
    on the schema being stable.
    """
    prompt = call.get("prompt") or {}
    available_tools = prompt.get("tools") or []
    output_text, tool_calls = _extract_output_fields(call.get("output"))

    fmt_reward, fmt_info = compute_format_reward(
        output_text, has_tool_call=bool(tool_calls), cfg=cfg.format
    )
    corr_reward, corr_info = compute_correctness_reward(
        tool_calls, available_tools, cfg=cfg.correctness
    )
    return {
        "format": fmt_reward,
        "correctness": corr_reward,
        "total": fmt_reward + corr_reward,
        "info": {"format": fmt_info, "correctness": corr_info},
    }
