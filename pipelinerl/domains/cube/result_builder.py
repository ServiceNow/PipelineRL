"""Cube-harness ↔ PipelineRL adapter.

Builds the payload for cube-harness `RolloutTaskRunner`, runs one rollout
synchronously, and converts the resulting RL event stream into a PipelineRL
`RolloutResult`. Token IDs, logprobs, and trainable-event tagging are taken
straight from cube-harness — we do not reconstruct them here.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from cube_harness.episode import MAX_STEPS
from cube_harness.rl.event_publisher import EventPublisher
from cube_harness.rl.events import EventContext, TerminalEvent
from cube_harness.rl.task_runner import RolloutTaskRunner

from pipelinerl.rollouts import BaseMetrics, RolloutResult, TrainingText
MASKED_TOKEN_ID = -100

logger = logging.getLogger(__name__)


def llm_config_dict(llm: dict[str, Any]) -> dict[str, Any]:
    """Translate a PipelineRL `llm` dict into a `RolloutLLMConfig`-shaped dict."""
    parameters = dict(llm.get("parameters") or {})
    config_data = {
        **parameters,
        "api_base": llm["base_url"],
        "api_key": "EMPTY",
        "model_name": llm.get("served_model_name") or llm["model_name"],
        "tokenizer_name": llm.get("tokenizer_name") or llm.get("served_model_name") or llm["model_name"],
    }
    if llm.get("collect_logprobs"):
        config_data.setdefault("logprobs", True)
        config_data.setdefault("include_stop_str_in_output", True)
        config_data.setdefault("skip_special_tokens", False)
    return config_data


def build_payload(
    *,
    request: Any,
    item: dict[str, Any],
    llm: dict[str, Any],
    output_dir: Path,
    persist_rollout: bool = False,
    service_name: str = "pipelinerl",
) -> dict[str, Any]:
    """Build the payload `RolloutTaskRunner` expects for one rollout."""
    domain = item.get("domain")
    task_id = str(item["task_id"])
    max_steps = request.max_steps or item.get("max_steps") or MAX_STEPS
    rl_request = {
        "request_id": request.request_id,
        "task_id": task_id,
        "llm_config": llm_config_dict(llm),
        "rollout_index": request.rollout_index,
        "group_id": request.group_id,
        "max_steps": max_steps,
        "model_version": request.model_version,
    }
    event_context = EventContext(
        request_id=request.request_id,
        trajectory_id=request.request_id,
        env_name=domain,
        task_id=task_id,
        group_id=request.group_id,
        rollout_index=request.rollout_index,
        model_version=request.model_version,
    )
    return {
        "request": rl_request,
        "task_config": item["task_config"],
        "agent_config": item["agent_cfg"],
        "runtime_context": item.get("runtime_context"),
        "output_dir": str(output_dir),
        "persist_rollout": persist_rollout,
        "service_name": service_name,
        "benchmark_name": domain,
        "max_steps": max_steps,
        "event_context": event_context,
    }


def run_rollout(payload: dict[str, Any], publisher: EventPublisher) -> None:
    """Run one rollout in-thread and guarantee a terminal event is published.

    Mirrors the synthetic-terminal policy from cube-harness `LocalRolloutExecutor`:
    on an unhandled exception or a worker that returns without emitting a
    terminal, publish a synthetic terminal so consumers always see one.
    """
    request = payload["request"]
    request_id = str(request["request_id"])
    try:
        RolloutTaskRunner(payload, publisher).run()
    except Exception as exc:
        if not publisher.has_terminal(request_id):
            _publish_synthetic_terminal(
                publisher,
                payload,
                status="agent_error",
                error={"type": type(exc).__name__, "message": str(exc)[:500]},
            )
        return
    if not publisher.has_terminal(request_id):
        _publish_synthetic_terminal(
            publisher,
            payload,
            status="event_error",
            error={"type": "MissingTerminal", "message": "worker completed without terminal event"},
        )


def rollout_result_from_events(
    events: list[dict[str, Any]],
    *,
    latency: float,
    dataset: str | None,
    domain: str | None,
) -> RolloutResult:
    """Map a cube-harness RL event stream to a PipelineRL `RolloutResult`."""
    training_texts: list[TrainingText] = []
    event_errors: list[Any] = []
    terminal: dict[str, Any] | None = None
    for event in events:
        event_type = event.get("type")
        if event_type == "terminal":
            terminal = event
        elif event_type == "llm_call":
            rl = event.get("rl") or {}
            if not rl.get("trainable"):
                payload = event.get("event") or {}
                if payload.get("error"):
                    event_errors.append(payload["error"])
                continue
            call = (event.get("event") or {}).get("call")
            if call is None:
                continue
            training_texts.append(_training_text_from_event(event, call))
        elif event_type in {"agent_error", "tool_call"}:
            payload = event.get("event") or {}
            if payload.get("error"):
                event_errors.append(payload["error"])

    if terminal is None:
        raise RuntimeError("Cube rollout ended without terminal event")

    final_reward = float(terminal.get("final_reward") or 0.0)
    rollout_status = str(terminal.get("rollout_status") or "event_error")
    finished = rollout_status == "completed"
    for text in training_texts:
        text.reward = final_reward
        text.finished = finished
        text.metadata["final_reward"] = final_reward
        text.metadata["rollout_status"] = rollout_status
        text.metadata["domain"] = domain
        text.metadata["dataset_name"] = dataset

    summary_info = ((terminal.get("summary") or {}).get("info") or {})
    filtered_info = {
        key: value
        for key, value in summary_info.items()
        if isinstance(value, (list, float, bool, int))
    }
    terminal_error = terminal.get("error")
    metrics = BaseMetrics(
        reward=final_reward,
        success=bool(terminal.get("outcome_success")),
        no_error=terminal_error is None and not event_errors,
        no_answer=not bool(training_texts),
        num_steps=len(training_texts),
        rollout_valid=bool(terminal.get("rollout_valid")),
        trainable=bool(terminal.get("trainable")),
        **filtered_info,
    )

    return RolloutResult(
        training_texts=training_texts,
        metrics=metrics,
        latency=latency,
        dataset_name=dataset,
        domain=domain,
    )


def apply_rollout_rewards(
    training_texts: list[TrainingText],
    *,
    agent_config: Any,
    buffer_tokens: int = 0,
    discount_factor: float = 1.0,
) -> None:
    """Apply discount + length-buffer reward shaping in-place."""
    total_output_tokens = sum(getattr(t, "output_tokens", 0) for t in training_texts)
    if discount_factor != 1.0:
        scale = discount_factor**total_output_tokens
        for text in training_texts:
            text.reward *= scale

    agent_llm_config = getattr(agent_config, "llm_config", None)
    max_completion_tokens = int(getattr(agent_llm_config, "max_completion_tokens", 0) or 0)
    if buffer_tokens and max_completion_tokens > 0:
        penalty = _length_penalty(max_completion_tokens, total_output_tokens, buffer_tokens)
        for text in training_texts:
            text.reward += penalty


def write_rollout_artifact(
    *,
    events: list[dict[str, Any]],
    artifact_dir: Path,
    worker_name: str,
    trajectory_id: str,
    task_id: str | None,
    domain: str | None,
    dataset: str | None,
) -> None:
    """Persist a compact per-rollout summary derived from the event stream."""
    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        llm_calls = []
        terminal: dict[str, Any] | None = None
        for event in events:
            event_type = event.get("type")
            if event_type == "llm_call":
                call = (event.get("event") or {}).get("call") or {}
                rl = event.get("rl") or {}
                llm_calls.append(
                    {
                        "step_index": event.get("event_index"),
                        "llm_call_index": rl.get("llm_call_index"),
                        "llm_call_id": call.get("id"),
                        "tag": call.get("tag"),
                        "timestamp": call.get("timestamp"),
                        "prompt_tokens": call.get("prompt_tokens"),
                        "output_tokens": call.get("output_tokens"),
                        "finish_reason": call.get("finish_reason"),
                        "metadata": call.get("metadata"),
                    }
                )
            elif event_type == "terminal":
                terminal = event

        payload = {
            "trajectory_id": trajectory_id,
            "domain": domain,
            "dataset_name": dataset,
            "task_id": task_id,
            "termination_reason": terminal.get("rollout_status") if terminal else None,
            "reward_info": {"final_reward": terminal.get("final_reward")} if terminal else None,
            "summary_stats": terminal.get("summary") if terminal else None,
            "n_steps": len(llm_calls),
            "llm_calls": llm_calls,
        }
        artifact_path = artifact_dir / f"{worker_name}_{trajectory_id}_{time.time_ns()}.json"
        artifact_path.write_text(json.dumps(payload, indent=2, default=str))
    except Exception:
        logger.exception("%s failed to write rollout artifact for %s", worker_name, trajectory_id)


def _training_text_from_event(event: dict[str, Any], call: dict[str, Any]) -> TrainingText:
    prompt_token_ids = call.get("prompt_token_ids") or []
    completion_token_ids = call.get("completion_token_ids") or []
    logprobs = call.get("logprobs") or []
    if not prompt_token_ids or not completion_token_ids or not logprobs:
        # RLEventSink only tags `trainable=True` when these are present and aligned;
        # falling into this branch means the contract was violated upstream.
        raise ValueError("trainable LLM call missing prompt_token_ids/completion_token_ids/logprobs")
    if len(completion_token_ids) != len(logprobs):
        raise ValueError(
            "completion_token_ids/logprobs length mismatch: "
            f"{len(completion_token_ids)} != {len(logprobs)}"
        )

    input_ids = list(prompt_token_ids) + list(completion_token_ids)
    labels = [MASKED_TOKEN_ID] * len(prompt_token_ids) + list(completion_token_ids)
    text, n_predicted = _inspection_text(call)
    rl = event.get("rl") or {}
    metadata = dict(call.get("metadata") or {})
    metadata.update(
        {
            "llm_call_id": call.get("id"),
            "llm_call_tag": call.get("tag"),
            "trajectory_id": event.get("trajectory_id"),
            "task_id": event.get("task_id"),
            "agent_step_index": event.get("event_index"),
            "llm_call_index": rl.get("llm_call_index"),
            "finish_reason": call.get("finish_reason"),
            "llm_prompt_tokens": call.get("prompt_tokens"),
            "llm_output_tokens": call.get("output_tokens"),
        }
    )
    return TrainingText(
        text=text,
        n_predicted=n_predicted,
        input_ids=input_ids,
        labels=labels,
        logprobs=list(logprobs),
        prompt_tokens=int(call.get("prompt_tokens") or 0),
        output_tokens=int(call.get("output_tokens") or 0),
        metadata=metadata,
    )


def _inspection_text(call: dict[str, Any]) -> tuple[str, int]:
    prompt = call.get("prompt") or {}
    prompt_text = json.dumps(
        {"messages": prompt.get("messages") or [], "tools": prompt.get("tools") or []},
        separators=(",", ":"),
        default=str,
    )
    output_text = json.dumps(call.get("output") or {}, separators=(",", ":"), default=str)
    return prompt_text + output_text, len(output_text)


def _length_penalty(max_length: int, sequence_length: int, buffer_tokens: int) -> float:
    if buffer_tokens <= 0:
        return 0.0
    if sequence_length > (max_length - buffer_tokens) and sequence_length <= max_length:
        return ((max_length - buffer_tokens) - sequence_length) / buffer_tokens
    return 0.0


def _publish_synthetic_terminal(
    publisher: EventPublisher,
    payload: dict[str, Any],
    *,
    status: str,
    error: dict[str, Any] | None,
) -> None:
    request = payload["request"]
    publisher.publish(
        TerminalEvent(
            event_index=-1,
            request_id=str(request["request_id"]),
            trajectory_id=str(request["request_id"]),
            env_name=payload.get("benchmark_name"),
            task_id=str(request.get("task_id") or ""),
            group_id=request.get("group_id"),
            rollout_index=int(request.get("rollout_index") or 0),
            model_version=request.get("model_version"),
            rollout_status=status,  # type: ignore[arg-type]
            outcome_success=False,
            final_reward=None,
            rollout_valid=False,
            trainable=False,
            error=error,
        )
    )
