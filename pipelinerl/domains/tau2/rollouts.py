import math
import pickle
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import aiohttp
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError
from transformers import PreTrainedTokenizerBase

from pipelinerl.domains.tau2.client import (
    Tau2BoundaryFailure,
    Tau2GymClient,
    Tau2GymSettings,
    Tau2RunResponse,
    normalize_openai_base_url,
)
from pipelinerl.llm import TrainableLLM
from pipelinerl.rollouts import (
    BaseMetrics,
    RolloutResult,
    TrainingText,
    cached_tokenizer_token_ids,
)


MASKED_TOKEN_ID = -100
MASKED_LOGPROB = 0.0
_TOKEN_FIELDS = {
    "prompt_token_ids",
    "generation_token_ids",
    "generation_log_probs",
}
_PROVENANCE_FIELDS = {
    "model_version_start",
    "model_version_end",
    "policy_endpoint",
}
_CALL_FIELDS = _TOKEN_FIELDS | _PROVENANCE_FIELDS
_FINISHED_REASONS = frozenset({"agent_stop", "user_stop"})
_TRAIN_ZERO_REASONS = frozenset({"max_steps", "context_window_exceeded"})
_CONDITIONAL_ZERO_REASONS = frozenset(
    {"empty_tool_calls_and_content", "agent_error"}
)
_SEMANTIC_DROP_REASONS = frozenset({"user_error", "empty_user_message"})
_FATAL_TERMINATION_REASONS = frozenset(
    {"timeout", "infrastructure_error", "unexpected_error"}
)
_KNOWN_TERMINATION_REASONS = (
    _FINISHED_REASONS
    | _TRAIN_ZERO_REASONS
    | _CONDITIONAL_ZERO_REASONS
    | _SEMANTIC_DROP_REASONS
    | _FATAL_TERMINATION_REASONS
    | {"too_many_errors"}
)
_TAU2_GYM_CLIENTS: dict[tuple[str, tuple[str, ...]], Tau2GymClient] = {}


class Tau2TrajectoryError(ValueError):
    """Raised when a Gym response cannot produce one exact TITO trajectory."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


class Tau2TerminationContractError(RuntimeError):
    """Raised when a completed Gym response violates Tau2 termination semantics."""


class Tau2Metrics(BaseMetrics):
    boundary_failure: bool = False


@dataclass(frozen=True)
class Tau2PolicyCall:
    prompt_token_ids: list[int]
    generation_token_ids: list[int]
    generation_log_probs: list[float]
    model_version_start: int
    model_version_end: int
    policy_endpoint: str

    def provenance(
        self,
        call_index: int,
        *,
        token_start: int,
        token_end: int,
    ) -> dict[str, Any]:
        return {
            "call_index": call_index,
            "version_start": self.model_version_start,
            "version_end": self.model_version_end,
            "endpoint": self.policy_endpoint,
            "token_start": token_start,
            "token_end": token_end,
        }


def _validated_token_ids(value: Any, field: str) -> list[int]:
    if not isinstance(value, list) or not all(isinstance(token_id, int) for token_id in value):
        raise Tau2TrajectoryError(
            f"invalid_{field}",
            f"Tau2 response has invalid {field}",
        )
    return list(value)


def _validated_logprobs(value: Any) -> list[float]:
    if not isinstance(value, list) or not all(
        isinstance(logprob, (int, float)) and math.isfinite(logprob) for logprob in value
    ):
        raise Tau2TrajectoryError(
            "invalid_generation_log_probs",
            "Tau2 response has invalid generation_log_probs",
        )
    return [float(logprob) for logprob in value]


def _validated_model_version(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise Tau2TrajectoryError(
            f"invalid_{field}",
            f"Tau2 response has invalid {field}",
        )
    return value


def _extract_policy_calls(
    run_response: Tau2RunResponse,
    *,
    expected_policy_endpoint: str,
) -> list[Tau2PolicyCall]:
    response_inputs = run_response.responses_create_params.get("input")
    if not isinstance(response_inputs, list):
        raise Tau2TrajectoryError(
            "missing_response_input",
            "Tau2 response is missing responses_create_params.input",
        )
    for item in response_inputs:
        if not isinstance(item, Mapping):
            continue
        if item.get("role") == "assistant" and _CALL_FIELDS.intersection(item):
            raise Tau2TrajectoryError(
                "seeded_assistant_capture",
                "Seeded Tau2 assistant input must not carry policy token labels or provenance",
            )

    output = run_response.response.get("output")
    if not isinstance(output, list):
        raise Tau2TrajectoryError(
            "missing_response_output",
            "Tau2 response is missing response.output",
        )

    expected_endpoint = normalize_openai_base_url(expected_policy_endpoint)
    calls = []
    for item in output:
        if not isinstance(item, Mapping):
            continue
        present_fields = _CALL_FIELDS.intersection(item)
        if not present_fields:
            continue
        if present_fields != _CALL_FIELDS:
            raise Tau2TrajectoryError(
                "incomplete_policy_capture",
                f"Tau2 policy output has incomplete capture: {sorted(present_fields)}",
            )

        prompt_ids = _validated_token_ids(item["prompt_token_ids"], "prompt_token_ids")
        generation_ids = _validated_token_ids(item["generation_token_ids"], "generation_token_ids")
        generation_logprobs = _validated_logprobs(item["generation_log_probs"])
        version_start = _validated_model_version(
            item["model_version_start"],
            "model_version_start",
        )
        version_end = _validated_model_version(
            item["model_version_end"],
            "model_version_end",
        )
        if version_end < version_start:
            raise Tau2TrajectoryError(
                "model_version_decreased",
                f"Tau2 policy call model version decreased: {version_start} -> {version_end}",
            )
        endpoint_value = item["policy_endpoint"]
        if not isinstance(endpoint_value, str) or not endpoint_value:
            raise Tau2TrajectoryError(
                "invalid_policy_endpoint",
                "Tau2 response has invalid policy_endpoint",
            )
        policy_endpoint = normalize_openai_base_url(endpoint_value)
        if policy_endpoint != expected_endpoint:
            raise Tau2TrajectoryError(
                "policy_endpoint_mismatch",
                f"Tau2 policy call endpoint {policy_endpoint} does not match {expected_endpoint}",
            )
        if not prompt_ids:
            raise Tau2TrajectoryError(
                "empty_prompt_capture",
                "Tau2 policy output has an empty captured prompt",
            )
        if not generation_ids:
            raise Tau2TrajectoryError(
                "empty_generation_capture",
                "Tau2 policy output has no generated tokens",
            )
        if len(generation_ids) != len(generation_logprobs):
            raise Tau2TrajectoryError(
                "generation_logprob_length_mismatch",
                "Tau2 generation token/logprob length mismatch: "
                f"{len(generation_ids)} != {len(generation_logprobs)}",
            )
        calls.append(
            Tau2PolicyCall(
                prompt_token_ids=prompt_ids,
                generation_token_ids=generation_ids,
                generation_log_probs=generation_logprobs,
                model_version_start=version_start,
                model_version_end=version_end,
                policy_endpoint=policy_endpoint,
            )
        )

    if run_response.num_agent_calls is not None:
        expected_calls = max(run_response.num_agent_calls - 1, 0)
        if len(calls) != expected_calls:
            raise Tau2TrajectoryError(
                "policy_call_count_mismatch",
                f"Tau2 captured {len(calls)} policy calls, expected {expected_calls}",
            )
    if not calls:
        raise Tau2TrajectoryError(
            "no_policy_calls",
            "Tau2 rollout has no captured policy calls",
        )
    return calls


def serialized_training_text_size(training_text: TrainingText) -> int:
    return len(pickle.dumps(training_text))


def build_tau2_training_text(
    run_response: Tau2RunResponse,
    tokenizer: PreTrainedTokenizerBase,
    *,
    expected_policy_endpoint: str,
    max_sequence_length: int,
    shared_memory_entry_size: int,
) -> TrainingText:
    """Merge exact response-side token captures into one all-turn trajectory.

    n_predicted=0 makes prompt_text empty and output_text the full merged text;
    metadata.model_calls token spans are the canonical assistant-only view.
    """
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    if shared_memory_entry_size <= 0:
        raise ValueError("shared_memory_entry_size must be positive")

    calls = _extract_policy_calls(
        run_response,
        expected_policy_endpoint=expected_policy_endpoint,
    )
    input_ids: list[int] = []
    labels: list[int] = []
    old_logprobs: list[float] = []
    model_calls = []

    for call_index, call in enumerate(calls):
        prompt_ids = call.prompt_token_ids
        generation_ids = call.generation_token_ids
        if call_index == 0:
            input_ids.extend(prompt_ids)
            labels.extend([MASKED_TOKEN_ID] * len(prompt_ids))
            old_logprobs.extend([MASKED_LOGPROB] * len(prompt_ids))
        else:
            if len(prompt_ids) < len(input_ids) or prompt_ids[: len(input_ids)] != input_ids:
                raise Tau2TrajectoryError(
                    "prompt_not_prefix_extension",
                    f"Tau2 prompt at policy call {call_index} is not an exact prefix extension",
                )
            prompt_extension = prompt_ids[len(input_ids) :]
            input_ids.extend(prompt_extension)
            labels.extend([MASKED_TOKEN_ID] * len(prompt_extension))
            old_logprobs.extend([MASKED_LOGPROB] * len(prompt_extension))

        span_start = len(input_ids)
        input_ids.extend(generation_ids)
        labels.extend(generation_ids)
        old_logprobs.extend(call.generation_log_probs)
        span_end = len(input_ids)

        model_calls.append(
            call.provenance(
                call_index,
                token_start=span_start,
                token_end=span_end,
            )
        )

        if len(input_ids) > max_sequence_length:
            raise Tau2TrajectoryError(
                "sequence_length_exceeded",
                f"Tau2 trajectory length {len(input_ids)} exceeds sequence limit {max_sequence_length}",
            )

    if not (len(input_ids) == len(labels) == len(old_logprobs)):
        raise AssertionError("Tau2 trajectory token fields are not aligned")

    valid_token_ids = cached_tokenizer_token_ids(tokenizer)
    invalid_token_ids = sorted(set(input_ids) - valid_token_ids)
    if invalid_token_ids:
        raise Tau2TrajectoryError(
            "oov_token_ids",
            f"Tau2 trajectory contains out-of-vocabulary token IDs: {invalid_token_ids}",
        )

    termination_reason = run_response.result.get("termination_reason")
    output_tokens = sum(label != MASKED_TOKEN_ID for label in labels)
    boundary_versions = [
        version
        for call in calls
        for version in (call.model_version_start, call.model_version_end)
    ]
    model_version = min(call.model_version_start for call in calls)
    version_min = min(boundary_versions)
    version_max = max(boundary_versions)
    training_text = TrainingText(
        text=tokenizer.decode(input_ids, skip_special_tokens=False),
        n_predicted=0,
        reward=run_response.reward,
        logprobs=old_logprobs,
        ref_logprobs=list(old_logprobs),
        input_ids=input_ids,
        labels=labels,
        finished=termination_reason in _FINISHED_REASONS,
        prompt_tokens=len(input_ids) - output_tokens,
        output_tokens=output_tokens,
        metadata={
            "num_policy_calls": len(calls),
            "termination_reason": termination_reason,
            "model_version": model_version,
            "model_version_min": version_min,
            "model_version_max": version_max,
            "model_version_spread": version_max - version_min,
            "model_calls": model_calls,
        },
    )
    serialized_size = serialized_training_text_size(training_text)
    if serialized_size > shared_memory_entry_size:
        raise Tau2TrajectoryError(
            "serialized_size_exceeded",
            f"Tau2 trajectory serialized size {serialized_size} exceeds shared-memory "
            f"entry limit {shared_memory_entry_size}",
        )
    return training_text


def _get_tau2_gym_client(cfg: DictConfig) -> Tau2GymClient:
    settings_payload = OmegaConf.to_container(cfg.tau2_gym, resolve=True)
    settings = Tau2GymSettings.model_validate(settings_payload)
    actor_llm_urls = tuple(str(cfg.me.llm_urls).split("+"))
    key = (settings.model_dump_json(), actor_llm_urls)
    if key not in _TAU2_GYM_CLIENTS:
        _TAU2_GYM_CLIENTS[key] = Tau2GymClient(settings, actor_llm_urls)
    return _TAU2_GYM_CLIENTS[key]


def _full_tau2_actions(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions = []
    messages = result.get("messages")
    if not isinstance(messages, list):
        return actions
    for message in messages:
        if not isinstance(message, Mapping) or not message.get("tool_calls"):
            continue
        actions.append(
            {
                "turn_idx": message.get("turn_idx"),
                "tool_calls": message["tool_calls"],
            }
        )
    return actions


def _audit_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _validated_termination_reason(result: Mapping[str, Any]) -> str:
    termination_reason = result.get("termination_reason")
    if type(termination_reason) is not str or termination_reason not in _KNOWN_TERMINATION_REASONS:
        raise Tau2TerminationContractError(
            f"Tau2 returned missing or unknown termination reason {termination_reason!r}"
        )
    if termination_reason in _FATAL_TERMINATION_REASONS:
        raise Tau2TerminationContractError(
            f"Tau2 returned fatal termination reason {termination_reason}"
        )
    return termination_reason


def _too_many_errors_evidence(result: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    tool_results: list[Any] = []
    malformed_tool_results = 0
    messages = result.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, Mapping) or message.get("role") != "tool":
                continue
            if "tool_messages" not in message:
                tool_results.append(message)
                continue
            nested = message["tool_messages"]
            if not isinstance(nested, list):
                malformed_tool_results += 1
                continue
            tool_results.extend(nested)

    total_errors = 0
    assistant_errors = 0
    user_errors = 0
    unattributed_errors = 0
    error_results = []
    for tool_result in tool_results:
        if not isinstance(tool_result, Mapping):
            malformed_tool_results += 1
            continue
        error = tool_result.get("error")
        requestor = tool_result.get("requestor")
        valid = type(error) is bool and requestor in {"assistant", "user"}
        if not valid:
            malformed_tool_results += 1
        if error is not True:
            continue
        total_errors += 1
        if requestor == "assistant":
            assistant_errors += 1
        elif requestor == "user":
            user_errors += 1
        else:
            unattributed_errors += 1
        error_results.append(
            {
                "id": tool_result.get("id"),
                "turn_idx": tool_result.get("turn_idx"),
                "requestor": requestor,
            }
        )

    evidence = {
        "tool_results_total": len(tool_results),
        "tool_errors_total": total_errors,
        "assistant_tool_errors": assistant_errors,
        "user_tool_errors": user_errors,
        "unattributed_tool_errors": unattributed_errors,
        "malformed_tool_results": malformed_tool_results,
        "error_results": error_results,
    }
    trusted = (
        total_errors > 0
        and malformed_tool_results == 0
        and assistant_errors == total_errors
    )
    return evidence, trusted


def _base_tau2_audit(
    problem: Mapping[str, Any],
    run_response: Any,
    *,
    policy_endpoint: str,
    evaluator_reward: float | None,
) -> dict[str, Any]:
    termination_reason = _validated_termination_reason(run_response.result)
    task = problem.get("task")
    task_id = problem.get("task_id")
    if task_id is None and isinstance(task, Mapping):
        task_id = task.get("id")
    verifier_result = run_response.result.get("reward_info", run_response.result)
    auxiliary_model_calls = [
        _audit_value(call)
        for call in getattr(run_response, "auxiliary_model_calls", [])
    ]
    judge_sampling = getattr(run_response, "judge_sampling", None)
    return {
        "task_id": task_id,
        "dataset_name": problem.get("dataset"),
        "domain": problem.get("domain", "tau2"),
        "policy_endpoint": policy_endpoint,
        "admission_policy_endpoint": policy_endpoint,
        "reward": evaluator_reward,
        "evaluator_reward": evaluator_reward,
        "training_reward": None,
        "reward_available": evaluator_reward is not None,
        "verifier_result": verifier_result,
        "actions": _full_tau2_actions(run_response.result),
        "submitted": termination_reason in _FINISHED_REASONS,
        "terminated": True,
        "finished": termination_reason in _FINISHED_REASONS,
        "termination_reason": termination_reason,
        "stop_reason": termination_reason,
        "disposition": None,
        "labelled_policy_tokens": None,
        "entered_training": False,
        "drop_reason": None,
        "model_calls": [],
        "auxiliary_model_calls": auxiliary_model_calls,
        "judge_sampling": (
            _audit_value(judge_sampling)
            if judge_sampling is not None
            else None
        ),
    }


def _update_capture_audit(audit: dict[str, Any], training_text: TrainingText) -> None:
    metadata = training_text.metadata
    audit.update(
        {
            "model_version": metadata["model_version"],
            "model_version_min": metadata["model_version_min"],
            "model_version_max": metadata["model_version_max"],
            "model_version_spread": metadata["model_version_spread"],
            "model_calls": metadata["model_calls"],
            "prompt_tokens": training_text.prompt_tokens,
            "output_tokens": training_text.output_tokens,
            "response_tokens": training_text.output_tokens,
            "labeled_tokens": training_text.output_tokens,
            "labelled_policy_tokens": training_text.output_tokens,
            "sequence_tokens": len(training_text.input_ids),
        }
    )


def _semantic_boundary_result(
    *,
    reason: str,
    detail: str,
    audit: dict[str, Any],
    latency: float,
    problem: Mapping[str, Any],
    labelled_policy_tokens: int,
    model_version: int | None = None,
    failure_fields: Mapping[str, Any] | None = None,
) -> RolloutResult:
    failure = {"reason": reason, "detail": detail}
    if failure_fields:
        failure.update(failure_fields)
    audit.update(
        {
            "training_reward": None,
            "reward_available": False,
            "submitted": False,
            "finished": False,
            "disposition": "drop_group",
            "labelled_policy_tokens": labelled_policy_tokens,
            "drop_reason": reason,
            "boundary_failure": failure,
        }
    )
    return RolloutResult(
        training_texts=[],
        metrics=None,
        latency=latency,
        model_version=model_version,
        dataset_name=problem.get("dataset"),
        domain=problem.get("domain", "tau2"),
        audit=audit,
        atomic_group=True,
    )


async def generate_tau2_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    policy_endpoint = normalize_openai_base_url(llm.get_base_url())
    start_time = time.perf_counter()
    client = _get_tau2_gym_client(cfg)

    def build_capture(
        response: Tau2RunResponse,
        *,
        shared_memory_entry_size: int,
    ) -> TrainingText:
        llm.load_tokenizer()
        assert llm.tokenizer is not None
        return build_tau2_training_text(
            response,
            llm.tokenizer,
            expected_policy_endpoint=policy_endpoint,
            max_sequence_length=int(cfg.finetune.seq_length),
            shared_memory_entry_size=shared_memory_entry_size,
        )

    try:
        run_response = await client.run(
            policy_endpoint,
            problem,
            session,
        )
    except Tau2BoundaryFailure as exc:
        latency = time.perf_counter() - start_time
        boundary = exc.boundary
        audit = _base_tau2_audit(
            problem,
            boundary,
            policy_endpoint=policy_endpoint,
            evaluator_reward=None,
        )
        captured_response = Tau2RunResponse(
            reward=0.0,
            responses_create_params=boundary.responses_create_params,
            response=boundary.response,
            result=boundary.result,
            duration=boundary.duration,
            num_steps=boundary.num_steps,
            num_agent_calls=boundary.num_agent_calls,
        )
        try:
            training_text = build_capture(
                captured_response,
                shared_memory_entry_size=2**63 - 1,
            )
        except Tau2TrajectoryError as capture_error:
            raise Tau2TerminationContractError(
                "Tau2 judge-invalid boundary has invalid policy capture"
            ) from capture_error
        _update_capture_audit(audit, training_text)
        return _semantic_boundary_result(
            reason=boundary.reason,
            detail=boundary.diagnostic,
            audit=audit,
            latency=latency,
            problem=problem,
            labelled_policy_tokens=training_text.output_tokens,
            model_version=training_text.metadata["model_version"],
            failure_fields={
                "producer_stage": boundary.producer_stage,
                "detail_code": boundary.detail_code,
                "attempt_count": boundary.attempt_count,
                "diagnostic": boundary.diagnostic,
            },
        )
    except ValidationError as exc:
        if any(
            "Tau2 response has no termination reason" in error["msg"]
            for error in exc.errors(include_input=False)
        ):
            raise Tau2TerminationContractError(
                "Tau2 completed response is missing termination_reason"
            ) from exc
        latency = time.perf_counter() - start_time
        task = problem.get("task")
        task_id = problem.get("task_id")
        if task_id is None and isinstance(task, Mapping):
            task_id = task.get("id")
        validation_errors = [
            {
                "type": error["type"],
                "location": [str(part) for part in error["loc"]],
                "message": error["msg"],
            }
            for error in exc.errors(include_input=False)
        ]
        reason = "malformed_gym_response"
        audit = {
            "task_id": task_id,
            "dataset_name": problem.get("dataset"),
            "domain": problem.get("domain", "tau2"),
            "policy_endpoint": policy_endpoint,
            "admission_policy_endpoint": policy_endpoint,
            "reward": None,
            "evaluator_reward": None,
            "training_reward": None,
            "reward_available": False,
            "verifier_result": None,
            "actions": [],
            "submitted": None,
            "terminated": None,
            "finished": None,
            "termination_reason": None,
            "stop_reason": None,
            "disposition": "drop_group",
            "labelled_policy_tokens": None,
            "entered_training": False,
            "drop_reason": reason,
            "model_calls": [],
            "auxiliary_model_calls": [],
            "judge_sampling": None,
            "boundary_failure": {
                "reason": reason,
                "detail": "Tau2 Gym /run response failed schema validation",
                "validation_errors": validation_errors,
            },
        }
        return RolloutResult(
            training_texts=[],
            metrics=None,
            latency=latency,
            dataset_name=problem.get("dataset"),
            domain=problem.get("domain", "tau2"),
            audit=audit,
            atomic_group=True,
        )

    latency = time.perf_counter() - start_time
    evaluator_reward = run_response.reward
    if type(evaluator_reward) not in (int, float) or not math.isfinite(evaluator_reward):
        raise Tau2TerminationContractError(
            f"Tau2 returned invalid evaluator reward {evaluator_reward!r}"
        )
    evaluator_reward = float(evaluator_reward)
    audit = _base_tau2_audit(
        problem,
        run_response,
        policy_endpoint=policy_endpoint,
        evaluator_reward=evaluator_reward,
    )
    termination_reason = audit["termination_reason"]
    metrics = Tau2Metrics(
        reward=evaluator_reward,
        success=evaluator_reward == 1.0,
        no_error=True,
        no_answer=False,
    )
    try:
        training_text = build_capture(
            run_response,
            shared_memory_entry_size=int(cfg.actor.shared_memory_entry_size),
        )
    except Tau2TrajectoryError as exc:
        if (
            termination_reason in _CONDITIONAL_ZERO_REASONS
            and exc.reason == "no_policy_calls"
        ):
            reason = f"termination_{termination_reason}_no_labeled_span"
            return _semantic_boundary_result(
                reason=reason,
                detail=(
                    f"Tau2 {termination_reason} response has no labelled policy span"
                ),
                audit=audit,
                latency=latency,
                problem=problem,
                labelled_policy_tokens=0,
            )
        audit.update(
            {
                "training_reward": None,
                "reward_available": True,
                "disposition": "drop_technical_boundary",
                "labelled_policy_tokens": None,
                "drop_reason": exc.reason,
                "boundary_failure": {
                    "reason": exc.reason,
                    "detail": str(exc),
                },
            }
        )
        metrics.no_error = False
        metrics.boundary_failure = True
        return RolloutResult(
            training_texts=[],
            metrics=metrics,
            latency=latency,
            dataset_name=problem.get("dataset"),
            domain=problem.get("domain", "tau2"),
            audit=audit,
            atomic_group=True,
        )

    _update_capture_audit(audit, training_text)
    model_version = training_text.metadata["model_version"]

    if termination_reason in _SEMANTIC_DROP_REASONS:
        reason = f"termination_{termination_reason}"
        return _semantic_boundary_result(
            reason=reason,
            detail=f"Tau2 termination {termination_reason} is not reward-valid",
            audit=audit,
            latency=latency,
            problem=problem,
            labelled_policy_tokens=training_text.output_tokens,
            model_version=model_version,
        )

    if termination_reason == "too_many_errors":
        tool_error_evidence, trusted = _too_many_errors_evidence(run_response.result)
        audit["tool_error_evidence"] = tool_error_evidence
        if not trusted:
            return _semantic_boundary_result(
                reason="termination_too_many_errors_untrusted_provenance",
                detail="Tau2 too_many_errors has untrusted tool-error provenance",
                audit=audit,
                latency=latency,
                problem=problem,
                labelled_policy_tokens=training_text.output_tokens,
                model_version=model_version,
            )

    if termination_reason in _FINISHED_REASONS:
        training_reward = evaluator_reward
        finished = True
        disposition = "train_evaluator_reward"
    else:
        training_reward = 0.0
        finished = False
        disposition = "train_zero"

    training_text.reward = training_reward
    training_text.finished = finished
    audit.update(
        {
            "training_reward": training_reward,
            "reward_available": True,
            "submitted": finished,
            "finished": finished,
            "disposition": disposition,
            "labelled_policy_tokens": training_text.output_tokens,
        }
    )
    metrics.reward = training_reward
    metrics.success = finished and training_reward == 1.0
    return RolloutResult(
        training_texts=[training_text],
        metrics=metrics,
        latency=latency,
        model_version=model_version,
        dataset_name=problem.get("dataset"),
        domain=problem.get("domain", "tau2"),
        audit=audit,
        atomic_group=True,
    )
