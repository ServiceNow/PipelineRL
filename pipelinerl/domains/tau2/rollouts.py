import math
import pickle
from collections.abc import Mapping
from typing import Any

from transformers import PreTrainedTokenizerBase

from pipelinerl.domains.tau2.client import Tau2RunResponse
from pipelinerl.rollouts import TrainingText


MASKED_TOKEN_ID = -100
MASKED_LOGPROB = 0.0
_TOKEN_FIELDS = {
    "prompt_token_ids",
    "generation_token_ids",
    "generation_log_probs",
}
_INCOMPLETE_REASONS = {"context_length", "max_steps"}


class Tau2TrajectoryError(ValueError):
    """Raised when a Gym response cannot produce one exact TITO trajectory."""


def _validated_token_ids(value: Any, field: str) -> list[int]:
    if not isinstance(value, list) or not all(isinstance(token_id, int) for token_id in value):
        raise Tau2TrajectoryError(f"Tau2 response has invalid {field}")
    return list(value)


def _validated_logprobs(value: Any) -> list[float]:
    if not isinstance(value, list) or not all(
        isinstance(logprob, (int, float)) and math.isfinite(logprob) for logprob in value
    ):
        raise Tau2TrajectoryError("Tau2 response has invalid generation_log_probs")
    return [float(logprob) for logprob in value]


def _extract_policy_calls(run_response: Tau2RunResponse) -> list[tuple[list[int], list[int], list[float]]]:
    response_inputs = run_response.responses_create_params.get("input")
    if not isinstance(response_inputs, list):
        raise Tau2TrajectoryError("Tau2 response is missing responses_create_params.input")
    for item in response_inputs:
        if not isinstance(item, Mapping):
            continue
        if item.get("role") == "assistant" and _TOKEN_FIELDS.intersection(item):
            raise Tau2TrajectoryError("Seeded Tau2 assistant input must not carry policy token labels")

    output = run_response.response.get("output")
    if not isinstance(output, list):
        raise Tau2TrajectoryError("Tau2 response is missing response.output")

    calls = []
    for item in output:
        if not isinstance(item, Mapping):
            continue
        present_fields = _TOKEN_FIELDS.intersection(item)
        if not present_fields:
            continue
        if present_fields != _TOKEN_FIELDS:
            raise Tau2TrajectoryError(
                f"Tau2 policy output has incomplete token capture: {sorted(present_fields)}"
            )

        prompt_ids = _validated_token_ids(item["prompt_token_ids"], "prompt_token_ids")
        generation_ids = _validated_token_ids(item["generation_token_ids"], "generation_token_ids")
        generation_logprobs = _validated_logprobs(item["generation_log_probs"])
        if not prompt_ids:
            raise Tau2TrajectoryError("Tau2 policy output has an empty captured prompt")
        if not generation_ids:
            raise Tau2TrajectoryError("Tau2 policy output has no generated tokens")
        if len(generation_ids) != len(generation_logprobs):
            raise Tau2TrajectoryError(
                "Tau2 generation token/logprob length mismatch: "
                f"{len(generation_ids)} != {len(generation_logprobs)}"
            )
        calls.append((prompt_ids, generation_ids, generation_logprobs))

    if run_response.num_agent_calls is not None:
        expected_calls = max(run_response.num_agent_calls - 1, 0)
        if len(calls) != expected_calls:
            raise Tau2TrajectoryError(
                f"Tau2 captured {len(calls)} policy calls, expected {expected_calls}"
            )
    if not calls:
        raise Tau2TrajectoryError("Tau2 rollout has no captured policy calls")
    return calls


def serialized_training_text_size(training_text: TrainingText) -> int:
    return len(pickle.dumps(training_text))


def build_tau2_training_text(
    run_response: Tau2RunResponse,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_sequence_length: int,
    shared_memory_entry_size: int,
) -> TrainingText:
    """Merge exact response-side token captures into one all-turn trajectory."""
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    if shared_memory_entry_size <= 0:
        raise ValueError("shared_memory_entry_size must be positive")

    calls = _extract_policy_calls(run_response)
    input_ids: list[int] = []
    labels: list[int] = []
    old_logprobs: list[float] = []

    for call_index, (prompt_ids, generation_ids, generation_logprobs) in enumerate(calls):
        if call_index == 0:
            input_ids.extend(prompt_ids)
            labels.extend([MASKED_TOKEN_ID] * len(prompt_ids))
            old_logprobs.extend([MASKED_LOGPROB] * len(prompt_ids))
        else:
            if len(prompt_ids) < len(input_ids) or prompt_ids[: len(input_ids)] != input_ids:
                raise Tau2TrajectoryError(
                    f"Tau2 prompt at policy call {call_index} is not an exact prefix extension"
                )
            prompt_extension = prompt_ids[len(input_ids) :]
            input_ids.extend(prompt_extension)
            labels.extend([MASKED_TOKEN_ID] * len(prompt_extension))
            old_logprobs.extend([MASKED_LOGPROB] * len(prompt_extension))

        input_ids.extend(generation_ids)
        labels.extend(generation_ids)
        old_logprobs.extend(generation_logprobs)

        if len(input_ids) > max_sequence_length:
            raise Tau2TrajectoryError(
                f"Tau2 trajectory length {len(input_ids)} exceeds sequence limit {max_sequence_length}"
            )

    if not (len(input_ids) == len(labels) == len(old_logprobs)):
        raise AssertionError("Tau2 trajectory token fields are not aligned")

    termination_reason = run_response.result.get("termination_reason")
    output_tokens = sum(label != MASKED_TOKEN_ID for label in labels)
    training_text = TrainingText(
        text=tokenizer.decode(input_ids, skip_special_tokens=False),
        n_predicted=0,
        reward=run_response.reward,
        logprobs=old_logprobs,
        ref_logprobs=list(old_logprobs),
        input_ids=input_ids,
        labels=labels,
        finished=termination_reason not in _INCOMPLETE_REASONS,
        prompt_tokens=len(input_ids) - output_tokens,
        output_tokens=output_tokens,
        metadata={
            "num_policy_calls": len(calls),
            "termination_reason": termination_reason,
        },
    )
    serialized_size = serialized_training_text_size(training_text)
    if serialized_size > shared_memory_entry_size:
        raise Tau2TrajectoryError(
            f"Tau2 trajectory serialized size {serialized_size} exceeds shared-memory "
            f"entry limit {shared_memory_entry_size}"
        )
    return training_text
