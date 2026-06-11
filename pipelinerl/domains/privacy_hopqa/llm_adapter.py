"""Async LLM adapter for the simplified privacy_hopqa domain."""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import aiohttp

from pipelinerl.async_llm import llm_async_generate, make_training_text as make_rl_training_text
from pipelinerl.llm import LLMCall, Prompt, TrainableLLM

logger = logging.getLogger(__name__)

# Planning-only RL trains the controllable retrieval policy. The resolver still
# runs during rollouts, but its reward is too entangled with reader quality and
# answer-variant noise to train in the current Privacy HopQA experiments.
PLANNING_LOG_NAMES = {"hop_plan", "doc_choose"}


class PrivacyHopQALLMInfrastructureError(RuntimeError):
    """Inference service failure that should be handled by PipelineRL, not the agent."""


class PrivacyHopQAContextLengthError(ValueError):
    """Prompt is too large for the configured rollout model context."""


def is_llm_infrastructure_error(exc: BaseException) -> bool:
    """Return true for rollout-LLM transport/server failures, not bad model outputs."""
    if isinstance(exc, PrivacyHopQALLMInfrastructureError):
        return True
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status >= 500
    if isinstance(
        exc,
        (
            aiohttp.ClientConnectionError,
            aiohttp.ClientPayloadError,
            aiohttp.ServerTimeoutError,
            asyncio.TimeoutError,
            TimeoutError,
        ),
    ):
        return True
    text = str(exc).lower()
    return "asyncenginedeaderror" in text or "engine background task failed" in text


@dataclass
class CapturedLLMCall:
    llm_call: LLMCall
    log_name: str
    requested_model: str | None
    hop_number: int | None = None
    iteration: int | None = None
    log_metadata: dict[str, Any] | None = None


class PrivacyHopQALLMAdapter:
    def __init__(
        self,
        llm: TrainableLLM,
        session: aiohttp.ClientSession,
        capture_mode: str = "all_calls",
        rollout_id: str = "",
        task_id: str = "",
        chain_id: str = "",
    ):
        if capture_mode not in {"all_calls", "planning_only"}:
            raise ValueError(f"Unsupported capture_mode: {capture_mode}")

        self.llm = llm
        self.session = session
        self.capture_mode = capture_mode
        self.rollout_id = rollout_id
        self.task_id = task_id
        self.chain_id = chain_id
        self.captured_calls: list[CapturedLLMCall] = []
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_output_tokens = 0
        self.captured_prompt_tokens = 0
        self.captured_output_tokens = 0
        self.max_prompt_tokens_by_log_name: dict[str, int] = {}
        self.last_captured_call_index_by_key: dict[tuple[str, int | None, int | None], int] = {}
        self.llm_calls_by_hop: dict[str, int] = {}
        self.llm_calls_by_hop_stage: dict[str, dict[str, int]] = {}

    def _should_capture(self, log_name: str) -> bool:
        if self.capture_mode == "all_calls":
            return True
        return log_name in PLANNING_LOG_NAMES

    def _log_prompt_summary(
        self,
        log_name: str,
        prompt_tokens: int,
        force: bool = False,
        log_metadata: dict[str, Any] | None = None,
    ) -> None:
        previous_max = self.max_prompt_tokens_by_log_name.get(log_name, 0)
        if not force and prompt_tokens <= previous_max and prompt_tokens < 4096:
            return
        record = {
            "rollout_id": self.rollout_id,
            "task_id": self.task_id,
            "chain_id": self.chain_id,
            "log_name": log_name,
            "prompt_tokens": int(prompt_tokens),
        }
        if log_metadata:
            record["metadata"] = log_metadata
        logger.info("privacy_hopqa prompt summary: %s", json.dumps(record, sort_keys=True))
        self.max_prompt_tokens_by_log_name[log_name] = max(previous_max, prompt_tokens)

    def _log_call_summary(
        self,
        log_name: str,
        prompt_tokens: int,
        output_tokens: int,
        requested_model: str | None,
        budget: dict[str, Any] | None = None,
        log_metadata: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "rollout_id": self.rollout_id,
            "task_id": self.task_id,
            "chain_id": self.chain_id,
            "log_name": log_name,
            "prompt_tokens": int(prompt_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(prompt_tokens) + int(output_tokens),
            "capture_mode": self.capture_mode,
            "captured": self._should_capture(log_name),
            "requested_model": requested_model or self.llm.model_name,
        }
        usage = getattr(getattr(self, "_last_llm_call", None), "llm_info", {}).get("usage")
        if isinstance(usage, dict):
            record["provider_usage"] = usage
        if budget:
            record["budget"] = budget
        if log_metadata:
            record["metadata"] = log_metadata
        logger.info("privacy_hopqa llm call summary: %s", json.dumps(record, sort_keys=True))

    def _base_log_record(
        self,
        log_name: str,
        hop_number: int | None,
        iteration: int | None,
        log_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "rollout_id": self.rollout_id,
            "task_id": self.task_id,
            "chain_id": self.chain_id,
            "log_name": log_name,
        }
        if hop_number is not None:
            record["hop_number"] = int(hop_number)
        if iteration is not None:
            record["iteration"] = int(iteration)
        if log_metadata:
            record["metadata"] = log_metadata
        return record

    async def _run_generation(
        self,
        prompt: Prompt,
        requested_model: str | None,
        parameters_override: dict[str, Any] | None = None,
    ) -> LLMCall:
        if requested_model and requested_model != self.llm.model_name:
            raise ValueError(
                "privacy_hopqa routes domain calls through the rollout LLM. "
                f"Requested model '{requested_model}' does not match rollout model '{self.llm.model_name}'."
            )
        # Per-call sampling overrides (e.g. a token budget) ride on a copy of the
        # rollout LLM, so the shared llm_async_generate keeps its upstream signature.
        llm = self.llm
        if parameters_override:
            llm = llm.model_copy(update={"parameters": {**llm.parameters, **parameters_override}})
        try:
            return await llm_async_generate(
                llm,
                prompt,
                self.session,
            )
        except Exception as exc:
            # Infrastructure errors are retried or failed by the actor; prompt
            # and parsing errors stay in the domain rollout path.
            if is_llm_infrastructure_error(exc):
                # Let PipelineRL's actor fail-fast path handle dead vLLM shards.
                raise PrivacyHopQALLMInfrastructureError(
                    f"LLM infrastructure failure at {self.llm.base_url}: {exc}"
                ) from exc
            raise

    async def generate_text(
        self,
        prompt: str,
        model: str | None = None,
        log_name: str = "llm",
        max_tokens: int | None = None,
        max_context_tokens: int | None = None,
        context_margin_tokens: int = 0,
        hop_number: int | None = None,
        iteration: int | None = None,
        log_metadata: dict[str, Any] | None = None,
    ) -> str:
        prompt_text = str(prompt)
        prompt_obj = Prompt(messages=[{"role": "user", "content": prompt_text}])
        estimated_prompt_tokens = max(0, self.llm.count_tokens(prompt_obj.messages))
        self._log_prompt_summary(
            log_name=log_name,
            prompt_tokens=estimated_prompt_tokens,
            log_metadata=log_metadata,
        )
        parameters_override: dict[str, Any] = {}
        budget_record: dict[str, Any] | None = None
        if max_tokens is not None:
            requested_max_tokens = max(1, int(max_tokens))
            safe_max_tokens = requested_max_tokens
            budget_record = {
                "requested_max_tokens": requested_max_tokens,
                "safe_max_tokens": safe_max_tokens,
                "estimated_prompt_tokens": estimated_prompt_tokens,
            }
            if max_context_tokens is not None:
                context_limit = int(max_context_tokens)
                margin = int(context_margin_tokens)
                remaining_tokens = context_limit - estimated_prompt_tokens - margin
                budget_record.update(
                    {
                        "context_limit": context_limit,
                        "context_margin_tokens": margin,
                        "remaining_tokens": remaining_tokens,
                    }
                )
                if remaining_tokens <= 0:
                    record = self._base_log_record(
                        log_name=log_name,
                        hop_number=hop_number,
                        iteration=iteration,
                        log_metadata=log_metadata,
                    )
                    record["budget"] = budget_record
                    logger.warning(
                        "privacy_hopqa skipped %s generation because prompt_tokens=%s exceeds context_limit=%s margin=%s",
                        log_name,
                        estimated_prompt_tokens,
                        context_limit,
                        margin,
                    )
                    logger.warning("privacy_hopqa generation skipped: %s", json.dumps(record, sort_keys=True))
                    raise PrivacyHopQAContextLengthError(
                        f"{log_name} prompt has {estimated_prompt_tokens} tokens, leaving no room in "
                        f"context limit {context_limit} with margin {margin}"
                    )
                safe_max_tokens = min(requested_max_tokens, max(1, remaining_tokens))
                budget_record["safe_max_tokens"] = safe_max_tokens
                budget_record["capped"] = safe_max_tokens < requested_max_tokens
                if safe_max_tokens < requested_max_tokens:
                    record = self._base_log_record(
                        log_name=log_name,
                        hop_number=hop_number,
                        iteration=iteration,
                        log_metadata=log_metadata,
                    )
                    record["budget"] = budget_record
                    logger.warning(
                        "privacy_hopqa capped %s max_tokens from %s to %s for prompt_tokens=%s context_limit=%s margin=%s",
                        log_name,
                        requested_max_tokens,
                        safe_max_tokens,
                        estimated_prompt_tokens,
                        context_limit,
                        margin,
                    )
                    logger.warning("privacy_hopqa generation capped: %s", json.dumps(record, sort_keys=True))
            parameters_override["max_tokens"] = safe_max_tokens

        try:
            llm_call = await self._run_generation(
                prompt_obj,
                requested_model=model,
                parameters_override=parameters_override or None,
            )
        except Exception as exc:
            # Context failures are easier to debug with the exact prompt size in
            # the rollout logs, so force one more prompt summary before re-raising.
            exc_text = str(exc).lower()
            if "maximum context length" in exc_text or "requested" in exc_text or "context" in exc_text:
                self._log_prompt_summary(
                    log_name=log_name,
                    prompt_tokens=estimated_prompt_tokens,
                    force=True,
                    log_metadata=log_metadata,
                )
            raise

        prompt_tokens = max(0, llm_call.prompt_length_tokens)
        output_tokens = max(0, llm_call.output_length_tokens)
        self._last_llm_call = llm_call
        self.total_calls += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_output_tokens += output_tokens
        if hop_number is not None:
            hop_key = str(hop_number)
            self.llm_calls_by_hop[hop_key] = self.llm_calls_by_hop.get(hop_key, 0) + 1
            stage_counts = self.llm_calls_by_hop_stage.setdefault(hop_key, {})
            stage_counts[log_name] = stage_counts.get(log_name, 0) + 1
        self._log_call_summary(
            log_name=log_name,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            requested_model=model,
            budget=budget_record,
            log_metadata=log_metadata,
        )

        if self._should_capture(log_name):
            self.captured_prompt_tokens += prompt_tokens
            self.captured_output_tokens += output_tokens
            captured_index = len(self.captured_calls)
            self.captured_calls.append(
                CapturedLLMCall(
                    llm_call=llm_call,
                    log_name=log_name,
                    requested_model=model,
                    hop_number=hop_number,
                    iteration=iteration,
                    log_metadata=dict(log_metadata or {}),
                )
            )
            self.last_captured_call_index_by_key[(log_name, hop_number, iteration)] = captured_index

        return llm_call.output.content or ""

    def update_captured_metadata(
        self,
        log_name: str,
        hop_number: int | None,
        iteration: int | None,
        metadata: dict[str, Any],
    ) -> None:
        """Attach post-generation metadata, such as normalized chooser IDs."""
        captured_index = self.last_captured_call_index_by_key.get((log_name, hop_number, iteration))
        if captured_index is None:
            return
        captured = self.captured_calls[captured_index]
        if captured.log_metadata is None:
            captured.log_metadata = {}
        captured.log_metadata.update(metadata)

    def make_training_texts(self, group_id: str | None, base_metadata: dict[str, Any]) -> list:
        traces = []
        for idx, captured in enumerate(self.captured_calls):
            hop_key = str(captured.hop_number) if captured.hop_number is not None else None
            if self.llm.collect_logprobs and captured.llm_call.logprobs:
                trace = make_rl_training_text(self.llm, captured.llm_call)
            else:
                trace = self.llm.make_training_text(
                    captured.llm_call.prompt,
                    captured.llm_call.output,
                )
                trace.prompt_tokens = max(0, captured.llm_call.prompt_length_tokens)
                trace.output_tokens = max(0, captured.llm_call.output_length_tokens)

            trace.group_id = group_id
            trace.metadata.setdefault("privacy_hopqa", {}).update(
                {
                    **base_metadata,
                    "captured_call_index": idx,
                    "log_name": captured.log_name,
                    "hop_number": captured.hop_number,
                    "iteration": captured.iteration,
                    "requested_model": captured.requested_model,
                    "capture_mode": self.capture_mode,
                    "hop_step_llm_call_count": (
                        self.llm_calls_by_hop.get(hop_key, 0) if hop_key is not None else None
                    ),
                    "hop_step_llm_call_counts_by_stage": (
                        dict(self.llm_calls_by_hop_stage.get(hop_key, {})) if hop_key is not None else {}
                    ),
                }
            )
            if captured.log_metadata:
                trace.metadata["privacy_hopqa"]["call_metadata"] = dict(captured.log_metadata)
            traces.append(trace)
        return traces
