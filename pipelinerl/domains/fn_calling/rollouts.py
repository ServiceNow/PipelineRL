"""Rollout generation for BFCL v3 function calling domain."""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional

import aiohttp
from omegaconf import DictConfig

from pipelinerl.llm import Prompt, TrainableLLM
from pipelinerl.async_llm import llm_async_generate, make_training_text
from pipelinerl.domains.math.rollouts import RewardTable, length_penalty
from pipelinerl.domains.fn_calling.verifier_api import (
    verify_fn_calling_answer,
    verify_fn_calling_answer_rpc,
)
from pipelinerl.rollouts import BaseMetrics, RolloutResult
from pipelinerl.utils import get_environment_jobs, resolve_environment_key


class FnCallingMetrics(BaseMetrics):
    """Metrics for function calling rollouts."""
    penalty: float


def _build_prompt(cfg: DictConfig, problem: Dict[str, Any]) -> Prompt:
    """Build the prompt for function calling.

    Args:
        cfg: Configuration object.
        problem: Problem dictionary with 'task' and 'extra_info'.

    Returns:
        Prompt with system message and tools.
    """
    messages: List[Dict[str, Any]] = []

    # Add system prompt if configured
    system_prompt = cfg.actor.get("system_prompt", "")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Add user message with the task
    task = problem.get("task", "")
    task_template = cfg.actor.get("task_template", "{task}")
    user_content = task_template.format(task=task)
    messages.append({"role": "user", "content": user_content})

    # Get tools from extra_info
    extra_info = problem.get("extra_info", {})
    oai_tools = extra_info.get("oai_tools", [])

    return Prompt(messages=messages, tools=oai_tools if oai_tools else None)


async def _run_verification(
    cfg: DictConfig,
    session: aiohttp.ClientSession,
    generation: str,
    reward_context: Dict[str, Any],
    tool_calls: Optional[List[Dict[str, Any]]],
    model_name: str,
) -> str:
    """Run verification either locally or via RPC.

    Args:
        cfg: Configuration object.
        session: aiohttp session for RPC calls.
        generation: Model's text generation.
        reward_context: Verification context.
        tool_calls: Structured tool calls from model output.
        model_name: Model name for verification.

    Returns:
        Answer status string.
    """
    use_local = cfg.actor.get("use_local_bfcl_verifier", False)

    if use_local:
        return verify_fn_calling_answer(
            generation=generation,
            reward_context=reward_context,
            tool_calls=tool_calls,
            model_name=model_name,
        )

    # Use RPC verification
    env_key = resolve_environment_key(cfg, default="fn_calling")
    env_jobs = get_environment_jobs(cfg, env_key)
    if not env_jobs:
        raise RuntimeError("No environment servers available for fn_calling domain")
    env_job = random.choice(env_jobs)
    if env_job.hostname is None or env_job.port is None:
        raise RuntimeError("fn_calling environment job is missing host/port information")

    return await verify_fn_calling_answer_rpc(
        session=session,
        host=env_job.hostname,
        port=env_job.port,
        generation=generation,
        reward_context=reward_context,
        tool_calls=tool_calls,
        model_name=model_name,
    )


async def generate_fn_calling_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: Dict[str, Any],
    session: aiohttp.ClientSession,
) -> RolloutResult:
    """Generate a rollout for a BFCL function calling problem.

    Args:
        cfg: Configuration object.
        llm: Language model to generate with.
        problem: Problem dictionary containing task, reward_context, extra_info.
        session: aiohttp session for RPC calls.

    Returns:
        RolloutResult with training texts and metrics.
    """
    # Build prompt with tools
    prompt = _build_prompt(cfg, problem)

    # Generate response
    start_time = time.time()
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - start_time
    assert llm_call.output.content is not None

    # Extract tool_calls from structured output if available
    tool_calls = None
    if hasattr(llm_call.output, "tool_calls") and llm_call.output.tool_calls:
        tool_calls = [
            {
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
                "id": getattr(tc, "id", None),
            }
            for tc in llm_call.output.tool_calls
        ]

    # Get reward context
    reward_context = problem.get("reward_context", {})

    # Get model name for verification
    model_name = getattr(llm, "model_name", "model")
    if hasattr(llm, "model"):
        model_name = llm.model

    # Run verification
    answer_status = await _run_verification(
        cfg=cfg,
        session=session,
        generation=llm_call.output.content,
        reward_context=reward_context,
        tool_calls=tool_calls,
        model_name=model_name,
    )

    # Calculate reward
    rewards = RewardTable(**dict(cfg.rewards))
    trace = make_training_text(llm, llm_call)

    match (answer_status, trace.finished):
        case ("wrong", False):
            reward = rewards.wrong_answer_not_finished
        case ("wrong", True):
            reward = rewards.wrong_answer_finished
        case ("no_answer", False):
            reward = rewards.no_answer_not_finished
        case ("no_answer", True):
            reward = rewards.no_answer_finished
        case ("unparsable", False):
            reward = rewards.unparsable_not_finished
        case ("unparsable", True):
            reward = rewards.unparsable_finished
        case ("correct", False):
            reward = rewards.correct_answer_not_finished
        case ("correct", True):
            reward = rewards.correct_answer_finished
        case _:
            raise ValueError(f"Unexpected fn_calling answer status '{answer_status}'")

    # Apply discount factor
    reward *= cfg.actor.discount_factor ** llm_call.output_length_tokens

    # Apply length penalty if configured
    overlong_penalty = 0.0
    try:
        max_tokens = llm.parameters["max_tokens"]
    except (KeyError, TypeError):
        max_tokens = None

    if rewards.buffer_tokens > 0 and max_tokens is not None:
        overlong_penalty = length_penalty(max_tokens, llm_call.output_length_tokens, rewards.buffer_tokens)
        reward += overlong_penalty

    trace.reward = reward
    trace.metadata.setdefault("fn_calling", {}).update({"answer_status": answer_status})

    metrics = FnCallingMetrics(
        reward=reward,
        success=answer_status == "correct",
        no_error=answer_status != "unparsable",
        no_answer=answer_status == "no_answer",
        penalty=overlong_penalty,
    )

    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset"),
    )
