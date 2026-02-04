"""Rollout generation for BFCL v3 function calling domain."""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, List, Optional

import aiohttp
from omegaconf import DictConfig

from pipelinerl.llm import Prompt, TrainableLLM
from pipelinerl.async_llm import llm_async_generate, make_training_text
from pipelinerl.domains.math.rollouts import RewardTable, length_penalty
from pipelinerl.domains.fn_calling.verifier_api import verify_fn_calling_answer_rpc
from pipelinerl.rollouts import BaseMetrics, RolloutResult
from pipelinerl.utils import get_environment_jobs, resolve_environment_key

logger = logging.getLogger(__name__)

_reward_config_logged = False


class FnCallingMetrics(BaseMetrics):
    penalty: float


def _format_tools_for_prompt(tools: List[Dict[str, Any]]) -> str:
    import json

    tool_descriptions = []
    for tool in tools:
        # Handle both OpenAI tool format and raw function format
        if "function" in tool:
            func = tool["function"]
        else:
            func = tool
        tool_descriptions.append(json.dumps(func, indent=2))

    tools_text = "\n".join(tool_descriptions)

    return f"""You are provided with function signatures within <available_tools></available_tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about the arguments. Here are the available tools:
<available_tools>
{tools_text}
</available_tools>

Return ALL required function calls as a JSON list within <tool_calls></tool_calls> XML tags. If multiple function calls are needed, include ALL of them in a single list:
<tool_calls>[{{"name": "function1", "arguments": {{"arg1": "val1"}}}}, {{"name": "function2", "arguments": {{"arg2": "val2"}}}}]</tool_calls>

Example - calling add twice: <tool_calls>[{{"name": "add", "arguments": {{"a": 1, "b": 2}}}}, {{"name": "add", "arguments": {{"a": 3, "b": 4}}}}]</tool_calls>"""


def _build_prompt(cfg: DictConfig, problem: Dict[str, Any]) -> Prompt:
    """Build the prompt for function calling.

    Args:
        cfg: Configuration object.
        problem: Problem dictionary with 'task' and 'extra_info'.

    Returns:
        Prompt with system message and tools embedded in text.
    """
    messages: List[Dict[str, Any]] = []

    # Get tools from extra_info
    extra_info = problem.get("extra_info", {})
    oai_tools = extra_info.get("oai_tools", [])

    # Build system prompt with tools embedded
    base_system_prompt = cfg.actor.get("system_prompt", "")
    if oai_tools:
        tools_text = _format_tools_for_prompt(oai_tools)
        if base_system_prompt:
            system_content = f"{base_system_prompt}\n\n{tools_text}"
        else:
            system_content = tools_text
        messages.append({"role": "system", "content": system_content})
    elif base_system_prompt:
        messages.append({"role": "system", "content": base_system_prompt})

    task = problem.get("task", "")
    task_template = cfg.actor.get("task_template", "{task}")
    user_content = task_template.format(task=task)
    messages.append({"role": "user", "content": user_content})

    # Don't pass tools to Prompt
    # we've embed them in the system message
    # This avoids vLLM requiring --enable-auto-tool-choice --tool-call-parser
    return Prompt(messages=messages, tools=None)


async def _run_verification(
    cfg: DictConfig,
    session: aiohttp.ClientSession,
    generation: str,
    reward_context: Dict[str, Any],
    tool_calls: Optional[List[Dict[str, Any]]],
) -> str:
    """Run verification via RPC to BFCLEnvironment server."""
    env_key = resolve_environment_key(cfg, default="fn_calling")
    env_jobs = get_environment_jobs(cfg, env_key)
    if not env_jobs:
        raise RuntimeError("No environment servers available for fn_calling domain")
    env_job = random.choice(env_jobs)
    assert env_job.port is not None

    return await verify_fn_calling_answer_rpc(
        session=session,
        host=env_job.hostname,
        port=env_job.port,
        generation=generation,
        reward_context=reward_context,
        tool_calls=tool_calls,
    )


async def generate_fn_calling_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: Dict[str, Any],
    session: aiohttp.ClientSession,
) -> RolloutResult:
    prompt = _build_prompt(cfg, problem)

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

    reward_context = problem.get("reward_context", {})

    answer_status = await _run_verification(
        cfg=cfg,
        session=session,
        generation=llm_call.output.content,
        reward_context=reward_context,
        tool_calls=tool_calls,
    )

    rewards = RewardTable(**dict(cfg.rewards))
    trace = make_training_text(llm, llm_call)

    global _reward_config_logged
    if not _reward_config_logged:
        rewards.log_config("fn_calling")
        _reward_config_logged = True

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

    reward *= cfg.actor.discount_factor ** llm_call.output_length_tokens

    overlong_penalty = 0.0
    try:
        max_tokens = llm.parameters["max_tokens"]
    except (KeyError, TypeError):
        max_tokens = None

    if rewards.buffer_tokens > 0 and max_tokens is not None:
        overlong_penalty = length_penalty(max_tokens, llm_call.output_length_tokens, rewards.buffer_tokens)
        reward += overlong_penalty

    # Warn if reward is negative (shouldn't happen with standard configs)
    if reward < 0:
        logger.warning(
            f"Negative reward {reward:.4f} for fn_calling: "
            f"status={answer_status}, finished={trace.finished}, "
            f"penalty={overlong_penalty:.4f}, tokens={llm_call.output_length_tokens}"
        )

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
        domain="fn_calling",
    )
