"""Rollout generation for the fn_calling domain."""

from __future__ import annotations

import json
import random
import time
from typing import Any

import aiohttp
from omegaconf import DictConfig

from pipelinerl.llm import Prompt, TrainableLLM
from pipelinerl.async_llm import llm_async_generate, make_training_text
from pipelinerl.domains.math.rollouts import RewardTable, length_penalty
from pipelinerl.domains.fn_calling.verifier_api import verify_fn_calling_answer_rpc
from pipelinerl.rollouts import BaseMetrics, RolloutResult
from pipelinerl.utils import get_environment_jobs, resolve_environment_key


class FnCallingMetrics(BaseMetrics):
    penalty: float


def _format_task(problem: dict[str, Any]) -> str:
    if problem.get("task"):
        return str(problem["task"])
    if problem.get("question"):
        return str(problem["question"])
    return str(problem)


def _coerce_dict(data: dict[str, Any] | str | None) -> dict[str, Any]:
    if data is None:
        return {}
    if isinstance(data, dict):
        return data
    if isinstance(data, str) and data.strip():
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return {}
    return {}


async def generate_fn_calling_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict[str, Any],
    session: aiohttp.ClientSession,
) -> RolloutResult:
    messages = []
    if cfg.actor.system_prompt:
        messages.append({"role": "system", "content": cfg.actor.system_prompt})
    messages.append({"role": "user", "content": cfg.actor.task_template.format(task=_format_task(problem))})
    prompt = Prompt(messages=messages)

    start_time = time.time()
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - start_time
    assert llm_call.output.content is not None

    env_key = resolve_environment_key(cfg, default="fn_calling")
    env_jobs = get_environment_jobs(cfg, env_key)
    if not env_jobs:
        raise RuntimeError("No environment servers available for fn_calling domain")
    env_job = random.choice(env_jobs)
    if env_job.hostname is None or env_job.port is None:
        raise RuntimeError("fn_calling environment job is missing host/port information")

    reward_context = _coerce_dict(problem.get("reward_context"))
    extra_info = _coerce_dict(problem.get("extra_info"))
    answer_status = await verify_fn_calling_answer_rpc(
        session=session,
        host=env_job.hostname,
        port=env_job.port,
        generation=llm_call.output.content,
        reward_context=reward_context,
        extra_info=extra_info,
    )

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

    reward *= cfg.actor.discount_factor ** llm_call.output_length_tokens
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


__all__ = ["generate_fn_calling_rollout", "FnCallingMetrics"]
