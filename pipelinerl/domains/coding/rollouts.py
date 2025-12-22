"""Rollout generation for the coding domain using the sandbox verifier."""

from __future__ import annotations

import json
import random
import time
from typing import Any, Literal

import aiohttp
from omegaconf import DictConfig

from pipelinerl.llm import Prompt, TrainableLLM
from pipelinerl.async_llm import llm_async_generate, make_training_text
from pipelinerl.rollouts import BaseMetrics, RolloutResult
from pipelinerl.utils import get_environment_jobs, resolve_environment_key
from pipelinerl.domains.math.rollouts import RewardTable, length_penalty

from .verifier_api import verify_coding_solution_rpc


class CodingMetrics(BaseMetrics):
    compile_error: bool = False
    runtime_error: bool = False
    timeout_error: bool = False
    passed: int = 0
    total: int = 0


def _format_task(problem: dict[str, Any]) -> str:
    if "task" in problem and problem["task"]:
        return str(problem["task"])
    if "question" in problem and problem["question"]:
        return str(problem["question"])
    extra_info = problem.get("extra_info") or {}
    if isinstance(extra_info, dict) and extra_info.get("question"):
        return str(extra_info["question"])
    return str(problem)


def _determine_answer_status(verification: dict[str, Any]) -> Literal["correct", "wrong", "no_answer", "unparsable"]:
    if verification.get("empty_response"):
        return "no_answer"
    if verification.get("compile_error") or verification.get("timeout_error"):
        return "unparsable"
    total = int(verification.get("total") or 0)
    passed = int(verification.get("passed") or 0)
    if total > 0 and passed == total:
        return "correct"
    return "wrong"


def _compute_reward(
    cfg: DictConfig,
    rewards: RewardTable,
    *,
    answer_status: Literal["correct", "wrong", "no_answer", "unparsable"],
    finished: bool,
    output_tokens: int,
    max_tokens: int | None,
) -> tuple[float, float]:
    match (answer_status, finished):
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
            raise ValueError(f"Invalid answer_status/finished combination: {answer_status}/{finished}")

    reward *= cfg.actor.discount_factor ** output_tokens
    overlong_penalty = 0.0
    if rewards.buffer_tokens and max_tokens is not None:
        overlong_penalty = length_penalty(max_tokens, output_tokens, rewards.buffer_tokens)
        reward += overlong_penalty
    return reward, overlong_penalty


async def _run_verification(
    cfg: DictConfig,
    *,
    session: aiohttp.ClientSession,
    prediction: str | None,
    reward_context: dict[str, Any] | str | None,
    extra_info: dict[str, Any] | str | None,
) -> dict[str, Any]:
    env_key = resolve_environment_key(cfg, default="coding")
    env_jobs = get_environment_jobs(cfg, env_key)
    if not env_jobs:
        raise RuntimeError("No coding environment servers registered")
    env_job = random.choice(env_jobs)
    if env_job.hostname is None or env_job.port is None:
        raise RuntimeError("Coding environment job is missing host/port information")
    return await verify_coding_solution_rpc(
        session,
        host=env_job.hostname,
        port=env_job.port,
        prediction=prediction,
        reward_context=reward_context,
        extra_info=extra_info,
    )


def _coerce_dict(data: dict[str, Any] | str | None) -> dict[str, Any]:
    if data is None:
        return {}
    if isinstance(data, dict):
        return data
    try:
        return json.loads(data)
    except Exception:
        return {}


def _get_system_prompt(cfg: DictConfig) -> str:
    """Get the system prompt, preferring domain-specific prompt if global is empty."""
    if cfg.actor.system_prompt:
        return cfg.actor.system_prompt
    # Fall back to domain-specific system prompt
    domain_prompts = getattr(cfg.actor, "domain_system_prompts", None)
    if domain_prompts:
        return domain_prompts.get("coding", "") or ""
    return ""


async def generate_coding_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict[str, Any],
    session: aiohttp.ClientSession,
) -> RolloutResult:
    messages = []
    system_prompt = _get_system_prompt(cfg)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    user_task = cfg.actor.task_template.format(task=_format_task(problem))
    messages.append({"role": "user", "content": user_task})
    prompt = Prompt(messages=messages)

    start_time = time.time()
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - start_time
    assert llm_call.output.content is not None
    trace = make_training_text(llm, llm_call)

    reward_context = _coerce_dict(problem.get("reward_context"))
    extra_info = _coerce_dict(problem.get("extra_info"))
    verification = await _run_verification(
        cfg,
        session=session,
        prediction=llm_call.output.content,
        reward_context=reward_context,
        extra_info=extra_info,
    )

    rewards_cfg = RewardTable(**dict(cfg.rewards))
    answer_status = _determine_answer_status(verification)
    try:
        max_tokens = llm.parameters["max_tokens"]
    except (KeyError, TypeError):
        max_tokens = None
    reward, _ = _compute_reward(
        cfg,
        rewards_cfg,
        answer_status=answer_status,
        finished=trace.finished,
        output_tokens=llm_call.output_length_tokens,
        max_tokens=max_tokens,
    )
    trace.reward = reward

    coding_metadata = {
        "passed": verification.get("passed"),
        "total": verification.get("total"),
        "compile_error": verification.get("compile_error"),
        "runtime_error": verification.get("runtime_error"),
        "timeout_error": verification.get("timeout_error"),
        "empty_response": verification.get("empty_response"),
        "call_type": verification.get("call_type"),
        "fn_name": verification.get("fn_name"),
        "error": verification.get("error"),
        "tests": (verification.get("tests") or [])[:5],
    }
    trace.metadata.setdefault("coding", {}).update(coding_metadata)

    metrics = CodingMetrics(
        reward=reward,
        success=(verification.get("passed") == verification.get("total") and verification.get("total", 0) > 0),
        no_error=not (
            verification.get("compile_error")
            or verification.get("runtime_error")
            or verification.get("timeout_error")
        ),
        no_answer=bool(verification.get("empty_response")),
        compile_error=bool(verification.get("compile_error")),
        runtime_error=bool(verification.get("runtime_error")),
        timeout_error=bool(verification.get("timeout_error")),
        passed=int(verification.get("passed") or 0),
        total=int(verification.get("total") or 0),
    )

    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset"),
    )
