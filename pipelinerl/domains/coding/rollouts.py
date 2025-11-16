"""Rollout generation for the coding domain using the sandbox verifier."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any

import aiohttp
from omegaconf import DictConfig, OmegaConf
from tapeagents.core import Prompt
from tapeagents.llms.trainable import TrainableLLM

from pipelinerl.async_llm import llm_async_generate, make_training_text
from pipelinerl.rollouts import BaseMetrics, RolloutResult
from pipelinerl.utils import get_environment_jobs, resolve_environment_key

from .verifier_api import verify_coding_solution_rpc


@dataclass
class CodingRewardConfig:
    compile_error_penalty: float
    runtime_error_penalty: float
    timeout_penalty: float
    empty_response_penalty: float
    partial_credit_weight: float
    full_pass_bonus: float


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


def _load_reward_cfg(cfg: DictConfig) -> CodingRewardConfig:
    rewards_cfg = cfg.actor.get("coding_rewards")
    if isinstance(rewards_cfg, DictConfig):
        rewards_dict = OmegaConf.to_container(rewards_cfg, resolve=True)
    else:
        rewards_dict = rewards_cfg or {}
    return CodingRewardConfig(
        compile_error_penalty=float(rewards_dict.get("compile_error_penalty", -1.0)),
        runtime_error_penalty=float(rewards_dict.get("runtime_error_penalty", -0.3)),
        timeout_penalty=float(rewards_dict.get("timeout_penalty", -0.5)),
        empty_response_penalty=float(rewards_dict.get("empty_response_penalty", -1.5)),
        partial_credit_weight=float(rewards_dict.get("partial_credit_weight", 1.0)),
        full_pass_bonus=float(rewards_dict.get("full_pass_bonus", 0.75)),
    )


def _compute_reward(result: dict[str, Any], rewards: CodingRewardConfig) -> float:
    if result.get("empty_response"):
        return rewards.empty_response_penalty
    if result.get("compile_error"):
        return rewards.compile_error_penalty
    if result.get("timeout_error"):
        return rewards.timeout_penalty

    passed = int(result.get("passed") or 0)
    total = int(result.get("total") or 0)
    reward = 0.0
    if total > 0:
        reward += (passed / total) * rewards.partial_credit_weight
        if passed == total:
            reward += rewards.full_pass_bonus
    if result.get("runtime_error") and passed < total:
        reward += rewards.runtime_error_penalty
    return reward


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


async def generate_coding_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict[str, Any],
    session: aiohttp.ClientSession,
) -> RolloutResult:
    messages = []
    if cfg.actor.system_prompt:
        messages.append({"role": "system", "content": cfg.actor.system_prompt})
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

    rewards_cfg = _load_reward_cfg(cfg)
    reward = _compute_reward(verification, rewards_cfg)
    reward *= cfg.actor.discount_factor ** llm_call.output_length_tokens
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


__all__ = ["generate_coding_rollout", "CodingMetrics"]
