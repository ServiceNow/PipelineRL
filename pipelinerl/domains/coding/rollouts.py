from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import aiohttp
from omegaconf import DictConfig
from pydantic import BaseModel

from pipelinerl.async_llm import llm_async_generate, make_training_text
from pipelinerl.rollouts import BaseMetrics, RolloutResult
from tapeagents.core import Prompt
from tapeagents.llms.trainable import TrainableLLM

from .executor import ExecutionResult, run_coding_submission

logger = logging.getLogger(__name__)
DOMAIN = "coding"


class _TemplateVars(dict):
    def __missing__(self, key):  # type: ignore[override]
        return ""


class CodingRewardTable(BaseModel):
    compile_error_penalty: float = -1.0
    runtime_error_penalty: float = -0.5
    timeout_penalty: float = -0.75
    empty_response_penalty: float = -1.0
    partial_credit_weight: float = 1.0
    full_pass_bonus: float = 0.5


def _format_prompt(problem: dict[str, Any], task_template: str | None) -> list[dict[str, str]]:
    question = problem.get("question") or ""
    starter = problem.get("starter_code") or ""
    metadata_lines = []
    if problem.get("title"):
        metadata_lines.append(f"Title: {problem['title']}")
    if problem.get("source"):
        metadata_lines.append(f"Source: {problem['source']}")
    if problem.get("difficulty"):
        metadata_lines.append(f"Difficulty: {problem['difficulty']}")
    metadata = "\n".join(metadata_lines)
    if task_template:
        template_values = _TemplateVars(
            question=question,
            starter_code=starter,
            metadata=metadata,
            task=question,
        )
        body = task_template.format_map(template_values)
    else:
        body = question
    return [{"role": "user", "content": body.strip()}]


def _compute_reward(result: ExecutionResult, rewards: CodingRewardTable, response_empty: bool) -> float:
    if result.total <= 0:
        base = 0.0
    else:
        base = (result.passed / result.total) * rewards.partial_credit_weight
        if result.passed == result.total:
            base += rewards.full_pass_bonus
    if response_empty:
        base += rewards.empty_response_penalty
    if result.compile_error:
        base += rewards.compile_error_penalty
    if result.runtime_error:
        base += rewards.runtime_error_penalty
    if result.timeout:
        base += rewards.timeout_penalty
    return base


async def _execute_submission(problem: dict[str, Any], code: str, cfg: DictConfig) -> ExecutionResult:
    def _pick_limit(key: str, default: float | int, caster):
        value = problem.get(key)
        if value is None:
            return default
        try:
            return caster(value)
        except (TypeError, ValueError):
            return default

    time_limit = _pick_limit(
        "time_limit_s",
        float(getattr(cfg.actor, "coding_time_limit_s", 10.0)),
        float,
    )
    per_test = _pick_limit(
        "per_test_timeout_s",
        float(getattr(cfg.actor, "coding_per_test_timeout_s", 4.0)),
        float,
    )
    memory_limit = _pick_limit(
        "memory_limit_bytes",
        int(getattr(cfg.actor, "coding_memory_limit_bytes", 512 * 1024 * 1024)),
        int,
    )
    return await asyncio.to_thread(
        run_coding_submission,
        code,
        problem.get("tests", []),
        entry_point=problem.get("entry_point"),
        time_limit_s=time_limit,
        per_test_timeout=per_test,
        memory_limit_bytes=memory_limit,
    )


async def generate_coding_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    messages = []
    if cfg.actor.system_prompt:
        messages.append({"role": "system", "content": cfg.actor.system_prompt})
    messages.extend(_format_prompt(problem, getattr(cfg.actor, "task_template", None)))

    prompt = Prompt(messages=messages)
    time_start = time.time()
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - time_start

    output_text = llm_call.output.content or ""
    rewards_cfg = getattr(cfg.actor, "coding_rewards", None)
    rewards = CodingRewardTable(**dict(rewards_cfg) if rewards_cfg else {})

    exec_result = await _execute_submission(problem, output_text, cfg)
    reward = _compute_reward(exec_result, rewards, response_empty=(not output_text.strip()))

    training_text = make_training_text(llm, llm_call)
    training_text.reward = reward
    training_text.metadata["coding_result"] = exec_result.as_dict()

    metrics = BaseMetrics(
        reward=reward,
        success=exec_result.total > 0 and exec_result.passed == exec_result.total,
        no_error=exec_result.compile_error is None and exec_result.runtime_error is None,
        no_answer=not output_text.strip(),
    )

    return RolloutResult(
        training_texts=[training_text],
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset"),
    )
