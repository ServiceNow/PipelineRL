"""Rollout generation for the IFEval instruction following domain."""

import logging
import random
import time

import aiohttp
from omegaconf import DictConfig

from pipelinerl.async_llm import llm_async_generate, make_training_text
from pipelinerl.llm import Prompt, TrainableLLM
from pipelinerl.rollouts import BaseMetrics, RolloutResult
from pipelinerl.utils import get_environment_jobs, resolve_environment_key
from pipelinerl.domains.math.rollouts import RewardTable, length_penalty

from .verifier_api import verify_answer_rpc

logger = logging.getLogger(__name__)


class Metrics(BaseMetrics):
    instructions_followed: int = 0
    instructions_total: int = 0
    partial_score: float = 0.0


async def generate_ifeval_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    # Build prompt
    messages = []
    if cfg.actor.system_prompt:
        messages.append({"role": "system", "content": cfg.actor.system_prompt})
    messages.append({
        "role": "user",
        "content": cfg.actor.task_template.format(task=problem["task"])
    })
    prompt = Prompt(messages=messages)

    time_start = time.time()
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - time_start

    assert llm_call.output.content is not None

    rewards = RewardTable(**dict(cfg.rewards))
    discount_factor = cfg.actor.discount_factor
    use_partial_credit = getattr(cfg.actor, "ifeval_partial_credit", True)

    env_key = resolve_environment_key(cfg, default="ifeval")
    env_jobs = get_environment_jobs(cfg, env_key)
    if not env_jobs:
        raise RuntimeError("No environment servers available for ifeval domain")
    env_job = random.choice(env_jobs)
    assert env_job.port is not None

    verification = await verify_answer_rpc(
        session=session,
        host=env_job.hostname,
        port=env_job.port,
        prediction=llm_call.output.content,
        reward_context=problem.get("reward_context", {}),
    )

    trace = make_training_text(llm, llm_call)

    answer_status = verification["answer_status"]
    followed_count = verification["followed_count"]
    total_count = verification["total_count"]
    score = verification["score"]

    if answer_status == "unparsable":
        if trace.finished:
            reward = rewards.unparsable_finished
        else:
            reward = rewards.unparsable_not_finished
        success = False
    elif answer_status == "no_answer":
        if trace.finished:
            reward = rewards.no_answer_finished
        else:
            reward = rewards.no_answer_not_finished
        success = False
    elif answer_status == "correct":
        if trace.finished:
            reward = rewards.correct_answer_finished
        else:
            reward = rewards.correct_answer_not_finished
        success = True
    elif answer_status == "partial" and use_partial_credit:
        if trace.finished:
            base_reward = rewards.wrong_answer_finished
            max_reward = rewards.correct_answer_finished
        else:
            base_reward = rewards.wrong_answer_not_finished
            max_reward = rewards.correct_answer_not_finished
        reward = base_reward + (max_reward - base_reward) * score
        success = False
    else:
        # Wrong answer
        if trace.finished:
            reward = rewards.wrong_answer_finished
        else:
            reward = rewards.wrong_answer_not_finished
        success = False

    reward *= discount_factor ** llm_call.output_length_tokens
    if rewards.buffer_tokens and llm.parameters.get("max_tokens") is not None:
        reward += length_penalty(
            llm.parameters["max_tokens"],
            llm_call.output_length_tokens,
            rewards.buffer_tokens,
        )
    trace.reward = reward

    metrics = Metrics(
        reward=reward,
        success=success,
        no_error=answer_status != "unparsable",
        no_answer=answer_status == "no_answer",
        instructions_followed=followed_count,
        instructions_total=total_count,
        partial_score=score,
    )

    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset"),
        domain="ifeval",
    )
