"""Rollout generation for the logic domain."""

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
    """Metrics for logic domain rollouts."""
    penalty: float = 0.0


async def generate_logic_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    """Generate a rollout for a logic problem."""
    # Build prompt
    messages = []
    if cfg.actor.system_prompt:
        messages.append({"role": "system", "content": cfg.actor.system_prompt})
    messages.append({
        "role": "user",
        "content": cfg.actor.task_template.format(task=problem["task"])
    })
    prompt = Prompt(messages=messages)

    # Generate response
    time_start = time.time()
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - time_start

    assert llm_call.output.content is not None

    # Get reward configuration
    rewards = RewardTable(**dict(cfg.rewards))
    discount_factor = cfg.actor.discount_factor

    # Verify answer via RPC
    env_key = resolve_environment_key(cfg, default="logic")
    env_jobs = get_environment_jobs(cfg, env_key)
    if not env_jobs:
        raise RuntimeError("No environment servers available for logic domain")
    env_job = random.choice(env_jobs)
    assert env_job.port is not None

    answer_status = await verify_answer_rpc(
        session=session,
        host=env_job.hostname,
        port=env_job.port,
        prediction=llm_call.output.content,
        reward_context=problem.get("reward_context", {}),
    )

    trace = make_training_text(llm, llm_call)

    # Determine reward based on answer status and finished state
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
            raise ValueError(f"Invalid answer_status/finished combination: {answer_status}/{trace.finished}")

    # Apply discount factor based on output length
    reward *= discount_factor ** llm_call.output_length_tokens
    overlong_penalty = 0.0
    if rewards.buffer_tokens and llm.parameters.get("max_tokens") is not None:
        overlong_penalty = length_penalty(
            llm.parameters["max_tokens"],
            llm_call.output_length_tokens,
            rewards.buffer_tokens,
        )
        reward += overlong_penalty
    trace.reward = reward

    metrics = Metrics(
        reward=reward,
        penalty=overlong_penalty,
        success=answer_status == "correct",
        no_error=answer_status != "unparsable",
        no_answer=answer_status == "no_answer",
    )

    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset"),
        domain="logic",
    )
