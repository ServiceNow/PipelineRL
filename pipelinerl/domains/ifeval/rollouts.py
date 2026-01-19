"""Rollout generation for the IFEval instruction following domain."""

from __future__ import annotations

import time

import aiohttp
from omegaconf import DictConfig

from pipelinerl.async_llm import llm_async_generate, make_training_text
from pipelinerl.llm import Prompt, TrainableLLM
from pipelinerl.rollouts import BaseMetrics, RolloutResult
from pipelinerl.domains.math.rollouts import RewardTable

from .verifier_api import verify_ifeval_from_context


class Metrics(BaseMetrics):
    """Metrics for IFEval domain rollouts."""

    # Additional metrics for instruction following
    instructions_followed: int = 0
    instructions_total: int = 0
    partial_score: float = 0.0


async def generate_ifeval_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    """Generate a rollout for an IFEval instruction following problem.

    Args:
        cfg: Hydra config.
        llm: Language model to generate responses.
        problem: Problem dict with task and reward_context.
        session: aiohttp session for any RPC calls.

    Returns:
        RolloutResult with training text and metrics.
    """
    # Build prompt
    messages = []

    # Use domain-specific system prompt if available
    system_prompt = cfg.actor.system_prompt
    if not system_prompt:
        domain_prompts = getattr(cfg.actor, "domain_system_prompts", None)
        if domain_prompts:
            system_prompt = getattr(domain_prompts, "ifeval", "")

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

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
    response = llm_call.output.content

    # Get reward configuration (uses shared RewardTable from math domain)
    rewards = RewardTable(**dict(cfg.get("rewards", {})))
    discount_factor = getattr(cfg.actor, "discount_factor", 1.0)
    # IFEval supports partial credit: interpolate between wrong and correct rewards
    use_partial_credit = getattr(cfg.actor, "ifeval_partial_credit", True)

    # Verify instruction following
    reward_context = problem.get("reward_context", {})
    strict = getattr(cfg.actor, "ifeval_strict", False)

    verification = verify_ifeval_from_context(
        response=response,
        reward_context=reward_context,
        strict=strict,
    )

    trace = make_training_text(llm, llm_call)

    # Calculate reward based on verification result
    # Mapping: all_followed -> correct, none_followed -> wrong, error -> unparsable
    if verification.error:
        # Verification error -> unparsable
        if trace.finished:
            reward = rewards.unparsable_finished
        else:
            reward = rewards.unparsable_not_finished
        success = False
    elif verification.followed_all:
        # All instructions followed -> correct
        if trace.finished:
            reward = rewards.correct_answer_finished
        else:
            reward = rewards.correct_answer_not_finished
        success = True
    elif verification.followed_count > 0 and use_partial_credit:
        # Partial credit: interpolate between wrong and correct based on fraction followed
        if trace.finished:
            base_reward = rewards.wrong_answer_finished
            max_reward = rewards.correct_answer_finished
        else:
            base_reward = rewards.wrong_answer_not_finished
            max_reward = rewards.correct_answer_not_finished
        reward = base_reward + (max_reward - base_reward) * verification.score
        success = False
    else:
        # No instructions followed -> wrong
        if trace.finished:
            reward = rewards.wrong_answer_finished
        else:
            reward = rewards.wrong_answer_not_finished
        success = False

    # Apply discount factor based on output length
    reward *= discount_factor ** llm_call.output_length_tokens
    trace.reward = reward

    metrics = Metrics(
        reward=reward,
        success=success,
        no_error=verification.error is None,
        no_answer=not response.strip(),
        instructions_followed=verification.followed_count,
        instructions_total=verification.total_count,
        partial_score=verification.score,
    )

    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset"),
        domain="ifeval",
    )
