"""Rollout generation for the logic domain.

Supports both local (in-process) verification using i3-logic verifiers
and RPC-based verification via a remote LogicEnvironment server.
"""

import random
import time

import aiohttp
from omegaconf import DictConfig
from pydantic import BaseModel

from pipelinerl.async_llm import llm_async_generate, make_training_text
from pipelinerl.llm import Prompt, TrainableLLM
from pipelinerl.rollouts import BaseMetrics, RolloutResult
from pipelinerl.utils import get_environment_jobs, resolve_environment_key

from .verifier_api import verify_logic_answer, verify_logic_answer_rpc, VerificationResult


class Metrics(BaseMetrics):
    """Metrics for logic domain rollouts."""
    pass


class RewardTable(BaseModel):
    """Reward values for different answer statuses."""
    wrong_answer_not_finished: float = 0.0
    wrong_answer_finished: float = 0.0
    no_answer_not_finished: float = 0.0
    no_answer_finished: float = 0.0
    error_not_finished: float = 0.0
    error_finished: float = 0.0
    correct_answer_not_finished: float = 0.5
    correct_answer_finished: float = 1.0


async def _run_verification(
    cfg: DictConfig,
    session: aiohttp.ClientSession,
    prediction: str,
    reward_context: dict,
) -> VerificationResult:
    """Run verification either locally (in-process) or via RPC.

    If cfg.actor.use_local_logic_verifier is True (default), uses i3-logic directly.
    Otherwise, sends verification request to a remote LogicEnvironment server.
    """
    use_local = getattr(cfg.actor, "use_local_logic_verifier", True)

    if use_local:
        return verify_logic_answer(prediction, reward_context)

    # Fall back to RPC-based verification
    env_key = resolve_environment_key(cfg, default="logic")
    env_jobs = get_environment_jobs(cfg, env_key)
    if not env_jobs:
        raise RuntimeError("No logic environment servers registered and use_local_logic_verifier=False")
    env_job = random.choice(env_jobs)
    if env_job.hostname is None or env_job.port is None:
        raise RuntimeError("Logic environment job is missing host/port information")
    return await verify_logic_answer_rpc(
        session,
        host=env_job.hostname,
        port=env_job.port,
        prediction=prediction,
        reward_context=reward_context,
    )


async def generate_logic_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    """Generate a rollout for a logic problem.

    Supports both local verification (default) and RPC-based verification.
    Set cfg.actor.use_local_logic_verifier=False to use remote LogicEnvironment.
    """
    # Build prompt
    messages = []

    # Use domain-specific system prompt if available
    system_prompt = cfg.actor.system_prompt
    if not system_prompt:
        domain_prompts = getattr(cfg.actor, "domain_system_prompts", None)
        if domain_prompts:
            system_prompt = getattr(domain_prompts, "logic", "")

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

    # Get reward configuration
    rewards = RewardTable(**dict(cfg.get("rewards", {})))
    discount_factor = getattr(cfg.actor, "discount_factor", 1.0)

    # Verify answer (local or RPC based on config)
    verification = await _run_verification(
        cfg,
        session,
        prediction=llm_call.output.content,
        reward_context=problem.get("reward_context", {}),
    )

    trace = make_training_text(llm, llm_call)

    # Determine reward based on answer status and finished state
    answer_status = verification.status
    match (answer_status, trace.finished):
        case ("wrong", False):
            reward = rewards.wrong_answer_not_finished
        case ("wrong", True):
            reward = rewards.wrong_answer_finished
        case ("no_answer", False):
            reward = rewards.no_answer_not_finished
        case ("no_answer", True):
            reward = rewards.no_answer_finished
        case ("error", False):
            reward = rewards.error_not_finished
        case ("error", True):
            reward = rewards.error_finished
        case ("correct", False):
            reward = rewards.correct_answer_not_finished
        case ("correct", True):
            reward = rewards.correct_answer_finished
        case _:
            reward = 0.0

    # Apply discount factor based on output length
    reward *= discount_factor ** llm_call.output_length_tokens
    trace.reward = reward

    metrics = Metrics(
        reward=reward,
        success=answer_status == "correct",
        no_error=answer_status not in ("error",),
        no_answer=answer_status == "no_answer",
    )

    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset"),
        domain="logic",
    )
