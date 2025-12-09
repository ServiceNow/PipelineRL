import asyncio
import logging
import os
import random
import time
import traceback
from typing import Literal

import aiohttp
from hydra.utils import instantiate
from omegaconf import DictConfig
from tapeagents.agent import DEFAULT, Agent
from tapeagents.core import LLMOutputParsingFailureAction, Observation
from tapeagents.io import save_json_tape
from tapeagents.orchestrator import async_execute_agent, execute_agent
from tapeagents.remote_environment import AsyncRemoteEnvironment
from tapeagents.tools.simple_browser import PageObservation

from pipelinerl.async_llm import make_training_text
from pipelinerl.domains.miniwob.environment import WebEnvironment
from pipelinerl.domains.miniwob.steps import WebTape
from pipelinerl.llm import LLMCall, TrainableLLM, TrainingText
from pipelinerl.rollouts import BaseMetrics, RolloutResult
from pipelinerl.world import Job

logger = logging.getLogger(__name__)


def task_id(problem: dict) -> str:
    """Format task identifier for logging."""
    return f"{problem['dataset']}/{problem['task']}/{problem['seed']}"


class MiniwobMetrics(BaseMetrics):
    reward: float = -1.0
    success: bool = False
    has_error: bool = False
    no_answer: bool = True
    overflow: bool = False
    n_llm_calls: int = 0
    n_step_errors: int = 0
    n_observations: int = 0
    n_steps: int = 0
    env_creation_time: float = 0.0
    agent_creation_time: float = 0.0
    env_start_time: float = 0.0
    env_close_time: float = 0.0
    agent_execution_time: float = 0.0
    total_execution_time: float = 0.0
    llm_call_time: float = 0.0
    env_step_time: float = 0.0
    total_llm_call_time: float = 0.0
    total_env_call_time: float = 0.0


def _tape_contains_an_error(tape: WebTape) -> bool:
    """
    Returns true if the tape ends with an error, ie if one of the following is true:
    - the last step is an LLMOutputParsingFailureAction
    - the tape metadata has an error
    - the last step is a PageObservation with an error
    """
    return (
        len(tape.steps) == 0
        or isinstance(tape.steps[-1], LLMOutputParsingFailureAction)
        or tape.metadata.result.get("error") is not None
        or (isinstance(tape.steps[-1], PageObservation) and bool(tape.steps[-1].error))
    )


def _compute_reward(
    tape: WebTape,
    reward_computation: Literal["uic", "default"] = "default",
    has_error: bool = False,
) -> tuple[float, bool]:
    """
    Compute reward from tape.

    Args:
        tape: The execution tape
        cfg: Configuration with reward_computation setting
        has_error: If there were errors during execution

    Returns:
        tuple of (reward, has_error)
    """
    # Extract raw reward from last observation
    obs_steps = [step for step in tape if isinstance(step, Observation)]
    if obs_steps:
        last_obs = obs_steps[-1]
        raw_reward = last_obs.metadata.other.get("info", {}).get("task_info", {}).get("REWARD_GLOBAL", -1.0)
    else:
        raw_reward = -1.0

    # Count errors and page observations
    n_step_errors = len([step for step in tape.steps if isinstance(step, LLMOutputParsingFailureAction)])
    n_observations = len([step for step in tape.steps if isinstance(step, Observation)])

    # Determine if tape has errors
    has_error = has_error or _tape_contains_an_error(tape)

    # Compute final reward based on configuration
    if reward_computation == "uic":
        reward = float(raw_reward > 0)
        if reward == 0.0:
            reward = -1.0
        reward *= 0.98**n_observations
    else:
        reward = raw_reward * 0.99**n_step_errors if not has_error and raw_reward >= 0 else -1.0

    return reward, has_error


def _extract_llm_calls(tape: WebTape) -> list[LLMCall]:
    """Extract LLM calls from tape steps."""
    return [
        LLMCall(**step.metadata.other["llm_call"])
        if isinstance(step.metadata.other["llm_call"], dict)
        else step.metadata.other["llm_call"]
        for step in tape.steps
        if "llm_call" in step.metadata.other
    ]


def _compute_metrics(
    tape: WebTape,
    training_texts: list[TrainingText],
    reward: float,
    has_error: bool,
    n_llm_calls: int,
) -> MiniwobMetrics:
    # Create training texts
    has_overflow = False
    for text in training_texts:
        text.reward = reward
        has_overflow |= not text.finished

    # Extract timing information
    llm_call_times = [float(step.metadata.other.get("llm_call_time", 0.0)) for step in tape.steps]
    env_call_times = [float(step.metadata.other.get("action_execution_time", 0.0)) for step in tape.steps]
    total_llm_call_time = sum(llm_call_times)
    total_env_call_time = sum(env_call_times)
    llm_call_time = total_llm_call_time / len(llm_call_times) if llm_call_times else -1.0
    env_step_time = total_env_call_time / len(env_call_times) if env_call_times else -1.0
    env_start_time = tape.metadata.result.get("env_start_time", -1.0)
    env_close_time = tape.metadata.result.get("env_close_time", -1.0)
    env_creation_time = tape.metadata.result.get("env_creation_time", -1)
    agent_creation_time = tape.metadata.result.get("agent_creation_time", -1)
    agent_execution_time = tape.metadata.result.get("agent_execution_time", -1.0)
    total_execution_time = tape.metadata.result.get("total_execution_time", -1.0)

    # Compute step counts
    n_observations = len([s for s in tape.steps if isinstance(s, Observation)])
    n_step_errors = len([step for step in tape.steps if isinstance(step, LLMOutputParsingFailureAction)])

    metrics = MiniwobMetrics(
        reward=reward,
        success=reward > 0.5,
        has_error=has_error,
        no_answer=reward < 0,
        overflow=has_overflow,
        n_llm_calls=n_llm_calls,
        n_step_errors=n_step_errors,
        n_steps=len(tape.steps),
        n_observations=n_observations,

        env_creation_time=env_creation_time,
        env_start_time=env_start_time,
        env_close_time=env_close_time,
        
        agent_creation_time=agent_creation_time,
        agent_execution_time=agent_execution_time,

        llm_call_time=llm_call_time,
        env_step_time=env_step_time,
        total_llm_call_time=total_llm_call_time,
        total_env_call_time=total_env_call_time,
        total_execution_time=total_execution_time,
    )
    return metrics


async def check_env_server_health(env_job: Job, session: aiohttp.ClientSession) -> dict:
    """Check environment server health via HTTP API."""
    try:
        url = f"http://{env_job.hostname}:{env_job.port}/health"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5.0)) as response:
            if response.status == 200:
                health_data = await response.json()
                return {"healthy": True, "health_data": health_data, "last_check": time.time()}
            else:
                error_text = await response.text()
                health_data = f"HTTP {response.status}: {error_text}"
                return {"healthy": False, "health_data": health_data, "last_check": time.time()}
    except Exception as e:
        exception_type = type(e).__name__
        exception_message = str(e) if str(e) else "No message available"
        logger.exception(
            f"Error checking environment server health: {exception_type}: {exception_message}", stack_info=True
        )
        return {
            "healthy": False,
            "health_data": f"Exception: {exception_type}: {exception_message}",
            "last_check": time.time(),
            "error_stacktrace": traceback.format_exc(),
        }


def generate_miniwob_rollout(cfg: DictConfig, llm: TrainableLLM, problem: dict) -> RolloutResult:
    """
    Generate a MiniWoB rollout. Steps:
    - make agent and env
    - set the llm
    - run the agent
    - get llm calls from tape
    - compute rewards
    - get training text from llm calls

    Args:
        cfg: Configuration for the rollout
        llm: The LLM to use
        problem: The problem dict
    Returns:
        RolloutResult with training texts and metrics
    """
    tid = task_id(problem)
    start_time = time.perf_counter()
    environment: WebEnvironment = instantiate(cfg.environment)
    environment.initialize()
    env_creation_time = time.perf_counter() - start_time
    logger.info(f"Environment tools: {environment.tools_description()}")
    t = time.perf_counter()
    agent: Agent = instantiate(
        cfg.agent,
        known_actions=environment.actions(),
        tools_description=environment.tools_description(),
        llms={DEFAULT: llm},
    )
    logger.info(f"Agent and environment loaded, using llm {llm.model_name} at {llm.get_base_url()}")
    agent_creation_time = time.perf_counter() - t
    try:
        start_attempts = cfg.start_attempts
        t = time.perf_counter()
        while True:
            try:
                tape, _ = environment.start_task(problem)
                break
            except Exception as e:
                logger.exception(f"Failed to start task {tid}: {e}")
                start_attempts -= 1
                if start_attempts <= 0:
                    raise Exception(f"Failed to start task {tid} after {cfg.start_attempts} attempts")
                else:
                    logger.warning("Retrying after 1 second")
                    time.sleep(1)
        env_start_time = time.perf_counter() - t
        logger.info(f"Task {tid} started in {env_start_time:.2f}s")
        t = time.perf_counter()
        tape = execute_agent(agent, tape, environment, max_loops=cfg.agent_max_loops)
        agent_execution_time = time.perf_counter() - t
    finally:
        t = time.perf_counter()
        environment.close()
        env_close_time = time.perf_counter() - t
    total_execution_time = time.perf_counter() - start_time
    logger.info(f"Task {tid} finished in {total_execution_time:.2f}s")
    tape.metadata.result.update(
        {
            "total_execution_time": total_execution_time,
            "env_creation_time": env_creation_time,
            "env_start_time": env_start_time,
            "env_close_time": env_close_time,
            "agent_creation_time": agent_creation_time,
            "agent_execution_time": agent_execution_time,
        }
    )

    # save the tape as we go
    if cfg.save_tapes:
        _save_tapes(cfg, problem, tape)

    # Compute reward and metrics
    reward, has_error = _compute_reward(tape, cfg.reward_computation)
    llm_calls = _extract_llm_calls(tape)
    training_texts = [make_training_text(llm, llm_call) for llm_call in llm_calls]
    metrics = _compute_metrics(
        tape,
        training_texts,
        reward,
        has_error,
        len(llm_calls),
    )
    latency = time.perf_counter() - start_time
    return RolloutResult(
        training_texts=training_texts,
        metrics=metrics,
        latency=latency,
        dataset_name=problem["dataset"],
    )

def _save_tapes(cfg, problem, tape):
    tape_name = problem.get("_task_id", tape.metadata.id)
    try:
        save_json_tape(tape, os.path.join(cfg.output_dir, "tapes"), tape_name)
    except Exception as e:
        logger.error(f"Error saving tape {tape_name}: {e}")


async def generate_miniwob_rollout_async(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    start_time = time.perf_counter()
    tid = task_id(problem)
    rollout_timeout = getattr(cfg, "rollout_timeout", 600)  # 10 minutes default

    env_jobs = [Job(**job) for job in cfg.jobs if job["kind"] == "environment"]
    env_jobs_url_tried = []

    for _ in range(len(env_jobs)):
        env_job = random.choice(
            [job for job in env_jobs if f"http://{job.hostname}:{job.port}" not in env_jobs_url_tried]
        )
        env_job_url = f"http://{env_job.hostname}:{env_job.port}"
        env_jobs_url_tried.append(env_job_url)

        health = await check_env_server_health(env_job, session)
        if not health["healthy"]:
            logger.warning(f"Env server {env_job_url} unhealthy: {health.get('health_data', 'unknown')}, skip to next one")
            continue
        logger.debug(f"Using env server {env_job_url}")

        try:
            return await asyncio.wait_for(
                _execute_rollout_with_timeout(cfg, llm, problem, session, start_time, env_job_url),
                timeout=rollout_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Task {tid} timed out after {rollout_timeout}s on {env_job_url}")
            continue
        except Exception as e:
            logger.warning(f"Task {tid} failed on {env_job_url}: {e}")
            continue

    logger.error(f"Task {tid}: all environment servers failed")
    # Return a failed rollout result
    return RolloutResult(
        training_texts=[],
        metrics=MiniwobMetrics(),
        latency=time.perf_counter() - start_time,
        dataset_name=problem["dataset"],
    )


async def _execute_rollout_with_timeout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
    start_time: float,
    env_job_url: str,
) -> RolloutResult:
    tid = task_id(problem)
    has_error = False
    start_time = time.perf_counter()
    environment = AsyncRemoteEnvironment(server_url=env_job_url)  # type: ignore
    async with environment.acontext(session, wait_for_env=True) as env:
        env_creation_time = time.perf_counter() - start_time
        agent_creation_time = 0.0
        start_attempts = cfg.start_attempts
        t = time.perf_counter()
        tape_dict = {}
        while start_attempts > 0:
            try:
                tape_dict, info = await env.start_task(problem)
                if info.get("error"):
                    raise ValueError(info["error"])
                break
            except Exception as e:
                start_attempts -= 1
                logger.warning(f"Task {tid} start failed, {start_attempts} attempts left: {e}")
                if start_attempts <= 0:
                    logger.error(f"Task {tid} start failed after all retries: {e}")
                    has_error = True
                    break
                else:
                    await asyncio.sleep(5)
        env_start_time = time.perf_counter() - t
        logger.info(f"Task {tid} started in {env_start_time:.2f}s (worker={env.worker_id})")
        tape: WebTape = WebTape(**tape_dict)
        t = time.perf_counter()
        agent_execution_time = 0.0
        if not has_error:
            agent_attempts = cfg.agent_attempts
            while agent_attempts > 0:
                try:
                    worker_status = await env.check_worker_alive()
                    if worker_status.get("status") == "starting":
                        logger.debug(f"Task {tid}: worker {env.worker_id} starting, waiting...")
                        await asyncio.sleep(5)
                        continue
                except Exception as e:
                    logger.exception(f"Task {tid}: worker {env.worker_id} dead: {e}")
                    has_error = True
                    break

                try:
                    t = time.perf_counter()
                    actions = await env.a_actions()
                    tools_description = await env.a_tools_description()
                    agent: Agent = instantiate(cfg.agent, known_actions=actions, tools_description=tools_description)
                    agent.llms = {DEFAULT: llm}  # type: ignore
                    agent_creation_time = time.perf_counter() - t
                    t = time.perf_counter()
                    tape = await async_execute_agent(agent, tape, env, session, max_loops=cfg.agent_max_loops)
                    agent_execution_time = time.perf_counter() - t
                    if tape.metadata.error:
                        logger.error(f"Task {tid}: agent error: {tape.metadata.error}")
                        raise ValueError(tape.metadata.error)
                    logger.info(f"Task {tid}: agent execution succeeded")
                    break
                except Exception as e:
                    agent_attempts -= 1
                    logger.warning(f"Task {tid}: agent error, {agent_attempts} attempts left: {e}")
                    if agent_attempts <= 0:
                        logger.error(f"Task {tid}: agent failed after all retries: {e}")
                        has_error = True
                        break
                    await asyncio.sleep(5)

            logger.info(f"Task {tid} finished in {time.perf_counter() - t:.2f}s (worker={env.worker_id})")
        t = time.perf_counter()
        await env.aclose()
        env_close_time = time.perf_counter() - t
        total_execution_time=time.perf_counter() - start_time
        tape.metadata.result.update({
            "total_execution_time": total_execution_time,
            "env_creation_time": env_creation_time,
            "env_start_time": env_start_time,
            "env_close_time": env_close_time,
            "agent_creation_time": agent_creation_time,
            "agent_execution_time": agent_execution_time,
        })

    if cfg.save_tapes:
        _save_tapes(cfg, problem, tape)

    # Compute reward and metrics
    reward, has_error = _compute_reward(tape, cfg.reward_computation, has_error)
    llm_calls = _extract_llm_calls(tape)
    training_texts = [make_training_text(llm, llm_call) for llm_call in llm_calls]
    metrics = _compute_metrics(
        tape,
        training_texts,
        reward,
        has_error,
        len(llm_calls),
    )
    latency = time.time() - start_time
    return RolloutResult(
        training_texts=training_texts,
        metrics=metrics,
        latency=latency,
        dataset_name=problem["dataset"],
    )
