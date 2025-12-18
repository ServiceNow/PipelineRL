import asyncio
import json
import logging
import os
import random
import time
import traceback

import aiohttp
from examples.rl_webagent.steps import WebTape
from hydra.utils import instantiate
from omegaconf import DictConfig
from tapeagents.agent import DEFAULT, Agent
from tapeagents.core import LLMCall, LLMOutputParsingFailureAction, Observation
from tapeagents.io import save_json_tape
from tapeagents.llms.trainable import TrainableLLM
from tapeagents.orchestrator import async_execute_agent
from tapeagents.remote_environment import AsyncRemoteEnvironment
from tapeagents.tools.simple_browser import PageObservation

from pipelinerl.async_llm import make_training_text
from pipelinerl.llm import LLMCall, TrainableLLM
from pipelinerl.rollouts import BaseMetrics, RolloutResult
from pipelinerl.world import Job

from .steps import WebTape

logger = logging.getLogger(__name__)


class MiniwobMetrics(BaseMetrics):
    reward: float
    success: bool
    no_error: bool
    no_answer: bool
    overflow: bool
    n_llm_calls: int
    n_step_errors: int
    n_page_observations: int
    n_steps: int
    total_execution_time: float
    agent_execution_time: float
    environment_execution_time: float
    env_step_time: float
    agent_step_time: float


def tape_contains_an_error(tape: WebTape) -> bool:
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
        or (isinstance(tape.steps[-1], PageObservation) and tape.steps[-1].error)
    )


async def check_env_server_health(env_job: Job, session: aiohttp.ClientSession) -> dict:
    """Check environment server health via HTTP API."""
    try:
        url = f"http://{env_job.hostname}:{env_job.port}/health"
        async with session.get(url, timeout=5) as response:
            if response.status == 200:
                health_data = await response.json()
                return {
                    "healthy": True,
                    "health_data": health_data,
                    "last_check": time.time()
                }
            else:
                error_text = await response.text()
                return {"healthy": False, "error_message": f"HTTP {response.status}: {error_text}", "last_check": time.time()}
    except Exception as e:
        exception_type = type(e).__name__
        exception_message = str(e) if str(e) else "No message available"
        logger.exception(f"Error checking environment server health: {exception_type}: {exception_message}", stack_info=True)
        return {"healthy": False, "error_message": f"Exception: {exception_type}: {exception_message}", "last_check": time.time(), "error_stacktrace": traceback.format_exc()}


async def generate_miniwob_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    # choose a random environment server
    # Generate environment
    # Generate TapeAgent
    # run the agent
    # get llm calls from tape
    # compute rewards
    # get training text from llm calls

    start_time = time.time()
    
    # Overall timeout for the entire rollout to prevent hanging
    rollout_timeout = getattr(cfg, 'rollout_timeout', 600)  # 10 minutes default

    env_jobs = [Job(**job) for job in cfg.jobs if job["kind"] == "environment"]
    env_jobs_url_tried = []

    # Try each environment server with health checks until one of them returns a rollout result
    for _ in range(len(env_jobs)):
        # Choose the next environment server to try randomly from the ones that have not been tried yet
        env_job = random.choice([job for job in env_jobs if f"http://{job.hostname}:{job.port}" not in env_jobs_url_tried])
        env_job_url = f"http://{env_job.hostname}:{env_job.port}"
        env_jobs_url_tried.append(env_job_url)

        # Check server health before using
        health = await check_env_server_health(env_job, session)
        if not health["healthy"]:
            logger.warning(f"Environment server {env_job_url} is unhealthy: {health}")
            logger.warning(f"Get health error stacktrace: {health['error_stacktrace']}")
            continue
        # Log health status for monitoring
        if health["healthy"]:
            logger.info(f"Using healthy environment server {env_job_url}: {health}")

        try:
            # Execute the entire rollout with a timeout
            return await asyncio.wait_for(
                _execute_rollout_with_timeout(cfg, llm, problem, session, start_time, env_job_url),
                timeout=rollout_timeout
            )
        except asyncio.TimeoutError:
            health = await check_env_server_health(env_job, session)
            if stack_trace := health.get("error_stacktrace"):
                logger.warning(f"Get health error stacktrace: {stack_trace}")
            logger.warning(f"Rollout timeout error stacktrace: {traceback.format_exc()}")
            logger.warning(f"Rollout timed out after {rollout_timeout} seconds for task {problem['dataset']}/{problem['task']}/{problem['seed']} on environment {env_job_url}. Health: {health}. Trying next server.")
            continue
        except Exception as e:
            health = await check_env_server_health(env_job, session)
            if stack_trace := health.get("error_stacktrace"):
                logger.warning(f"Get health error stacktrace: {stack_trace}")
            logger.warning(f"Rollout failed error stacktrace: {traceback.format_exc()}")
            logger.warning(f"Rollout failed for task {problem['dataset']}/{problem['task']}/{problem['seed']} on environment {env_job_url}. Health: {health}. Trying next server.")
            continue
    # If all servers failed
    logger.error(f"All environment servers failed for task {problem['dataset']}/{problem['task']}/{problem['seed']}. Returning a failed rollout result.")
    return _create_failed_rollout_result(problem, start_time, "all environment servers failed")


async def _execute_rollout_with_timeout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
    start_time: float,
    env_job_url: str,
) -> RolloutResult:
    # (2) Generate environment, TapeAgent, and run them to get a Tape
    no_error = True  # track if there was an error in the tape
    environment = AsyncRemoteEnvironment(server_url=env_job_url)  # type: ignore
    async with environment.acontext(session, wait_for_env=True) as env:
        start_attempts = cfg.start_attempts
        t = time.perf_counter()
        while start_attempts > 0:
            try:
                tape_dict, info = await env.start_task(problem)
                if info.get("error"):
                    raise ValueError(info['error'])
                break
            except Exception as e:
                start_attempts -= 1
                logger.warning(f"Failed to start task {problem['dataset']}/{problem['task']}/{problem['seed']}. {start_attempts} attempts remaining. Error: {e}")
                if start_attempts <= 0:
                    logger.error(f"Failed to start task after all retry attempts: {e}")
                    no_error = False
                    tape_dict = {}
                    break
                else:
                    logger.warning("Retry start task after 5 seconds.")
                    await asyncio.sleep(5)
        logger.info(
            f"Task {problem['dataset']}/{problem['task']}/{problem['seed']} started in {time.perf_counter() - t:.2f} seconds. Worker ID: {env.worker_id}. Tape dict: {tape_dict}"
        )
        tape: WebTape = WebTape(**tape_dict)  # convert http response dict to WebTape object
        t = time.perf_counter()
        if no_error:  # only run the agent if the task started successfully
            logger.info(f"Running agent for task {problem['dataset']}/{problem['task']}/{problem['seed']} with worker ID: {env.worker_id} and tape ID {tape.metadata.id}")
            agent_attempts = cfg.agent_attempts
            while agent_attempts > 0:
                # check if the worker is alive.
                try:
                    # this will either raise RuntimeError if worker is not alive anymore, or return a dictionary with the worker status
                    worker_status = await env.check_worker_alive()
                    if worker_status.get("status") == "starting":
                        logger.warning(f"Worker {env.worker_id} for task {problem['dataset']}/{problem['task']}/{problem['seed']} and tape ID {tape.metadata.id} is starting, waiting 5 seconds for it to be fully started.")
                        await asyncio.sleep(5)
                        continue
                except Exception as e:
                    # if worker is dead, no need to retry
                    logger.exception(f"Worker {env.worker_id} for task {problem['dataset']}/{problem['task']}/{problem['seed']} and tape ID {tape.metadata.id} is dead. Error: {e}", stack_info=True)
                    no_error = False
                    break
                # if worker is alive, run the agent
                try:
                    actions = await env.a_actions()
                    tools_description = await env.a_tools_description()
                    agent: Agent = instantiate(cfg.agent, known_actions=actions, tools_description=tools_description)
                    agent.llms = {DEFAULT: llm}
                    tape = await async_execute_agent(agent, tape, env, session, max_loops=cfg.agent_max_loops)
                    # Check if the tape has an error from the orchestrator (e.g., SocketTimeoutError, RuntimeError: Worker is not alive, etc.)
                    if tape.metadata.error:
                        logger.error(f"Agent execution for task {problem['dataset']}/{problem['task']}/{problem['seed']} with worker ID: {env.worker_id} and tape ID {tape.metadata.id} returned a tape with error: {tape.metadata.error}")
                        raise ValueError(tape.metadata.error)
                    else:
                        # Success - break out of retry loop
                        logger.info(f"Agent execution for task {problem['dataset']}/{problem['task']}/{problem['seed']} with worker ID: {env.worker_id} and tape ID {tape.metadata.id} finished successfully")
                        break
                except Exception as e:
                    agent_attempts -= 1
                    logger.warning(f"Error occurred while running agent for task {problem['dataset']}/{problem['task']}/{problem['seed']} with worker ID: {env.worker_id} and tape ID {tape.metadata.id}. {agent_attempts} attempts remaining. Error: {e}")
                    if agent_attempts <= 0:
                        logger.error(f"Agent execution failed after all retry attempts for task {problem['dataset']}/{problem['task']}/{problem['seed']} with worker ID: {env.worker_id} and tape ID {tape.metadata.id}: {e}")
                        no_error = False
                        break
                    else:
                        logger.warning(f"Retry agent execution after 5 seconds for task {problem['dataset']}/{problem['task']}/{problem['seed']} with worker ID: {env.worker_id} and tape ID {tape.metadata.id}.")
                        await asyncio.sleep(5)
            logger.info(
                f"Agent finished task {problem['dataset']}/{problem['task']}/{problem['seed']} in {time.perf_counter() - t:.2f} seconds with worker ID: {env.worker_id} and tape ID {tape.metadata.id}"
            )
        tape.metadata.result.update({"total_execution_time": time.perf_counter() - t})

    # save the tape as we go
    if cfg.save_tapes:
        save_json_tape(tape, os.path.join(cfg.output_dir, "tapes"), tape.metadata.id)

    # (3) Compute rewards
    obs_steps = [step for step in tape if isinstance(step, Observation)]
    if obs_steps:
        last_obs = obs_steps[-1]
        # in Miniwob, the observation "reward" is defined as RAW_REWARD_GLOBAL > 0
        # see here: https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/miniwob/src/browsergym/miniwob/base.py#L188
        # Let's take directly the RAW_REWARD_GLOBAL from the metadata
        # raw_reward = last_obs.metadata.other.get("reward", 0.0)
        raw_reward = last_obs.metadata.other.get("info", {}).get("task_info", {}).get("REWARD_GLOBAL", -1.0)
    else:
        raw_reward = -1.0

    no_error = no_error and not tape_contains_an_error(tape)
    # get the number of LLMOutputParsingFailureAction in the tape
    n_step_errors = len([step for step in tape.steps if isinstance(step, LLMOutputParsingFailureAction)])
    # get the number of PageObservation steps in the tape
    n_page_observations = len([step for step in tape.steps if isinstance(step, PageObservation)])

    if cfg.reward_computation == "nico":
        reward = raw_reward * 0.99**n_step_errors if no_error and raw_reward >= 0 else -1.0
    elif cfg.reward_computation == "uic":
        reward = float(raw_reward>0)
        if reward == 0.0:
            reward = -1.0
        reward *= 0.98 ** n_page_observations
    else:
        raise ValueError(f"Invalid reward configuration: {cfg.reward_computation}")

    # (3) Get LLM calls from Tape
    llm_calls = [step for step in tape.steps if step.metadata.other.get("llm_call") is not None]
    n_llm_calls = len(llm_calls)
    llm_calls: list[LLMCall] = [
        LLMCall(**step.metadata.other["llm_call"]) if isinstance(step.metadata.other["llm_call"], dict)
        else step.metadata.other["llm_call"]
        for step in llm_calls
    ]

    # (4) # For each LLM interaction in the tape, make a training example.
    all_finished = 1
    prompt_tokens = [llm_call.prompt_length_tokens for llm_call in llm_calls]
    output_tokens = [llm_call.output_length_tokens for llm_call in llm_calls]
    training_texts = [make_training_text(llm, llm_call) for llm_call in llm_calls]
    for text in training_texts:
        text.reward = reward
        all_finished &= 1 if text.input_ids[-1] == llm.tokenizer.eos_token_id else 0

    latency = time.time() - start_time
    agent_time = tape.metadata.result.get("agent_execution_time", -1.0)
    env_time = tape.metadata.result.get("environment_execution_time", -1.0)
    n_observations = len([s for s in tape.steps if isinstance(s, Observation)])  # TODO: is this not the same n_page_observations??
    n_other_steps = len(tape.steps) - n_observations
    metrics = MiniwobMetrics(
        reward=reward,
        success=reward > 0.5,
        no_error=no_error,
        no_answer=reward < 0,
        overflow=not all_finished,
        n_llm_calls=n_llm_calls,
        n_step_errors=n_step_errors,
        n_page_observations=n_page_observations,
        n_steps=len(tape.steps),
        total_execution_time=tape.metadata.result.get("total_execution_time", -1.0),
        agent_execution_time=agent_time,
        environment_execution_time=env_time,
        env_step_time=env_time / n_observations if env_time > 0 and n_observations > 0 else -1.0,
        agent_step_time=agent_time / n_other_steps if agent_time > 0 and n_other_steps > 0 else -1.0,
    )

    return RolloutResult(
        training_texts=training_texts,
        metrics=metrics,
        latency=latency,
        dataset_name=problem["dataset"],
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
    )


def _create_failed_rollout_result(problem: dict, start_time: float, error_type: str) -> RolloutResult:
    """Create a failed rollout result for timeout or other errors."""
    latency = time.time() - start_time
    
    # Create empty training texts and metrics for failed rollout
    metrics = MiniwobMetrics(
        reward=-1.0,
        success=False,
        no_error=False,
        no_answer=True,
        overflow=False,
        n_llm_calls=0,
        n_step_errors=0,
        n_page_observations=0,
        n_steps=0,
        total_execution_time=latency,
        agent_execution_time=-1.0,
        environment_execution_time=-1.0,
        env_step_time=-1.0,
        agent_step_time=-1.0,
    )
    
    return RolloutResult(
        training_texts=[],
        metrics=metrics,
        latency=latency,
        dataset_name=problem["dataset"],
        prompt_tokens=[],
        output_tokens=[],
    )
