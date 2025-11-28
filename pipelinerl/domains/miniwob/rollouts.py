import asyncio
import logging
import os
import random
import time
import traceback

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
from pipelinerl.llm import LLMCall, TrainableLLM
from pipelinerl.rollouts import BaseMetrics, RolloutResult
from pipelinerl.world import Job

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
    env_start_time: float
    env_close_time: float
    env_agent_creation_time: float
    agent_execution_time: float
    environment_execution_time: float
    env_step_time: float
    agent_step_time: float
    llm_call_time: float
    env_call_time: float
    total_llm_call_time: float
    total_env_call_time: float


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
        or (isinstance(tape.steps[-1], PageObservation) and bool(tape.steps[-1].error))
    )


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
                return {
                    "healthy": False,
                    "error_message": f"HTTP {response.status}: {error_text}",
                    "last_check": time.time(),
                }
    except Exception as e:
        exception_type = type(e).__name__
        exception_message = str(e) if str(e) else "No message available"
        logger.exception(
            f"Error checking environment server health: {exception_type}: {exception_message}", stack_info=True
        )
        return {
            "healthy": False,
            "error_message": f"Exception: {exception_type}: {exception_message}",
            "last_check": time.time(),
            "error_stacktrace": traceback.format_exc(),
        }


def generate_miniwob_rollout(cfg: DictConfig, llm: TrainableLLM, problem: dict) -> RolloutResult:
    # make agent and env
    # set the llm
    # run the agent
    # get llm calls from tape
    # compute rewards
    # get training text from llm calls

    start_time = time.perf_counter()
    environment: WebEnvironment = instantiate(cfg.environment)
    environment.initialize()
    logger.info(f"Environment tools: {environment.tools_description()}")
    agent: Agent = instantiate(
        cfg.agent,
        known_actions=environment.actions(),
        tools_description=environment.tools_description(),
        llms={DEFAULT: llm},
    )
    logger.info(f"Agent and environment loaded, using llm {llm.model_name} at {llm.get_base_url()}")
    env_agent_creation_time = time.perf_counter() - start_time
    try:
        start_attempts = cfg.start_attempts
        t = time.perf_counter()
        while True:
            try:
                tape, _ = environment.start_task(problem)
                break
            except Exception as e:
                logger.exception(f"Failed to start task {problem['dataset']}/{problem['task']}/{problem['seed']}: {e}")
                start_attempts -= 1
                if start_attempts <= 0:
                    raise Exception(
                        f"Failed to start task {problem['dataset']}/{problem['task']}/{problem['seed']} after {cfg.start_attempts} attempts"
                    )
                else:
                    logger.warning("retry after 1 seconds")
                    time.sleep(1)
        env_start_time = time.perf_counter() - t
        logger.info(
            f"Task {problem['dataset']}/{problem['task']}/{problem['seed']} started in {env_start_time:.2f} seconds"
        )
        logger.info(f"Running agent for task {problem['dataset']}/{problem['task']}/{problem['seed']}")
        ex_t = time.perf_counter()
        tape = execute_agent(agent, tape, environment, max_loops=cfg.agent_max_loops)
        execution_time = time.perf_counter() - ex_t
    finally:
        close_t = time.perf_counter()
        environment.close()
        env_close_time = time.perf_counter() - close_t
    logger.info(
        f"Agent finished task {problem['dataset']}/{problem['task']}/{problem['seed']}, times: start {env_start_time:.2f} sec, exec {execution_time:.2f} sec, close {env_close_time:.2f} sec, produced tape with {len(tape.steps)} steps"
    )
    total_execution_time = time.perf_counter() - t
    tape.metadata.result.update(
        {
            "total_execution_time": total_execution_time,
            "env_start_time": env_start_time,
            "env_agent_creation_time": env_agent_creation_time,
            "execution_time": execution_time,
            "env_close_time": env_close_time,
        }
    )

    # save the tape as we go
    if cfg.save_tapes:
        tape_name = problem.get("_task_id", tape.metadata.id)
        try:
            save_json_tape(tape, os.path.join(cfg.output_dir, "tapes"), tape_name)
        except Exception as e:
            logger.error(f"Error saving tape {tape_name}: {e}")

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

    # get the number of LLMOutputParsingFailureAction in the tape
    n_step_errors = len([step for step in tape.steps if isinstance(step, LLMOutputParsingFailureAction)])
    # get the number of PageObservation steps in the tape
    n_page_observations = len([step for step in tape.steps if isinstance(step, PageObservation)])

    if obs_steps:
        last_obs = obs_steps[-1]
        # in Miniwob, the observation "reward" is defined as RAW_REWARD_GLOBAL > 0
        # see here: https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/miniwob/src/browsergym/miniwob/base.py#L188
        # Let's take directly the RAW_REWARD_GLOBAL from the metadata
        # raw_reward = last_obs.metadata.other.get("reward", 0.0)
        raw_reward = last_obs.metadata.other.get("info", {}).get("task_info", {}).get("REWARD_GLOBAL", -1.0)
    else:
        raw_reward = -1.0

    no_error = not tape_contains_an_error(tape)
    # get the number of LLMOutputParsingFailureAction in the tape
    n_step_errors = len([step for step in tape.steps if isinstance(step, LLMOutputParsingFailureAction)])
    # get the number of PageObservation steps in the tape
    n_page_observations = len([step for step in tape.steps if isinstance(step, PageObservation)])

    if cfg.reward_computation == "uic":
        reward = float(raw_reward > 0)
        if reward == 0.0:
            reward = -1.0
        reward *= 0.98**n_page_observations
    else:
        reward = raw_reward * 0.99**n_step_errors if no_error and raw_reward >= 0 else -1.0

    # (3) Get LLM calls from Tape
    llm_calls: list[LLMCall] = [
        LLMCall(**step.metadata.other["llm_call"])
        if isinstance(step.metadata.other["llm_call"], dict)
        else step.metadata.other["llm_call"]
        for step in tape.steps
        if "llm_call" in step.metadata.other
    ]
    llm_call_times = [float(step.metadata.other.get("llm_call_time", 0.0)) for step in tape.steps]
    env_call_times = [float(step.metadata.other.get("action_execution_time", 0.0)) for step in tape.steps]
    total_llm_call_time = sum(llm_call_times)
    total_env_call_time = sum(env_call_times)
    llm_call_time = total_llm_call_time / len(llm_call_times) if len(llm_call_times) > 0 else -1.0
    env_call_time = total_env_call_time / len(env_call_times) if len(env_call_times) > 0 else -1.0

    # (4) # For each LLM interaction in the tape, make a training example.
    all_finished = 1
    training_texts = [make_training_text(llm, llm_call) for llm_call in llm_calls]
    for text in training_texts:
        text.reward = reward
        all_finished &= 1 if text.input_ids[-1] == llm.tokenizer.eos_token_id else 0

    latency = time.perf_counter() - start_time
    agent_time = tape.metadata.result.get("agent_execution_time", -1.0)
    env_time = tape.metadata.result.get("environment_execution_time", -1.0)
    n_observations = len([s for s in tape.steps if isinstance(s, Observation)])
    n_other_steps = len(tape.steps) - n_observations
    metrics = MiniwobMetrics(
        reward=reward,
        success=reward > 0.5,
        no_error=no_error,
        no_answer=reward < 0,
        overflow=not all_finished,
        n_llm_calls=len(llm_calls),
        n_step_errors=n_step_errors,
        n_page_observations=n_page_observations,
        n_steps=len(tape.steps),
        total_execution_time=total_execution_time,
        env_start_time=env_start_time,
        env_close_time=env_close_time,
        env_agent_creation_time=env_agent_creation_time,
        agent_execution_time=agent_time,
        environment_execution_time=env_time,
        env_step_time=env_time / n_observations if env_time > 0 and n_observations > 0 else -1.0,
        agent_step_time=agent_time / n_other_steps if agent_time > 0 and n_other_steps > 0 else -1.0,
        llm_call_time=llm_call_time,
        env_call_time=env_call_time,
        total_llm_call_time=total_llm_call_time,
        total_env_call_time=total_env_call_time,
    )

    return RolloutResult(
        training_texts=training_texts,
        metrics=metrics,
        latency=latency,
        dataset_name=problem["dataset"],
    )


async def generate_miniwob_rollout_async(
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
    rollout_timeout = getattr(cfg, "rollout_timeout", 600)  # 10 minutes default

    env_jobs = [Job(**job) for job in cfg.jobs if job["kind"] == "environment"]
    env_jobs_url_tried = []

    # Try each environment server with health checks until one of them returns a rollout result
    for _ in range(len(env_jobs)):
        # Choose the next environment server to try randomly from the ones that have not been tried yet
        env_job = random.choice(
            [job for job in env_jobs if f"http://{job.hostname}:{job.port}" not in env_jobs_url_tried]
        )
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
                timeout=rollout_timeout,
            )
        except asyncio.TimeoutError:
            health = await check_env_server_health(env_job, session)
            if stack_trace := health.get("error_stacktrace"):
                logger.warning(f"Get health error stacktrace: {stack_trace}")
            logger.warning(f"Rollout timeout error stacktrace: {traceback.format_exc()}")
            logger.warning(
                f"Rollout timed out after {rollout_timeout} seconds for task {problem['dataset']}/{problem['task']}/{problem['seed']} on environment {env_job_url}. Health: {health}. Trying next server."
            )
            continue
        except Exception as e:
            health = await check_env_server_health(env_job, session)
            if stack_trace := health.get("error_stacktrace"):
                logger.warning(f"Get health error stacktrace: {stack_trace}")
            logger.warning(f"Rollout failed error stacktrace: {traceback.format_exc()}")
            logger.warning(
                f"Rollout failed for task {problem['dataset']}/{problem['task']}/{problem['seed']} on environment {env_job_url}. Health: {health}. Trying next server."
            )
            continue
    # If all servers failed
    logger.error(
        f"All environment servers failed for task {problem['dataset']}/{problem['task']}/{problem['seed']}. Returning a failed rollout result."
    )
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
    t = time.perf_counter()
    environment = AsyncRemoteEnvironment(server_url=env_job_url)  # type: ignore
    async with environment.acontext(session, wait_for_env=True) as env:
        env_agent_creation_time = time.perf_counter() - t
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
                logger.warning(
                    f"Failed to start task {problem['dataset']}/{problem['task']}/{problem['seed']}. {start_attempts} attempts remaining. Error: {e}"
                )
                if start_attempts <= 0:
                    logger.error(f"Failed to start task after all retry attempts: {e}")
                    no_error = False
                    break
                else:
                    logger.warning("Retry start task after 5 seconds.")
                    await asyncio.sleep(5)
        env_start_time = time.perf_counter() - t
        logger.info(
            f"Task {problem['dataset']}/{problem['task']}/{problem['seed']} started in {env_start_time:.2f} seconds. Worker ID: {env.worker_id}. Tape dict: {tape_dict}"
        )
        tape: WebTape = WebTape(**tape_dict)  # convert http response dict to WebTape object
        t = time.perf_counter()
        if no_error:  # only run the agent if the task started successfully
            logger.info(
                f"Running agent for task {problem['dataset']}/{problem['task']}/{problem['seed']} with worker ID: {env.worker_id} and tape ID {tape.metadata.id}"
            )
            agent_attempts = cfg.agent_attempts
            while agent_attempts > 0:
                # check if the worker is alive.
                try:
                    # this will either raise RuntimeError if worker is not alive anymore, or return a dictionary with the worker status
                    worker_status = await env.check_worker_alive()
                    if worker_status.get("status") == "starting":
                        logger.warning(
                            f"Worker {env.worker_id} for task {problem['dataset']}/{problem['task']}/{problem['seed']} and tape ID {tape.metadata.id} is starting, waiting 5 seconds for it to be fully started."
                        )
                        await asyncio.sleep(5)
                        continue
                except Exception as e:
                    # if worker is dead, no need to retry
                    logger.exception(
                        f"Worker {env.worker_id} for task {problem['dataset']}/{problem['task']}/{problem['seed']} and tape ID {tape.metadata.id} is dead. Error: {e}",
                        stack_info=True,
                    )
                    no_error = False
                    break
                # if worker is alive, run the agent
                try:
                    t = time.perf_counter()
                    actions = await env.a_actions()
                    tools_description = await env.a_tools_description()
                    agent: Agent = instantiate(cfg.agent, known_actions=actions, tools_description=tools_description)
                    agent.llms = {DEFAULT: llm}  # type: ignore
                    env_agent_creation_time += time.perf_counter() - t
                    tape = await async_execute_agent(agent, tape, env, session, max_loops=cfg.agent_max_loops)
                    # Check if the tape has an error from the orchestrator (e.g., SocketTimeoutError, RuntimeError: Worker is not alive, etc.)
                    if tape.metadata.error:
                        logger.error(
                            f"Agent execution for task {problem['dataset']}/{problem['task']}/{problem['seed']} with worker ID: {env.worker_id} and tape ID {tape.metadata.id} returned a tape with error: {tape.metadata.error}"
                        )
                        raise ValueError(tape.metadata.error)
                    else:
                        # Success - break out of retry loop
                        logger.info(
                            f"Agent execution for task {problem['dataset']}/{problem['task']}/{problem['seed']} with worker ID: {env.worker_id} and tape ID {tape.metadata.id} finished successfully"
                        )
                        break
                except Exception as e:
                    agent_attempts -= 1
                    logger.warning(
                        f"Error occurred while running agent for task {problem['dataset']}/{problem['task']}/{problem['seed']} with worker ID: {env.worker_id} and tape ID {tape.metadata.id}. {agent_attempts} attempts remaining. Error: {e}"
                    )
                    if agent_attempts <= 0:
                        logger.error(
                            f"Agent execution failed after all retry attempts for task {problem['dataset']}/{problem['task']}/{problem['seed']} with worker ID: {env.worker_id} and tape ID {tape.metadata.id}: {e}"
                        )
                        no_error = False
                        break
                    else:
                        logger.warning(
                            f"Retry agent execution after 5 seconds for task {problem['dataset']}/{problem['task']}/{problem['seed']} with worker ID: {env.worker_id} and tape ID {tape.metadata.id}."
                        )
                        await asyncio.sleep(5)
            logger.info(
                f"Agent finished task {problem['dataset']}/{problem['task']}/{problem['seed']} in {time.perf_counter() - t:.2f} seconds with worker ID: {env.worker_id} and tape ID {tape.metadata.id}"
            )
        tape.metadata.result.update({"total_execution_time": time.perf_counter() - t})
        t = time.perf_counter()
        await env.aclose()
        env_close_time = time.perf_counter() - t

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

    if cfg.reward_computation == "uic":
        reward = float(raw_reward > 0)
        if reward == 0.0:
            reward = -1.0
        reward *= 0.98**n_page_observations
    else:
        reward = raw_reward * 0.99**n_step_errors if no_error and raw_reward >= 0 else -1.0

    # (3) Get LLM calls from Tape
    llm_calls: list[LLMCall] = [
        LLMCall(**step.metadata.other["llm_call"])
        if isinstance(step.metadata.other["llm_call"], dict)
        else step.metadata.other["llm_call"]
        for step in tape.steps
        if "llm_call" in step.metadata.other
    ]

    # (4) # For each LLM interaction in the tape, make a training example.
    all_finished = 1
    training_texts = [make_training_text(llm, llm_call) for llm_call in llm_calls]
    for text in training_texts:
        text.reward = reward
        all_finished &= 1 if text.input_ids[-1] == llm.tokenizer.eos_token_id else 0

    latency = time.time() - start_time
    agent_time = tape.metadata.result.get("agent_execution_time", -1.0)
    env_time = tape.metadata.result.get("environment_execution_time", -1.0)
    n_observations = len([s for s in tape.steps if isinstance(s, Observation)])
    n_other_steps = len(tape.steps) - n_observations

    llm_call_times = [float(step.metadata.other.get("llm_call_time", 0.0)) for step in tape.steps]
    env_call_times = [float(step.metadata.other.get("action_execution_time", 0.0)) for step in tape.steps]
    total_llm_call_time = sum(llm_call_times)
    total_env_call_time = sum(env_call_times)
    llm_call_time = total_llm_call_time / len(llm_call_times) if len(llm_call_times) > 0 else -1.0
    env_call_time = total_env_call_time / len(env_call_times) if len(env_call_times) > 0 else -1.0

    metrics = MiniwobMetrics(
        reward=reward,
        success=reward > 0.5,
        no_error=no_error,
        no_answer=reward < 0,
        overflow=not all_finished,
        n_llm_calls=len(llm_calls),
        n_step_errors=n_step_errors,
        n_page_observations=n_page_observations,
        n_steps=len(tape.steps),
        total_execution_time=tape.metadata.result.get("total_execution_time", -1.0),
        agent_execution_time=agent_time,
        environment_execution_time=env_time,
        env_step_time=env_time / n_observations if env_time > 0 and n_observations > 0 else -1.0,
        agent_step_time=agent_time / n_other_steps if agent_time > 0 and n_other_steps > 0 else -1.0,
        llm_call_time=llm_call_time,
        env_call_time=env_call_time,
        total_llm_call_time=total_llm_call_time,
        total_env_call_time=total_env_call_time,
        env_start_time=env_start_time,
        env_close_time=env_close_time,
        env_agent_creation_time=env_agent_creation_time,
    )

    return RolloutResult(
        training_texts=training_texts,
        metrics=metrics,
        latency=latency,
        dataset_name=problem["dataset"],
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
        agent_execution_time=0.0,
        environment_execution_time=0.0,
        env_step_time=0.0,
        agent_step_time=0.0,
        llm_call_time=0.0,
        env_call_time=0.0,
        total_llm_call_time=0.0,
        total_env_call_time=0.0,
        env_start_time=0.0,
        env_close_time=0.0,
        env_agent_creation_time=0.0,
    )

    return RolloutResult(
        training_texts=[],
        metrics=metrics,
        latency=latency,
        dataset_name=problem["dataset"],
    )
