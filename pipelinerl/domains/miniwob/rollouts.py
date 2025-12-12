import asyncio
import logging
import os
import random
import time
from typing import Any

import aiohttp
from examples.rl_webagent.environment import WebEnvironment
from examples.rl_webagent.generic_agent import GenericWebAgent
from examples.rl_webagent.steps import WebTape
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tapeagents.agent import DEFAULT
from tapeagents.core import LLMCall, LLMOutputParsingFailureAction, Observation
from tapeagents.io import save_json_tape
from tapeagents.llms.trainable import TrainableLLM
from tapeagents.orchestrator import async_execute_agent, execute_agent, get_agent_and_env_from_config
from tapeagents.remote_environment import AsyncRemoteEnvironment
from tapeagents.tools.simple_browser import PageObservation

from pipelinerl.async_llm import make_training_text
from pipelinerl.rollouts import BaseMetrics, RolloutResult
from pipelinerl.world import Job

logger = logging.getLogger(__name__)


class MiniwobMetrics(BaseMetrics):
    reward: float
    raw_success: float  # 1.0 if raw_reward > 0, else 0.0 (used for raw advantage calculation)
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
        or (isinstance(tape.steps[-1], PageObservation) and tape.steps[-1].error)
    )


def _create_generic_agent(
    cfg: DictConfig,
    llm: TrainableLLM,
    actions: Any,
    tools_description: Any,
) -> GenericWebAgent:
    generic_cfg: dict[str, Any] = {}
    if "generic_agent" in cfg:
        raw_cfg = cfg.generic_agent
        if isinstance(raw_cfg, DictConfig):
            generic_cfg = OmegaConf.to_container(raw_cfg, resolve=True)  # type: ignore[assignment]
        elif isinstance(raw_cfg, dict):
            generic_cfg = raw_cfg
    use_examples = generic_cfg.get("use_examples", True)
    max_iterations = generic_cfg.get("max_iterations", 10)
    max_retries = generic_cfg.get("max_retries")
    agent = GenericWebAgent.create(llm, max_iterations=max_iterations, use_examples=use_examples)
    agent.known_actions = actions
    agent.tools_description = tools_description

    max_chars = generic_cfg.get("max_chars_page_observation")
    include_think_in_history = generic_cfg.get("include_think_in_history", True)
    for node in agent.nodes:
        if max_chars is not None and hasattr(node, "max_chars_page_observation"):
            node.max_chars_page_observation = max_chars
        if max_retries is not None and hasattr(node, "max_retries"):
            node.max_retries = max_retries
            node.current_retries = 0
        if hasattr(node, "include_think_in_history"):
            node.include_think_in_history = include_think_in_history

    return agent


def generate_miniwob_rollout(cfg: DictConfig, llm: TrainableLLM, problem: dict) -> RolloutResult:
    # make agent and env
    # set the llm
    # run the agent
    # get llm calls from tape
    # compute rewards
    # get training text from llm calls

    start_time = time.perf_counter()

    creation_start = time.perf_counter()
    if getattr(cfg, "use_generic_agent", False):
        environment = instantiate(cfg.environment)
        environment.initialize()
        logger.info(f"Environment tools: {environment.tools_description()}")
        agent = _create_generic_agent(cfg, llm, environment.actions(), environment.tools_description())
    else:
        agent, environment = get_agent_and_env_from_config(cfg)
    env_agent_creation_time = time.perf_counter() - creation_start
    try:
        agent.llms = {DEFAULT: llm}
        logger.info(f"Agent and environment loaded, using llm {llm.model_name} at {llm.get_base_url()}")
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
                    raise Exception(f"Failed to start task {problem['dataset']}/{problem['task']}/{problem['seed']} after {cfg.start_attempts} attempts")
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
    tape.metadata.result.update({"total_execution_time": total_execution_time, "env_start_time": env_start_time, "env_agent_creation_time": env_agent_creation_time, "execution_time": execution_time, "env_close_time": env_close_time})

    # save the tape as we go
    if cfg.save_tapes:
        try:
            # Add metadata for tracking
            tape.metadata.other.update({
                "split": "train" if problem.get("_is_training", True) else "test",
                "model_version": problem.get("_model_version", -1),
                "llm_model_name": llm.model_name,
                "llm_base_url": llm.get_base_url(),
            })
            tape_name = problem.get("_task_id", tape.metadata.id)
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
        raw_reward = last_obs.metadata.other.get("info", {}).get("task_info", {}).get("REWARD_GLOBAL", -1.0)
    else:
        raw_reward = -1.0

    no_error = not tape_contains_an_error(tape)
    # get the number of LLMOutputParsingFailureAction in the tape
    n_step_errors = len([step for step in tape.steps if isinstance(step, LLMOutputParsingFailureAction)])
    # get the number of PageObservation steps in the tape
    n_page_observations = len([step for step in tape.steps if isinstance(step, PageObservation)])

    if cfg.reward_computation == "nico":
        reward = raw_reward * 0.99**n_step_errors if no_error and raw_reward >= 0 else -1.0
    elif cfg.reward_computation == "massimo":
        discount_factor = cfg.actor.discount_factor
        reward = float(raw_reward > 0)
        if reward == 0.0:
            reward = -1.0
        reward *= discount_factor ** n_page_observations
    else:
        raise ValueError(f"Invalid reward configuration: {cfg.reward_computation}")

    # Raw success: 1.0 if task succeeded, 0.0 otherwise (for raw advantage calculation)
    raw_success = 1.0 if raw_reward > 0 else 0.0

    # (3) Get LLM calls from Tape
    llm_calls = [step for step in tape.steps if step.metadata.other.get("llm_call") is not None]
    n_llm_calls = len(llm_calls)
    llm_calls: list[LLMCall] = [
        LLMCall(**step.metadata.other["llm_call"])
        if isinstance(step.metadata.other["llm_call"], dict)
        else step.metadata.other["llm_call"]
        for step in llm_calls
    ]
    llm_call_times = [step.metadata.other.get("llm_call_time") for step in tape.steps if"llm_call_time" in step.metadata.other]
    env_call_times = [step.metadata.other.get("action_execution_time") for step in tape.steps if"action_execution_time" in step.metadata.other]
    total_llm_call_time = sum(llm_call_times)
    total_env_call_time = sum(env_call_times)
    llm_call_time = total_llm_call_time / len(llm_call_times) if len(llm_call_times) > 0 else -1.0
    env_call_time = total_env_call_time / len(env_call_times) if len(env_call_times) > 0 else -1.0

    # (4) # For each LLM interaction in the tape, make a training example.
    all_finished = 1
    prompt_tokens = [llm_call.prompt_length_tokens for llm_call in llm_calls]
    output_tokens = [llm_call.output_length_tokens for llm_call in llm_calls]
    training_texts = [make_training_text(llm, llm_call) for llm_call in llm_calls]
    for text in training_texts:
        text.reward = reward
        text.metadata["raw_success"] = raw_success  # Store raw success for raw advantage calculation
        all_finished &= 1 if text.input_ids[-1] == llm.tokenizer.eos_token_id else 0

    latency = time.perf_counter() - start_time
    agent_time = tape.metadata.result.get("agent_execution_time", -1.0)
    env_time = tape.metadata.result.get("environment_execution_time", -1.0)
    n_observations = len(
        [s for s in tape.steps if isinstance(s, Observation)]
    )  # TODO: is this not the same n_page_observations??
    n_other_steps = len(tape.steps) - n_observations
    metrics = MiniwobMetrics(
        reward=reward,
        raw_success=raw_success,
        success=reward > 0.5,
        no_error=no_error,
        no_answer=reward < 0,
        overflow=not all_finished,
        n_llm_calls=n_llm_calls,
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
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
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

    # (1) Choose a random environment server
    env_jobs = [Job(**job) for job in cfg.jobs if job["kind"] == "environment"]
    # choose the env job randomly
    env_job = random.choice(env_jobs)
    assert env_job.port is not None
    env_job_url = f"http://{env_job.hostname}:{env_job.port}"

    # (2) Generate environment, TapeAgent, and run them to get a Tape
    no_error = True  # track if there was an error in the tape
    environment = AsyncRemoteEnvironment(server_url=env_job_url)  # type: ignore
    async with environment.acontext(session, wait_for_env=True) as env:
        start_attempts = cfg.start_attempts
        t = time.perf_counter()
        while True:
            try:
                tape_dict, _ = await env.start_task(problem)
                break
            except Exception as e:
                logger.warning(f"Failed to start task {problem['dataset']}/{problem['task']}/{problem['seed']}")
                start_attempts -= 1
                if start_attempts <= 0:
                    no_error = False
                    tape_dict = {}
                    break
                else:
                    logger.warning(f"retry after 5 seconds: {e}")
                    await asyncio.sleep(5)
        logger.info(
            f"Task {problem['dataset']}/{problem['task']}/{problem['seed']} started in {time.perf_counter() - t:.2f} seconds"
        )
        tape: WebTape = WebTape(**tape_dict)  # convert http response dict to WebTape object
        t = time.perf_counter()
        if no_error:  # only run the agent if the task started successfully
            logger.info(f"Running agent for task {problem['dataset']}/{problem['task']}/{problem['seed']}")
            try:
                actions = await env.a_actions()
                tools_description = await env.a_tools_description()
                logger.debug(f"Available tools: {tools_description}")
                if getattr(cfg, "use_generic_agent", False):
                    agent = _create_generic_agent(cfg, llm, actions, tools_description)
                else:
                    agent = instantiate(cfg.agent, known_actions=actions, tools_description=tools_description)
                agent.llms = {DEFAULT: llm}
                tape = await async_execute_agent(agent, tape, env, session, max_loops=cfg.agent_max_loops)
            except Exception as e:
                logger.error(f"Error occurred while running agent: {e}")
                no_error = False
            logger.info(
                f"Agent finished task {problem['dataset']}/{problem['task']}/{problem['seed']} in {time.perf_counter() - t:.2f} seconds"
            )
        tape.metadata.result.update({"total_execution_time": time.perf_counter() - t})

    # save the tape as we go
    if cfg.save_tapes:
        # Add metadata for tracking
        tape.metadata.other.update({
            "split": "train" if problem.get("_is_training", True) else "test",
            "model_version": problem.get("_model_version", -1),
            "llm_model_name": llm.model_name,
            "llm_base_url": llm.get_base_url(),
        })
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
    discount_factor = cfg.actor.discount_factor
    if cfg.reward_computation == "nico":
        reward = raw_reward * 0.99**n_step_errors if no_error and raw_reward >= 0 else -1.0
    elif cfg.reward_computation == "massimo":
        reward = float(raw_reward > 0)
        if reward <= 0.0:
            reward = -1.0
        reward *= discount_factor**n_page_observations

    # Raw success: 1.0 if task succeeded, 0.0 otherwise (for raw advantage calculation)
    raw_success = 1.0 if raw_reward > 0 else 0.0

    # (3) Get LLM calls from Tape
    llm_calls = [step for step in tape.steps if step.metadata.other.get("llm_call") is not None]
    n_llm_calls = len(llm_calls)
    llm_calls: list[LLMCall] = [
        LLMCall(**step.metadata.other["llm_call"])
        if isinstance(step.metadata.other["llm_call"], dict)
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
        text.metadata["raw_success"] = raw_success  # Store raw success for raw advantage calculation
        all_finished &= 1 if text.input_ids[-1] == llm.tokenizer.eos_token_id else 0

    latency = time.time() - start_time
    agent_time = tape.metadata.result.get("agent_execution_time", -1.0)
    env_time = tape.metadata.result.get("environment_execution_time", -1.0)
    n_observations = len([s for s in tape.steps if isinstance(s, Observation)])
    n_other_steps = len(tape.steps) - n_observations
    metrics = MiniwobMetrics(
        reward=reward,
        raw_success=raw_success,
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
