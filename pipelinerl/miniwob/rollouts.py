
import asyncio
import logging
import os
import random
import time
import aiohttp
from hydra.utils import instantiate
from omegaconf import DictConfig

from pipelinerl.async_llm import llm_async_generate, make_training_text
from pipelinerl.rollouts import RolloutResult
from pipelinerl.world import Job
from tapeagents.agent import Agent, DEFAULT
from tapeagents.core import LLMOutputParsingFailureAction, Observation, LLMCall
from tapeagents.llms.trainable import TrainableLLM
from tapeagents.remote_environment import AsyncRemoteEnvironment
from tapeagents.tools.simple_browser import PageObservation
from tapeagents.orchestrator import async_execute_agent
from tapeagents.io import save_json_tape
from examples.rl_webagent.steps import WebTape


logger = logging.getLogger(__name__)


def tape_contains_an_error(tape: WebTape) -> bool:
    """
    Returns true if the tape ends with an error, ie if one of the following is true:
    - the last step is an LLMOutputParsingFailureAction
    - the tape metadata has an error
    - the last step is a PageObservation with an error
    """
    return (
        isinstance(tape.steps[-1], LLMOutputParsingFailureAction)
        or tape.metadata.result.get("error") is not None
        or (isinstance(tape.steps[-1], PageObservation) and tape.steps[-1].error)
    )


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

    # (1) Choose a random environment server
    env_jobs = [Job(**job) for job in cfg.jobs if job["kind"] == "environment"]
    # choose the env job randomly
    env_job = random.choice(env_jobs)
    assert env_job.port is not None
    env_job_url = f"http://{env_job.hostname}:{env_job.port}"

    # (2) Generate environment, TapeAgent, and run them to get a Tape
    environment = AsyncRemoteEnvironment(server_url=env_job_url)  # type: ignore
    async with environment.acontext(session, wait_for_env=True) as env:
        start_attempts = cfg.start_attempts
        t = time.perf_counter()
        while True:
            try:
                tape_dict, _ = await env.start_task(problem)
                break
            except Exception as e:
                start_attempts -= 1
                if start_attempts <= 0:
                    raise e
                logger.warning(f"Failed to start task, retry after 5 seconds: {e}")
                await asyncio.sleep(5)
        logger.info(f"Task {problem['dataset']}/{problem['task']}/{problem['seed']} started in {time.perf_counter() - t:.2f} seconds")
        tape: WebTape = WebTape(**tape_dict)  # convert http response dict to WebTape object
        t = time.perf_counter()
        try:
            actions = await env.a_actions()
            tools_description = await env.a_tools_description()
            logger.debug(f"Available tools: {tools_description}")
            agent: Agent = instantiate(cfg.agent, known_actions=actions, tools_description=tools_description)
            agent.llms = {DEFAULT: llm}
            tape = await async_execute_agent(agent, tape, env, session, max_loops=cfg.agent_max_loops)
        except Exception as e:
            logger.error(f"Error occurred while running agent: {e}")
        tape.metadata.result = {"execution_time": time.perf_counter() - t}

    # save the tape as we go
    if cfg.save_tapes:
        save_json_tape(tape, os.path.join(cfg.output_dir, "tapes"), tape.metadata.id)

    # (3) Compute rewards
    last_obs = [step for step in tape if isinstance(step, Observation)][-1]
    # in Miniwob, the observation "reward" is defined as RAW_REWARD_GLOBAL > 0
    # see here: https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/miniwob/src/browsergym/miniwob/base.py#L183
    # Let's take directly the RAW_REWARD_GLOBAL from the metadata
    # raw_reward = last_obs.metadata.other.get("reward", 0.0)
    raw_reward = last_obs.metadata.other.get("info", {}).get("task_info", {}).get("REWARD_GLOBAL", -1.0)
    no_error = not tape_contains_an_error(tape)
    # get the number of LLMOutputParsingFailureAction in the tape
    n_step_errors = len([step for step in tape.steps if isinstance(step, LLMOutputParsingFailureAction)])
    # get the number of PageObservation steps in the tape
    n_page_observations = len([step for step in tape.steps if isinstance(step, PageObservation)])

    reward = raw_reward * 0.99**n_step_errors if no_error and raw_reward >= 0 else -1.0

    # (3) Get LLM calls from Tape
    llm_calls = [step for step in tape.steps if step.metadata.other.get("llm_call") is not None]
    n_llm_calls = len(llm_calls)
    llm_calls: list[LLMCall] = [
        LLMCall(**step.metadata.other["llm_call"]) if isinstance(step.metadata.other["llm_call"], dict)
        else step.metadata.other["llm_call"]
        for step in llm_calls
    ]

    # (4) # For each LLM interaction in the tape, make a training example.
    all_finished = 0
    prompt_tokens = [llm_call.prompt_length_tokens for llm_call in llm_calls]
    output_tokens = [llm_call.output_length_tokens for llm_call in llm_calls]
    training_texts = [make_training_text(llm, llm_call) for llm_call in llm_calls]
    for text in training_texts:
        text.reward = reward
        all_finished &= 1 if text.input_ids[-1] == llm.tokenizer.eos_token_id else 0

    latency = time.time() - start_time

    metrics = {
        "reward": reward,
        "success": 1 if reward > 0.5 else 0,
        "no_error": no_error,
        "no_answer": 1 if reward < 0 else 0,
        "overflow": 0 if all_finished else 1,
        "n_llm_calls": n_llm_calls,
        "n_step_errors": n_step_errors,
        "n_page_observations": n_page_observations,
        "n_steps": len(tape.steps),
    }

    return RolloutResult(
        training_texts=training_texts,
        metrics=metrics,
        latency=latency,
        dataset_name=problem["dataset"],
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
    )

