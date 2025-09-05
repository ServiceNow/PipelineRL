import asyncio
from urllib.parse import urlparse
import time
import random
import logging 
from collections import Counter
from typing import List, Dict

import aiohttp
from omegaconf import DictConfig
from pipelinerl.domains.mcp.steps import MathAnswer
from pipelinerl.world import Job
from tapeagents.llms.trainable import TrainableLLM
from tapeagents.remote_environment import AsyncRemoteEnvironment
from pipelinerl.async_llm import make_training_text
from tapeagents.orchestrator import async_execute_agent
from tapeagents.agent import DEFAULT, Agent
from hydra.utils import instantiate
from tapeagents.core import Tape
from tapeagents.dialog_tape import UserStep
from tapeagents.core import LLMCall

from pipelinerl.domains.math import verify_answer_rpc, RewardTable, get_reward
from pipelinerl.rollouts import RolloutResult, BaseMetrics

logger = logging.getLogger(__name__)


def count_tool_calls_by_category(llm_calls: List[LLMCall]) -> Dict[str, int]:
    """
    Count the number of tool calls for each function name category.
    
    Args:
        llm_calls: List of LLMCall objects
        
    Returns:
        Dictionary mapping function names to their counts
    """
    tool_call_names = []
    
    for llm_call in llm_calls:
        if llm_call.output.tool_calls:
            for tool_call in llm_call.output.tool_calls:
                tool_call_names.append(tool_call.function.name)
    
    return dict(Counter(tool_call_names))


class Metrics(BaseMetrics):
    num_python_calls: int = 0
    num_steps: int = 0
    n_llm_calls: int = 0
    total_execution_time: float = -1.0
    agent_execution_time: float = -1.0
    environment_execution_time: float = -1.0

async def generate_mcp_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    # choose and retry env servers if one is saturated
    start = time.perf_counter()
    env_jobs = [Job(**job) for job in cfg.jobs if job["kind"] == "environment"]
    if not env_jobs:
        raise RuntimeError("No environment servers available")

    # shuffle to avoid dead-locking a single server
    env_urls_all = [f"http://{job.hostname}:{job.port}" for job in env_jobs if job.port is not None]
    if not env_urls_all:
        raise RuntimeError("Environment server definitions missing ports")

    while True:
        env_urls = env_urls_all[:]
        random.shuffle(env_urls)
        chosen_url = None
        for env_url in env_urls:
            try:
                environment = AsyncRemoteEnvironment(
                    server_url=env_url, start_timeout_sec=600, start_repeat_delay=5)
                context_manager = environment.acontext(session, wait_for_env=True)
                env = await context_manager.__aenter__()
                try:
                    await env.start_task(problem)
                    chosen_url = env_url
                    actions = await env.a_actions()
                    tools_description = await env.a_tools_description()
                    logger.debug(f"Available tools: {tools_description}")
                    agent: Agent = instantiate(cfg.agent, known_actions=actions, tools_description=tools_description)
                    agent.llms = {DEFAULT: llm}

                    tape = Tape(steps=[
                        UserStep(content=f"{problem['task']}. You have access to the following tools: {tools_description}")
                    ])
                    t_exec = time.perf_counter()
                    while True:
                        try:
                            tape = await async_execute_agent(agent, tape, env, session, max_loops=cfg.agent_max_loops)
                            tape.metadata.result.update({"total_execution_time": time.perf_counter() - t_exec})
                            break
                        except Exception:
                            await asyncio.sleep(5)
                    break  # success
                finally:
                    await context_manager.__aexit__(None, None, None)
            except Exception as e:
                # try the next server on errors (503: busyslots)
                logger.warning(f"Env start failed at {env_url}: {e}")
                continue
        if chosen_url is not None:
            break  # success
        # if none succeeded backoff and retry the whole list
        await asyncio.sleep(1.0)

    reward_table = RewardTable(**dict(cfg.rewards))

    llm_calls: list[LLMCall] = [
        LLMCall(**step.metadata.other["llm_call"])
        if isinstance(step.metadata.other["llm_call"], dict)
        else step.metadata.other["llm_call"]
        for step in tape.steps if step.metadata.other.get("llm_call") is not None
    ]
    assert len(llm_calls) > 0, "No LLM calls found"
    tool_call_counts = count_tool_calls_by_category(llm_calls)
    training_texts = [make_training_text(llm, llm_call) for llm_call in llm_calls]
    n_llm_calls = len(llm_calls)
    parsed = urlparse(chosen_url)
    assert parsed.hostname is not None and parsed.port is not None
    answer_status = await verify_answer_rpc(
        session=session,
        host=parsed.hostname,
        port=parsed.port,
        prediction=llm_calls[-1].output.content,  # type: ignore
        gold=problem["answer"],
        strict=True,
    )
    # Tape should finish with an answer
    tape_finished = True if isinstance(tape.steps[-1], MathAnswer) else False
    reward = get_reward(answer_status, tape_finished, reward_table)
    for text in training_texts:
        text.reward = reward

    latency = time.perf_counter() - start

    agent_time = tape.metadata.result.get("agent_execution_time", -1.0)
    env_time = tape.metadata.result.get("environment_execution_time", -1.0)
    total_time = tape.metadata.result.get("total_execution_time", -1.0)
    
    
    metrics = Metrics(
        reward=reward,
        success=answer_status == "correct",
        no_error=answer_status != "unparsable",
        no_answer=answer_status == "no_answer",
        num_steps=len(tape.steps),
        num_python_calls=tool_call_counts.get("run_python_code", 0),
        n_llm_calls=n_llm_calls,
        total_execution_time=total_time,
        agent_execution_time=agent_time,
        environment_execution_time=env_time,
    )

    return RolloutResult(
        training_texts=training_texts,
        metrics=metrics,
        latency=latency,
        dataset_name=problem["dataset"],
    )
