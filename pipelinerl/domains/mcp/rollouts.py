import asyncio
import time
import random
import logging 
from collections import Counter
from typing import List, Dict

import aiohttp
from omegaconf import DictConfig
from pydantic import BaseModel
from pipelinerl.domains.mcp.steps import MathAnswer
from pipelinerl.world import Job
from tapeagents.core import Prompt
from tapeagents.llms.trainable import TrainableLLM
from tapeagents.remote_environment import AsyncRemoteEnvironment
from pipelinerl.async_llm import llm_async_generate, make_training_text
from tapeagents.orchestrator import async_execute_agent
from tapeagents.agent import DEFAULT, Agent
from hydra.utils import instantiate
from tapeagents.core import StopStep, Tape
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

async def generate_mcp_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    # (1) Choose a random environment server
    start = time.perf_counter()
    env_jobs = [Job(**job) for job in cfg.jobs if job["kind"] == "environment"]
    # choose the env job randomly
    env_job = random.choice(env_jobs)
    assert env_job.port is not None
    env_job_url = f"http://{env_job.hostname}:{env_job.port}"
    environment = AsyncRemoteEnvironment(server_url=env_job_url)  # type: ignore
    async with environment.acontext(session, wait_for_env=True) as env:
        await env.start_task(problem)
        actions = await env.a_actions()
        tools_description = await env.a_tools_description()
        logger.debug(f"Available tools: {tools_description}")
        agent: Agent = instantiate(cfg.agent, known_actions=actions, tools_description=tools_description)
        agent.llms = {DEFAULT: llm}

        tape = Tape(steps=[
            UserStep(content=f"{problem['task']}. You have access to the following tools: {tools_description}")
            ])
        while True:
            try:
                tape = await async_execute_agent(agent, tape, env, session, max_loops=cfg.agent_max_loops)
                break
            except Exception:
                await asyncio.sleep(5)

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
    answer_status = await verify_answer_rpc(
        session=session,
        host=env_job.hostname,
        port=env_job.port, # type: ignore
        prediction=llm_calls[-1].output.content, # type: ignore
        gold=problem["answer"],
        strict=True,
    )
    # Tape should finish with an answer
    tape_finished = True if isinstance(tape.steps[-1], MathAnswer) else False
    reward = get_reward(answer_status, tape_finished, reward_table)
    for text in training_texts:
        text.reward = reward

    latency = time.perf_counter() - start

    
    metrics = Metrics(
        reward=reward,
        success=answer_status == "correct",
        no_error=answer_status != "unparsable",
        no_answer=answer_status == "no_answer",
        num_steps=len(tape.steps),
        num_python_calls=tool_call_counts.get("run_python_code", 0),
    )

    return RolloutResult(
        training_texts=training_texts,
        metrics=metrics,
        latency=latency,
        dataset_name=problem["dataset"],
    )
