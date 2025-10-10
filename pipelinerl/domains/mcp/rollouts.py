import asyncio
import time
import random
import logging 
from collections import Counter
from typing import Dict, List

import aiohttp
from urllib.parse import urlparse
from omegaconf import DictConfig
from pipelinerl.domains.mcp.steps import MathAnswer
from pipelinerl.world import Job
from tapeagents.llms.trainable import TrainableLLM
from pipelinerl.async_llm import make_training_text
from tapeagents.environment import Environment
from tapeagents.orchestrator import async_execute_agent
from tapeagents.agent import DEFAULT, Agent
from hydra.utils import instantiate
from tapeagents.core import Tape
from tapeagents.dialog_tape import UserStep
from tapeagents.core import LLMCall
from tapeagents.remote_environment import AsyncRemoteEnvironment

from pipelinerl.async_llm import make_training_text
from pipelinerl.domains.mcp.env_server import EmbeddedEnvironmentWorker
from pipelinerl.domains.mcp.steps import MathAnswer
from pipelinerl.world import Job
from pipelinerl.domains.math import RewardTable, get_reward, verify_answer, verify_answer_rpc, length_penalty
from pipelinerl.rollouts import RolloutResult, BaseMetrics

logger = logging.getLogger(__name__)


_embedded_worker: EmbeddedEnvironmentWorker | None = None


def _get_embedded_worker(env_cfg: DictConfig, concurrency: int) -> EmbeddedEnvironmentWorker:
    global _embedded_worker
    concurrency = max(1, concurrency)
    if _embedded_worker is None or not _embedded_worker.matches(env_cfg):
        _embedded_worker = EmbeddedEnvironmentWorker(env_cfg, concurrency=concurrency)
    else:
        _embedded_worker.set_concurrency(concurrency)
    return _embedded_worker


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
    overflow: bool = False

async def generate_mcp_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    start = time.perf_counter()

    chosen_url: str | None = None
    env_host: str | None = None
    env_port: int | None = None

    if cfg.world.environment_mode == "remote":
        env_jobs = [Job(**job) for job in cfg.jobs if job["kind"] == "environment"]
        if not env_jobs:
            raise RuntimeError("No environment servers available")

        env_urls_all = [f"http://{job.hostname}:{job.port}" for job in env_jobs if job.port is not None]
        if not env_urls_all:
            raise RuntimeError("Environment server definitions missing ports")

        while True:
            env_urls = env_urls_all[:]
            random.shuffle(env_urls)
            chosen_url = None
            for env_url in env_urls:
                jitter = random.randint(3, 12)
                try:
                    environment = AsyncRemoteEnvironment(
                        server_url=env_url, start_timeout_sec=600, start_repeat_delay=jitter)
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
                    logger.warning(f"Env start failed at {env_url}: {e}")
                    continue
            if chosen_url is not None:
                break  # success
            await asyncio.sleep(1.0)

        parsed = urlparse(chosen_url)
        env_host, env_port = parsed.hostname, parsed.port
    else:
        concurrency = max(1, int(getattr(cfg.world, "env_replicas_per_actor", 1)))
        env_worker = _get_embedded_worker(cfg.environment, concurrency)
        async with env_worker.alifecycle() as environment:
            start_result = environment.start_task(problem)
            tape_metadata = start_result if isinstance(start_result, dict) else {}

            actions = environment.actions()
            tools_description = environment.tools_description()
            logger.debug(f"Embedded tools: {tools_description}")
            agent: Agent = instantiate(cfg.agent, known_actions=actions, tools_description=tools_description)
            agent.llms = {DEFAULT: llm}
            tape = Tape(
                steps=[
                    UserStep(
                        content=f"{problem['task']}. You have access to the following tools: {tools_description}"
                    )
                ]
            )
            if tape_metadata:
                tape.metadata.other.update(tape_metadata)

            t_exec = time.perf_counter()
            tape = await async_execute_agent(agent, tape, environment, session, max_loops=cfg.agent_max_loops)
            tape.metadata.result.update({"total_execution_time": time.perf_counter() - t_exec})
        env_host = env_port = None

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
    if env_host and env_port:
        answer_status = await verify_answer_rpc(
            session=session,
            host=env_host,
            port=env_port,
            prediction=llm_calls[-1].output.content,  # type: ignore
            gold=problem["answer"],
            strict=True,
        )
    else:
        answer_status = verify_answer(
            prediction=llm_calls[-1].output.content,  # type: ignore
            gold=problem["answer"],
            strict=True,
        )
    # Tape should finish with an answer
    tape_finished = True if isinstance(tape.steps[-1], MathAnswer) else False
    base_reward = get_reward(answer_status, tape_finished, reward_table)

    reward = base_reward

    discount_factor = float(getattr(cfg.actor, "discount_factor", 1.0))
    if discount_factor != 1.0:
        total_generated_tokens = sum(getattr(call, "output_length_tokens", 0) for call in llm_calls)
        reward *= discount_factor ** total_generated_tokens

    buffer_tokens = getattr(reward_table, "buffer_tokens", 0)
    if buffer_tokens:
        max_tokens = int(llm.parameters.get("max_tokens", 0))
        total_output_tokens = sum(getattr(text, "output_tokens", 0) for text in training_texts)
        if max_tokens > 0:
            reward += length_penalty(max_tokens, total_output_tokens, buffer_tokens)

    # Assign identical reward to all steps in the rollout (pipeline expects uniform rollout_reward)
    for text in training_texts:
        text.reward = reward
        text.finished = tape_finished

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
        overflow=not tape_finished,
    )

    return RolloutResult(
        training_texts=training_texts,
        metrics=metrics,
        latency=latency,
        dataset_name=problem["dataset"],
    )
