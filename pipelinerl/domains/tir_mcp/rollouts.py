import time
import random
import logging 

import aiohttp
from omegaconf import DictConfig
from pydantic import BaseModel
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

from pipelinerl.domains.math import verify_answer_rpc, RewardTable, get_reward
from pipelinerl.rollouts import RolloutResult, BaseMetrics

logger = logging.getLogger(__name__)



async def generate_math_rollout2(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    # (1) Choose a random environment server
    start = time.perf_counter()
    env_jobs = [Job(**job) for job in cfg.jobs if job["kind"] == "environment"]
    math_job, mcp_jobs = env_jobs[:1], env_jobs[1:]
    # choose the env job randomly
    mcp_job = random.choice(mcp_jobs)
    assert mcp_job.port is not None
    mcp_job_url = f"http://{mcp_job.hostname}:{mcp_job.port}"
    environment = AsyncRemoteEnvironment(server_url=mcp_job_url)  # type: ignore
    async with environment.acontext(session, wait_for_env=True) as env:
        actions = await env.a_actions()
        tools_description = await env.a_tools_description()
        logger.debug(f"Available tools: {tools_description}")
        agent: Agent = instantiate(cfg.agent, known_actions=actions, tools_description=tools_description)
        agent.llms = {DEFAULT: llm}
        tape = Tape(steps=[UserStep(content=problem["task"])])
        tape = await async_execute_agent(agent, tape, env, session, max_loops=cfg.agent_max_loops)

    reward_table = RewardTable(**dict(cfg.rewards))
    answer_status = await verify_answer_rpc(
        session=session,
        host=math_job.hostname,
        port=math_job.port,
        prediction=llm_call.output.content,
        gold=problem["answer"],
        strict=True,
    )
    reward = get_reward(answer_status, tape.finished, reward_table)

    metrics = BaseMetrics(
        reward=reward,
        success=answer_status == "correct",
        no_error=answer_status != "unparsable",
        no_answer=answer_status == "no_answer",
    )

    training_texts = [make_training_text(llm, llm_call) for llm_call in llm_calls]
    for text in training_texts:
        text.reward = reward
    latency = time.perf_counter() - start
    return RolloutResult(
        training_texts=training_texts,
        metrics=metrics,
        latency=latency,
        dataset_name=problem["dataset"],
    )
