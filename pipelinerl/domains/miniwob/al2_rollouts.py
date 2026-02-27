import asyncio
import logging
import time

import aiohttp
from agentlab2.agents.react import ReactAgentConfig
from agentlab2.benchmarks.miniwob.task import MiniWobTask
from agentlab2.environment import EnvConfig
from agentlab2.llm import LLMConfig
from agentlab2.rl.rollout import RolloutResult as AL2RolloutResult
from agentlab2.rl.rollout import rollout
from agentlab2.tools.playwright import PlaywrightConfig
from omegaconf import DictConfig

from pipelinerl.llm import TrainableLLM
from pipelinerl.rollouts import BaseMetrics, RolloutResult, TrainingText

logger = logging.getLogger(__name__)


class MiniwobMetrics(BaseMetrics):
    reward: float
    success: bool
    no_error: bool
    no_answer: bool
    overflow: bool
    n_llm_calls: int
    n_step_errors: int
    n_observations: int
    n_steps: int
    total_execution_time: float
    agent_execution_time: float
    environment_execution_time: float
    env_step_time: float
    agent_step_time: float


async def generate_miniwob_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    task = task_dict_to_al2_task(problem)
    tool_config = PlaywrightConfig(use_screenshot=True, headless=True)
    env_config = EnvConfig(task=task, tool_config=tool_config)
    llm_config = LLMConfig(
        model_name=llm.model_name,
        api_base=llm.base_url,
        api_key=llm.api_token,
        temperature=llm.parameters["temperature"],
    )
    agent_config = ReactAgentConfig(llm_config=llm_config)

    start = time.perf_counter()
    al2_result = await asyncio.to_thread(rollout, agent_config, env_config, cfg.max_steps)
    latency = time.perf_counter() - start

    return convert_al2_rollout_to_pipelinerl(al2_result, latency)


def task_dict_to_al2_task(task: dict) -> MiniWobTask:
    return MiniWobTask(
        id=task["subdomain"],
        desc=task["desc"],
        subdomain=task["subdomain"],
        base_url=task["base_url"],
    )


def convert_al2_rollout_to_pipelinerl(
    al2_result: AL2RolloutResult,
    latency: float,
) -> RolloutResult:
    reward = al2_result.reward
    text_pairs = al2_result.text_pairs
    n_llm_calls = len(text_pairs)

    training_texts = [
        TrainingText(
            text=tp.prompt_text + tp.response_text,
            n_predicted=len(tp.response_text),
            reward=tp.reward if tp.reward is not None else reward,
            metadata={"prompt_text": tp.prompt_text, "response_text": tp.response_text},
        )
        for tp in text_pairs
    ]

    m = al2_result.metrics
    metrics = MiniwobMetrics(
        reward=reward,
        success=reward > 0.5,
        no_error=m.n_step_errors == 0,
        no_answer=n_llm_calls == 0,
        overflow=False,
        n_llm_calls=m.n_llm_calls,
        n_step_errors=m.n_step_errors,
        n_observations=m.n_observations,
        n_steps=m.n_steps,
        total_execution_time=m.total_execution_time,
        agent_execution_time=m.agent_execution_time,
        environment_execution_time=m.environment_execution_time,
        env_step_time=m.env_step_time,
        agent_step_time=m.agent_step_time,
    )

    return RolloutResult(
        training_texts=training_texts,
        metrics=metrics,
        latency=latency,
    )
