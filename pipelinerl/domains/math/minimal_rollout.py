import time
import random

import aiohttp
from omegaconf import DictConfig
from pydantic import BaseModel
from pipelinerl.rollouts import RolloutResult, BaseMetrics
from pipelinerl.world import Job
from tapeagents.core import Prompt
from tapeagents.llms.trainable import TrainableLLM

from pipelinerl.async_llm import llm_async_generate, make_training_text
from .verifier_api import verify_answer_rpc

class Metrics(BaseMetrics):
    pass

class RewardTable(BaseModel):
    wrong_answer_not_finished: float
    wrong_answer_finished: float
    no_answer_not_finished: float
    no_answer_finished: float
    unparsable_not_finished: float
    unparsable_finished: float
    correct_answer_not_finished: float
    correct_answer_finished: float
    buffer_tokens: int = 0 # 0 means no overlong reward shaping

def length_penalty(max_length: int, sequence_length: int, buffer_tokens: int) -> float:
    """
    Compute the overlong penalty
    """
    if sequence_length > (max_length - buffer_tokens) and sequence_length <= max_length:
        return ((max_length - buffer_tokens) - sequence_length) / buffer_tokens
    return 0.

def get_reward(trace, answer_status: str, rewards: RewardTable) -> float:
    pass


async def generate_math_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    messages = []
    if cfg.actor.system_prompt:
        messages.append({"role": "system", "content": cfg.actor.system_prompt})
    messages.append({"role": "user", "content": f"{problem['task']} \n{cfg.actor.task_prompt}"})
    prompt = Prompt(messages=messages)

    time_start = time.time()
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - time_start

    assert llm_call.output.content is not None
    rewards = RewardTable(**dict(cfg.rewards))

    env_jobs = [Job(**job) for job in cfg.jobs if job["kind"] == "environment"]
    env_job = random.choice(env_jobs)
    assert env_job.port is not None
    answer_status = await verify_answer_rpc(session=session, host=env_job.hostname, port=env_job.port, prediction=llm_call.output.content, gold=problem["answer"])

    trace = make_training_text(llm, llm_call)
    reward = get_reward(trace, answer_status, rewards)
    trace.reward = reward

    metrics = Metrics(reward=reward, success=answer_status == "correct", no_error=answer_status != "unparsable", no_answer=answer_status == "no_answer")
    

    return RolloutResult(training_texts=[trace], metrics=metrics, latency=latency, dataset_name=problem.get("dataset"))
