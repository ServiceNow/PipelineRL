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
from pipelinerl.math.verifier_api import verify_answer_rpc

class Metrics(BaseMetrics):
    penalty: float

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

def length_penalty(max_length: int, sequence_length: int, buffer_tokens: int):
    """
    Compute the overlong penalty
    """
    if sequence_length > (max_length - buffer_tokens) and sequence_length <= max_length:
        return ((max_length - buffer_tokens) - sequence_length) / buffer_tokens
    return 0

async def generate_math_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    messages = []
    if cfg.actor.system_prompt:
        messages.append({"role": "system", "content": cfg.actor.system_prompt})
    messages.append({"role": "user", "content": cfg.actor.task_template.format(task=problem["task"])})
    prompt = Prompt(messages=messages)

    time_start = time.time()
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - time_start

    assert llm_call.output.content is not None
    rewards = RewardTable(**dict(cfg.rewards))
    discount_factor = cfg.actor.discount_factor

    # math_verify is a fast environment, no support for environment replicas for now
    env_jobs = [Job(**job) for job in cfg.jobs if job["kind"] == "environment"]
    # choose the job randomly
    env_job = random.choice(env_jobs)
    assert env_job.port is not None
    answer_status = await verify_answer_rpc(
        session=session,
        host=env_job.hostname,
        port=env_job.port,
        prediction=llm_call.output.content,
        gold=problem["answer"],
        strict=True,
    )

    trace = make_training_text(llm, llm_call)
    # Check if the generation is finished (ended with EOS token)
    finished = 1 if trace.input_ids[-1] == llm.tokenizer.eos_token_id else 0

    # Determine reward based on answer status and finished state
    match (answer_status, finished):
        case ("wrong", 0):
            reward = rewards.wrong_answer_not_finished
        case ("wrong", 1):
            reward = rewards.wrong_answer_finished
        case ("no_answer", 0):
            reward = rewards.no_answer_not_finished
        case ("no_answer", 1):
            reward = rewards.no_answer_finished
        case ("unparsable", 0):
            reward = rewards.unparsable_not_finished
        case ("unparsable", 1):
            reward = rewards.unparsable_finished
        case ("correct", 0):
            reward = rewards.correct_answer_not_finished
        case ("correct", 1):
            reward = rewards.correct_answer_finished
        case _:
            raise ValueError(f"Invalid answer_status/finished combination: {answer_status}/{finished}")

    # Apply discount factor based on output length
    reward *= discount_factor**llm_call.output_length_tokens
    overlong_penalty = 0
    if rewards.buffer_tokens > 0:
        overlong_penalty = length_penalty(llm.parameters['max_tokens'], llm_call.output_length_tokens, rewards.buffer_tokens)
    reward += overlong_penalty
    trace.reward = reward

    metrics = Metrics(
        reward=reward,
        success=answer_status == "correct",
        no_error=answer_status != "unparsable",
        no_answer=answer_status == "no_answer",
        overflow=0 if finished else 1,
        prompt_tokens=[llm_call.prompt_length_tokens],
        output_tokens=[llm_call.output_length_tokens],
        penalty=overlong_penalty,
    )

    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=latency, 
        dataset_name=problem.get("dataset"),
    )
