import time
import random

import aiohttp
from omegaconf import DictConfig
from pydantic import BaseModel
from pipelinerl.rollouts import RolloutResult, BaseMetrics
from pipelinerl.world import Job
from tapeagents.core import Prompt
from tapeagents.llms.trainable import TrainableLLM
import json

from pipelinerl.async_llm import llm_async_generate, make_training_text
from .verifier_api import verify_answer_rpc

import logging
logging.basicConfig(
    level=logging.DEBUG,  # Or INFO, WARNING, etc.
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Logs to console
    ],
)

logger = logging.getLogger(__name__)

class Metrics(BaseMetrics):
    partially_correct: bool

class RewardTable(BaseModel):
    correct: float
    incorrect: float
    partially_correct: float
    buffer_tokens: int = 0 # 0 means no overlong reward shaping


# TODO: This is LLM specific. It should be moved to a separate module or made configurable.
def postprocess_generation(generation: str) -> str:
    try:
        if '[BEGIN FINAL RESPONSE]' in generation:
            generation = generation.split('[BEGIN FINAL RESPONSE]')[1]
            generation = generation.replace('[END FINAL RESPONSE]', '')
    except Exception as e:
        logger.warning(f"Postprocessing failed: {e}")
            
    return generation.strip()



async def generate_agentic_fn_calling_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    messages = []
    # if cfg.actor.system_prompt:
    #     messages.append({"role": "system", "content": cfg.actor.system_prompt})

    # messages.append({"role": "user", "content": cfg.actor.task_template.format(task=problem["task"])})
    messages = problem['prompt']
    prompt = Prompt(messages=messages)

    time_start = time.time()
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - time_start
    # logger.info(f"Shiva-LLM generation: {llm_call.output.content}")
    logger.info(f"Shiva-LLM generation took {latency} seconds")

    postprocessed_generation = postprocess_generation(llm_call.output.content)
    # logger.info(f"Shiva-LLM post processed output: {postprocessed_generation}")

    assert llm_call.output.content is not None
    rewards = RewardTable(**dict(cfg.rewards))
    logger.info(f"Shiva-Using rewards: {rewards}")

    discount_factor = cfg.actor.discount_factor

    # code verification happens on a remote job, no support for environment replicas for now
    env_jobs = [Job(**job) for job in cfg.jobs if job["kind"] == "environment"]
    # choose the job randomly
    env_job = random.choice(env_jobs)

    answer_status = await verify_answer_rpc(
        session=session,
        host=env_job.hostname,
        port=env_job.port,
        generation=postprocessed_generation,
        reward_context=problem.get('reward_context')
    )


    match(answer_status):
        case "correct":
            reward = rewards.correct
        case "incorrect":
            reward = rewards.incorrect
        case "partially_correct":
            reward = rewards.partially_correct
        case _:
            raise ValueError(f"Unknown answer status: {answer_status}")

    logger.info(f"Shiva-Code verification result: {answer_status}, reward: {reward}")
    trace = make_training_text(llm, llm_call)

    reward *= discount_factor**llm_call.output_length_tokens

    trace.reward = reward

    metrics = Metrics(
       reward=reward,
        success=answer_status == "correct",
        no_error=answer_status != "incorrect" and answer_status != "partially_correct",
        no_answer=answer_status == "no_answer",
        partially_correct=answer_status == "partially_correct",
    )

    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=latency, 
        dataset_name=problem.get("dataset"),
    )

    