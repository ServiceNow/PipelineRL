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
    pass

class RewardTable(BaseModel):
    ifeval_correct: float
    ifeval_incorrect: float
    coding_correct: float
    coding_incorrect: float
    agentic_fn_calling_correct: float
    agentic_fn_calling_incorrect: float
    agentic_fn_calling_partially_correct: float
    math_correct: float
    math_incorrect: float
    math_unparsable: float
    buffer_tokens: int = 0 # 0 means no overlong reward shaping

# TODO: This is LLM specific. It should be moved to a separate module or made configurable.
# TODO: Return thinking tokens count as well, if penalizing based on that.
def postprocess_generation(generation: str) -> str:
    try:
        if '[BEGIN FINAL RESPONSE]' in generation:
            generation = generation.split('[BEGIN FINAL RESPONSE]')[1]
            generation = generation.replace('[END FINAL RESPONSE]', '')
    except Exception as e:
        logger.warning(f"Postprocessing failed: {e}")
            
    return generation.strip()

async def generate_apriel_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    # Conversation is already prepared in the dataset itself.
    messages = problem['prompt']
    prompt = Prompt(messages=messages)

    time_start = time.time()
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - time_start

    assert llm_call.output.content is not None
    rewards = RewardTable(**dict(cfg.rewards))
    discount_factor = cfg.actor.discount_factor

    # code verification happens on a remote job, no support for environment replicas for now
    env_jobs = [Job(**job) for job in cfg.jobs if job["kind"] == "environment"]
    env_job = random.choice(env_jobs)

    answer_status = await verify_answer_rpc(
        session=session,
        host=env_job.hostname,
        port=env_job.port,
        domain=problem.get('domain'),
        generation=postprocess_generation(llm_call.output.content),
        reward_context=problem.get('reward_context')
    )


    match(answer_status):
        case "ifeval_correct":
            reward = rewards.ifeval_correct
        case "ifeval_incorrect":
            reward = rewards.ifeval_incorrect
        case "coding_correct":
            reward = rewards.coding_correct
        case "coding_incorrect":
            reward = rewards.coding_incorrect
        case "agentic_fn_calling_correct":
            reward = rewards.agentic_fn_calling_correct
        case "agentic_fn_calling_incorrect":
            reward = rewards.agentic_fn_calling_incorrect
        case "agentic_fn_calling_partially_correct":
            reward = rewards.agentic_fn_calling_partially_correct
        case "math_correct":
            reward = rewards.math_correct
        case "math_incorrect":
            reward = rewards.math_incorrect
        case "math_unparsable":
            reward = rewards.math_unparsable
        case _:
            raise ValueError(f"Unknown answer status: {answer_status}")

    logger.info(f"Shiva-Code verification result: {answer_status}, reward: {reward}")
    trace = make_training_text(llm, llm_call)

    reward *= discount_factor**llm_call.output_length_tokens
    trace.reward = reward

    metrics = Metrics(
       reward=reward,
        success=answer_status == "correct",
        no_error=answer_status != "incorrect",
        no_answer=answer_status == "no_answer",
    )

    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=latency, 
        dataset_name=problem.get("dataset"),
    )

