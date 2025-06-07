
import os
import json
import random
import re
import time
import aiohttp
from omegaconf import DictConfig

from pipelinerl.async_llm import llm_async_generate, make_training_text
from pipelinerl.rollouts import RolloutResult
from pipelinerl.world import Job
from tapeagents.core import Prompt
from tapeagents.llms.trainable import TrainableLLM


async def generate_miniwob_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    ### STARTING POINT #1
    # Generate environment
    # choose a random environment server
    # Generate TapeAgent
    # run the agent
    # get llm calls from tape
    # compute rewards
    # get training text from llm calls

    start_time = time.time()

    env_jobs = [Job(**job) for job in cfg.jobs if job["kind"] == "environment"]
    # choose the env job randomly
    env_job = random.choice(env_jobs)
    assert env_job.port is not None
    


    return RolloutResult(
        training_texts=[],
        metrics={},
        latency=time.time() - start_time,
        dataset_name=problem["dataset"],
    )

