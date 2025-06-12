import os
import json
import re
import time
import aiohttp
from omegaconf import DictConfig

from pipelinerl.async_llm import llm_async_generate, make_training_text
from pipelinerl.rollouts import RolloutResult, BaseMetrics
from tapeagents.core import Prompt
from tapeagents.llms.trainable import TrainableLLM


async def generate_counting_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    letter = problem["letter"]
    word = problem["word"]
    count = problem["count"]
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant",
        },
        {
            "role": "user",
            "content": f"How many {letter}'s are there in the word '{word}'? You can think step by step. Output the answer as <answer>number</answer>.", 
        }
    ]
    time_start = time.time()
    llm_call = await llm_async_generate(llm, Prompt(messages=messages), session)
    latency = time.time() - time_start
    output_text = llm_call.output.content 

    reward = 0
    error = 0
    if output_text:
        answer = re.search("<answer>(\d+)</answer>", output_text)
        if answer:
            answer = int(answer.group(1))
            if answer == count:
                reward = 1
        else:
            error = 1
    else:
        error = 1

    training_text = make_training_text(llm, llm_call)
    training_text.reward = reward

    finished = 1 if training_text.input_ids[-1] == llm.tokenizer.eos_token_id else 0

    metrics = BaseMetrics(
        reward=reward,
        success=reward,
        no_error=not error,
        no_answer=error,
        overflow=0 if finished else 1,
        prompt_tokens= [llm_call.prompt_length_tokens],
        output_tokens=[llm_call.output_length_tokens],
    )

    return RolloutResult(
        training_texts=[training_text],
        metrics=metrics,
        latency=latency,
        dataset_name=problem["dataset"],
    )
    

def load_problems(dataset_names: list[str]):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    problems = []
    for name in dataset_names:
        file_path = os.path.join(dir_path, f"{name}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file {file_path} does not exist.")
        with open(file_path, "r") as f:
            dataset = json.load(f)
            if not isinstance(dataset, list):
                raise ValueError(f"Dataset {name} should be a list of problems.")
            for problem in dataset:
                if not isinstance(problem, dict) or "letter" not in problem or "word" not in problem or "count" not in problem:
                    raise ValueError(f"Problem {problem} in dataset {name} is invalid.")
                problem["dataset"] = name
                problems.append(problem)
    return problems
    
