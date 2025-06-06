import os
import json
import re
import time
import aiohttp
from omegaconf import DictConfig

from pipelinerl.async_llm import llm_async_generate, make_training_text
from pipelinerl.rollouts import RolloutResult
from tapeagents.core import Prompt
from tapeagents.llms.trainable import TrainableLLM


async def generate_guessing_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    initial_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant",
        },
        {
            "role": "user",
            "content": f"You must guess a number. Output the answer as <answer>number</answer>."
                        " After each guess I will tell you if your answer is higher or lower than the target number."
        }
    ]
    time_start = time.time()
    llm_calls = []
    guess_history = []
    reward = 0
    error = 0
    for i in range(3):
        messages = initial_messages.copy()
        if i > 0:
            last_message = f"Your {i} previous guesses:"
            for guess in guess_history:
                relation = "lower" if guess < problem["answer"] else "higher"
                last_message += f"\n{guess}, which is {relation} than the target number."
            else:
                last_message += "\n<wrong output>"
            messages.append({
                "role": "user",
                "content": last_message
            })
        llm_call = await llm_async_generate(llm, Prompt(messages=messages), session)
        llm_calls.append(llm_call)

        output_text = llm_call.output.content 
        if output_text:
            answer = re.search("<answer>(\d+)</answer>", output_text)
            if answer:
                answer = int(answer.group(1))
                if answer == problem["answer"]:
                    reward = 1 - i / 10
                    break
                else:
                    guess_history.append(answer)                            
            else:
                reward = -1
                error = 1
                break
        else:
            error = 1
            reward = -1
            break
    latency = time.time() - time_start        

    all_finished = 0
    prompt_tokens = sum(llm_call.prompt_length_tokens for llm_call in llm_calls)
    output_tokens = sum(llm_call.output_length_tokens for llm_call in llm_calls)
    training_texts = [make_training_text(llm, llm_call) for llm_call in llm_calls]
    for text in training_texts:
        text.reward = reward
        all_finished &= 1 if text.input_ids[-1] == llm.tokenizer.eos_token_id else 0

    metrics = {
        "reward": reward,
        "success": reward,
        "no_error": not error,
        "no_answer": error,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "overflow": 0 if all_finished else 1,
    }

    return RolloutResult(
        training_texts=training_texts,
        metrics=metrics,
        latency=latency,
        dataset_name=problem["dataset"]
    )
    

def load_problems(dataset_names: list[str]):
    problems = []
    for name in dataset_names:
        if name == "train":
            problems.extend([
                {"answer": i, "dataset": "train"} for i in range(0, 128, 2)
            ])
        elif name == "test":
            problems.extend([
                {"answer": i, "dataset": "test"} for i in range(1, 128, 2)
            ])
    return problems
