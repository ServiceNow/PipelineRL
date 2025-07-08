import re
import time
from typing import Any, Literal
import aiohttp
from omegaconf import DictConfig

from pipelinerl.async_llm import make_training_text
from pipelinerl.rollouts import RolloutResult
from tapeagents.core import Prompt, Tape, Observation, AgentStep 
from tapeagents.llms import LLMStream
from tapeagents.llms.trainable import TrainableLLM
from tapeagents.agent import Node, Agent

class ProblemStep(Observation):
    kind: Literal["problem"] = "problem"
    problem: dict


class ResultsStep(AgentStep):
    kind: Literal["result"] = "result"
    results: dict


class CountingNode(Node):   
    def make_prompt(self, agent: Any, tape: Tape):
        assert isinstance(tape.steps[0], ProblemStep)
        problem = tape.steps[0].problem
        letter = problem["letter"]
        word = problem["word"]
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
        return Prompt(messages=messages)
    
    def generate_steps(self, agent: Any, tape: Tape, llm_stream: LLMStream):
        assert isinstance(tape.steps[0], ProblemStep)
        groundtruth_count = tape.steps[0].problem["count"]
        reward = 0
        error = 0
        output_text = llm_stream.get_output().content
        if output_text:
            answer = re.search("<answer>(\d+)</answer>", output_text)
            if answer:
                answer = int(answer.group(1))
                if answer == groundtruth_count:
                    reward = 1
            else:
                error = 1
        else:
            error = 1
        yield ResultsStep(
            results={
                "reward": reward,
                "success": reward,
                "no_error": not error,
            }
        )

async def generate_counting_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    time_start = time.time()
    agent = Agent.create(nodes=[CountingNode()], llms=llm, store_llm_calls=True)
    start_tape = Tape(steps=[ProblemStep(problem=problem)])
    final_tape = None
    async for event in agent.arun(start_tape, session, max_iterations=1):
        if (final_tape := event.final_tape):
            break
    latency = time.time() - time_start
    assert final_tape is not None
    assert len(final_tape.steps) == 2
    llm_call = final_tape.steps[1].metadata.other["llm_call"]    
    training_text = make_training_text(llm, llm_call)
    assert isinstance(final_tape.steps[1], ResultsStep)
    results = final_tape.steps[1].results
    training_text.reward = results["reward"]

    finished = 1 if training_text.input_ids[-1] == llm.tokenizer.eos_token_id else 0

    metrics = {
        "reward": results["reward"],
        "success": results["reward"],
        "no_error": results["no_error"],
        "no_answer": not results["no_error"],
        "overflow": 0 if finished else 1,
    }

    return RolloutResult(
        training_texts=[training_text],
        metrics=metrics,
        latency=latency,
        dataset_name=problem["dataset"],
        prompt_tokens= [llm_call.prompt_length_tokens],
        output_tokens=[llm_call.output_length_tokens],
    )