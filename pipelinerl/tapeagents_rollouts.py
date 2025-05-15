import asyncio
import time

import aiohttp
from omegaconf import DictConfig
from tapeagents.agent import Agent, LLMEvent, LLMStream
from tapeagents.core import Prompt, StopStep, TrainingText
from tapeagents.dialog_tape import DialogTape, UserStep
from tapeagents.environment import Environment
from tapeagents.llms.trainable import TrainableLLM
from tapeagents.orchestrator import get_agent_and_env_from_config, main_loop

from pipelinerl.async_llm import llm_async_generate
from pipelinerl.math_rollouts import RolloutResult


def run_tapeagent(
    task: str, agent: Agent, environment: Environment, max_loops: int
) -> tuple[list[TrainingText], dict[str, float]]:
    start_tape = DialogTape(steps=[UserStep(content=task)])
    tape: DialogTape | None = None
    for event in main_loop(agent, start_tape, environment, max_loops):
        if event.agent_tape:
            tape = event.agent_tape
        elif event.env_tape:
            tape = event.env_tape
    assert tape is not None, "No tape generated"
    has_errors = any([1 for s in tape.steps if s.llm_dict().get("error")])
    has_answer = any([isinstance(s, StopStep) for s in tape.steps])
    _, llm_calls = agent.reuse(tape)
    samples = [agent.make_training_text(llm_call) for llm_call in llm_calls]
    reward = 0  # TODO: implement verifier usage and reward calculation
    metrics = {
        "reward": reward,
        "success": reward > 0,
        "no_error": not has_errors,
        "no_answer": not has_answer,
        "prompt_tokens": sum([llm_call.prompt_length_tokens for llm_call in llm_calls]),
        "output_tokens": sum([llm_call.output_length_tokens for llm_call in llm_calls]),
        "overflow": 0,  # TODO: should we treat max_loops stop as overflow?
    }
    return samples, metrics


async def generate_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    def generate(self, prompt: Prompt):
        # !! should be called in a separate thread only !!
        # 'self' here is the llm instance (agent.llms["default"])
        # 'session' is captured from the outer scope of generate_rollout
        def _implementation():
            llm_call = asyncio.run(llm_async_generate(self, prompt, session))
            yield LLMEvent(output=llm_call.output, llm_call=llm_call)

        return LLMStream(_implementation(), prompt)

    time_start = time.time()
    task: str = cfg.task_template.format(task=problem["task"])
    agent, environment = get_agent_and_env_from_config(cfg)
    agent.llms = {"default": llm.model_copy()}
    agent.llms["default"].generate = generate  # type: ignore
    samples, metrics = await asyncio.to_thread(run_tapeagent, task, agent, environment, cfg.max_loops)
    latency = time.time() - time_start
    return RolloutResult(training_texts=samples, metrics=metrics, latency=latency, dataset_name=problem.get("dataset"))
