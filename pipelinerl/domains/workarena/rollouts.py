import logging
import os
import time

from examples.rl_webagent.steps import WebTape
from examples.workarena.agent import WorkArenaAgent
from examples.workarena.environment import WorkArenaEnvironment
from hydra.utils import instantiate
from omegaconf import DictConfig
from tapeagents.core import LLMCall, LLMOutputParsingFailureAction, Observation
from tapeagents.io import save_json_tape
from tapeagents.llms.trainable import TrainableLLM
from tapeagents.orchestrator import execute_agent

from pipelinerl.async_llm import make_training_text
from pipelinerl.domains.workarena.load_tasks import get_task_by_id
from pipelinerl.rollouts import BaseMetrics, RolloutResult

logger = logging.getLogger(__name__)


class WorkarenaMetrics(BaseMetrics):
    reward: float
    success: bool
    no_error: bool
    no_answer: bool
    overflow: bool
    n_llm_calls: int
    n_step_errors: int
    n_page_observations: int
    n_steps: int
    total_execution_time: float
    env_start_time: float
    env_close_time: float
    env_agent_creation_time: float
    agent_execution_time: float
    environment_execution_time: float
    env_step_time: float
    agent_step_time: float
    llm_call_time: float
    env_call_time: float
    total_llm_call_time: float
    total_env_call_time: float


def tape_contains_an_error(tape: WebTape) -> bool:
    """
    Returns true if the tape ends with an error, ie if one of the following is true:
    - the last step is an LLMOutputParsingFailureAction
    - the tape metadata has an error
    - the last step is a PageObservation with an error
    """
    return (
        len(tape.steps) == 0
        or isinstance(tape.steps[-1], LLMOutputParsingFailureAction)
        or tape.metadata.result.get("error") is not None
        or (tape.steps[-1].__class__.__name__ == "PageObservation" and tape.steps[-1].error)
    )


def compute_reward(tape: WebTape, success: bool, result: dict) -> float:
    """
    TODO: Improve this
    """
    return 1.0 if success else -1.0


def generate_workarena_rollout(cfg: DictConfig, llm: TrainableLLM, problem: dict) -> RolloutResult:
    # make agent and env
    # set the llm
    # run the agent
    # get llm calls from tape
    # compute rewards
    # get training text from llm calls

    start_time = time.perf_counter()

    environment: WorkArenaEnvironment = instantiate(cfg.environment)
    environment.initialize()
    agent = WorkArenaAgent.create(llm)
    logger.info(f"Agent and environment loaded, using llm {llm.model_name} at {llm.get_base_url()}")
    env_agent_creation_time = time.perf_counter() - start_time
    try:
        task_entrypoint = get_task_by_id(problem["task"])
        start_attempts = cfg.start_attempts
        t = time.perf_counter()
        while True:
            try:
                tape, _ = environment.start_task(task_entrypoint)
                break
            except Exception as e:
                logger.exception(f"Failed to start task {problem['dataset']}/{problem['task']}/{problem['seed']}: {e}")
                start_attempts -= 1
                if start_attempts <= 0:
                    raise Exception(
                        f"Failed to start task {problem['dataset']}/{problem['task']}/{problem['seed']} after {cfg.start_attempts} attempts"
                    )
                else:
                    logger.warning("retry after 1 seconds")
                    time.sleep(1)
        env_start_time = time.perf_counter() - t
        logger.info(
            f"Task {problem['dataset']}/{problem['task']}/{problem['seed']} started in {env_start_time:.2f} seconds"
        )
        logger.info(f"Running agent for task {problem['dataset']}/{problem['task']}/{problem['seed']}")
        ex_t = time.perf_counter()
        tape = execute_agent(agent, tape, environment, max_loops=cfg.agent_max_loops)
        execution_time = time.perf_counter() - ex_t
    finally:
        close_t = time.perf_counter()
        environment.close()
        env_close_time = time.perf_counter() - close_t
    logger.info(
        f"Agent finished task {problem['dataset']}/{problem['task']}/{problem['seed']}, times: start {env_start_time:.2f} sec, exec {execution_time:.2f} sec, close {env_close_time:.2f} sec, produced tape with {len(tape.steps)} steps"
    )
    total_execution_time = time.perf_counter() - t
    tape.metadata.result.update(
        {
            "total_execution_time": total_execution_time,
            "env_start_time": env_start_time,
            "env_agent_creation_time": env_agent_creation_time,
            "execution_time": execution_time,
            "env_close_time": env_close_time,
        }
    )

    # save the tape as we go
    if cfg.save_tapes:
        try:
            tape_name = problem.get("_task_id", tape.metadata.id)
            save_json_tape(tape, os.path.join(cfg.output_dir, "tapes"), tape_name)
        except Exception as e:
            logger.error(f"Error saving tape {tape_name}: {e}")

    success, result = environment.validate_task(tape)
    reward = compute_reward(tape, success, result)

    # (3) Get LLM calls from Tape
    llm_calls = [step for step in tape.steps if step.metadata.other.get("llm_call") is not None]
    n_llm_calls = len(llm_calls)
    llm_calls: list[LLMCall] = [
        LLMCall(**step.metadata.other["llm_call"])
        if isinstance(step.metadata.other["llm_call"], dict)
        else step.metadata.other["llm_call"]
        for step in llm_calls
    ]
    llm_call_times = [
        step.metadata.other.get("llm_call_time") for step in tape.steps if "llm_call_time" in step.metadata.other
    ]
    env_call_times = [
        step.metadata.other.get("action_execution_time")
        for step in tape.steps
        if "action_execution_time" in step.metadata.other
    ]
    total_llm_call_time = sum(llm_call_times)
    total_env_call_time = sum(env_call_times)
    llm_call_time = total_llm_call_time / len(llm_call_times) if len(llm_call_times) > 0 else -1.0
    env_call_time = total_env_call_time / len(env_call_times) if len(env_call_times) > 0 else -1.0

    # (4) # For each LLM interaction in the tape, make a training example.
    all_finished = 1
    prompt_tokens = [llm_call.prompt_length_tokens for llm_call in llm_calls]
    output_tokens = [llm_call.output_length_tokens for llm_call in llm_calls]
    training_texts = [make_training_text(llm, llm_call) for llm_call in llm_calls]
    for text in training_texts:
        text.reward = reward
        all_finished &= 1 if text.input_ids[-1] == llm.tokenizer.eos_token_id else 0

    latency = time.perf_counter() - start_time
    agent_time = tape.metadata.result.get("agent_execution_time", -1.0)
    env_time = tape.metadata.result.get("environment_execution_time", -1.0)
    n_observations = len(
        [s for s in tape.steps if isinstance(s, Observation)]
    )  # TODO: is this not the same n_page_observations??
    n_other_steps = len(tape.steps) - n_observations
    n_step_errors = len([step for step in tape.steps if isinstance(step, LLMOutputParsingFailureAction)])
    n_page_observations = len([step for step in tape.steps if step.__class__.__name__ == "PageObservation"])
    no_error = not tape_contains_an_error(tape)
    metrics = WorkarenaMetrics(
        reward=reward,
        success=reward > 0.5,
        no_error=no_error,
        no_answer=reward < 0,
        overflow=not all_finished,
        n_llm_calls=n_llm_calls,
        n_step_errors=n_step_errors,
        n_page_observations=n_page_observations,
        n_steps=len(tape.steps),
        total_execution_time=total_execution_time,
        env_start_time=env_start_time,
        env_close_time=env_close_time,
        env_agent_creation_time=env_agent_creation_time,
        agent_execution_time=agent_time,
        environment_execution_time=env_time,
        env_step_time=env_time / n_observations if env_time > 0 and n_observations > 0 else -1.0,
        agent_step_time=agent_time / n_other_steps if agent_time > 0 and n_other_steps > 0 else -1.0,
        llm_call_time=llm_call_time,
        env_call_time=env_call_time,
        total_llm_call_time=total_llm_call_time,
        total_env_call_time=total_env_call_time,
    )

    return RolloutResult(
        training_texts=training_texts,
        metrics=metrics,
        latency=latency,
        dataset_name=problem["dataset"],
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
    )
