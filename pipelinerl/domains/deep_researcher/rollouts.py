"""Multi-turn rollout generation for DeepResearcher domain."""
import time
import random
import logging
from omegaconf import DictConfig
from pydantic import BaseModel
import aiohttp

from pipelinerl.rollouts import RolloutResult, BaseMetrics
from pipelinerl.world import Job
from tapeagents.llms.trainable import TrainableLLM
from pipelinerl.async_llm import make_training_text

from .tools import ToolRegistry
from .orchestration.registry import OrchestrationRegistry
from .verifier_api import verify_answer_rpc

logger = logging.getLogger(__name__)


class Metrics(BaseMetrics):
    num_turns: int
    num_tool_calls: int
    strategy_used: str


class RewardTable(BaseModel):
    correct: float
    incorrect: float
    max_turns_exceeded: float


async def generate_deep_researcher_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    rewards = RewardTable(**dict(cfg.rewards))
    tool_registry = ToolRegistry()
    strategy_name = cfg.actor.get("orchestration_strategy", "react")
    logger.info(f"Using orchestration strategy: {strategy_name}")
    
    try:
        orchestrator_class = OrchestrationRegistry.get(strategy_name)
        orchestrator = orchestrator_class(cfg, llm, tool_registry)
    except ValueError as e:
        logger.error(f"Failed to load orchestration strategy: {e}")
        raise

    time_start = time.time()
    try:
        result = await orchestrator.execute(
            question=problem["question"],
            session=session
        )
    except Exception as e:
        logger.error(f"Orchestration execution failed: {e}")
        raise
    
    latency = time.time() - time_start
    
    llm_calls = result["llm_calls"]
    final_answer = result["final_answer"]
    metadata = result["metadata"]
    
    logger.info(
        f"Orchestration completed: {metadata['num_turns']} turns, "
        f"{metadata['num_tool_calls']} tool calls"
    )

    if not final_answer or final_answer.strip() == "":
        answer_status = "max_turns_exceeded"
        reward = rewards.max_turns_exceeded
        logger.warning("No final answer provided")
    else:
        env_jobs = [Job(**job) for job in cfg.jobs if job["kind"] == "environment"]
        if not env_jobs:
            logger.error("No environment jobs configured")
            raise ValueError("No environment jobs found in configuration")
        
        env_job = random.choice(env_jobs)

        try:
            answer_status = await verify_answer_rpc(
                session=session,
                host=env_job.hostname,
                port=env_job.port,
                generation=final_answer,
                reward_context=problem.get('reward_context', {})
            )
            logger.info(f"Verification result: {answer_status}")
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            answer_status = "incorrect"
        
        if answer_status == "correct":
            reward = rewards.correct
        else:
            reward = rewards.incorrect
    
    training_texts = [make_training_text(llm, llm_call) for llm_call in llm_calls]
    
    for text in training_texts:
        text.reward = reward
    
    logger.info(
        f"Rollout completed: reward={reward}, "
        f"success={answer_status == 'correct'}, "
        f"turns={len(training_texts)}"
    )
    
    metrics = Metrics(
        reward=reward,
        success=answer_status == "correct",
        no_error=answer_status != "incorrect",
        no_answer=not final_answer,
        num_turns=metadata["num_turns"],
        num_tool_calls=metadata["num_tool_calls"],
        strategy_used=metadata["strategy_used"]
    )
    
    return RolloutResult(
        training_texts=training_texts,
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset", "deep_researcher"),
    )
