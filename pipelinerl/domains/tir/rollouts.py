"""Rollout generation for TIR domain."""

import logging
import time
import json
import os
from typing import Any, List
from collections import Counter
import aiohttp
from omegaconf import DictConfig
from tapeagents.llms import TrainableLLM
from pipelinerl.rollouts import RolloutResult, BaseMetrics
from pydantic import BaseModel
from tapeagents.steps import ActionExecutionFailure

logger = logging.getLogger(__name__)

# cache environments globally to avoid recreating them
_cached_environments = {}


class TIRMetrics(BaseMetrics):
    """TIR-specific metrics extending the base metrics."""
    overflow: int = 0
    timeout: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    reached_answer_action: int = 0


class TIRRewardTable(BaseModel):
    """Reward table for TIR domain"""
    correct_answer: float = 1.0
    wrong_answer: float = 0.0
    no_answer: float = -0.1
    unparsable: float = -0.1
    execution_failure: float = -0.05
    timeout_penalty: float = -0.2
    buffer_tokens: int = 0
    iteration_penalty: float = -0.02


def length_penalty(max_length: int, sequence_length: int, buffer_tokens: int) -> float:
    """Compute the overlong penalty"""
    if sequence_length > (max_length - buffer_tokens) and sequence_length <= max_length:
        return ((max_length - buffer_tokens) - sequence_length) / buffer_tokens
    return 0.


async def generate_tir_rollout(cfg: DictConfig, llm: TrainableLLM, problem: dict, session: aiohttp.ClientSession) -> RolloutResult:
    """Generate a rollout for TIR domain with iterative reasoning."""
    from pipelinerl.async_llm import make_training_text
    from tapeagents.orchestrator import async_main_loop
    from .agent import Task, TIRMathTape, AnswerAction, TIRMathAgent
    from .environment import MCPPythonEnvironment
    
    time_start = time.time()
    
    # Create or reuse environment
    env_key = str(cfg.environment)
    if env_key not in _cached_environments:
        _cached_environments[env_key] = MCPPythonEnvironment()
        logger.info("Created new cached MCP environment")
    environment = _cached_environments[env_key]
    
    max_reasoning_steps = getattr(cfg.actor, 'max_reasoning_steps', 8)
    logger.info(f"Running TIR with max {max_reasoning_steps} reasoning steps")
    
    # Create agent
    agent = TIRMathAgent(
        system_prompt=cfg.actor.system_prompt,
        max_iterations=max_reasoning_steps
    )
    agent.llms = {"default": llm}
    
    # Debug: Check what tokenizer is being used
    if hasattr(llm, 'tokenizer') and llm.tokenizer:
        logger.info(f"LLM using tokenizer: {llm.tokenizer.__class__.__name__} from {llm.tokenizer.name_or_path}")
        logger.info(f"LLM tokenizer vocab size: {llm.tokenizer.vocab_size}")
    else:
        logger.warning("LLM has no tokenizer loaded")
    
    # Use task template if provided
    task_template = getattr(cfg.actor, 'task_template', '{task}')
    task_step = Task(task=problem["task"], template=task_template)
    start_tape = TIRMathTape(steps=[task_step], context=None)
    
    # Run agent-environment interaction
    final_tape = None
    
    async for event in async_main_loop(agent, start_tape, environment, session, cfg.max_loops):
        if event.agent_tape:
            final_tape = event.agent_tape
        elif event.env_tape:
            final_tape = event.env_tape
    
    if final_tape is None:
        logger.warning("Failed to generate tape")
        metrics = TIRMetrics(
            reward=0.0,
            success=False,
            no_error=False,
            no_answer=True,
        )
        return RolloutResult(
            training_texts=[],
            metrics=metrics,
            latency=time.time() - time_start,
            dataset_name=problem.get("dataset", "unknown"),
        )
    
    final_answer = None
    reached_answer_action = False
    if final_tape and isinstance(final_tape.steps[-1], AnswerAction):
        final_answer = final_tape.steps[-1].value
        reached_answer_action = True
    
    predicted_answer = final_answer if final_answer is not None else "No answer"
    ground_truth = problem.get("answer", "Unknown")
    logger.info(f"Problem: {problem.get('id', 'unknown')} | Predicted: {predicted_answer} | Ground truth: {ground_truth} | Reached AnswerAction: {reached_answer_action}")
    
    training_samples = []

    llm_calls = []

    if getattr(agent, "llm_calls", None):
        llm_calls.extend(agent.llm_calls)

    if final_tape and not llm_calls:
        for step in final_tape.steps:
            if (
                step.metadata
                and step.metadata.other
                and "llm_call" in step.metadata.other
            ):
                llm_calls.append(step.metadata.other["llm_call"])

    for llm_call in llm_calls:
        if isinstance(llm_call, dict):
            from tapeagents.core import LLMCall
            llm_call = LLMCall(**llm_call)

        training_samples.append(make_training_text(llm, llm_call))

    if not llm_calls:
        logger.warning(
            "No LLM calls were captured for this rollout; no training samples will be produced. "
            "Check that `agent.store_llm_calls=True` and that the orchestrator "
            "is not stripping metadata."
        )
    
    # save debug info if requested
    if getattr(cfg, 'save_tapes', False):
        debug_dir = os.path.join(cfg.output_dir, "debug_tapes") 
        os.makedirs(debug_dir, exist_ok=True)
        
        debug_file = os.path.join(debug_dir, f"problem_{problem.get('id', 'unknown')}.json")
        debug_data = {
            "problem": problem,
            "answer": final_answer,
            "num_llm_calls": len(llm_calls),
            "target_answer": problem.get("answer", ""),
        }
        
        with open(debug_file, "w") as f:
            json.dump(debug_data, f, indent=2)
    
    success = False
    answer_status = "no_answer"
    
    if final_answer is not None:
        try:
            from pipelinerl.domains.math.verifier_api import verify_math
            predicted_answer = f"\\boxed{{{final_answer}}}"
            target_answer = problem.get("answer", "")
            answer_status = verify_math(predicted_answer, target_answer, strict=True)
            success = (answer_status == "correct")
        except Exception as e:
            task_value = problem.get("value")
            if task_value is not None:
                success = abs(float(task_value) - float(final_answer)) < 1e-6
                answer_status = "correct" if success else "wrong"
            else:
                answer_status = "unparsable"
    
    if training_samples:
        import numpy as np
        avg_prompt_tokens = np.mean([t.prompt_tokens for t in training_samples])
        avg_output_tokens = np.mean([t.output_tokens for t in training_samples])
        logger.info(
            f"🎓 Generated {len(training_samples)} training samples | "
            f"avg prompt={avg_prompt_tokens:.0f} tokens | "
            f"avg output={avg_output_tokens:.0f} tokens | "
            f"answer_status={answer_status} | "
            f"reached_answer={reached_answer_action}"
        )
    
    rewards = TIRRewardTable(**dict(cfg.get('rewards', {})))
    
    num_llm_calls = len(llm_calls)
    iteration_penalty_total = num_llm_calls * rewards.iteration_penalty
    
    if training_samples:
        if answer_status == "correct":
            base_reward = rewards.correct_answer
        elif answer_status == "wrong":
            base_reward = rewards.wrong_answer
        elif answer_status == "no_answer":
            base_reward = rewards.no_answer
        elif answer_status == "unparsable":
            base_reward = rewards.unparsable
        else:
            base_reward = rewards.wrong_answer  # fallback
        
        has_execution_errors = any(
            isinstance(step, ActionExecutionFailure) for step in final_tape.steps
        )
        if has_execution_errors:
            base_reward += rewards.execution_failure
        
        base_reward += iteration_penalty_total
        
        for sample in training_samples:
            sample.reward = base_reward
    
    if cfg.actor.discount_factor and llm_calls:
        total_output_tokens = sum(llm_call.output_length_tokens for llm_call in llm_calls)
        discount_multiplier = cfg.actor.discount_factor ** total_output_tokens
        
        overlong_penalty = 0
        if rewards.buffer_tokens > 0:
            max_tokens = getattr(cfg.actor, 'max_tokens', 4096)
            overlong_penalty = length_penalty(max_tokens, total_output_tokens, rewards.buffer_tokens)
        
        for sample in training_samples:
            sample.reward *= discount_multiplier
            sample.reward += overlong_penalty
        
        avg_reward = sum(sample.reward for sample in training_samples) / len(training_samples)
    else:
        avg_reward = sum(sample.reward for sample in training_samples) / len(training_samples) if training_samples else 0.0
    
    if training_samples:
        reward_values = [sample.reward for sample in training_samples]
        logger.info(
            f"🏆 Final rewards: mean={np.mean(reward_values):.3f} | "
            f"min={np.min(reward_values):.3f} | max={np.max(reward_values):.3f} | "
            f"base_reward={base_reward:.3f} | "
            f"iter_penalty={iteration_penalty_total:.3f} | "
            f"llm_calls={num_llm_calls} | "
            f"discount_factor={cfg.actor.discount_factor if cfg.actor.discount_factor else 'N/A'}"
        )
    
    has_errors = any(
        any(1 for s in final_tape.steps if hasattr(s, 'error') and s.error) 
        for s in [final_tape]
    )
    
    metrics = TIRMetrics(
        reward=avg_reward,
        success=success,
        no_error=not has_errors,
        no_answer=(answer_status == "no_answer"),
        overflow=0,
        timeout=0,
        prompt_tokens=sum(llm_call.prompt_length_tokens for llm_call in llm_calls) if llm_calls else 0,
        output_tokens=sum(llm_call.output_length_tokens for llm_call in llm_calls) if llm_calls else 0,
        reached_answer_action=int(reached_answer_action),
    )
    
    # guard against assertion error when we have insufficient samples or inconsistent text
    if len(training_samples) >= 2 and training_samples[0].text not in training_samples[1].text:
        logger.debug("Rollout consistency check failed; continuing without assertion")
    return RolloutResult(
        training_texts=training_samples,
        metrics=metrics,
        latency=time.time() - time_start,
        dataset_name=problem.get("dataset", "unknown"),
        prompt_tokens=[llm_call.prompt_length_tokens for llm_call in llm_calls] if llm_calls else [],
        output_tokens=[llm_call.output_length_tokens for llm_call in llm_calls] if llm_calls else [],
    )


def apply_majority_voting(candidate_answers: List[Any]) -> Any:
    """Apply majority voting to select final answer from candidates."""
    valid_answers = [ans for ans in candidate_answers if ans is not None]
    
    if not valid_answers:
        return None
    
    normalized_answers = []
    for ans in valid_answers:
        if isinstance(ans, (int, float)):
            normalized_answers.append(float(ans))
        elif isinstance(ans, str):
            try:
                normalized_answers.append(float(ans))
            except ValueError:
                normalized_answers.append(ans.strip())
        else:
            normalized_answers.append(ans)
    
    answer_counts = Counter(normalized_answers)
    if answer_counts:
        most_common = answer_counts.most_common(1)[0][0]
        logger.info(f"Majority voting: {dict(answer_counts)} -> {most_common}")
        return most_common
    
    return None 