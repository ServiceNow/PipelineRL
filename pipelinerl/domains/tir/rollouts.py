import logging
import time
import json
import os
from typing import Dict, Any, List, Union
from collections import Counter
import aiohttp
from omegaconf import DictConfig
from tapeagents.llms import TrainableLLM
from pipelinerl.rollouts import RolloutResult

logger = logging.getLogger(__name__)

# Cache environments globally
_cached_environments = {}

async def generate_tir_rollout(cfg: DictConfig, llm: TrainableLLM, problem: dict, session: aiohttp.ClientSession) -> RolloutResult:
    """Generate a rollout for TIR domain with fast or sc_tir modes."""
    from pipelinerl.async_llm import make_training_text
    from tapeagents.orchestrator import main_loop
    from .agent import Task, TIRMathTape, AnswerAction, TIRMathAgent
    from .environment import MCPPythonEnvironment
    
    time_start = time.time()
    
    env_key = str(cfg.environment)
    if env_key not in _cached_environments:
        _cached_environments[env_key] = MCPPythonEnvironment()
        logger.info("Created new cached MCP environment")
    environment = _cached_environments[env_key]
    
    mode = getattr(cfg.actor, 'mode', 'fast')
    num_candidates = getattr(cfg.actor, 'num_candidates', 4) if mode == 'sc_tir' else 1
    max_reasoning_steps = getattr(cfg.actor, 'max_reasoning_steps', 8)
    
    logger.info(f"Running {mode} mode with {num_candidates} candidates, max {max_reasoning_steps} steps")
    
    all_final_tapes = []
    all_llm_calls = []
    all_training_samples = []
    candidate_answers = []
    
    for candidate_idx in range(num_candidates):
        logger.info(f"Generating candidate {candidate_idx + 1}/{num_candidates}")
        
        agent = TIRMathAgent(
            system_prompt=cfg.actor.system_prompt,
            max_iterations=max_reasoning_steps
        )
        agent.llms = {"default": llm}
        
        task_step = Task(task=problem["task"])
        start_tape = TIRMathTape(steps=[task_step], context=None)
        
        # agent-environment interaction
        final_tape = None
        for event in main_loop(agent, start_tape, environment, cfg.max_loops):
            if event.agent_tape:
                final_tape = event.agent_tape
            elif event.env_tape:
                final_tape = event.env_tape
        
        if final_tape is not None:
            all_final_tapes.append(final_tape)
            
            answer_step = None
            for step in reversed(final_tape.steps):
                if isinstance(step, AnswerAction):
                    answer_step = step
                    break
            
            if answer_step is not None:
                candidate_answers.append(answer_step.value)
                logger.info(f"Candidate {candidate_idx + 1} answer: {answer_step.value}")
            else:
                candidate_answers.append(None)
                logger.warning(f"Candidate {candidate_idx + 1} produced no answer")
            
            candidate_llm_calls = []
            candidate_samples = []
            
            for step in final_tape.steps:
                if step.metadata and step.metadata.other:
                    llm_call_data = step.metadata.other.get("llm_call")
                    if llm_call_data:
                        training_text = make_training_text(llm, llm_call_data)
                        candidate_samples.append(training_text)
                        candidate_llm_calls.append(llm_call_data)
            
            if not candidate_llm_calls:
                _, candidate_llm_calls = agent.reuse(final_tape)
                candidate_samples = [agent.make_training_text(llm_call) for llm_call in candidate_llm_calls]
            
            all_llm_calls.extend(candidate_llm_calls)
            all_training_samples.extend(candidate_samples)
        else:
            candidate_answers.append(None)
            logger.warning(f"Candidate {candidate_idx + 1} failed")
    
    # majority voting or single answer
    if mode == 'sc_tir':
        final_answer = apply_majority_voting(candidate_answers)
        logger.info(f"Candidates: {candidate_answers} -> Majority: {final_answer}")
    else:
        final_answer = candidate_answers[0] if candidate_answers else None
        logger.info(f"Fast mode answer: {final_answer}")
    
    if getattr(cfg, 'save_tapes', False):
        debug_dir = os.path.join(cfg.output_dir, "debug_tapes") 
        os.makedirs(debug_dir, exist_ok=True)
        
        debug_file = os.path.join(debug_dir, f"problem_{problem.get('id', 'unknown')}.json")
        debug_data = {
            "problem": problem,
            "mode": mode,
            "num_candidates": num_candidates,
            "candidate_answers": candidate_answers,
            "majority_answer": final_answer,
            "num_tapes": len(all_final_tapes),
            "total_llm_calls": len(all_llm_calls),
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
    
    # rewards
    reward = 1.0 if success else 0.0
    for sample in all_training_samples:
        sample.reward = reward
    
    # discount factor
    if cfg.actor.discount_factor and all_llm_calls:
        total_output_tokens = sum(llm_call.output_length_tokens for llm_call in all_llm_calls)
        reward *= cfg.actor.discount_factor ** total_output_tokens
        for sample in all_training_samples:
            sample.reward = reward
    
    has_errors = any(
        any(1 for s in tape.steps if hasattr(s, 'error') and s.error) 
        for tape in all_final_tapes
    )
    
    valid_answers = [ans for ans in candidate_answers if ans is not None]
    if mode == 'sc_tir' and len(valid_answers) > 1:
        answer_counts = Counter(valid_answers)
        most_common_count = answer_counts.most_common(1)[0][1] if answer_counts else 0
        agreement_rate = most_common_count / len(valid_answers)
    else:
        agreement_rate = 1.0 if valid_answers else 0.0
    
    metrics = {
        "reward": reward,
        "success": 1 if success else 0,
        "no_error": 1 if not has_errors else 0,
        "no_answer": 1 if answer_status == "no_answer" else 0,
        "overflow": 0,  # TODO: detect max_loops
        "prompt_tokens": sum(llm_call.prompt_length_tokens for llm_call in all_llm_calls) if all_llm_calls else 0,
        "output_tokens": sum(llm_call.output_length_tokens for llm_call in all_llm_calls) if all_llm_calls else 0,
        "mode": mode,
        "num_candidates": num_candidates,
        "candidates_with_answers": len(valid_answers),
        "agreement_rate": agreement_rate,
        "majority_answer": final_answer,
        "candidate_answers": candidate_answers,
    }
    
    return RolloutResult(
        training_texts=all_training_samples,
        metrics=metrics,
        latency=time.time() - time_start,
        dataset_name=problem.get("dataset", "unknown"),
        prompt_tokens=[llm_call.prompt_length_tokens for llm_call in all_llm_calls] if all_llm_calls else [],
        output_tokens=[llm_call.output_length_tokens for llm_call in all_llm_calls] if all_llm_calls else [],
    )


def apply_majority_voting(candidate_answers: List[Any]) -> Any:
    """Apply majority voting to select final answer from candidates."""
    valid_answers = [ans for ans in candidate_answers if ans is not None]
    
    if not valid_answers:
        return None
    
    # normalise answers
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