import logging
import math
import time

import aiohttp
from omegaconf import DictConfig

from pipelinerl.async_llm import llm_async_generate, make_training_text
from pipelinerl.llm import Prompt, TrainableLLM
from pipelinerl.rollouts import BaseMetrics, RolloutResult

from pipelinerl.domains.swe.repair import build_messages, parse_edits
from pipelinerl.domains.swe.reward import calculate_precise_reward

logger = logging.getLogger(__name__)


class SWEMetrics(BaseMetrics):
    format_error: bool = False


async def generate_swe_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    file_contents = problem.get("file_contents", {})
    problem_statement = problem["problem_statement"]
    gold_patch = problem.get("patch", "")

    messages = build_messages(problem_statement, file_contents)
    prompt = Prompt(messages=messages)

    time_start = time.time()
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - time_start

    raw_output = llm_call.output.content or ""
    edits = parse_edits(raw_output)

    reward, reward_meta = calculate_precise_reward(file_contents, gold_patch, edits)

    if hasattr(cfg.actor, 'discount_factor'):
        reward *= cfg.actor.discount_factor ** llm_call.output_length_tokens

    trace = make_training_text(llm, llm_call)
    trace.reward = reward if not math.isnan(reward) else 0.0
    trace.metadata["stage"] = "repair"
    trace.metadata["dataset"] = problem.get("dataset")
    trace.metadata["problem_id"] = problem.get("id") or problem.get("instance_id")

    format_error = bool(reward_meta.get("format_error", False))
    success_threshold = getattr(cfg.actor, 'success_threshold', 0.8)
    success = (not format_error) and reward >= success_threshold

    metrics = SWEMetrics(
        reward=trace.reward,
        success=success,
        no_error=not format_error,
        no_answer=len(edits) == 0,
        format_error=format_error,
    )

    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset"),
        domain="swe",
    )
