import re
import time
import random
import logging

import aiohttp
from omegaconf import DictConfig
from pydantic import BaseModel
from pipelinerl.rollouts import RolloutResult, BaseMetrics
from pipelinerl.world import Job
from tapeagents.core import Prompt
from tapeagents.llms.trainable import TrainableLLM

from pipelinerl.async_llm import llm_async_generate, make_training_text
from .verifier_api import verify_answer_rpc


class Metrics(BaseMetrics):
    penalty: float
    overflow: bool = False
    auto_boxed: bool = False

class RewardTable(BaseModel):
    wrong_answer_not_finished: float
    wrong_answer_finished: float
    no_answer_not_finished: float
    no_answer_finished: float
    unparsable_not_finished: float
    unparsable_finished: float
    correct_answer_not_finished: float
    correct_answer_finished: float
    buffer_tokens: int = 0 # 0 means no overlong reward shaping


_BOXED_PREFIX = "\\boxed{"


def _find_last_boxed_span(text: str) -> tuple[int, int] | None:
    start = text.rfind(_BOXED_PREFIX)
    if start < 0:
        return None
    depth = 0
    for idx in range(start + len(_BOXED_PREFIX), len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:
                return start, idx + 1
            depth -= 1
    return None


_ANSWER_PREFIX_RE = re.compile(
    r"^(final answer|answer|ans\.?|thus.*?is|therefore.*?is|so the answer is)[:=\-\s]*",
    re.IGNORECASE,
)


def _strip_answer_prefix(line: str) -> str:
    return _ANSWER_PREFIX_RE.sub("", line).strip()


_EXPRESSION_RE = re.compile(r"([-+]?\s*[^\s]+(?:\s*[^\s]+)*)")


def _extract_candidate_expression(text: str) -> str | None:
    for raw_line in reversed(text.strip().splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        line = _strip_answer_prefix(line.rstrip(".;!"))
        if not line:
            continue
        if any(char.isdigit() for char in line) or "\\" in line:
            return line
    match = _EXPRESSION_RE.search(text.strip())
    return match.group(1).strip() if match else None


def ensure_boxed_answer(text: str) -> tuple[str, bool]:
    """Return text contained in the last \boxed{} block."""
    if not text:
        return text, False

    cleaned = text.rstrip()
    span = _find_last_boxed_span(cleaned)
    if span is not None:
        start, end = span
        prefix = cleaned[:start].rstrip()
        boxed = cleaned[start:end]
        suffix_adjusted = f"{prefix}\n{boxed}" if prefix else boxed
        changed = suffix_adjusted != cleaned
        return suffix_adjusted, changed

    candidate = _extract_candidate_expression(cleaned)
    if not candidate:
        return cleaned, False

    candidate_boxed = f"\\boxed{{{candidate}}}"
    if candidate in cleaned:
        prefix, sep, suffix = cleaned.rpartition(candidate)
        if sep:
            adjusted = f"{prefix}{candidate_boxed}{suffix}".rstrip()
        else:
            adjusted = cleaned
    else:
        adjusted = cleaned
    if adjusted == cleaned:
        adjusted = f"{cleaned}\n{candidate_boxed}" if cleaned else candidate_boxed
    return adjusted, adjusted != cleaned
def get_reward(answer_status: str, finished: bool, reward_table: RewardTable) -> float:
    match (answer_status, finished):
        case ("wrong", False):
            return reward_table.wrong_answer_not_finished
        case ("wrong", True):
            return reward_table.wrong_answer_finished
        case ("no_answer", False):
            return reward_table.no_answer_not_finished
        case ("no_answer", True):
            return reward_table.no_answer_finished
        case ("unparsable", False):
            return reward_table.unparsable_not_finished
        case ("unparsable", True):
            return reward_table.unparsable_finished
        case ("correct", False):
            return reward_table.correct_answer_not_finished
        case ("correct", True):
            return reward_table.correct_answer_finished
        case _:
            raise ValueError(f"Invalid answer_status/finished combination: {answer_status}/{finished}")


def length_penalty(max_length: int, sequence_length: int, buffer_tokens: int) -> float:
    """
    Compute the overlong penalty
    """
    if sequence_length > (max_length - buffer_tokens) and sequence_length <= max_length:
        return ((max_length - buffer_tokens) - sequence_length) / buffer_tokens
    return 0.

async def generate_math_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    messages = []
    if cfg.actor.system_prompt:
        messages.append({"role": "system", "content": cfg.actor.system_prompt})
    user_content = problem["task"]
    if cfg.actor.task_prompt:
        user_content = f"{user_content} \n{cfg.actor.task_prompt}"
    messages.append({"role": "user", "content": user_content})
    prompt = Prompt(messages=messages)

    time_start = time.time()
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - time_start

    assert llm_call.output.content is not None
    auto_boxed = False
    if getattr(cfg.actor, "ensure_boxed_answers", False):
        sanitized, changed = ensure_boxed_answer(llm_call.output.content)
        if changed:
            llm_call.output.content = sanitized
            auto_boxed = True
    reward_table = RewardTable(**dict(cfg.rewards))
    discount_factor = cfg.actor.discount_factor

    # math_verify is a fast environment, no support for environment replicas for now
    env_jobs = [Job(**job) for job in cfg.jobs if job["kind"] == "environment"]
    # choose the job randomly
    env_job = random.choice(env_jobs)
    assert env_job.port is not None
    try:
        answer_status = await verify_answer_rpc(
            session=session,
            host=env_job.hostname,
            port=env_job.port,
            prediction=llm_call.output.content,
            gold=problem["answer"],
            strict=True,
        )
    except Exception as exc:
        answer_status = "unparsable"

    trace = make_training_text(llm, llm_call)
    # Determine reward based on answer status and finished state
    reward = get_reward(answer_status, trace.finished, reward_table)
    # Apply discount factor based on output length
    reward *= discount_factor**llm_call.output_length_tokens
    overlong_penalty = 0
    if reward_table.buffer_tokens > 0:
        overlong_penalty = length_penalty(
            llm.parameters["max_tokens"],
            llm_call.output_length_tokens,
            reward_table.buffer_tokens,
        )
    reward += overlong_penalty
    trace.reward = reward

    # Prefer backend-provided finish reason if available; normalize for comparisons
    if isinstance(trace.metadata, dict):
        finish_reason_raw = trace.metadata.get("finish_reason")
    else:
        finish_reason_raw = None

    finish_reason = (
        str(finish_reason_raw).strip().lower() if finish_reason_raw is not None else None
    )

    # Overflow is true when generation hit the backend length cap explicitly
    overflow_by_reason = finish_reason == "length"

    # Only fall back to heuristics when we lack a reliable finish reason (e.g., old backends)
    use_fallback = finish_reason is None or finish_reason not in {"stop", "length"}

    overflow_by_length = False
    overflow_by_eos = False
    if use_fallback:
        max_tokens = int(llm.parameters.get("max_tokens", 0) or 0)
        if max_tokens > 0:
            try:
                overflow_by_length = trace.output_tokens >= max_tokens
            except Exception:
                overflow_by_length = False

        try:
            eos_token_id = getattr(llm.tokenizer, "eos_token_id", None)
            overflow_by_eos = bool(
                trace.input_ids
                and eos_token_id is not None
                and trace.input_ids[-1] != eos_token_id
            )
        except Exception:
            overflow_by_eos = False

    metrics = Metrics(
        reward=reward,
        success=answer_status == "correct",
        no_error=answer_status != "unparsable",
        no_answer=answer_status == "no_answer",
        penalty=overlong_penalty,
        overflow=bool(overflow_by_reason or overflow_by_length or overflow_by_eos),
        auto_boxed=auto_boxed,
    )

    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=latency, 
        dataset_name=problem.get("dataset"),
    )
