import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

import aiohttp
from omegaconf import DictConfig

from sandbox_fusion import RunCodeRequest, set_sandbox_endpoint, run_code_async

from pipelinerl.async_llm import llm_async_generate, make_training_texts_from_llm_calls
from pipelinerl.domains.math import RewardTable, get_reward, length_penalty, verify_answer_rpc
from pipelinerl.llm import Prompt, TrainableLLM
from pipelinerl.rollouts import BaseMetrics, RolloutResult
from pipelinerl.utils import get_environment_jobs, resolve_environment_key

logger = logging.getLogger(__name__)

_SANDBOX_CONFIGURED = False

_BLOCKED_PATTERNS = [
    re.compile(r"\bsys\.exit\b"),
    re.compile(r"\bos\._exit\b"),
    re.compile(r"\bos\.system\b"),
    re.compile(r"\bsubprocess\b"),
    re.compile(r"\bos\.popen\b"),
    re.compile(r"\bos\.exec\w*\b"),
    re.compile(r"\bos\.spawn\w*\b"),
    re.compile(r"\bos\.kill\b"),
    re.compile(r"\bshutil\.rmtree\b"),
    re.compile(r"\bos\.remove\b"),
    re.compile(r"\bos\.unlink\b"),
]


def build_tool_definitions() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "run_python_code",
                "description": "Execute Python code. Print only the final result.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "MathAnswer",
                "description": "Submit the final answer in LaTeX \\boxed{} format.",
                "parameters": {
                    "type": "object",
                    "properties": {"answer": {"type": "string", "description": "The final answer"}},
                    "required": ["answer"],
                },
            },
        },
    ]


def _check_code_safety(code: str) -> str | None:
    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(code):
            return f"Blocked: code contains forbidden pattern '{pattern.pattern}'"
    return None


async def execute_python_sandbox(code: str, endpoint: str, timeout: float) -> str:
    """Execute Python code via SandboxFusion and return formatted output."""
    global _SANDBOX_CONFIGURED
    if not _SANDBOX_CONFIGURED:
        set_sandbox_endpoint(endpoint)
        _SANDBOX_CONFIGURED = True
        logger.info("Configured SandboxFusion endpoint: %s", endpoint)

    rejection = _check_code_safety(code)
    if rejection is not None:
        return rejection

    try:
        request = RunCodeRequest(code=code, language="python", run_timeout=timeout)
        response = await run_code_async(request)

        stdout = ""
        stderr = ""
        if response.run_result:
            stdout = response.run_result.stdout or ""
            stderr = response.run_result.stderr or ""

        status = response.status.value if hasattr(response.status, "value") else str(response.status)
        is_timeout = "timeout" in status.lower() or "timeout" in (response.message or "").lower()

        parts = []
        if stdout:
            parts.append(stdout.rstrip())
        if stderr:
            parts.append(f"[stderr]\n{stderr.rstrip()}")
        if is_timeout:
            parts.append("[execution timed out]")
        if not parts:
            parts.append("[no output]")
        return "\n".join(parts)

    except asyncio.TimeoutError:
        return "[execution timed out]"
    except Exception as exc:
        logger.warning("SandboxFusion error: %s", exc)
        return f"[execution error: {exc}]"


def _serialize_tool_calls(tool_calls) -> list[dict]:
    """Serialize litellm tool call objects to dicts for conversation history."""
    return [
        {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            },
        }
        for tc in tool_calls
    ]


def _parse_tool_arguments(arguments: str, *, fallback_key: str | None = None) -> dict:
    """Parse tool-call arguments into an object payload.

    Valid JSON that is not an object should not crash the rollout loop. A bare
    string can still be recovered for simple single-field tool schemas.
    """
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return {}
    if isinstance(parsed, dict):
        return parsed
    if fallback_key is not None and isinstance(parsed, str):
        return {fallback_key: parsed}
    return {}


@dataclass
class _ToolContext:
    sandbox_endpoint: str
    sandbox_timeout: float
    messages: list[dict]
    final_answer: str | None = None
    submitted_final_answer: bool = False
    num_python_calls: int = 0


ToolHandler = Callable[[object, _ToolContext], Awaitable[None]]


async def _handle_math_answer(tc, ctx: _ToolContext) -> None:
    args = _parse_tool_arguments(tc.function.arguments, fallback_key="answer")
    ctx.final_answer = args.get("answer", "")
    ctx.submitted_final_answer = True
    ctx.messages.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": f"Answer submitted: {ctx.final_answer}",
    })


async def _handle_run_python_code(tc, ctx: _ToolContext) -> None:
    args = _parse_tool_arguments(tc.function.arguments, fallback_key="code")
    code = args.get("code") or args.get("python_code", "")
    result = await execute_python_sandbox(code, ctx.sandbox_endpoint, ctx.sandbox_timeout)
    ctx.num_python_calls += 1
    ctx.messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})


async def _handle_unknown_tool(tc, ctx: _ToolContext) -> None:
    ctx.messages.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": f"Unknown tool: {tc.function.name}",
    })


_TOOL_HANDLERS: dict[str, ToolHandler] = {
    "MathAnswer": _handle_math_answer,
    "run_python_code": _handle_run_python_code,
}


class RewardShaper:
    def __init__(self, cfg: DictConfig, llm: TrainableLLM):
        self._python_cfg = getattr(cfg, "python_tool_shaping", None)
        self._length_cfg = getattr(cfg, "length_shaping", None)
        self._max_gen_tokens = int(llm.parameters.get("max_tokens", 2048))

    def compute(self, answer_status: str, num_python_calls: int, llm_calls: list) -> float:
        return (
            self._python_tool_bonus(answer_status, num_python_calls)
            + self._length_adjustment(answer_status, llm_calls)
        )

    def _python_tool_bonus(self, answer_status: str, num_python_calls: int) -> float:
        cfg = self._python_cfg
        if cfg is None:
            return 0.0
        bonus = float(getattr(cfg, "bonus_on_correct_with_python", 0.0))
        penalty = float(getattr(cfg, "penalty_on_incorrect_without_python", 0.0))
        max_abs = float(getattr(cfg, "max_abs", 0.2))
        total = 0.0
        if answer_status == "correct" and num_python_calls >= 1:
            total += bonus
        if answer_status in ("wrong", "unparsable") and num_python_calls == 0:
            total -= penalty
        return max(-max_abs, min(max_abs, total))

    def _length_adjustment(self, answer_status: str, llm_calls: list) -> float:
        cfg = self._length_cfg
        if cfg is None or not llm_calls:
            return 0.0
        if hasattr(cfg, "target_ratio"):
            ratio = float(getattr(cfg, "target_ratio"))
            target = int(max(1, ratio * self._max_gen_tokens))
            target = max(int(getattr(cfg, "min_target_tokens", 0)), target)
            target = min(int(getattr(cfg, "max_target_tokens", 10**9)), target)
        else:
            target = int(getattr(cfg, "target_output_tokens", 512))
        slope = float(getattr(cfg, "slope", 0.0))
        max_penalty = float(getattr(cfg, "max_penalty", 0.0))
        bonus_short_correct = float(getattr(cfg, "bonus_on_short_correct", 0.0))

        avg_out = sum(getattr(c, "output_length_tokens", 0) for c in llm_calls) / len(llm_calls)
        total = 0.0
        if slope > 0.0 and max_penalty > 0.0 and avg_out > target:
            total -= min(max_penalty, slope * (avg_out - target))
        if bonus_short_correct > 0.0 and answer_status == "correct" and avg_out <= target:
            total += bonus_short_correct
        return total


class Metrics(BaseMetrics):
    num_python_calls: int = 0
    num_steps: int = 0
    overflow: bool = False


async def generate_tir_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    start = time.perf_counter()

    messages: list[dict] = []
    if cfg.actor.system_prompt:
        messages.append({"role": "system", "content": cfg.actor.system_prompt})
    messages.append({"role": "user", "content": cfg.actor.task_template.format(task=problem["task"])})

    tools = build_tool_definitions()

    llm_calls = []
    ctx = _ToolContext(
        sandbox_endpoint=str(cfg.sandbox_endpoint),
        sandbox_timeout=float(cfg.sandbox_timeout),
        messages=messages,
    )
    agent_max_loops = int(getattr(cfg.actor, "agent_max_loops", 3))
    configured_max_tokens = int(llm.parameters.get("max_tokens", 16000))
    max_model_len = int(cfg.vllm_config.vllm_kwargs.get("max_model_len", 32000))
    min_generation_tokens = 256

    for _turn in range(agent_max_loops):
        prompt = Prompt(messages=list(messages), tools=tools)

        llm.load_tokenizer()
        prompt_token_ids = llm.tokenizer.apply_chat_template(
            messages,
            add_special_tokens=True,
            add_generation_prompt=True,
            tools=tools,
        )
        prompt_len = len(prompt_token_ids)
        remaining = max_model_len - prompt_len
        if remaining < min_generation_tokens:
            logger.warning(
                "Prompt length %d leaves only %d tokens for generation (max_model_len=%d), stopping loop",
                prompt_len, remaining, max_model_len,
            )
            break
        max_tokens_this_turn = min(configured_max_tokens, remaining)
        if max_tokens_this_turn < configured_max_tokens:
            logger.warning(
                "Turn %d: capping max_tokens from %d to %d (prompt_len=%d, max_model_len=%d)",
                _turn, configured_max_tokens, max_tokens_this_turn, prompt_len, max_model_len,
            )

        llm_call = await llm_async_generate(llm, prompt, session, max_tokens_override=max_tokens_this_turn)
        llm_calls.append(llm_call)

        if not llm_call.output.tool_calls:
            break

        assistant_msg: dict = {"role": "assistant", "content": llm_call.output.content or ""}
        assistant_msg["tool_calls"] = _serialize_tool_calls(llm_call.output.tool_calls)
        messages.append(assistant_msg)

        for tc in llm_call.output.tool_calls:
            handler = _TOOL_HANDLERS.get(tc.function.name, _handle_unknown_tool)
            await handler(tc, ctx)
            if ctx.submitted_final_answer:
                break

        if ctx.submitted_final_answer:
            break

    if ctx.final_answer is not None:
        prediction = ctx.final_answer
    elif llm_calls:
        prediction = llm_calls[-1].output.content or ""
    else:
        prediction = ""

    env_key = resolve_environment_key(cfg, default="math")
    env_jobs = get_environment_jobs(cfg, env_key)
    if not env_jobs:
        raise RuntimeError("No environment servers available for math domain")
    env_job = random.choice(env_jobs)
    assert env_job.port is not None
    answer_status = await verify_answer_rpc(
        session=session,
        host=env_job.hostname,
        port=env_job.port,
        prediction=prediction,
        gold=problem["answer"],
        strict=True,
    )

    reward_table = RewardTable(**dict(cfg.rewards))
    base_reward = get_reward(answer_status, ctx.submitted_final_answer, reward_table)

    discount_factor = float(getattr(cfg.actor, "discount_factor", 1.0))
    if discount_factor != 1.0:
        total_generated_tokens = sum(getattr(c, "output_length_tokens", 0) for c in llm_calls)
        base_reward *= discount_factor ** total_generated_tokens

    buffer_tokens = getattr(reward_table, "buffer_tokens", 0)
    if buffer_tokens:
        max_tokens = int(llm.parameters.get("max_tokens", 0))
        total_output_tokens = sum(getattr(c, "output_length_tokens", 0) for c in llm_calls)
        if max_tokens > 0:
            base_reward += length_penalty(max_tokens, total_output_tokens, buffer_tokens)

    shaping = RewardShaper(cfg, llm).compute(answer_status, ctx.num_python_calls, llm_calls)
    reward = base_reward + shaping

    training_texts = make_training_texts_from_llm_calls(llm, llm_calls, reward=reward)
    for text in training_texts:
        text.finished = ctx.submitted_final_answer

    latency = time.perf_counter() - start

    metrics = Metrics(
        reward=reward,
        success=answer_status == "correct",
        no_error=answer_status != "unparsable",
        no_answer=answer_status == "no_answer",
        num_python_calls=ctx.num_python_calls,
        num_steps=len(llm_calls),
        overflow=not ctx.submitted_final_answer,
    )

    return RolloutResult(
        training_texts=training_texts,
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset"),
        domain="tir",
    )
