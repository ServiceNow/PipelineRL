import asyncio
import json
import logging
import random
import re
import time

import aiohttp
from omegaconf import DictConfig
from pydantic import BaseModel

from sandbox_fusion import RunCodeRequest, set_sandbox_endpoint, run_code_async

from pipelinerl.async_llm import llm_async_generate, make_training_text_with_tools
from pipelinerl.domains.math import RewardTable, get_reward, length_penalty, verify_answer_rpc
from pipelinerl.llm import Prompt, TrainableLLM
from pipelinerl.rollouts import BaseMetrics, RolloutResult
from pipelinerl.utils import get_environment_jobs, resolve_environment_key

logger = logging.getLogger(__name__)

_SANDBOX_CONFIGURED = False

# Python safety blocklist: patterns that must not appear in user-submitted code
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
    """Return a rejection message if the code matches a blocked pattern, else None."""
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


def _compute_shaping(
    cfg: DictConfig,
    answer_status: str,
    num_python_calls: int,
    llm_calls: list,
    llm: TrainableLLM,
) -> float:
    """Compute reward shaping (python tool bonus + length shaping)."""
    total = 0.0

    shaping_cfg = getattr(cfg, "python_tool_shaping", None)
    if shaping_cfg is not None:
        bonus = float(getattr(shaping_cfg, "bonus_on_correct_with_python", 0.0))
        penalty = float(getattr(shaping_cfg, "penalty_on_incorrect_without_python", 0.0))
        max_abs = float(getattr(shaping_cfg, "max_abs", 0.2))

        if answer_status == "correct" and num_python_calls >= 1:
            total += bonus
        if answer_status in ("wrong", "unparsable") and num_python_calls == 0:
            total -= penalty

        total = max(-max_abs, min(max_abs, total))

    length_cfg = getattr(cfg, "length_shaping", None)
    if length_cfg is not None:
        try:
            if hasattr(length_cfg, "target_ratio"):
                ratio = float(getattr(length_cfg, "target_ratio"))
                max_gen = int(llm.parameters.get("max_tokens", 2048))
                target_tokens = int(max(1, ratio * max_gen))
                min_t = int(getattr(length_cfg, "min_target_tokens", 0))
                max_t = int(getattr(length_cfg, "max_target_tokens", 10**9))
                target_tokens = max(min_t, min(max_t, target_tokens))
            else:
                target_tokens = int(getattr(length_cfg, "target_output_tokens", 512))
            slope = float(getattr(length_cfg, "slope", 0.0))
            max_penalty = float(getattr(length_cfg, "max_penalty", 0.0))
            bonus_short_correct = float(getattr(length_cfg, "bonus_on_short_correct", 0.0))
        except Exception:
            target_tokens, slope, max_penalty, bonus_short_correct = 512, 0.0, 0.0, 0.0

        total_output_tokens = sum(getattr(c, "output_length_tokens", 0) for c in llm_calls)
        avg_output_tokens = total_output_tokens / max(1, len(llm_calls))

        if slope > 0.0 and max_penalty > 0.0 and avg_output_tokens > target_tokens:
            over_by = float(avg_output_tokens - target_tokens)
            total -= min(max_penalty, slope * over_by)

        if bonus_short_correct > 0.0 and answer_status == "correct" and avg_output_tokens <= target_tokens:
            total += bonus_short_correct

    return total


class Metrics(BaseMetrics):
    num_python_calls: int = 0
    num_steps: int = 0
    n_llm_calls: int = 0
    overflow: bool = False


async def generate_tir_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    start = time.perf_counter()

    # 1. Build initial messages
    messages: list[dict] = []
    if cfg.actor.system_prompt:
        messages.append({"role": "system", "content": cfg.actor.system_prompt})
    messages.append({"role": "user", "content": cfg.actor.task_template.format(task=problem["task"])})

    # 2. Tool definitions
    tools = build_tool_definitions()

    # 3. Multi-turn loop
    llm_calls = []
    final_answer = None
    submitted_final_answer = False
    num_python_calls = 0
    agent_max_loops = int(getattr(cfg.actor, "agent_max_loops", 3))
    sandbox_endpoint = str(cfg.sandbox_endpoint)
    sandbox_timeout = float(cfg.sandbox_timeout)

    for _turn in range(agent_max_loops):
        prompt = Prompt(messages=list(messages), tools=tools)
        llm_call = await llm_async_generate(llm, prompt, session)
        llm_calls.append(llm_call)

        if not llm_call.output.tool_calls:
            # Text-only response, no tool call -- end the loop
            break

        # Append assistant message with tool_calls to conversation history
        assistant_msg: dict = {"role": "assistant", "content": llm_call.output.content or ""}
        assistant_msg["tool_calls"] = _serialize_tool_calls(llm_call.output.tool_calls)
        messages.append(assistant_msg)

        # Execute each tool call
        for tc in llm_call.output.tool_calls:
            try:
                parsed = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                parsed = None
            args = parsed if isinstance(parsed, dict) else {}

            if tc.function.name == "MathAnswer":
                final_answer = args.get("answer", "")
                submitted_final_answer = True
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"Answer submitted: {final_answer}",
                })
                break
            elif tc.function.name == "run_python_code":
                code = args.get("code") or args.get("python_code", "")
                result = await execute_python_sandbox(code, sandbox_endpoint, sandbox_timeout)
                num_python_calls += 1
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"Unknown tool: {tc.function.name}",
                })

        if submitted_final_answer:
            break

    # 4. Determine prediction for grading
    if final_answer is not None:
        prediction = final_answer
    elif llm_calls:
        prediction = llm_calls[-1].output.content or ""
    else:
        prediction = ""

    # 5. Verify answer via math verifier
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

    # 6. Compute reward
    reward_table = RewardTable(**dict(cfg.rewards))
    base_reward = get_reward(answer_status, submitted_final_answer, reward_table)

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

    shaping = _compute_shaping(cfg, answer_status, num_python_calls, llm_calls, llm)
    reward = base_reward + shaping

    # 7. Build training texts (tool-call aware)
    training_texts = [make_training_text_with_tools(llm, call) for call in llm_calls]
    for text in training_texts:
        text.reward = reward
        text.finished = submitted_final_answer

    latency = time.perf_counter() - start

    metrics = Metrics(
        reward=reward,
        success=answer_status == "correct",
        no_error=answer_status != "unparsable",
        no_answer=answer_status == "no_answer",
        num_python_calls=num_python_calls,
        num_steps=len(llm_calls),
        n_llm_calls=len(llm_calls),
        overflow=not submitted_final_answer,
    )

    return RolloutResult(
        training_texts=training_texts,
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset"),
        domain="tir",
    )
