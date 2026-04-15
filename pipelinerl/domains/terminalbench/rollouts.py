
import asyncio
import logging
import random
import re
import time
import traceback
from typing import Any

import aiohttp
from omegaconf import DictConfig

from pipelinerl.llm import LLMCall, Prompt, TrainableLLM
from pipelinerl.async_llm import llm_async_generate, make_training_text
from pipelinerl.rollouts import BaseMetrics, RolloutResult
from pipelinerl.utils import get_environment_jobs, resolve_environment_key


logger = logging.getLogger(__name__)

# From hello_terminalbench_toolkit.py, extended with bash-block output format
# since llm_async_generate works with text content, not tool-use function calls.
DEFAULT_SYSTEM_PROMPT = """\
You are an expert software engineer working in a Linux terminal.
Work in the /app directory. Read existing files, test your solutions before declaring completion.

To run a shell command, write it in a bash code block:
```bash
<command>
```

You will receive the command output after each step.
When the task is complete, write TASK_COMPLETE on a line by itself."""


class TerminalBenchMetrics(BaseMetrics):
    reward: float
    success: bool
    no_error: bool
    no_answer: bool
    overflow: bool
    n_llm_calls: int
    n_step_errors: int
    n_steps: int
    total_execution_time: float


# ── Environment server HTTP helpers ───────────────────────────────────────────

async def _start_task(
    env_url: str,
    task_data: dict,
    session: aiohttp.ClientSession,
    timeout: int = 300,
) -> dict:
    """Acquire a free env and set up the task in one call. Returns session_id + task info."""
    async with session.post(
        f"{env_url}/start_task",
        json={"task_data": task_data},
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as resp:
        if resp.status != 200:
            raise RuntimeError(f"start_task failed (HTTP {resp.status}): {await resp.text()}")
        return await resp.json()


async def _release(env_url: str, session_id: str, session: aiohttp.ClientSession) -> None:
    try:
        async with session.post(
            f"{env_url}/release",
            json={"session_id": session_id},
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            await resp.read()
    except Exception as e:
        logger.warning(f"release failed for session {session_id}: {e}")


async def _exec(
    env_url: str,
    session_id: str,
    command: str,
    session: aiohttp.ClientSession,
    timeout: int = 60,
) -> dict:
    async with session.post(
        f"{env_url}/exec",
        json={"session_id": session_id, "command": command, "timeout": timeout},
        timeout=aiohttp.ClientTimeout(total=timeout + 10),
    ) as resp:
        return await resp.json()


async def _evaluate(
    env_url: str,
    session_id: str,
    archive_b64: str,
    session: aiohttp.ClientSession,
    test_timeout_sec: int = 900,
) -> float:
    async with session.post(
        f"{env_url}/evaluate",
        json={
            "session_id": session_id,
            "archive_b64": archive_b64,
            "test_timeout_sec": test_timeout_sec,
        },
        timeout=aiohttp.ClientTimeout(total=test_timeout_sec + 60),
    ) as resp:
        data = await resp.json()
    return float(data.get("reward", 0.0))


# ── Rollout entry point ────────────────────────────────────────────────────────

async def generate_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict[str, Any],
    session: aiohttp.ClientSession,
) -> RolloutResult:
    start_time = time.time()
    rollout_timeout = getattr(cfg, "rollout_timeout", 600)

    env_key = resolve_environment_key(cfg, default="terminalbench")
    env_jobs = get_environment_jobs(cfg, env_key)
    if not env_jobs:
        raise RuntimeError("No environment servers available for terminalbench domain")

    shuffled = list(env_jobs)
    random.shuffle(shuffled)

    for env_job in shuffled:
        env_url = f"http://{env_job.hostname}:{env_job.port}"

        # Health check
        try:
            async with session.get(
                f"{env_url}/health", timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"Env server {env_url} unhealthy (status {resp.status})")
                    continue
        except Exception as e:
            logger.warning(f"Health check failed for {env_url}: {e}")
            continue

        try:
            return await asyncio.wait_for(
                _run_rollout(cfg, llm, problem, session, env_url, start_time),
                timeout=rollout_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Rollout timed out after {rollout_timeout}s on {env_url} "
                f"for task {problem.get('task_id', '?')}"
            )
        except Exception:
            logger.warning(
                f"Rollout failed on {env_url} for task {problem.get('task_id', '?')}:\n"
                f"{traceback.format_exc()}"
            )

    logger.error(
        f"All env servers failed for task {problem.get('task_id', '?')}. Returning failed rollout."
    )
    return _failed_rollout(problem, start_time)


# ── Agent loop ─────────────────────────────────────────────────────────────────

async def _run_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
    env_url: str,
    start_time: float,
) -> RolloutResult:
    max_steps = getattr(cfg.actor, "max_steps", 100)
    exec_timeout = getattr(cfg, "exec_timeout", 60)
    test_timeout_sec = problem.get("max_test_timeout_sec", 900)
    archive_b64: str | None = problem.get("archive_b64")

    task_info = await _start_task(
        env_url,
        {
            "id": problem.get("task_id", ""),
            "description": problem.get("instruction", ""),
            "archive_b64": archive_b64,
            "container_config": problem.get("container_config", {}),
        },
        session,
    )
    session_id = task_info["session_id"]
    description = task_info.get("description") or problem.get("instruction", "")

    try:
        system_prompt = getattr(cfg.actor, "system_prompt", None) or DEFAULT_SYSTEM_PROMPT
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": description},
        ]

        llm_calls: list[LLMCall] = []
        n_step_errors = 0

        for _step in range(max_steps):
            prompt = Prompt(messages=list(messages))
            llm_call = await llm_async_generate(llm, prompt, session)
            llm_calls.append(llm_call)

            response_text = llm_call.output.content or ""
            messages.append({"role": "assistant", "content": response_text})

            if "TASK_COMPLETE" in response_text:
                break

            command = _extract_bash_command(response_text)
            if command is None:
                n_step_errors += 1
                messages.append({
                    "role": "user",
                    "content": "No bash command found. Please write a command in a ```bash block.",
                })
                continue

            exec_result = await _exec(env_url, session_id, command, session, timeout=exec_timeout)
            messages.append({"role": "user", "content": _format_exec_result(exec_result)})

        # Compute reward by running the task's test suite inside the sandbox
        reward = 0.0
        if archive_b64:
            try:
                reward = await _evaluate(
                    env_url, session_id, archive_b64, session, test_timeout_sec=test_timeout_sec
                )
            except Exception as e:
                logger.warning(f"Evaluation failed for task {problem.get('task_id', '?')}: {e}")
        else:
            logger.warning(
                f"No archive_b64 in problem dict for task {problem.get('task_id', '?')}; "
                "reward defaults to 0."
            )

    finally:
        await _release(env_url, session_id, session)

    latency = time.time() - start_time
    n_llm_calls = len(llm_calls)
    max_tokens = int(llm.parameters.get("max_tokens", 16000))
    overflow = any(c.output_length_tokens >= max_tokens for c in llm_calls)

    training_texts = [make_training_text(llm, c) for c in llm_calls]
    for t in training_texts:
        t.reward = reward

    metrics = TerminalBenchMetrics(
        reward=reward,
        success=reward > 0.5,
        no_error=n_step_errors == 0,
        no_answer=n_llm_calls == 0,
        overflow=overflow,
        n_llm_calls=n_llm_calls,
        n_step_errors=n_step_errors,
        n_steps=n_llm_calls,
        total_execution_time=latency,
    )

    return RolloutResult(
        training_texts=training_texts,
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset", "terminalbench"),
        domain="terminalbench",
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_bash_command(text: str) -> str | None:
    match = re.search(r"```(?:bash|sh|shell)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _format_exec_result(result: dict) -> str:
    parts = []
    if result.get("stdout"):
        parts.append(result["stdout"])
    if result.get("stderr"):
        parts.append(f"[stderr]: {result['stderr']}")
    exit_code = result.get("exit_code", 0)
    if exit_code != 0:
        parts.append(f"[exit code: {exit_code}]")
    return "\n".join(parts) if parts else "(no output)"


def _failed_rollout(problem: dict, start_time: float) -> RolloutResult:
    latency = time.time() - start_time
    metrics = TerminalBenchMetrics(
        reward=0.0,
        success=False,
        no_error=False,
        no_answer=True,
        overflow=False,
        n_llm_calls=0,
        n_step_errors=0,
        n_steps=0,
        total_execution_time=latency,
    )
    return RolloutResult(
        training_texts=[],
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset", "terminalbench"),
        domain="terminalbench",
    )
