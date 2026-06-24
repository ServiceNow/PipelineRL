"""Rollout for the TMax-style terminal-agent domain.

A multi-turn bash agent (mini-SWE-agent shape: one ``bash`` tool, persistent
shell, submit marker, output truncation) drives a proot sandbox hosted on a
remote env-server job, then a pytest verifier scores the final state. Reward is
outcome-only and broadcast to every turn, matching the TMax recipe; PipelineRL's
LOO group advantage plus zero-advantage filtering supply the rest.

The sandbox runs on ``kind="environment"`` jobs (placed across the actor nodes by
``WorldMap._place_environments``) and is reached over plain HTTP with the
actor's shared ``aiohttp`` session. No TapeAgents dependency.
"""
from __future__ import annotations

import asyncio
import logging
import random
import re
import time
import traceback
from typing import List, Optional

import aiohttp
from omegaconf import DictConfig

from pipelinerl.async_llm import llm_async_generate, make_training_texts_from_llm_calls
from pipelinerl.llm import LLMCall, Prompt, TrainableLLM
from pipelinerl.rollouts import BaseMetrics, RolloutResult, summarize_training_texts
from pipelinerl.utils import get_environment_jobs

logger = logging.getLogger(__name__)

_BASH_RE = re.compile(r"```(?:bash|sh)?\s*\n(.*?)```", re.DOTALL)
_SUBMIT_MARKER = "TASK_COMPLETE"

SYSTEM_PROMPT = (
    "You are a terminal agent. You solve a task by running shell commands in a "
    "persistent bash session. Respond each turn with exactly one command inside a "
    "```bash``` code block. You will then see its output. When the task is fully "
    f"done, reply with a final line containing only {_SUBMIT_MARKER} and no code block."
)


class TerminalMetrics(BaseMetrics):
    verifier_pass: bool = False
    build_ok: bool = True
    init_ok: bool = True
    overflow: bool = False
    n_turns: int = 0
    n_llm_calls: int = 0


def _parse_command(content: str) -> Optional[str]:
    """Return the bash command for this turn, or None if the agent is done."""
    if _SUBMIT_MARKER in content:
        return None
    matches = _BASH_RE.findall(content)
    if not matches:
        return None
    return matches[-1].strip()


def _failed_result(problem: dict, start_time: float, metrics: TerminalMetrics) -> RolloutResult:
    return RolloutResult(
        training_texts=[],
        metrics=metrics,
        latency=time.time() - start_time,
        dataset_name=problem.get("dataset"),
        domain="terminal",
    )


class EnvironmentCapacityError(RuntimeError):
    pass


async def _post(session: aiohttp.ClientSession, url: str, payload: dict, timeout: float) -> dict:
    async with session.post(url, json=payload, timeout=timeout) as resp:
        if resp.status != 200:
            text = await resp.text()
            if resp.status == 503 and "capacity reached" in text:
                raise EnvironmentCapacityError(f"{url} -> HTTP {resp.status}: {text}")
            raise RuntimeError(f"{url} -> HTTP {resp.status}: {text}")
        return await resp.json()


async def _check_env_health(url: str, session: aiohttp.ClientSession) -> bool:
    try:
        async with session.get(f"{url}/health", timeout=5) as resp:
            return resp.status == 200
    except Exception as e:
        logger.warning("env health check failed for %s: %s", url, e)
        return False


async def generate_terminal_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    start_time = time.time()
    tcfg = cfg.terminal
    rollout_timeout = getattr(tcfg, "rollout_timeout", 900)

    env_jobs = get_environment_jobs(cfg, "terminal")
    if not env_jobs:
        logger.error("no terminal environment jobs registered in cfg.jobs")
        return _failed_result(problem, start_time, TerminalMetrics(
            reward=tcfg.reward_fail, success=False, no_error=False, no_answer=True, build_ok=False))

    capacity_retry_sleep = float(getattr(tcfg, "capacity_retry_sleep", 2.0))
    deadline = time.time() + rollout_timeout
    logged_capacity_wait = False
    logged_no_healthy_wait = False
    while time.time() < deadline:
        urls = [f"http://{job.hostname}:{job.port}" for job in env_jobs]
        random.shuffle(urls)
        saw_capacity = False
        saw_healthy = False
        for url in urls:
            if not await _check_env_health(url, session):
                continue
            saw_healthy = True
            try:
                return await asyncio.wait_for(
                    _execute_rollout(cfg, llm, problem, session, start_time, url),
                    timeout=max(1.0, deadline - time.time()),
                )
            except EnvironmentCapacityError:
                saw_capacity = True
                continue
            except asyncio.TimeoutError:
                logger.warning("rollout timed out for %s on %s, trying next server", problem.get("task_id"), url)
                continue
            except Exception:
                logger.warning("rollout failed for %s on %s: %s", problem.get("task_id"), url, traceback.format_exc())
                continue

        if saw_capacity:
            if not logged_capacity_wait:
                logger.info("terminal env capacity saturated for %s; waiting", problem.get("task_id"))
                logged_capacity_wait = True
            await asyncio.sleep(min(capacity_retry_sleep, max(0.0, deadline - time.time())))
            continue

        if not saw_healthy:
            if not logged_no_healthy_wait:
                logger.warning(
                    "no healthy terminal environment servers for %s; waiting",
                    problem.get("task_id"),
                )
                logged_no_healthy_wait = True
            await asyncio.sleep(min(capacity_retry_sleep, max(0.0, deadline - time.time())))
            continue

        break

    logger.error("all terminal environment servers failed for %s", problem.get("task_id"))
    return _failed_result(problem, start_time, TerminalMetrics(
        reward=tcfg.reward_fail, success=False, no_error=False, no_answer=True))


async def _execute_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
    start_time: float,
    env_url: str,
) -> RolloutResult:
    tcfg = cfg.terminal
    call_timeout = getattr(tcfg, "env_call_timeout", 300)
    # /start_task triggers the one-time per-task rootfs build, which reads the
    # base over NFS and can take several minutes. Give it a longer timeout than
    # /step so a cold build is not abandoned mid-flight (which would leak the
    # server-side session and its capacity slot).
    start_timeout = getattr(tcfg, "env_start_timeout", 900)

    start = await _post(session, f"{env_url}/start_task", {"task_data": problem}, start_timeout)
    session_id = start.get("session_id")
    if not session_id or not start.get("started") or not start.get("init_ok"):
        logger.warning("task %s not runnable (start=%s), dropping", problem.get("task_id"), start)
        return _failed_result(problem, start_time, TerminalMetrics(
            reward=tcfg.reward_fail, success=False, no_error=False, no_answer=True,
            build_ok=start.get("build_ok", False), init_ok=start.get("init_ok", False)))

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem["task"]},
        ]
        llm_calls: List[LLMCall] = []
        for _ in range(tcfg.max_turns):
            llm_call = await llm_async_generate(llm, Prompt(messages=messages), session)
            llm_calls.append(llm_call)
            content = llm_call.output.content or ""
            messages.append({"role": "assistant", "content": content})

            command = _parse_command(content)
            if command is None:
                break
            obs = await _post(session, f"{env_url}/step", {"session_id": session_id, "command": command}, call_timeout)
            messages.append({"role": "user", "content": obs["output"]})

        verifier = await _post(session, f"{env_url}/finish", {"session_id": session_id}, call_timeout)
        verifier_pass = bool(verifier["passed"])
    finally:
        try:
            await _post(session, f"{env_url}/close", {"session_id": session_id}, 30)
        except Exception:
            logger.warning("failed to close session %s on %s", session_id, env_url)

    reward = tcfg.reward_pass if verifier_pass else tcfg.reward_fail
    training_texts = make_training_texts_from_llm_calls(llm, llm_calls, reward=reward)
    summary = summarize_training_texts(training_texts)

    metrics = TerminalMetrics(
        reward=reward,
        success=verifier_pass,
        no_error=True,
        no_answer=len(llm_calls) == 0,
        verifier_pass=verifier_pass,
        overflow=summary.overflow,
        n_turns=len(messages) // 2,
        n_llm_calls=len(llm_calls),
    )
    return RolloutResult(
        training_texts=training_texts,
        metrics=metrics,
        latency=time.time() - start_time,
        dataset_name=problem.get("dataset"),
        domain="terminal",
    )
