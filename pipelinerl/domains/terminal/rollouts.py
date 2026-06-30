"""Rollout for the TMax-style terminal-agent domain.

A multi-turn bash agent (mini-SWE-agent shape: one ``bash`` tool, persistent
shell, submit marker, output truncation) drives a proot sandbox hosted on a
remote env-server job, then a pytest verifier scores the final state. Reward is
outcome-only and broadcast to every valid action turn, matching the TMax recipe;
PipelineRL's LOO group advantage plus zero-advantage filtering supply the rest.

The sandbox runs on ``kind="environment"`` jobs (placed across the actor nodes by
``WorldMap._place_environments``) and is reached over plain HTTP with the
actor's shared ``aiohttp`` session. No TapeAgents dependency.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import traceback
from dataclasses import dataclass
from typing import List

import aiohttp
from omegaconf import DictConfig

from pipelinerl.async_llm import llm_async_generate, make_training_texts_from_llm_calls
from pipelinerl.llm import LLMCall, Prompt, TrainableLLM
from pipelinerl.rollouts import BaseMetrics, RolloutResult, summarize_training_texts
from pipelinerl.utils import get_environment_jobs

logger = logging.getLogger(__name__)

_BASH_TOOL_NAME = "bash"
_SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
_SUBMIT_COMMAND = f"echo {_SUBMIT_MARKER}"
_FORMAT_ERROR_MESSAGE = (
    "FORMAT_ERROR: Call exactly one bash tool with a non-empty `command` string. "
    f"When the task is complete, call bash with command `{_SUBMIT_COMMAND}`. "
    "Do not write prose or use markdown."
)

SYSTEM_PROMPT = (
    "You are a terminal agent. You solve a task by running shell commands in a "
    "persistent bash session. Each turn, call exactly one bash tool with the "
    "command to execute. Do not write prose, markdown, or more than one tool call. "
    "You will then see the command output. When the task is fully done, call the "
    f"bash tool with command `{_SUBMIT_COMMAND}`."
)


def build_terminal_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": _BASH_TOOL_NAME,
                "description": "Execute one bash command in the persistent terminal session.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute.",
                        }
                    },
                    "required": ["command"],
                },
            },
        }
    ]


class TerminalMetrics(BaseMetrics):
    verifier_pass: bool = False
    passed_tests: int = 0
    total_tests: int = 0
    pass_fraction: float = 0.0
    build_ok: bool = True
    init_ok: bool = True
    overflow: bool = False
    disk_aborted: bool = False
    n_turns: int = 0
    n_llm_calls: int = 0
    n_total_llm_calls: int = 0
    n_format_errors: int = 0
    format_error_rate: float = 0.0
    format_errors_missing_tool: int = 0
    format_errors_multiple_tools: int = 0
    format_errors_wrong_tool: int = 0
    format_errors_bad_arguments: int = 0
    format_errors_empty_command: int = 0
    format_errors_prose_with_tool: int = 0
    max_format_retries_exceeded: bool = False


@dataclass(frozen=True)
class TerminalAction:
    command: str | None = None
    tool_call_id: str | None = None
    error: str | None = None


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


def _tool_function(tool_call) -> object | None:
    if isinstance(tool_call, dict):
        return tool_call.get("function")
    return getattr(tool_call, "function", None)


def _tool_call_id(tool_call) -> str:
    if isinstance(tool_call, dict):
        return str(tool_call.get("id") or "call_0")
    return str(getattr(tool_call, "id", None) or "call_0")


def _tool_name(tool_call) -> str | None:
    function = _tool_function(tool_call)
    if isinstance(function, dict):
        name = function.get("name")
    else:
        name = getattr(function, "name", None)
    return str(name) if name is not None else None


def _tool_arguments(tool_call):
    function = _tool_function(tool_call)
    if isinstance(function, dict):
        return function.get("arguments")
    return getattr(function, "arguments", None)


def _parse_tool_arguments(arguments) -> dict | None:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return None
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_bash_action(llm_call: LLMCall) -> TerminalAction:
    content = (llm_call.output.content or "").strip()
    tool_calls = list(getattr(llm_call.output, "tool_calls", None) or [])
    if not tool_calls:
        return TerminalAction(error="missing_tool")
    if len(tool_calls) != 1:
        return TerminalAction(error="multiple_tools")
    if content:
        return TerminalAction(error="prose_with_tool")

    tool_call = tool_calls[0]
    if _tool_name(tool_call) != _BASH_TOOL_NAME:
        return TerminalAction(error="wrong_tool")

    arguments = _parse_tool_arguments(_tool_arguments(tool_call))
    if arguments is None:
        return TerminalAction(error="bad_arguments")
    command = arguments.get("command")
    if not isinstance(command, str):
        return TerminalAction(error="bad_arguments")
    command = command.strip()
    if not command:
        return TerminalAction(error="empty_command")
    return TerminalAction(command=command, tool_call_id=_tool_call_id(tool_call))


def _is_submit_command(command: str) -> bool:
    return command.strip() == _SUBMIT_COMMAND


def _assistant_tool_message(action: TerminalAction) -> dict:
    assert action.command is not None
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": action.tool_call_id or "call_0",
                "type": "function",
                "function": {
                    "name": _BASH_TOOL_NAME,
                    "arguments": json.dumps({"command": action.command}),
                },
            }
        ],
    }


def _format_feedback(category: str) -> str:
    return f"{_FORMAT_ERROR_MESSAGE} Error: {category}."


def _new_format_counts() -> dict[str, int]:
    return {
        "missing_tool": 0,
        "multiple_tools": 0,
        "wrong_tool": 0,
        "bad_arguments": 0,
        "empty_command": 0,
        "prose_with_tool": 0,
    }


# Per-URL rate limit for health-check failure warnings. A dead env-fleet endpoint
# (e.g. a fleet pod that was platform-killed) is otherwise re-checked by every
# concurrent rollout every loop iteration, flooding the actor log with thousands of
# identical "Name or service not known" lines. Warn at most once per URL per window;
# the health check itself still runs every time, so a recovered fleet is rediscovered.
_HEALTH_WARN_WINDOW = 60.0
_last_health_warn: dict[str, float] = {}


async def _check_env_health(url: str, session: aiohttp.ClientSession) -> bool:
    try:
        async with session.get(f"{url}/health", timeout=5) as resp:
            return resp.status == 200
    except Exception as e:
        now = time.monotonic()
        if now - _last_health_warn.get(url, 0.0) >= _HEALTH_WARN_WINDOW:
            logger.warning("env health check failed for %s: %s (further warns rate-limited 60s)", url, e)
            _last_health_warn[url] = now
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

    n_actions = 0
    n_total_llm_calls = 0
    format_counts = _new_format_counts()
    max_format_retries = int(getattr(tcfg, "max_format_retries", 3))
    max_format_retries_exceeded = False
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem["task"]},
        ]
        tools = build_terminal_tools()
        llm_calls: List[LLMCall] = []
        disk_aborted = False
        submitted = False
        while n_actions < tcfg.max_turns:
            llm_call = await llm_async_generate(llm, Prompt(messages=messages, tools=tools), session)
            n_total_llm_calls += 1
            action = _extract_bash_action(llm_call)
            if action.error is not None:
                format_counts[action.error] += 1
                messages.append({"role": "assistant", "content": llm_call.output.content or ""})
                messages.append({"role": "user", "content": _format_feedback(action.error)})
                if sum(format_counts.values()) >= max_format_retries:
                    max_format_retries_exceeded = True
                    break
                continue

            messages.append(_assistant_tool_message(action))
            llm_calls.append(llm_call)
            n_actions += 1
            assert action.command is not None
            if _is_submit_command(action.command):
                submitted = True
                break

            obs = await _post(session, f"{env_url}/step", {"session_id": session_id, "command": action.command}, call_timeout)
            messages.append({"role": "tool", "tool_call_id": action.tool_call_id or "call_0", "content": obs["output"]})
            if obs.get("disk_exceeded"):
                disk_aborted = True
                break

        if max_format_retries_exceeded:
            verifier_pass = False
            passed_tests = 0
            total_tests = 0
        else:
            verifier = await _post(session, f"{env_url}/finish", {"session_id": session_id}, call_timeout)
            verifier_pass = bool(verifier["passed"])
            passed_tests = int(verifier.get("passed_tests", 0))
            total_tests = int(verifier.get("total_tests", 0))
            disk_aborted = disk_aborted or bool(verifier.get("disk_exceeded"))
    finally:
        try:
            await _post(session, f"{env_url}/close", {"session_id": session_id}, 30)
        except Exception:
            logger.warning("failed to close session %s on %s", session_id, env_url)

    # Graded reward (opt-in): map the pytest pass fraction onto [reward_fail,
    # reward_pass] so partially-correct rollouts give within-group variance and
    # fewer groups are zero-advantage filtered. Falls back to binary when disabled
    # or when no tests resolved (collection error / disk abort -> total_tests 0).
    pass_fraction = passed_tests / total_tests if total_tests > 0 else 0.0
    if max_format_retries_exceeded:
        reward = tcfg.reward_fail
    elif getattr(tcfg, "graded_reward", False):
        # No resolved tests (collection error / skipped-only / parse-empty) -> fail,
        # never the binary fallback, so a skipped-only pytest exit 0 can't score pass.
        reward = (
            tcfg.reward_fail + (tcfg.reward_pass - tcfg.reward_fail) * pass_fraction
            if total_tests > 0
            else tcfg.reward_fail
        )
    else:
        reward = tcfg.reward_pass if verifier_pass else tcfg.reward_fail
    training_texts = make_training_texts_from_llm_calls(llm, llm_calls, reward=reward)
    summary = summarize_training_texts(training_texts)

    n_format_errors = sum(format_counts.values())
    metrics = TerminalMetrics(
        reward=reward,
        success=verifier_pass,
        no_error=not max_format_retries_exceeded,
        no_answer=len(llm_calls) == 0,
        verifier_pass=verifier_pass,
        passed_tests=passed_tests,
        total_tests=total_tests,
        pass_fraction=pass_fraction,
        overflow=summary.overflow,
        disk_aborted=disk_aborted,
        n_turns=n_actions,
        n_llm_calls=len(llm_calls),
        n_total_llm_calls=n_total_llm_calls,
        n_format_errors=n_format_errors,
        format_error_rate=n_format_errors / max(n_total_llm_calls, 1),
        format_errors_missing_tool=format_counts["missing_tool"],
        format_errors_multiple_tools=format_counts["multiple_tools"],
        format_errors_wrong_tool=format_counts["wrong_tool"],
        format_errors_bad_arguments=format_counts["bad_arguments"],
        format_errors_empty_command=format_counts["empty_command"],
        format_errors_prose_with_tool=format_counts["prose_with_tool"],
        max_format_retries_exceeded=max_format_retries_exceeded,
    )
    return RolloutResult(
        training_texts=training_texts,
        metrics=metrics,
        latency=time.time() - start_time,
        dataset_name=problem.get("dataset"),
        domain="terminal",
    )
