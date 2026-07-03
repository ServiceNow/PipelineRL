"""Rollout for the TMax-style terminal-agent domain.

A multi-turn bash agent (mini-SWE-agent shape: one ``bash`` tool, persistent
shell, submit marker, output truncation) drives a proot sandbox hosted on a
remote env-server job, then a pytest verifier scores the final state. By
default reward is outcome-only and broadcast to every valid action turn, matching
the TMax recipe; opt-in event rewards can retain malformed turns with
per-turn rewards. PipelineRL's LOO group advantage plus zero-advantage filtering
supply the rest.

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

from pipelinerl.async_llm import (
    RetryableAbortedCompletionError,
    _normalize_tool_call_messages,
    llm_async_generate,
    make_training_text,
    make_training_texts_from_llm_calls,
)
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
    "Brief reasoning is allowed, but do not use markdown."
)
_FINISH_STDOUT_TAIL_CHARS = 2000


def _terminal_audit_base(problem: dict) -> dict:
    return {
        "task_id": problem.get("task_id"),
        "domain": problem.get("tmax_domain") or problem.get("domain") or "terminal",
        "complexity": problem.get("task_complexity", problem.get("complexity")),
        "group_id": None,
        "rollout_index": None,
        "reward": None,
        "verifier_pass": False,
        "passed_tests": 0,
        "total_tests": 0,
        "pass_fraction": 0.0,
        "abort_kind": None,
        "abort_phase": None,
        "build_ok": True,
        "init_ok": True,
        "submitted": False,
        "format_retry_exceeded": False,
        "context_exhausted": False,
        "contamination_result": None,
        "finish_stdout_tail": "",
        "dropped": False,
        "drop_reason": None,
    }


def _contamination_audit(value) -> dict | None:
    if not isinstance(value, dict):
        return None
    return {
        "sampled": bool(value.get("sampled", False)),
        "count": int(value.get("count", 0)),
    }


def _finish_stdout_tail(output: str) -> str:
    return output[-_FINISH_STDOUT_TAIL_CHARS:]


SYSTEM_PROMPT = (
    "You are a terminal agent. You solve a task by running shell commands in a "
    "persistent bash session. Each turn, call exactly one bash tool with the "
    "command to execute. You may include brief reasoning, but do not use markdown "
    "or more than one tool call. "
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
    timeout_aborted: bool = False
    rss_aborted: bool = False
    submitted: bool = False
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
    tool_calls_with_prose: int = 0
    tool_call_prose_rate: float = 0.0
    max_format_retries_exceeded: bool = False
    format_error_texts_dropped: int = 0
    context_exhausted: bool = False


@dataclass(frozen=True)
class TerminalAction:
    command: str | None = None
    tool_call_id: str | None = None
    error: str | None = None
    has_prose: bool = False


def _failed_result(
    problem: dict,
    start_time: float,
    metrics: TerminalMetrics,
    audit: dict | None = None,
) -> RolloutResult:
    audit_data = _terminal_audit_base(problem)
    audit_data.update(
        {
            "reward": metrics.reward,
            "verifier_pass": metrics.verifier_pass,
            "passed_tests": metrics.passed_tests,
            "total_tests": metrics.total_tests,
            "pass_fraction": metrics.pass_fraction,
            "build_ok": metrics.build_ok,
            "init_ok": metrics.init_ok,
            "submitted": metrics.submitted,
            "format_retry_exceeded": metrics.max_format_retries_exceeded,
            "context_exhausted": metrics.context_exhausted,
            "dropped": True,
            "drop_reason": "failed_result",
        }
    )
    if audit:
        audit_data.update(audit)
    return RolloutResult(
        training_texts=[],
        metrics=metrics,
        latency=time.time() - start_time,
        dataset_name=problem.get("dataset"),
        domain="terminal",
        audit=audit_data,
    )


class EnvironmentCapacityError(RuntimeError):
    pass


class EnvironmentConnectionError(RuntimeError):
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
    return TerminalAction(command=command, tool_call_id=_tool_call_id(tool_call), has_prose=bool(content))


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
    }


# Per-URL rate limit for /start_task connection warnings. A dead env-fleet
# endpoint can be hit by every concurrent rollout every loop iteration.
_START_WARN_WINDOW = 60.0
_last_start_warn: dict[str, float] = {}


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
            try:
                return await asyncio.wait_for(
                    _execute_rollout(cfg, llm, problem, session, start_time, url),
                    timeout=max(1.0, deadline - time.time()),
                )
            except EnvironmentConnectionError:
                continue
            except EnvironmentCapacityError:
                saw_capacity = True
                saw_healthy = True
                continue
            except asyncio.TimeoutError:
                saw_healthy = True
                logger.warning("rollout timed out for %s on %s, trying next server", problem.get("task_id"), url)
                continue
            except Exception:
                saw_healthy = True
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
    audit = _terminal_audit_base(problem)
    # /start_task triggers the one-time per-task rootfs build, which reads the
    # base over NFS and can take several minutes. Once the server returns a
    # session_id, the finally block must close it even on an early failed rollout.
    start_timeout = getattr(tcfg, "env_start_timeout", 900)

    session_id = None
    try:
        try:
            start_timeout_cfg = aiohttp.ClientTimeout(total=start_timeout, connect=10)
            start = await _post(session, f"{env_url}/start_task", {"task_data": problem}, start_timeout_cfg)
        except aiohttp.ClientConnectionError as e:
            now = time.monotonic()
            if now - _last_start_warn.get(env_url, 0.0) >= _START_WARN_WINDOW:
                logger.warning("env start_task failed for %s: %s (further warns rate-limited 60s)", env_url, e)
                _last_start_warn[env_url] = now
            raise EnvironmentConnectionError(str(e)) from e
        session_id = start.get("session_id")
        if not session_id or not start.get("started") or not start.get("init_ok"):
            logger.warning("task %s not runnable (start=%s), dropping", problem.get("task_id"), start)
            audit.update(
                {
                    "reward": tcfg.reward_fail,
                    "build_ok": start.get("build_ok", False),
                    "init_ok": start.get("init_ok", False),
                    "dropped": True,
                    "drop_reason": "not_runnable",
                }
            )
            return _failed_result(problem, start_time, TerminalMetrics(
                reward=tcfg.reward_fail, success=False, no_error=False, no_answer=True,
                build_ok=start.get("build_ok", False), init_ok=start.get("init_ok", False)), audit=audit)

        n_actions = 0
        n_total_llm_calls = 0
        format_counts = _new_format_counts()
        tool_calls_with_prose = 0
        max_format_retries = int(getattr(tcfg, "max_format_retries", 3))
        format_error_reward = getattr(tcfg, "format_error_reward", None)
        max_format_retries_exceeded = False
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem["task"]},
        ]
        tools = build_terminal_tools()
        llm_calls: List[LLMCall] = []
        llm_call_events: list[tuple[LLMCall, bool]] = []
        disk_aborted = False
        timeout_aborted = False
        rss_aborted = False
        abort_kind = None
        abort_phase = None
        finish_output = ""
        contamination_result = None
        verifier_ran = False
        submitted = False
        context_exhausted = False
        # Context budget: vLLM rejects the request outright (400) when prompt +
        # max_tokens exceeds max_model_len, which failed the whole rollout once the
        # multi-turn history grew past that line. Predict it with the same tokenizer
        # and template the server uses and end the rollout cleanly instead.
        vllm_kwargs = getattr(getattr(cfg, "vllm_config", None), "vllm_kwargs", None)
        max_model_len = int(vllm_kwargs.get("max_model_len") or 0) if vllm_kwargs is not None else 0
        max_new_tokens = int((getattr(llm, "parameters", None) or {}).get("max_tokens") or 0)
        context_margin = 64
        if max_model_len and max_new_tokens:
            # The tokenizer is lazily loaded by llm_async_generate; the precheck
            # runs before the first generation and must load it itself.
            llm.load_tokenizer()
        while n_actions < tcfg.max_turns:
            if max_model_len and max_new_tokens:
                chat_kwargs = dict(getattr(llm, "chat_template_kwargs", None) or {})
                prompt_ids = llm.tokenizer.apply_chat_template(
                    _normalize_tool_call_messages(messages),
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                    **chat_kwargs,
                )
                if len(prompt_ids) + max_new_tokens + context_margin > max_model_len:
                    context_exhausted = True
                    break
            llm_call = await llm_async_generate(llm, Prompt(messages=messages, tools=tools), session)
            n_total_llm_calls += 1
            action = _extract_bash_action(llm_call)
            if action.error is not None:
                format_counts[action.error] += 1
                if format_error_reward is not None:
                    llm_call_events.append((llm_call, True))
                messages.append({"role": "assistant", "content": llm_call.output.content or ""})
                messages.append({"role": "user", "content": _format_feedback(action.error)})
                if sum(format_counts.values()) >= max_format_retries:
                    max_format_retries_exceeded = True
                    break
                continue

            if action.has_prose:
                tool_calls_with_prose += 1

            messages.append(_assistant_tool_message(action))
            if format_error_reward is not None:
                llm_call_events.append((llm_call, False))
            llm_calls.append(llm_call)
            n_actions += 1
            assert action.command is not None
            if _is_submit_command(action.command):
                submitted = True
                break

            obs = await _post(session, f"{env_url}/step", {"session_id": session_id, "command": action.command}, call_timeout)
            messages.append({"role": "tool", "tool_call_id": action.tool_call_id or "call_0", "content": obs["output"]})
            step_abort_kind = obs.get("abort_kind")
            if step_abort_kind:
                abort_kind = step_abort_kind
                abort_phase = "step"
                disk_aborted = disk_aborted or step_abort_kind == "disk"
                timeout_aborted = timeout_aborted or step_abort_kind == "timeout"
                rss_aborted = rss_aborted or step_abort_kind == "rss"
                break

        if max_format_retries_exceeded:
            verifier_pass = False
            passed_tests = 0
            total_tests = 0
        else:
            verifier_ran = True
            verifier = await _post(session, f"{env_url}/finish", {"session_id": session_id}, call_timeout)
            verifier_pass = bool(verifier["passed"])
            passed_tests = int(verifier.get("passed_tests", 0))
            total_tests = int(verifier.get("total_tests", 0))
            finish_output = str(verifier.get("output", ""))
            contamination_result = _contamination_audit(verifier.get("contamination_result"))
            finish_abort_kind = verifier.get("abort_kind")
            if finish_abort_kind:
                abort_kind = finish_abort_kind
                abort_phase = "finish"
            disk_aborted = disk_aborted or finish_abort_kind == "disk"
            timeout_aborted = timeout_aborted or finish_abort_kind == "timeout"
            rss_aborted = rss_aborted or finish_abort_kind == "rss"
    finally:
        if session_id:
            try:
                await _post(session, f"{env_url}/close", {"session_id": session_id}, 30)
            except Exception:
                logger.warning("failed to close session %s on %s", session_id, env_url)

    # Graded reward (opt-in): map the pytest pass fraction onto [reward_fail,
    # reward_pass] so partially-correct rollouts give within-group variance and
    # fewer groups are zero-advantage filtered. Falls back to binary when disabled
    # or when no tests resolved (collection error / abort -> total_tests 0).
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

    no_submit_penalty = float(getattr(tcfg, "no_submit_penalty", 0.0))
    if (
        no_submit_penalty > 0
        and (n_actions >= tcfg.max_turns or context_exhausted)
        and not submitted
        and not disk_aborted
        and not timeout_aborted
        and not rss_aborted
        and not max_format_retries_exceeded
    ):
        reward = max(tcfg.reward_fail, reward - no_submit_penalty)

    drop_reason = None
    if verifier_ran and abort_phase == "finish":
        drop_reason = "finish_abort"
    elif verifier_ran and total_tests == 0:
        drop_reason = "no_tests_resolved"

    audit.update(
        {
            "reward": reward,
            "verifier_pass": verifier_pass,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "pass_fraction": pass_fraction,
            "abort_kind": abort_kind,
            "abort_phase": abort_phase,
            "submitted": submitted,
            "format_retry_exceeded": max_format_retries_exceeded,
            "context_exhausted": context_exhausted,
            "contamination_result": contamination_result,
            "finish_stdout_tail": _finish_stdout_tail(finish_output),
            "dropped": drop_reason is not None,
            "drop_reason": drop_reason,
        }
    )

    format_error_texts_dropped = 0
    if drop_reason is not None:
        logger.info(
            "dropping terminal rollout %s from training: %s",
            problem.get("task_id"), drop_reason,
        )
        training_texts = []
    elif format_error_reward is None:
        training_texts = make_training_texts_from_llm_calls(llm, llm_calls, reward=reward)
    else:
        training_texts = []
        for llm_call, is_format_error in llm_call_events:
            try:
                training_text = make_training_text(llm, llm_call)
            except RetryableAbortedCompletionError:
                raise
            except Exception:
                if not is_format_error:
                    raise
                # A malformed tool call (e.g. unparseable `arguments`) can defeat the
                # chat-template reconstruction (jinja `|items` on a non-mapping). Drop
                # just this penalty turn instead of failing the whole rollout.
                format_error_texts_dropped += 1
                logger.warning(
                    "dropping unreconstructable format-error turn for %s",
                    problem.get("task_id"), exc_info=True,
                )
                continue
            training_text.reward = reward if max_format_retries_exceeded or not is_format_error else format_error_reward
            training_texts.append(training_text)
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
        timeout_aborted=timeout_aborted,
        rss_aborted=rss_aborted,
        submitted=submitted,
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
        tool_calls_with_prose=tool_calls_with_prose,
        tool_call_prose_rate=tool_calls_with_prose / max(len(llm_calls), 1),
        max_format_retries_exceeded=max_format_retries_exceeded,
        format_error_texts_dropped=format_error_texts_dropped,
        context_exhausted=context_exhausted,
    )
    return RolloutResult(
        training_texts=training_texts,
        metrics=metrics,
        latency=time.time() - start_time,
        dataset_name=problem.get("dataset"),
        domain="terminal",
        audit=audit,
    )
