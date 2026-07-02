import asyncio
import time
from types import SimpleNamespace

from pipelinerl.domains.terminal.environment_server import TerminalEnvironmentServer
from pipelinerl.domains.terminal.rollouts import (
    _SUBMIT_COMMAND,
    _assistant_tool_message,
    _execute_rollout,
    _extract_bash_action,
    _is_submit_command,
)
from pipelinerl.llm import LLMCall, LLMOutput, Prompt
from pipelinerl.rollouts import TrainingText


def _tool_call(name="bash", arguments=None, call_id="call_0"):
    if arguments is None:
        arguments = {"command": "ls -la"}
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _llm_call(content="", tool_calls=None):
    output = LLMOutput(content=content)
    if tool_calls is not None:
        output.tool_calls = tool_calls
    return LLMCall(prompt=Prompt.from_user_message("task"), output=output, cached=False)


def test_extracts_single_bash_tool_call_from_json_arguments():
    action = _extract_bash_action(_llm_call(tool_calls=[_tool_call(arguments='{"command": "pwd"}')]))

    assert action.error is None
    assert action.command == "pwd"
    assert action.tool_call_id == "call_0"
    assert not action.has_prose


def test_accepts_prose_around_single_bash_tool_call():
    action = _extract_bash_action(_llm_call(content="I will inspect files.", tool_calls=[_tool_call()]))

    assert action.error is None
    assert action.command == "ls -la"
    assert action.has_prose


def test_rejects_missing_multiple_wrong_and_empty_tools():
    assert _extract_bash_action(_llm_call(content="hello")).error == "missing_tool"
    assert _extract_bash_action(_llm_call(tool_calls=[_tool_call(), _tool_call(call_id="call_1")])).error == "multiple_tools"
    assert _extract_bash_action(_llm_call(tool_calls=[_tool_call(name="python")])).error == "wrong_tool"
    assert _extract_bash_action(_llm_call(tool_calls=[_tool_call(arguments={"command": ""})])).error == "empty_command"


def test_submit_command_is_exact_and_never_stepped():
    assert _is_submit_command(_SUBMIT_COMMAND)
    assert not _is_submit_command(f"{_SUBMIT_COMMAND} && echo extra")


def test_assistant_tool_message_uses_string_arguments_for_vllm_wire():
    action = _extract_bash_action(_llm_call(tool_calls=[_tool_call(arguments='{"command": "ls"}', call_id="call_7")]))
    message = _assistant_tool_message(action)

    tool_call = message["tool_calls"][0]
    assert tool_call["id"] == "call_7"
    assert tool_call["function"]["name"] == "bash"
    assert tool_call["function"]["arguments"] == '{"command": "ls"}'


def _terminal_cfg(**overrides):
    values = {
        "max_turns": 1,
        "env_call_timeout": 1,
        "env_start_timeout": 1,
        "max_format_retries": 3,
        "reward_pass": 1.0,
        "reward_fail": -1.0,
        "graded_reward": False,
        "format_error_reward": None,
        "no_submit_penalty": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(terminal=SimpleNamespace(**values))


def _patch_rollout_fakes(monkeypatch, llm_calls, *, verifier_pass=True, step_response=None):
    pending_calls = list(llm_calls)

    async def fake_generate(llm, prompt, session):
        assert pending_calls
        return pending_calls.pop(0)

    async def fake_post(session, url, payload, timeout):
        if url.endswith("/start_task"):
            return {"session_id": "session-1", "started": True, "init_ok": True, "build_ok": True}
        if url.endswith("/step"):
            return step_response or {"output": "ok", "disk_exceeded": False, "timeout_aborted": False}
        if url.endswith("/finish"):
            return {
                "passed": verifier_pass,
                "passed_tests": int(verifier_pass),
                "total_tests": 1,
                "disk_exceeded": False,
                "timeout_aborted": False,
            }
        if url.endswith("/close"):
            return {"status": "ok"}
        raise AssertionError(url)

    def fake_make_training_text(llm, llm_call):
        return TrainingText(text=llm_call.output.content or "tool", n_predicted=1)

    def fake_make_training_texts(llm, llm_calls, reward=None):
        return [
            TrainingText(text=llm_call.output.content or "tool", n_predicted=1, reward=reward)
            for llm_call in llm_calls
        ]

    monkeypatch.setattr("pipelinerl.domains.terminal.rollouts.llm_async_generate", fake_generate)
    monkeypatch.setattr("pipelinerl.domains.terminal.rollouts._post", fake_post)
    monkeypatch.setattr("pipelinerl.domains.terminal.rollouts.make_training_text", fake_make_training_text)
    monkeypatch.setattr(
        "pipelinerl.domains.terminal.rollouts.make_training_texts_from_llm_calls",
        fake_make_training_texts,
    )


def test_format_error_reward_retains_error_turn_in_chronological_order(monkeypatch):
    llm_calls = [
        _llm_call(content="bad format"),
        _llm_call(content="submit", tool_calls=[_tool_call(arguments={"command": _SUBMIT_COMMAND})]),
    ]
    _patch_rollout_fakes(monkeypatch, llm_calls)

    result = asyncio.run(
        _execute_rollout(
            _terminal_cfg(format_error_reward=-0.2),
            object(),
            {"task": "fix it", "task_id": "task-1"},
            object(),
            time.time(),
            "http://env",
        )
    )

    assert [text.text for text in result.training_texts] == ["bad format", "submit"]
    assert [text.reward for text in result.training_texts] == [-0.2, 1.0]
    assert result.metrics.n_format_errors == 1
    assert result.metrics.submitted


def test_format_error_reward_respects_max_retry_failure(monkeypatch):
    llm_calls = [
        _llm_call(content="bad format 1"),
        _llm_call(content="bad format 2"),
    ]
    _patch_rollout_fakes(monkeypatch, llm_calls)

    result = asyncio.run(
        _execute_rollout(
            _terminal_cfg(format_error_reward=-0.2, max_format_retries=2),
            object(),
            {"task": "fix it", "task_id": "task-1"},
            object(),
            time.time(),
            "http://env",
        )
    )

    assert [text.text for text in result.training_texts] == ["bad format 1", "bad format 2"]
    assert [text.reward for text in result.training_texts] == [-1.0, -1.0]
    assert result.metrics.max_format_retries_exceeded


def test_null_format_error_reward_drops_error_turn(monkeypatch):
    llm_calls = [
        _llm_call(content="bad format"),
        _llm_call(content="submit", tool_calls=[_tool_call(arguments={"command": _SUBMIT_COMMAND})]),
    ]
    _patch_rollout_fakes(monkeypatch, llm_calls)

    result = asyncio.run(
        _execute_rollout(
            _terminal_cfg(),
            object(),
            {"task": "fix it", "task_id": "task-1"},
            object(),
            time.time(),
            "http://env",
        )
    )

    assert [text.text for text in result.training_texts] == ["submit"]
    assert [text.reward for text in result.training_texts] == [1.0]
    assert result.metrics.n_format_errors == 1
    assert result.metrics.submitted


def test_no_submit_penalty_applies_only_to_clean_max_turn_exit(monkeypatch):
    _patch_rollout_fakes(
        monkeypatch,
        [_llm_call(content="inspect", tool_calls=[_tool_call(arguments={"command": "ls"})])],
    )

    result = asyncio.run(
        _execute_rollout(
            _terminal_cfg(no_submit_penalty=0.4),
            object(),
            {"task": "fix it", "task_id": "task-1"},
            object(),
            time.time(),
            "http://env",
        )
    )

    assert result.metrics.reward == 0.6
    assert [text.reward for text in result.training_texts] == [0.6]
    assert not result.metrics.submitted

    _patch_rollout_fakes(
        monkeypatch,
        [_llm_call(content="submit", tool_calls=[_tool_call(arguments={"command": _SUBMIT_COMMAND})])],
    )

    submitted_result = asyncio.run(
        _execute_rollout(
            _terminal_cfg(no_submit_penalty=0.4),
            object(),
            {"task": "fix it", "task_id": "task-1"},
            object(),
            time.time(),
            "http://env",
        )
    )

    assert submitted_result.metrics.reward == 1.0
    assert [text.reward for text in submitted_result.training_texts] == [1.0]
    assert submitted_result.metrics.submitted


class DummySession:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_reaper_removes_expired_sessions_and_closes_in_background():
    async def run_case():
        server = TerminalEnvironmentServer(
            bases_dir="/tmp",
            n_envs=1,
            session_ttl_seconds=1.0,
            session_reap_interval_seconds=60.0,
        )
        expired = DummySession()
        fresh = DummySession()
        server._sessions["expired"] = expired
        server._sessions["fresh"] = fresh
        server._session_last_activity["expired"] = time.monotonic() - 2.0
        server._session_last_activity["fresh"] = time.monotonic()

        await server._reap_expired_sessions()
        if server._bg_tasks:
            await asyncio.gather(*list(server._bg_tasks))
        server._executor.shutdown(wait=True)

        assert "expired" not in server._sessions
        assert "expired" not in server._session_last_activity
        assert expired.closed
        assert "fresh" in server._sessions
        assert not fresh.closed

    asyncio.run(run_case())
