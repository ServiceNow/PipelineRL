import asyncio
import time
from types import SimpleNamespace

from pipelinerl.domains.terminal.environment_server import TerminalEnvironmentServer
from pipelinerl.domains.terminal.rollouts import (
    _SUBMIT_COMMAND,
    _assistant_tool_message,
    _extract_bash_action,
    _is_submit_command,
)
from pipelinerl.llm import LLMCall, LLMOutput, Prompt


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


def test_rejects_prose_around_tool_call():
    action = _extract_bash_action(_llm_call(content="I will inspect files.", tool_calls=[_tool_call()]))

    assert action.error == "prose_with_tool"


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
