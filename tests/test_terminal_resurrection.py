import asyncio
import time
from types import SimpleNamespace

import pytest

from pipelinerl.domains.terminal import rollouts
from pipelinerl.domains.terminal.rollouts import (
    _SUBMIT_COMMAND,
    _execute_rollout,
    generate_terminal_rollout,
    RolloutResult,
    TerminalMetrics,
)
from pipelinerl.llm import LLMCall, LLMOutput, Prompt
from pipelinerl.rollouts import TrainingText


def _tool_call(command, call_id="call_0"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name="bash", arguments={"command": command}),
    )


def _llm_call(command, content="tool"):
    output = LLMOutput(content=content)
    output.tool_calls = [_tool_call(command)]
    return LLMCall(prompt=Prompt.from_user_message("task"), output=output, cached=False)


def _terminal_cfg(**overrides):
    values = {
        "max_turns": 4,
        "env_call_timeout": 1,
        "env_start_timeout": 1,
        "rollout_timeout": 5,
        "capacity_retry_sleep": 0,
        "max_format_retries": 3,
        "reward_pass": 1.0,
        "reward_fail": -1.0,
        "graded_reward": False,
        "format_error_reward": None,
        "no_submit_penalty": 0.0,
        "session_resurrection": False,
        "max_resurrections": 1,
    }
    values.update(overrides)
    return SimpleNamespace(terminal=SimpleNamespace(**values))


def _patch_llm_and_training(monkeypatch, llm_calls):
    pending = list(llm_calls)

    async def fake_generate(llm, prompt, session):
        assert pending
        return pending.pop(0)

    def fake_make_training_texts(llm, calls, reward=None):
        return [TrainingText(text=call.output.content or "tool", n_predicted=1, reward=reward) for call in calls]

    monkeypatch.setattr(rollouts, "llm_async_generate", fake_generate)
    monkeypatch.setattr(rollouts, "make_training_texts_from_llm_calls", fake_make_training_texts)


def _finish(passed=True):
    return {
        "passed": passed,
        "passed_tests": int(passed),
        "total_tests": 1,
        "abort_kind": None,
        "output": "pytest output",
        "contamination_result": {"sampled": False, "count": 0},
    }


def test_step_connection_error_resurrects_replays_prefix_and_retries_failed_command_once(monkeypatch):
    _patch_llm_and_training(
        monkeypatch,
        [_llm_call("first"), _llm_call("second"), _llm_call(_SUBMIT_COMMAND)],
    )
    calls = []

    async def fake_post(session, url, payload, timeout):
        calls.append((url, dict(payload)))
        if url == "http://old/start_task":
            return {"session_id": "old-session", "started": True, "init_ok": True, "build_ok": True}
        if url == "http://old/step" and payload["command"] == "first":
            return {"output": "first ok", "success": True, "abort_kind": None}
        if url == "http://old/step" and payload["command"] == "second":
            raise rollouts.aiohttp.ClientConnectionError("old server died")
        if url == "http://new/start_task":
            return {"session_id": "new-session", "started": True, "init_ok": True, "build_ok": True}
        if url == "http://new/replay":
            assert payload == {"session_id": "new-session", "commands": ["first"]}
            return {"n_executed": 1, "successes": [True], "abort_kind": None, "last_output": "first ok"}
        if url == "http://new/step" and payload["command"] == "second":
            return {"output": "second ok", "success": True, "abort_kind": None}
        if url == "http://new/finish":
            return _finish(True)
        if url == "http://new/close":
            return {"status": "ok"}
        raise AssertionError((url, payload))

    monkeypatch.setattr(rollouts, "_post", fake_post)

    result = asyncio.run(
        _execute_rollout(
            _terminal_cfg(session_resurrection=True),
            object(),
            {"task": "fix it", "task_id": "task-1"},
            object(),
            time.time(),
            "http://old",
            candidate_env_urls=["http://old", "http://new"],
        )
    )

    assert result.metrics.success
    assert result.audit["resurrected"]
    assert result.audit["resurrection_turn"] == 1
    assert not result.audit["resurrection_divergence"]
    assert result.audit["commands"] == ["first", "second", _SUBMIT_COMMAND]
    assert result.audit["command_errors"] == [False, False, False]
    assert result.audit["n_command_errors"] == 0
    assert [(url, payload.get("command")) for url, payload in calls if url.endswith("/step")] == [
        ("http://old/step", "first"),
        ("http://old/step", "second"),
        ("http://new/step", "second"),
    ]
    assert [url for url, _ in calls if url.endswith("/close")] == ["http://new/close"]


def test_finish_connection_error_resurrects_and_retries_finish(monkeypatch):
    _patch_llm_and_training(monkeypatch, [_llm_call("prepare"), _llm_call(_SUBMIT_COMMAND)])
    calls = []

    async def fake_post(session, url, payload, timeout):
        calls.append((url, dict(payload)))
        if url == "http://old/start_task":
            return {"session_id": "old-session", "started": True, "init_ok": True, "build_ok": True}
        if url == "http://old/step":
            return {"output": "prepared", "success": True, "abort_kind": None}
        if url == "http://old/finish":
            raise rollouts.aiohttp.ClientConnectionError("finish lost")
        if url == "http://new/start_task":
            return {"session_id": "new-session", "started": True, "init_ok": True, "build_ok": True}
        if url == "http://new/replay":
            assert payload["commands"] == ["prepare", _SUBMIT_COMMAND]
            return {"n_executed": 2, "successes": [True, True], "abort_kind": None, "last_output": "submit"}
        if url == "http://new/finish":
            return _finish(True)
        if url == "http://new/close":
            return {"status": "ok"}
        raise AssertionError((url, payload))

    monkeypatch.setattr(rollouts, "_post", fake_post)

    result = asyncio.run(
        _execute_rollout(
            _terminal_cfg(session_resurrection=True),
            object(),
            {"task": "fix it", "task_id": "task-1"},
            object(),
            time.time(),
            "http://old",
            candidate_env_urls=["http://old", "http://new"],
        )
    )

    assert result.audit["resurrected"]
    assert result.audit["resurrection_turn"] == 2
    assert result.metrics.success
    assert [url for url, _ in calls if url.endswith("/finish")] == ["http://old/finish", "http://new/finish"]


def test_resurrection_flag_off_preserves_connection_error(monkeypatch):
    _patch_llm_and_training(monkeypatch, [_llm_call("first"), _llm_call("second")])
    calls = []

    async def fake_post(session, url, payload, timeout):
        calls.append((url, dict(payload)))
        if url == "http://old/start_task":
            return {"session_id": "old-session", "started": True, "init_ok": True, "build_ok": True}
        if url == "http://old/step" and payload["command"] == "first":
            return {"output": "first ok", "success": True, "abort_kind": None}
        if url == "http://old/step" and payload["command"] == "second":
            raise rollouts.aiohttp.ClientConnectionError("dead")
        if url == "http://old/close":
            return {"status": "ok"}
        raise AssertionError((url, payload))

    monkeypatch.setattr(rollouts, "_post", fake_post)

    with pytest.raises(rollouts.aiohttp.ClientConnectionError):
        asyncio.run(
            _execute_rollout(
                _terminal_cfg(session_resurrection=False),
                object(),
                {"task": "fix it", "task_id": "task-1"},
                object(),
                time.time(),
                "http://old",
                candidate_env_urls=["http://old", "http://new"],
            )
        )

    assert not any(url.startswith("http://new/") for url, _ in calls)
    assert [url for url, _ in calls if url.endswith("/close")] == ["http://old/close"]


def test_resurrection_budget_exhausted_propagates_connection_error(monkeypatch):
    _patch_llm_and_training(monkeypatch, [_llm_call("first"), _llm_call("second")])
    calls = []

    async def fake_post(session, url, payload, timeout):
        calls.append((url, dict(payload)))
        if url == "http://old/start_task":
            return {"session_id": "old-session", "started": True, "init_ok": True, "build_ok": True}
        if url == "http://old/step" and payload["command"] == "first":
            return {"output": "first ok", "success": True, "abort_kind": None}
        if url == "http://old/step" and payload["command"] == "second":
            raise rollouts.aiohttp.ClientConnectionError("dead")
        if url == "http://old/close":
            return {"status": "ok"}
        raise AssertionError((url, payload))

    monkeypatch.setattr(rollouts, "_post", fake_post)

    with pytest.raises(rollouts.aiohttp.ClientConnectionError):
        asyncio.run(
            _execute_rollout(
                _terminal_cfg(session_resurrection=True, max_resurrections=0),
                object(),
                {"task": "fix it", "task_id": "task-1"},
                object(),
                time.time(),
                "http://old",
                candidate_env_urls=["http://old", "http://new"],
            )
        )

    assert not any(url.startswith("http://new/") for url, _ in calls)


def test_resurrection_divergence_is_audit_only(monkeypatch):
    _patch_llm_and_training(monkeypatch, [_llm_call("first"), _llm_call("second"), _llm_call(_SUBMIT_COMMAND)])

    async def fake_post(session, url, payload, timeout):
        if url == "http://old/start_task":
            return {"session_id": "old-session", "started": True, "init_ok": True, "build_ok": True}
        if url == "http://old/step" and payload["command"] == "first":
            return {"output": "first ok", "success": True, "abort_kind": None}
        if url == "http://old/step" and payload["command"] == "second":
            raise rollouts.aiohttp.ClientConnectionError("dead")
        if url == "http://new/start_task":
            return {"session_id": "new-session", "started": True, "init_ok": True, "build_ok": True}
        if url == "http://new/replay":
            return {"n_executed": 1, "successes": [False], "abort_kind": None, "last_output": "different"}
        if url == "http://new/step":
            return {"output": "second ok", "success": True, "abort_kind": None}
        if url == "http://new/finish":
            return _finish(True)
        if url == "http://new/close":
            return {"status": "ok"}
        raise AssertionError((url, payload))

    monkeypatch.setattr(rollouts, "_post", fake_post)

    result = asyncio.run(
        _execute_rollout(
            _terminal_cfg(session_resurrection=True),
            object(),
            {"task": "fix it", "task_id": "task-1"},
            object(),
            time.time(),
            "http://old",
            candidate_env_urls=["http://old", "http://new"],
        )
    )

    assert result.audit["resurrected"]
    assert result.audit["resurrection_divergence"]
    assert not result.audit["dropped"]
    assert result.training_texts


def test_replay_non_200_fails_resurrection_without_trying_another_replay(monkeypatch):
    _patch_llm_and_training(monkeypatch, [_llm_call("first"), _llm_call("second")])
    calls = []
    monkeypatch.setattr(rollouts.random, "shuffle", lambda urls: None)

    async def fake_post(session, url, payload, timeout):
        calls.append((url, dict(payload)))
        if url == "http://old/start_task":
            return {"session_id": "old-session", "started": True, "init_ok": True, "build_ok": True}
        if url == "http://old/step" and payload["command"] == "first":
            return {"output": "first ok", "success": True, "abort_kind": None}
        if url == "http://old/step" and payload["command"] == "second":
            raise rollouts.aiohttp.ClientConnectionError("dead")
        if url == "http://new/start_task":
            return {"session_id": "new-session", "started": True, "init_ok": True, "build_ok": True}
        if url == "http://new/replay":
            raise RuntimeError("HTTP 500: session closed mid-replay")
        if url == "http://new/close":
            return {"status": "ok"}
        if url == "http://old/close":
            return {"status": "ok"}
        if url.startswith("http://third/"):
            raise AssertionError("should not try another server after replay failure")
        raise AssertionError((url, payload))

    monkeypatch.setattr(rollouts, "_post", fake_post)

    with pytest.raises(rollouts.aiohttp.ClientConnectionError):
        asyncio.run(
            _execute_rollout(
                _terminal_cfg(session_resurrection=True),
                object(),
                {"task": "fix it", "task_id": "task-1"},
                object(),
                time.time(),
                "http://old",
                candidate_env_urls=["http://old", "http://new", "http://third"],
            )
        )

    assert [url for url, _ in calls if url.endswith("/replay")] == ["http://new/replay"]
    assert [url for url, _ in calls if url.endswith("/close")] == ["http://new/close", "http://old/close"]


def test_generate_rollout_passes_candidate_urls_only_when_resurrection_enabled(monkeypatch):
    jobs = [
        SimpleNamespace(hostname="env-a", port=7777),
        SimpleNamespace(hostname="env-b", port=7778),
    ]
    seen = []

    async def fake_execute(
        cfg,
        llm,
        problem,
        session,
        start_time,
        env_url,
        model_version_provider=None,
        candidate_env_urls=None,
    ):
        seen.append((env_url, candidate_env_urls))
        return RolloutResult(
            training_texts=[],
            metrics=TerminalMetrics(reward=1.0, success=True, no_error=True, no_answer=False),
            latency=0.0,
            dataset_name=None,
            domain="terminal",
        )

    monkeypatch.setattr(rollouts, "get_environment_jobs", lambda cfg, key: jobs)
    monkeypatch.setattr(rollouts.random, "shuffle", lambda urls: None)
    monkeypatch.setattr(rollouts, "_execute_rollout", fake_execute)

    asyncio.run(generate_terminal_rollout(_terminal_cfg(session_resurrection=False), object(), {"task": "fix"}, object()))
    asyncio.run(generate_terminal_rollout(_terminal_cfg(session_resurrection=True), object(), {"task": "fix"}, object()))

    assert seen == [
        ("http://env-a:7777", None),
        ("http://env-a:7777", ["http://env-a:7777", "http://env-b:7778"]),
    ]
