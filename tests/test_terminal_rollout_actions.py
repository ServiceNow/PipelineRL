import asyncio
import json
import os

import pytest
import time
from types import SimpleNamespace

from pipelinerl.domains.terminal import rollouts
from pipelinerl.domains.terminal.environment_server import TerminalEnvironmentServer
from pipelinerl.domains.terminal.rollouts import (
    _SUBMIT_COMMAND,
    _assistant_tool_message,
    _execute_rollout,
    _extract_bash_action,
    _is_submit_command,
    EnvironmentConnectionError,
    TerminalMetrics,
    generate_terminal_rollout,
)
from pipelinerl.llm import LLMCall, LLMOutput, Prompt, TrainableLLM
from pipelinerl.rollouts import RolloutResult, TrainingText


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


def _patch_rollout_fakes(
    monkeypatch,
    llm_calls,
    *,
    verifier_pass=True,
    step_response=None,
    finish_response=None,
    start_response=None,
):
    pending_calls = list(llm_calls)

    async def fake_generate(llm, prompt, session):
        assert pending_calls
        return pending_calls.pop(0)

    async def fake_post(session, url, payload, timeout):
        if url.endswith("/start_task"):
            return start_response or {"session_id": "session-1", "started": True, "init_ok": True, "build_ok": True}
        if url.endswith("/step"):
            return step_response or {"output": "ok", "abort_kind": None}
        if url.endswith("/finish"):
            if finish_response is not None:
                return finish_response
            return {
                "passed": verifier_pass,
                "passed_tests": int(verifier_pass),
                "total_tests": 1,
                "abort_kind": None,
                "output": "pytest output",
                "contamination_result": {"sampled": False, "count": 0},
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
    assert not result.audit["dropped"]


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


def test_execute_rollout_closes_started_session_on_init_failure(monkeypatch):
    calls = []

    async def fake_post(session, url, payload, timeout):
        calls.append((url, payload))
        if url.endswith("/start_task"):
            return {"session_id": "session-1", "started": True, "init_ok": False, "build_ok": True}
        if url.endswith("/close"):
            return {"status": "ok"}
        raise AssertionError(url)

    monkeypatch.setattr("pipelinerl.domains.terminal.rollouts._post", fake_post)

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

    assert result.metrics.reward == -1.0
    assert result.metrics.build_ok
    assert not result.metrics.init_ok
    assert calls == [
        ("http://env/start_task", {"task_data": {"task": "fix it", "task_id": "task-1"}}),
        ("http://env/close", {"session_id": "session-1"}),
    ]


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


def test_timeout_abort_breaks_loop_and_sets_metric(monkeypatch):
    _patch_rollout_fakes(
        monkeypatch,
        [_llm_call(content="run", tool_calls=[_tool_call(arguments={"command": "make"})])],
        verifier_pass=False,
        step_response={"output": "command timed out", "abort_kind": "timeout"},
    )

    result = asyncio.run(
        _execute_rollout(
            _terminal_cfg(max_turns=2),
            object(),
            {"task": "fix it", "task_id": "task-1"},
            object(),
            time.time(),
            "http://env",
        )
    )

    assert result.metrics.timeout_aborted
    assert not result.metrics.disk_aborted
    assert not result.metrics.rss_aborted
    assert result.metrics.n_llm_calls == 1
    assert result.metrics.n_total_llm_calls == 1
    assert [text.reward for text in result.training_texts] == [-1.0]
    assert result.audit["abort_kind"] == "timeout"
    assert result.audit["abort_phase"] == "step"
    assert not result.audit["dropped"]


def test_rss_abort_sets_metric_and_skips_no_submit_penalty(monkeypatch):
    _patch_rollout_fakes(
        monkeypatch,
        [_llm_call(content="run", tool_calls=[_tool_call(arguments={"command": "python train.py"})])],
        verifier_pass=True,
        step_response={"output": "rss cap exceeded", "abort_kind": "rss"},
    )

    result = asyncio.run(
        _execute_rollout(
            _terminal_cfg(max_turns=1, no_submit_penalty=0.4),
            object(),
            {"task": "fix it", "task_id": "task-1"},
            object(),
            time.time(),
            "http://env",
        )
    )

    assert result.metrics.rss_aborted
    assert not result.metrics.disk_aborted
    assert not result.metrics.timeout_aborted
    assert result.metrics.reward == 1.0
    assert [text.reward for text in result.training_texts] == [1.0]


def test_finish_abort_drops_rollout_and_records_audit(monkeypatch):
    _patch_rollout_fakes(
        monkeypatch,
        [_llm_call(content="submit", tool_calls=[_tool_call(arguments={"command": _SUBMIT_COMMAND})])],
        finish_response={
            "passed": False,
            "passed_tests": 0,
            "total_tests": 1,
            "abort_kind": "timeout",
            "output": "verifier timed out after partial stdout",
            "contamination_result": {"sampled": True, "count": 2},
        },
    )

    result = asyncio.run(
        _execute_rollout(
            _terminal_cfg(),
            object(),
            {"task": "fix it", "task_id": "task-1", "task_complexity": "hard"},
            object(),
            time.time(),
            "http://env",
        )
    )

    assert result.training_texts == []
    assert result.metrics.timeout_aborted
    assert result.audit["task_id"] == "task-1"
    assert result.audit["complexity"] == "hard"
    assert result.audit["abort_kind"] == "timeout"
    assert result.audit["abort_phase"] == "finish"
    assert result.audit["dropped"]
    assert result.audit["drop_reason"] == "finish_abort"
    assert result.audit["contamination_result"] == {"sampled": True, "count": 2}
    assert "verifier timed out" in result.audit["finish_stdout_tail"]


def test_no_tests_resolved_after_step_abort_drops_rollout(monkeypatch):
    _patch_rollout_fakes(
        monkeypatch,
        [_llm_call(content="run", tool_calls=[_tool_call(arguments={"command": "make"})])],
        step_response={"output": "command timed out", "abort_kind": "timeout"},
        finish_response={
            "passed": False,
            "passed_tests": 0,
            "total_tests": 0,
            "abort_kind": None,
            "output": "no tests resolved",
        },
    )

    result = asyncio.run(
        _execute_rollout(
            _terminal_cfg(max_turns=2),
            object(),
            {"task": "fix it", "task_id": "task-1"},
            object(),
            time.time(),
            "http://env",
        )
    )

    assert result.training_texts == []
    assert result.metrics.timeout_aborted
    assert result.audit["abort_kind"] == "timeout"
    assert result.audit["abort_phase"] == "step"
    assert result.audit["dropped"]
    assert result.audit["drop_reason"] == "no_tests_resolved"


def test_context_exhausted_with_verdict_stays_scored(monkeypatch):
    class TinyBudgetLLM:
        parameters = {"max_tokens": 16}
        chat_template_kwargs = {}

        def load_tokenizer(self):
            self.tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: list(range(100)))

    _patch_rollout_fakes(monkeypatch, [])
    cfg = _terminal_cfg(no_submit_penalty=0.4, max_turns=8)
    cfg.vllm_config = SimpleNamespace(vllm_kwargs={"max_model_len": 128})

    result = asyncio.run(
        _execute_rollout(
            cfg,
            TinyBudgetLLM(),
            {"task": "fix it", "task_id": "task-1"},
            object(),
            time.time(),
            "http://env",
        )
    )

    assert result.metrics.context_exhausted
    assert result.metrics.total_tests == 1
    assert result.metrics.reward == 0.6
    assert not result.audit["dropped"]
    assert result.audit["context_exhausted"]


class DummySession:
    def __init__(self):
        self.closed = False
        self.finished = False
        self.close_count = 0

    def finish(self):
        self.finished = True
        return {"passed": True}

    def sample_contamination(self):
        return 0, 0

    def close(self, contamination_sample=True):
        self.closed = True
        self.close_count += 1
        return 0, 0


class DummyRequest:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


def test_close_background_records_contamination_in_health():
    class ContaminatedSession(DummySession):
        def close(self, contamination_sample=True):
            super().close(contamination_sample=contamination_sample)
            return (1, 3) if contamination_sample else (0, 0)

    async def run_case():
        server = TerminalEnvironmentServer(
            bases_dir="/tmp",
            n_envs=1,
            session_ttl_seconds=60.0,
            session_reap_interval_seconds=60.0,
        )
        session = ContaminatedSession()
        server._sessions["session-1"] = session
        server._session_last_activity["session-1"] = time.monotonic()

        close_response = await server.close(DummyRequest({"session_id": "session-1"}))
        if server._bg_tasks:
            await asyncio.gather(*list(server._bg_tasks))
        health_response = await server.health(DummyRequest({}))
        server._executor.shutdown(wait=True)

        assert close_response.status == 200
        assert session.closed
        health = json.loads(health_response.text)
        assert health["contamination_events"] == 1
        assert health["contamination_closes_sampled"] == 1
        assert health["contamination_contaminated_sampled"] == 1

    asyncio.run(run_case())


def test_contamination_sampling_records_one_in_n_closes():
    class SamplingSession(DummySession):
        def __init__(self):
            super().__init__()
            self.samples = []

        def close(self, contamination_sample=True):
            super().close(contamination_sample=contamination_sample)
            self.samples.append(contamination_sample)
            return (1, 2) if contamination_sample else (0, 0)

    async def run_case():
        server = TerminalEnvironmentServer(
            bases_dir="/tmp",
            n_envs=3,
            contamination_sample_every=3,
            session_ttl_seconds=60.0,
            session_reap_interval_seconds=60.0,
        )
        sessions = [SamplingSession() for _ in range(3)]
        for i, session in enumerate(sessions):
            server._sessions[f"session-{i}"] = session
            server._session_last_activity[f"session-{i}"] = time.monotonic()

        for i in range(3):
            response = await server.close(DummyRequest({"session_id": f"session-{i}"}))
            assert response.status == 200
        if server._bg_tasks:
            await asyncio.gather(*list(server._bg_tasks))
        health_response = await server.health(DummyRequest({}))
        server._executor.shutdown(wait=True)

        assert [session.samples for session in sessions] == [[False], [False], [True]]
        health = json.loads(health_response.text)
        assert health["contamination_events"] == 1
        assert health["contamination_closes_sampled"] == 1
        assert health["contamination_contaminated_sampled"] == 1

    asyncio.run(run_case())


def test_finish_removes_session_and_close_stays_idempotent():
    async def run_case():
        server = TerminalEnvironmentServer(
            bases_dir="/tmp",
            n_envs=1,
            session_ttl_seconds=60.0,
            session_reap_interval_seconds=60.0,
        )
        session = DummySession()
        server._sessions["session-1"] = session
        server._session_last_activity["session-1"] = time.monotonic()

        response = await server.finish(DummyRequest({"session_id": "session-1"}))
        close_response = await server.close(DummyRequest({"session_id": "session-1"}))
        if server._bg_tasks:
            await asyncio.gather(*list(server._bg_tasks))
        server._executor.shutdown(wait=True)

        assert response.status == 200
        assert close_response.status == 200
        assert "session-1" not in server._sessions
        assert "session-1" not in server._session_last_activity
        assert session.finished
        assert session.closed
        assert session.close_count == 1

    asyncio.run(run_case())


def test_finish_returns_sampled_contamination_result_without_resampling_close():
    class SampledSession(DummySession):
        def __init__(self):
            super().__init__()
            self.close_samples = []

        def sample_contamination(self):
            return 1, 4

        def close(self, contamination_sample=True):
            self.close_samples.append(contamination_sample)
            super().close(contamination_sample=contamination_sample)
            return (1, 9) if contamination_sample else (0, 0)

    async def run_case():
        server = TerminalEnvironmentServer(
            bases_dir="/tmp",
            n_envs=1,
            contamination_sample_every=1,
            session_ttl_seconds=60.0,
            session_reap_interval_seconds=60.0,
        )
        session = SampledSession()
        server._sessions["session-1"] = session
        server._session_last_activity["session-1"] = time.monotonic()

        response = await server.finish(DummyRequest({"session_id": "session-1"}))
        if server._bg_tasks:
            await asyncio.gather(*list(server._bg_tasks))
        health_response = await server.health(DummyRequest({}))
        server._executor.shutdown(wait=True)

        body = json.loads(response.text)
        health = json.loads(health_response.text)
        assert body["contamination_result"] == {"sampled": True, "count": 4}
        assert session.close_samples == [False]
        assert health["contamination_events"] == 1
        assert health["contamination_closes_sampled"] == 1
        assert health["contamination_contaminated_sampled"] == 1

    asyncio.run(run_case())


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


def test_execute_rollout_wraps_start_task_connection_error(monkeypatch):
    seen_timeouts = []

    async def fake_post(session, url, payload, timeout):
        seen_timeouts.append(timeout)
        raise rollouts.aiohttp.ClientConnectionError("dead")

    monkeypatch.setattr(rollouts, "_post", fake_post)

    try:
        asyncio.run(
            _execute_rollout(
                _terminal_cfg(env_start_timeout=30),
                object(),
                {"task": "fix it", "task_id": "task-1"},
                object(),
                time.time(),
                "http://dead-env",
            )
        )
    except EnvironmentConnectionError:
        pass
    else:
        raise AssertionError("expected start_task connection error")

    assert seen_timeouts
    assert seen_timeouts[0].total == 30
    assert seen_timeouts[0].connect == 10


def test_generate_rollout_tries_start_task_without_health_probe(monkeypatch):
    jobs = [
        SimpleNamespace(hostname="dead-env", port=7777),
        SimpleNamespace(hostname="live-env", port=7778),
    ]
    attempts = []

    class NoHealthSession:
        def get(self, *args, **kwargs):
            raise AssertionError("health probe should not run")

    async def fake_execute(cfg, llm, problem, session, start_time, env_url):
        attempts.append(env_url)
        if env_url == "http://dead-env:7777":
            raise EnvironmentConnectionError("dead")
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

    result = asyncio.run(
        generate_terminal_rollout(
            _terminal_cfg(rollout_timeout=5, capacity_retry_sleep=0),
            object(),
            {"task": "fix it", "task_id": "task-1"},
            NoHealthSession(),
        )
    )

    assert result.metrics.success
    assert attempts == ["http://dead-env:7777", "http://live-env:7778"]


def test_unreconstructable_format_error_turn_is_dropped_not_fatal(monkeypatch):
    llm_calls = [
        _llm_call(content="bad format"),
        _llm_call(content="submit", tool_calls=[_tool_call(arguments={"command": _SUBMIT_COMMAND})]),
    ]
    _patch_rollout_fakes(monkeypatch, llm_calls)

    def exploding_make_training_text(llm, llm_call):
        if (llm_call.output.content or "") == "bad format":
            raise TypeError("Can only get item pairs from a mapping.")
        return TrainingText(text=llm_call.output.content or "tool", n_predicted=1)

    monkeypatch.setattr(
        "pipelinerl.domains.terminal.rollouts.make_training_text", exploding_make_training_text
    )

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

    assert [text.text for text in result.training_texts] == ["submit"]
    assert [text.reward for text in result.training_texts] == [1.0]
    assert result.metrics.format_error_texts_dropped == 1
    assert result.metrics.n_format_errors == 1


MODEL_PATH = "/mnt/llmd/base_models/Qwen3.5-9B"


@pytest.mark.skipif(not os.path.isdir(MODEL_PATH), reason="base model tokenizer not available")
def test_context_budget_with_real_tokenizer_ends_rollout_cleanly(monkeypatch):
    """Real TrainableLLM + real Qwen tokenizer through the precheck: catches the
    lazy-tokenizer lifecycle and template compatibility with wire-format messages."""
    llm = TrainableLLM(
        base_url="http://unused",
        model_name=MODEL_PATH,
        tokenizer_name=MODEL_PATH,
        parameters={"max_tokens": 16000},
    )
    llm.chat_template_kwargs = {"enable_thinking": True}

    # Budget the real system+task prompt cannot fit -> immediate clean exhaustion.
    _patch_rollout_fakes(monkeypatch, [])
    cfg = _terminal_cfg(no_submit_penalty=0.4, max_turns=8)
    cfg.vllm_config = SimpleNamespace(vllm_kwargs={"max_model_len": 512})
    result = asyncio.run(
        _execute_rollout(cfg, llm, {"task": "fix it", "task_id": "t"}, object(), time.time(), "http://env")
    )
    assert result.metrics.context_exhausted
    assert result.metrics.n_turns == 0
    assert result.metrics.reward == 0.6

    # Ample budget -> precheck passes, and the REAL template must render a history
    # containing a wire-format (string-arguments) tool call without crashing.
    _patch_rollout_fakes(
        monkeypatch,
        [
            _llm_call(content="run it", tool_calls=[_tool_call(arguments={"command": "ls"})]),
            _llm_call(content="done", tool_calls=[_tool_call(arguments={"command": _SUBMIT_COMMAND})]),
        ],
    )
    cfg.vllm_config = SimpleNamespace(vllm_kwargs={"max_model_len": 65536})
    result = asyncio.run(
        _execute_rollout(cfg, llm, {"task": "fix it", "task_id": "t"}, object(), time.time(), "http://env")
    )
    assert not result.metrics.context_exhausted
    assert result.metrics.submitted
    assert result.metrics.n_turns == 2


