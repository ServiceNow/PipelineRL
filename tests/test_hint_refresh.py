"""Unit tests for the in-training hint refresh logic (``pipelinerl.cube_rl.hint_refresh``).

Fast, no Ray / no LLM server: exercise the plain :class:`HintRefreshState` with a fake
LLM returning canned ```json fenced miner responses (mirrors cube-harness's
``tests/test_jefhinter.py`` style).

Run: .venv/bin/python -m pytest tests/test_hint_refresh.py
"""

from __future__ import annotations

from typing import Any

import pytest

from pipelinerl.cube_rl.hint_refresh import HintRefreshState, build_miner_llm_config


def _fenced(text: str) -> str:
    return (
        '```json\n{"hints": [{"hint_type": "task_specific", "text": "'
        + text
        + '", "rationale": "r", "confidence": 4}]}\n```'
    )


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeLLM:
    """Stand-in for cube_harness.llm.LLM — cycles through canned responses."""

    def __init__(self, contents: list[str]) -> None:
        self._contents = contents
        self.calls = 0

    def __call__(self, prompt: Any) -> _FakeResponse:
        content = self._contents[min(self.calls, len(self._contents) - 1)]
        self.calls += 1
        return _FakeResponse(content)


def _state(llm: _FakeLLM, min_episodes: int = 2, max_per_task: int = 4, seed: int = 0) -> HintRefreshState:
    return HintRefreshState(llm=llm, refresh_min_episodes=min_episodes, max_per_task=max_per_task, seed=seed)  # type: ignore[arg-type]


def test_build_miner_llm_config_from_actor_llm_info() -> None:
    cfg = build_miner_llm_config({"base_url": "http://localhost:8000", "model_name": "Qwen2.5-7B-Instruct"})
    assert cfg.api_base == "http://localhost:8000/v1"
    assert cfg.model_name == "openai/Qwen2.5-7B-Instruct"
    assert cfg.temperature == 0.6
    assert cfg.max_completion_tokens == 1024


def test_initial_state_is_version_zero_with_empty_maps() -> None:
    state = _state(_FakeLLM([_fenced("x")]))
    assert state.get_version() == 0
    assert state.get_hints("good") == (0, {})
    assert state.get_hints("none") == (0, {})
    with pytest.raises(ValueError):
        state.get_hints("bogus")


def test_refresh_skips_tasks_below_min_episodes_or_without_failures() -> None:
    llm = _FakeLLM([_fenced("h")])
    state = _state(llm, min_episodes=2)
    state.report("too-few", 0.0, "t")  # only 1 episode < min_episodes
    state.report("all-pass", 1.0, "t")
    state.report("all-pass", 1.0, "t")  # enough episodes but no failure
    assert state.refresh() == 0  # nothing eligible -> version unchanged
    assert llm.calls == 0


def test_refresh_mines_good_and_builds_distractor() -> None:
    llm = _FakeLLM([_fenced("click the funnel icon")])
    state = _state(llm, min_episodes=2)
    for task_id in ("task-a", "task-b"):
        state.report(task_id, 0.0, f"{task_id} failed transcript")
        state.report(task_id, 1.0, f"{task_id} passed transcript")

    version = state.refresh()
    assert version == 1
    assert llm.calls == 2  # one mining call per eligible task

    good_version, good = state.get_hints("good")
    assert good_version == 1
    assert good == {"task-a": "click the funnel icon", "task-b": "click the funnel icon"}
    _, distractor = state.get_hints("distractor")
    # build_hint_bank rule: each task gets a real hint mined for a DIFFERENT task.
    assert set(distractor) == {"task-a", "task-b"}
    assert distractor["task-a"] == good["task-b"] and distractor["task-b"] == good["task-a"]
    assert state.get_hints("none") == (1, {})


def test_refresh_replaces_hints_and_pins_old_versions() -> None:
    llm = _FakeLLM([_fenced("old hint"), _fenced("new hint")])
    state = _state(llm, min_episodes=1)
    state.report("t", 0.0, "failed")
    assert state.refresh() == 1
    state.report("t", 0.0, "failed again")
    assert state.refresh() == 2

    assert state.get_hints("good") == (2, {"t": "new hint"})
    # Version pinning (GRPO group consistency): version 1's map is still served.
    assert state.get_hints("good", version=1) == (1, {"t": "old hint"})
    # Unknown versions resolve to the latest.
    assert state.get_hints("good", version=99) == (2, {"t": "new hint"})


def test_refresh_without_change_keeps_version() -> None:
    llm = _FakeLLM([_fenced("same hint")])
    state = _state(llm, min_episodes=1)
    state.report("t", 0.0, "failed")
    assert state.refresh() == 1
    assert state.refresh() == 1  # same mined text -> no map change -> no version bump


def test_report_buffer_caps_at_max_per_task() -> None:
    llm = _FakeLLM([_fenced("h")])
    state = _state(llm, min_episodes=1, max_per_task=3)
    for i in range(10):
        state.report("t", 0.0, f"transcript {i}")
    assert len(state._buffers["t"]) == 3  # rolling buffer keeps the most recent N
    assert state._buffers["t"][-1] == (0.0, "transcript 9")
