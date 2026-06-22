"""Smoke-test script for miniwob-cube — validates infrastructure and task solving.

Verifies that the MiniWob HTTP server starts, the browser connects, the task
page loads and JS initialises, and the hardcoded agents achieve reward=1.0 for
the debug tasks.

Public API (cube.testing protocol)
-----------------------------------
get_debug_benchmark()              -> MiniWobBenchmarkConfig
make_debug_agent(task_id: str)     -> ClickButtonAgent | ClickCheckboxesAgent

Usage:
    uv run python -m miniwob_cube.debug
"""

from __future__ import annotations

import logging
import re
import sys

from cube.core import Action, ActionSchema, Observation, TextContent
from cube.testing import run_debug_suite

from cube_browser_tool import PlaywrightConfig

from miniwob_cube.benchmark import MiniWobBenchmarkConfig


logger = logging.getLogger(__name__)

# A small set of representative tasks that cover the JS setup / observation path.
_DEBUG_TASK_IDS = ["click-button", "click-checkboxes"]


class ClickButtonAgent:
    def __init__(self) -> None:
        self._done = False

    def _parse_button_text(self, obs: Observation) -> str:
        for content in obs.contents:
            if isinstance(content, TextContent):
                match = re.search(r'Click on the "(.+?)" button', content.data, re.IGNORECASE)
                assert match
                return match.group(1)

    def __call__(self, obs: Observation, action_set: list[ActionSchema]) -> Action:
        if not self._done:
            self._done = True
            text = self._parse_button_text(obs)
            return Action(name="browser_click", arguments={"selector": f"button:has-text('{text}')"})
        return Action(name="final_step", arguments={})


class ClickCheckboxesAgent:
    def __init__(self) -> None:
        self._step = 0
        self._targets: list[str] = []

    def _parse_targets(self, obs: Observation) -> list[str]:
        for content in obs.contents:
            if isinstance(content, TextContent):
                match = re.search(r"Select (.+?) and click Submit", content.data, re.IGNORECASE)
                assert match
                words_str = match.group(1)
                if words_str.lower() == "nothing":
                    return []
                return [w.strip() for w in words_str.split(",")]

    def __call__(self, obs: Observation, action_set: list[ActionSchema]) -> Action:
        if self._step == 0:
            self._targets = self._parse_targets(obs)
        idx = self._step
        self._step += 1
        if idx < len(self._targets):
            word = self._targets[idx]
            return Action(
                name="browser_click", arguments={"selector": f"label:has-text('{word}') input[type='checkbox']"}
            )
        if idx == len(self._targets):
            return Action(name="browser_click", arguments={"selector": "button#subbtn"})
        return Action(name="final_step", arguments={})


def make_debug_agent(task_id: str) -> ClickButtonAgent | ClickCheckboxesAgent:
    if task_id == "click-button":
        return ClickButtonAgent()
    if task_id == "click-checkboxes":
        return ClickCheckboxesAgent()
    raise ValueError(f"No hardcoded agent for task: {task_id}")


def get_debug_benchmark() -> MiniWobBenchmarkConfig:
    return MiniWobBenchmarkConfig(
        tool_config=PlaywrightConfig(headless=True, use_html=True, use_axtree=False, use_screenshot=False),
    ).subset_from_list(_DEBUG_TASK_IDS, benchmark_name_suffix="debug")


if __name__ == "__main__":
    import miniwob_cube.debug as _this_module

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    results = run_debug_suite("miniwob-cube", _this_module)
    failed = [r for r in results if r["error"] or not r["done"] or r["reward"] < 1.0]
    sys.exit(1 if failed else 0)
