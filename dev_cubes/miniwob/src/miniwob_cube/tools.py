"""MiniWob-specific BgymTool variants that preserve the last agent action.

`BgymTool.execute_action` writes the action to `_last_info["action"]` via
`_execute_bgym_step`, then calls `page_obs()` before returning. `page_obs()`
resets `_last_info` to ``{"source": "page_obs"}`` — so by the time the
cube-harness `MonitoredTool` flow reaches `task.evaluate(obs)`, the action
record is gone.

`MiniWobBgymTool` captures the cube `Action` on a dedicated attribute
*before* delegating to the parent's `execute_action`, so the in-method
`page_obs()` reset cannot clobber it. `MiniWobTask.evaluate()` reads
``self.tool.agent_last_action`` to feed action-conditioned local-verifier
signals.

This is purely additive — no changes to cube-standard / cube-harness /
cube-browser-tool source.
"""

from __future__ import annotations

from typing import Any

from cube.core import Action, Observation, StepError
from cube_browser_tool.bgym_tool import BgymTool, BgymToolConfig


class MiniWobBgymTool(BgymTool):
    """`BgymTool` that exposes the most recent agent `Action`.

    The cube `Action` object is stored on `agent_last_action` immediately on
    entry to `execute_action` / `async_execute_action`, before the parent's
    in-method `page_obs()` resets `_last_info`. `reset()` clears it.
    """

    def __init__(self, config: BgymToolConfig) -> None:
        super().__init__(config)
        self.agent_last_action: Action | None = None

    def execute_action(self, action: Action) -> Observation | StepError:
        self.agent_last_action = action
        return super().execute_action(action)

    async def async_execute_action(self, action: Action) -> Observation | StepError:
        self.agent_last_action = action
        return await super().async_execute_action(action)

    def reset(self) -> None:
        self.agent_last_action = None
        super().reset()


class MiniWobBgymToolConfig(BgymToolConfig):
    """`BgymToolConfig` that builds `MiniWobBgymTool` instead of `BgymTool`.

    Inherits all `BgymToolConfig` fields unchanged — only `make()` changes.
    """

    def make(self, container: Any = None) -> MiniWobBgymTool:
        return MiniWobBgymTool(self)
