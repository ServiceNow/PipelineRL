"""WorkArena-specific tools for tasks that may be unsolvable or require cheating."""

from typing import Any, Protocol, runtime_checkable

from browsergym.workarena.tasks.base import AbstractServiceNowTask
from cube.core import Observation
from cube.tool import Tool, ToolConfig, tool_action
from cube_browser_playwright import PlaywrightSession
from cube_browser_tool import PlaywrightConfig, SyncPlaywrightTool
from cube_browser_playwright import PlaywrightSessionConfig
from playwright.sync_api import Page


@runtime_checkable
class WorkarenaBrowserToolConfig(Protocol):
    """
    Protocol for browser tool configs used by WorkArenaTask — requires a `browser` attribute and a `make()` method.
    Both BgymToolConfig and PlaywrightConfig satisfy this protocol, so WorkArenaTask can work with either.
    """

    browser: PlaywrightSessionConfig

    def make(self, container: Any = None) -> "WorkArenaBrowserTool": ...


@runtime_checkable
class WorkArenaBrowserTool(Protocol):
    """
    Protocol for browser tools used by WorkArena tasks — requires a Playwright `page` attribute.
    Both BgymTool and SyncPlaywrightTool satisfy this protocol, so WorkArenaTask can work with either.
    """

    config: WorkarenaBrowserToolConfig

    @property
    def page(self) -> Page: ...

    def noop(self) -> Any: ...

    def page_obs(self) -> Observation: ...


class WorkArenaInfeasibleTool(Tool):
    """WorkArena-specific tool exposing the report_infeasible action."""

    def __init__(self) -> None:
        self._messages: list[dict[str, str]] = []

    def reset(self) -> None:
        self._messages = []

    def close(self) -> None:
        pass

    @tool_action
    def report_infeasible(self, explanation: str) -> str:
        """Report that this task is genuinely impossible to complete. Use only when the task is objectively infeasible.

        Args:
            explanation: Brief explanation of why the task cannot be completed.
        """
        self._messages.append({"role": "infeasible", "message": explanation})
        return "Reported task as infeasible."

    @property
    def messages(self) -> list[dict[str, str]]:
        """Return stored infeasible messages."""
        return list(self._messages)


class WorkArenaInfeasibleToolConfig(ToolConfig):
    """Configuration for WorkArenaInfeasibleTool."""

    def make(self, container: Any = None) -> WorkArenaInfeasibleTool:
        return WorkArenaInfeasibleTool()


class WorkArenaCheatTool(SyncPlaywrightTool):
    """SyncPlaywrightTool with an additional workarena_cheat action — for debug use only."""

    def __init__(self, config: PlaywrightConfig, session: PlaywrightSession) -> None:
        super().__init__(config, session)
        self._workarena_task: AbstractServiceNowTask | None = None
        self._chat_messages_ref: list[dict] = []

    def reset(self) -> None:
        super().reset()
        self._workarena_task = None
        self._chat_messages_ref = []

    @tool_action
    def workarena_cheat(self) -> str:
        """
        Execute the WorkArena built-in cheat to solve the task automatically.
        The .cheat() call mutates self._chat_messages_ref in-place by appending the answer.
        """
        if self._workarena_task is None:
            return "No WorkArena task initialized — cheat unavailable."
        self._workarena_task.cheat(self.page, self._chat_messages_ref)  # type: ignore : Workarena validators expect list[dict] despite the protocol specifying list[str].
        return "WorkArena cheat executed."


class WorkArenaCheatToolConfig(PlaywrightConfig):
    """PlaywrightConfig variant that creates a WorkArenaCheatTool."""

    def make(self, container: Any = None) -> WorkArenaCheatTool:
        session = self.browser.make()
        return WorkArenaCheatTool(self, session)
