"""WorkArena task implementation for the CUBE framework."""

import logging
import time
from typing import Any, List, Literal, override

import browsergym.workarena
from browsergym.workarena.tasks.base import AbstractServiceNowTask
from cube.benchmark import RuntimeContext
from cube.core import Action, EnvironmentOutput, Observation
from cube.task import Task, TaskConfig, TaskMetadata
from cube.tool import Toolbox
from cube.tools.browser import BrowserTool
from cube_browser_playwright import Viewport
from cube_chat_tool import ChatTool
from workarena_cube.tools import WorkArenaCheatTool, WorkArenaInfeasibleTool, WorkArenaBrowserTool
from pydantic import PrivateAttr


logger = logging.getLogger(__name__)


class WorkArenaTaskMetadata(TaskMetadata):
    """TaskMetadata subclass for WorkArena ServiceNow tasks.

    Public fields shipped in task_metadata.json (available at import time).
    WorkArena has no heavy execution data — all task logic is available from
    the browsergym-workarena library at runtime via task_class_path.
    """

    level: Literal["l1", "l2", "l3"]
    """Task level: l1 = atomic, l2 = compositional, l3 = extended compositional."""

    in_human_curriculum: bool
    """Whether this task type is part of the human evaluation curriculum."""

    task_class_path: str
    """Dotted path to the WorkArena task class, e.g. 'browsergym.workarena.tasks.dashboard.MultiChartValueRetrievalTask'."""


class WorkArenaTask(Task):
    """CUBE Task wrapper for WorkArena ServiceNow tasks."""

    metadata: WorkArenaTaskMetadata  # type: ignore[assignment]
    seed: int
    wait_first_page_time: float = 10.0
    validate_per_step: bool = True

    _workarena_task: AbstractServiceNowTask | None = PrivateAttr(default=None)
    _validate_cache: tuple[Any, ...] | None = PrivateAttr(default=None)

    @property
    def _browser_tool(self) -> WorkArenaBrowserTool:
        """Resolve the browser tool whether it's direct or inside a Toolbox."""
        if isinstance(self.tool, Toolbox):
            tool = self.tool.find_tool(BrowserTool)
            if tool is None:
                raise RuntimeError("No BrowserTool found in Toolbox")
        else:
            tool = self.tool
        if not isinstance(tool, WorkArenaBrowserTool):
            raise RuntimeError(
                f"The browser tool must satisfy the WorkArenaBrowserTool protocol (e.g., BgymTool or SyncPlaywrightTool), got {type(tool).__name__}"
            )
        return tool

    @property
    def _chat_tool(self) -> ChatTool | None:
        """Return the ChatTool if present in a Toolbox, else None."""
        if isinstance(self.tool, Toolbox):
            return self.tool.find_tool(ChatTool)  # type: ignore
        return None

    @property
    def _infeasible_tool(self) -> WorkArenaInfeasibleTool | None:
        """Return the WorkArenaInfeasibleTool if present in a Toolbox, else None."""
        if isinstance(self.tool, Toolbox):
            tool = self.tool.find_tool(WorkArenaInfeasibleTool)
            return tool if isinstance(tool, WorkArenaInfeasibleTool) else None
        return None

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        """Instantiate and set up the WorkArena task, returning the initial observation."""
        task_class = _load_task_class(self.metadata.task_class_path)
        self._workarena_task = task_class(seed=self.seed)
        if self._workarena_task is None:
            raise RuntimeError("Failed to initialize WorkArena task.")
        _apply_task_runtime_preferences(self._browser_tool, self._workarena_task)
        self.tool.reset()
        self._validate_cache = None
        if isinstance(self._browser_tool, WorkArenaCheatTool):
            self._browser_tool._workarena_task = self._workarena_task
        page = self._browser_tool.page
        goal, task_info = self._workarena_task.setup(page)

        logger.info(f"WorkArena page URL after setup: {page.url}")
        logger.info(f"WorkArena page title: {page.title()}")
        logger.info(f"WorkArena task class: {self._workarena_task.__class__.__name__}")

        self._browser_tool.noop()
        time.sleep(self.wait_first_page_time)
        logger.info(f"WorkArena task goal: {goal}")

        page_obs = self._browser_tool.page_obs()
        if self._chat_tool is not None:
            self._chat_tool.add_message("user", goal)
            obs = Observation.from_text(self._chat_tool.chat_obs()) + page_obs
        else:
            obs = Observation.from_text(goal) + page_obs
        info = {
            "task_id": self.id,
            "task_class": task_class.__name__,
            "seed": self.seed,
            "goal": goal,
            **task_info,
        }
        return obs, info

    @property
    def _chat_messages(self) -> list[dict]:
        """
        Return combined chat and infeasible messages.

        Normal path (ChatTool): a copy of session history — safe for parallel episodes,
        always current because send_message() writes before evaluate() runs.

        Cheat path (WorkArenaCheatTool, no ChatTool): the live _chat_messages_ref list.
        cheat() appends directly to whatever list it receives, so cheat() and validate()
        must share the same list instance.
        """
        messages: list[dict] = []
        if self._chat_tool is None and isinstance(self._browser_tool, WorkArenaCheatTool):
            messages.extend(self._browser_tool._chat_messages_ref)
        elif (chat := self._chat_tool) is not None:
            messages.extend(chat.messages)
        if (infeasible := self._infeasible_tool) is not None:
            messages.extend(infeasible.messages)
        return messages

    def _validate(self) -> tuple[float, bool, str, dict]:
        """Call WorkArena's validate() with per-step caching.

        Both evaluate() and finished() call this on every step. The cache avoids
        duplicate ServiceNow REST calls within the same step. It is cleared after
        the first consumer reads it, so the next step gets a fresh call.
        """
        if self._workarena_task is None:
            raise RuntimeError("WorkArena task is not initialized. Call reset() first.")
        if self._validate_cache is None:
            page = self._browser_tool.page
            self._validate_cache = self._workarena_task.validate(page, self._chat_messages)  # type: ignore : Workarena validators expect list[dict] despite the protocol specifying list[str].
        return self._validate_cache  # type: ignore[return-value]

    @override
    def step(self, action: Action | List[Action]) -> EnvironmentOutput:
        self._validate_cache = None
        return super().step(action)

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict[str, Any]]:
        """Score the current task state via WorkArena's validate()."""
        reward, done, _user_message, task_info = self._validate()
        return reward, {"done": done, **task_info}

    def finished(self, obs: Observation | None = None) -> bool:
        """Check if the task is done via WorkArena's validate()."""
        if self._workarena_task is None:
            return False
        _reward, done, _user_message, _task_info = self._validate()
        return done

    def close(self) -> None:
        """Teardown the WorkArena task and close the tool."""
        if self._workarena_task is not None:
            try:
                self._workarena_task.teardown()
            except Exception as e:
                logger.warning(f"Error during WorkArena task teardown: {e}")
            finally:
                self._workarena_task = None
        super().close()


class WorkArenaTaskConfig(TaskConfig[WorkArenaTaskMetadata]):
    """Serializable configuration for a single WorkArena task."""

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
    ) -> WorkArenaTask:
        _ = runtime_context
        assert self.tool_config, f"WorkArenaTaskConfig requires a tool_config, got {self.tool_config}"
        return WorkArenaTask(
            metadata=self.metadata,
            tool_config=self.tool_config,
            seed=self.seed if self.seed is not None else 42,
        )


def _load_task_class(class_path: str) -> type:
    """Reconstruct a task class from its dotted module-qualified name."""
    matches = [
        cls for cls in browsergym.workarena.ALL_WORKARENA_TASKS if f"{cls.__module__}.{cls.__name__}" == class_path
    ]
    if len(matches) == 0:
        raise ValueError(f"{class_path!r} is not a registered WorkArena task class")
    assert len(matches) == 1, f"Duplicate task class registered for {class_path!r}"
    return matches[0]


def _apply_task_runtime_preferences(tool: WorkArenaBrowserTool, workarena_task: AbstractServiceNowTask) -> None:
    """Apply WorkArena task runtime defaults to the tool config when not explicitly set."""
    browser_config = tool.config.browser
    explicitly_set = browser_config.model_fields_set
    updates: dict[str, Any] = {}
    for field in ("slow_mo", "timeout", "locale", "timezone_id"):
        if field not in explicitly_set and getattr(workarena_task, field, None) is not None:
            updates[field] = getattr(workarena_task, field)
    if "viewport" not in explicitly_set:
        raw_vp = getattr(workarena_task, "viewport", None)
        if isinstance(raw_vp, dict):
            updates["viewport"] = Viewport(**raw_vp)
        elif isinstance(raw_vp, Viewport):
            updates["viewport"] = raw_vp
    if updates:
        tool.config.browser = browser_config.model_copy(update=updates)
