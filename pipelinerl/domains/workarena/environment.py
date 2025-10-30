import logging
import os
from typing import Any

from browsergym.workarena.tasks.base import AbstractServiceNowTask

from tapeagents.core import LLMOutputParsingFailureAction, Observation
from tapeagents.environment import Environment
from tapeagents.steps import ActionExecutionFailure
from tapeagents.tools.browser import Browser
from tapeagents.utils import FatalError

from examples.workarena.steps import Action, FinalAnswerAction, ReflectionThought, WorkArenaTape, WorkArenaTask

logger = logging.getLogger(__name__)


class WorkArenaEnvironment(Environment):
    """
    WorkArena environment for running tasks.
    Translates action steps into gym browser python commands in the form of a string.
    """

    def __init__(self, exp_path: str, headless: bool = True) -> None:
        super().__init__()
        os.makedirs(exp_path, exist_ok=True)
        self.exp_path = exp_path
        self.headless = headless

    def initialize(self):
        self.browser = Browser(headless=self.headless, exp_path=self.exp_path)

    def start_task(
        self, task_entrypoint: type[AbstractServiceNowTask], seed: int = 42
    ) -> tuple[WorkArenaTape, dict[str, Any]]:
        task_id = f"browsergym/{task_entrypoint.get_task_id()}"
        info = self.browser.start_task(task_id, seed, wait_for_user_message=False)  # type: ignore
        obs = self.browser.run_browser_action("noop()")
        tape = WorkArenaTape(steps=[obs, WorkArenaTask(task=info["goal"])])
        return tape, info

    def actions(self) -> tuple[type[Action], ...]:
        return self.browser.actions

    def validate_task(self, tape: WorkArenaTape) -> tuple[bool, dict]:
        answer = tape.steps[-1].text if isinstance(tape.steps[-1], FinalAnswerAction) else "Task finished"
        self.browser._env.unwrapped.chat.add_message(role="assistant", msg=answer)
        assert self.browser._env.unwrapped.task is not None
        reward, stop, message, info = self.browser._env.unwrapped.task.validate(
            self.browser._env.unwrapped.page, self.browser._env.unwrapped.chat.messages
        )
        result_dict = {
            "reward": reward,
            "stop": stop,
            "message": message,
            "info": info,
        }
        return bool(reward > 0), result_dict

    def react(self, tape: WorkArenaTape) -> WorkArenaTape:
        actions = []
        for step in tape.steps[-tape.metadata.n_added_steps :]:
            if isinstance(step, Action):
                actions.append(step)
            elif isinstance(step, ReflectionThought):
                # send reflection to chat for user to see
                self.browser._env.unwrapped.chat.add_message(
                    role="assistant", msg=f"{step.last_action_achieved_effect}\nTodo: {step.next_action}"
                )
        for action in actions:
            try:
                if isinstance(action, LLMOutputParsingFailureAction):
                    continue
                observation = self.step(action)
                tape = tape.append(observation)  # type: ignore
            except FatalError:
                raise
            except Exception as e:
                logger.exception(f"Error during action execution: {e}")
                tape = tape.append(ActionExecutionFailure(error=str(e)))
                break
        return tape

    def step(self, action: Action) -> Observation:
        return self.browser.run(action)

    def reset(self):
        self.browser.reset()

    def close(self) -> None:
        try:
            self.browser.close()
        except Exception as e:
            logger.error(f"Failed to properly close task: {e}")
