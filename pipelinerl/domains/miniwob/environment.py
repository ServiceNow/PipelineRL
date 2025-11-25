import logging
import os
import time
from typing import Any, Literal

from browsergym.core.task import AbstractBrowserTask
from browsergym.miniwob import ALL_MINIWOB_TASKS
from browsergym.miniwob.base import AbstractMiniwobTask

from tapeagents.core import Action, FinalObservation, LLMOutputParsingFailureAction, Observation
from tapeagents.environment import Environment
from tapeagents.steps import ActionExecutionFailure
from tapeagents.tools.browser import Browser
from tapeagents.utils import FatalError

from .steps import (
    FinalAnswerAction,
    ReflectionThought,
    WebTape,
    WebTapeMetadata,
    WebTask,
)

from browsergym.core.chat import Chat

logger = logging.getLogger(__name__)

# Mock the Chat class to avoid huge slowdown caused by it
def mock_chat_init(self, *args, **kwargs):
    logger.info("Mocked Chat.__init__")
    self.messages = []

def mock_wait_for_user_message(self, *args, **kwargs):
    logger.info("Mocked Chat.wait_for_user_message")
    pass

def mock_add_message(self, role: str, msg: str):
    logger.info("Mocked Chat.add_message")
    self.messages.append({"role": role, "timestamp": time.time(), "message": msg})

def mock_close(self, *args, **kwargs):
    logger.info("Mocked Chat.close")
    pass

Chat.__init__ = mock_chat_init
Chat.wait_for_user_message = mock_wait_for_user_message
Chat.add_message = mock_add_message
Chat.close = mock_close

logger.info("Mocked Chat class initialized")


class WebEnvironment(Environment):
    """
    Web environment for running tasks.
    Translates action steps into gym browser python commands in the form of a string.
    """

    def __init__(
        self,
        exp_path: str | None,
        headless: bool = True,
        observation_format: Literal["axtree", "html", "markdown_html"] = "html",
    ) -> None:
        super().__init__()
        if exp_path:
            os.makedirs(exp_path, exist_ok=True)
        self.exp_path = exp_path
        self.headless = headless
        self.observation_format = observation_format

    def initialize(self):
        self.timers = {}  # reset timers
        self.browser = Browser(
            headless=self.headless,
            exp_path=self.exp_path,
            mock=True,
            observation_format=self.observation_format,  # type: ignore
            timeout_ms=5000,  # 5 seconds timeout for browser actions
        )

    def start_task(self, task_data: dict) -> tuple[WebTape, dict[str, Any]]:
        # if task_data["task"] is a string, it is the task_id of a registered task
        if isinstance(task_data["task"], str):
            # reconstruct the task entrypoint from the dataset and task_id
            assert "dataset" in task_data, "Task data must contain 'dataset' key."
            if "miniwob" in task_data["dataset"]:
                task_entrypoint = [t for t in ALL_MINIWOB_TASKS if t.get_task_id() == task_data["task"]][0]
            else:
                raise ValueError(f"Unknown dataset {task_data['dataset']} for task {task_data['task']}.")
        else:
            task_entrypoint: AbstractBrowserTask = task_data["task"]
        assert issubclass(task_entrypoint, AbstractBrowserTask)
        seed: int = task_data.get("seed", 42)

        self.timers = {}  # reset timers
        start_start_task = time.perf_counter()
        task_id = f"browsergym/{task_entrypoint.get_task_id()}"
        # logger.info(f"WebEnv.start_task {task_id} starting task...")
        _zero = time.perf_counter()
        info = self.browser.start_task(task_id, seed, wait_for_user_message=False)  # type: ignore
        zero = time.perf_counter() - _zero
        logger.info(f"WebEnv.start_task {task_id} browser.start_task took {zero:.2f}s")
        # Check if browser failed to start task
        if isinstance(info, dict) and "error" in info:
            logger.warning(f"Browser failed to start task {task_id}: {info['error']}")
            error_tape = WebTape(
                metadata=WebTapeMetadata(task_name=task_entrypoint.get_task_id(), seed=seed),
                steps=[ActionExecutionFailure(error=info["error"])],
            )
            return error_tape, info
        _one = time.perf_counter()
        obs = self.browser.run_browser_action("noop()")
        one = time.perf_counter() - _one
        logger.info(f"WebEnv.start_task {task_id} browser.run_browser_action took {one:.2f}s")
        _two = time.perf_counter()
        tape = WebTape(
            metadata=WebTapeMetadata(task_name=task_entrypoint.get_task_id(), seed=seed),
            steps=[obs, WebTask(task=info["goal"])],
        )
        two = time.perf_counter() - _two
        logger.info(f"WebEnv.start_task {task_id} WebTape creation took {two:.2f}s")
        self.timers["start_task"] = time.perf_counter() - start_start_task
        logger.info(f"total start_task {task_id}: {self.timers['start_task']:.2f}s")
        return tape, info

    def finish_task(self) -> None:
        start_finish_task = time.perf_counter()
        try:
            self.browser.close()
        except Exception as e:
            logger.exception(f"Failed to properly close task: {e}", stack_info=True)
        self.timers["finish_task"] = time.perf_counter() - start_finish_task

    def validate_task(self, tape: WebTape) -> tuple[bool, dict]:
        start_validate_task = time.perf_counter()
        answer = tape.steps[-1].text if isinstance(tape.steps[-1], FinalAnswerAction) else "Task finished"
        self.browser._env.unwrapped.chat.add_message(role="assistant", msg=answer)
        assert self.browser._env.unwrapped.task is not None
        try:
            reward, stop, message, info = self.browser._env.unwrapped.task.validate(
                self.browser._env.unwrapped.page, self.browser._env.unwrapped.chat.messages
            )
        except Exception as e:
            logger.exception(f"Error during task validation: {e}")
            reward = 0
            stop = True
            message = f"Task validation failed with error: {e}"
            info = {}
        # in AbstractMiniwobTask.validate() the reward is defined as float(info["RAW_REWARD_GLOBAL"] > 0) so it will alwayd be 0 or 1
        # let's keep the raw reward for now and define success as reward > 0.5
        if isinstance(self.browser._env.unwrapped.task, AbstractMiniwobTask) and "RAW_REWARD_GLOBAL" in info:
            reward = info["RAW_REWARD_GLOBAL"]
        result_dict = {
            "reward": reward,
            "stop": stop,
            "message": message,
            "info": info,
        }
        self.timers["validate_task"] = time.perf_counter() - start_validate_task
        return bool(reward > 0.5), result_dict

    def react(self, tape: WebTape) -> WebTape:
        if "react" not in self.timers:
            self.timers["react"] = []
        start_react = time.perf_counter()

        actions: list[Action] = []
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
                if isinstance(observation, ActionExecutionFailure):
                    logger.exception(f"Error during action execution: {observation.error}")
                    tape = tape.append(observation)
                    break
                tape = tape.append(observation)  # type: ignore
            except FatalError:
                self.timers["react"].append(time.perf_counter() - start_react)
                raise
            except Exception as e:
                logger.exception(f"Error during action execution {action}: {e}", stack_info=True)
                tape = tape.append(ActionExecutionFailure(error=str(e)))
                break
        self.timers["react"].append(time.perf_counter() - start_react)
        return tape
        # TODO: MAYBE make sure to update parent_id, author_name, etc... in the new tape.metadata just like in agent.run()

    def step(self, action: Action) -> Observation:
        t = time.perf_counter()
        obs = self.browser.run(action)
        if obs.metadata.other.get("env_finished", False):
            obs = FinalObservation(metadata=obs.metadata)
        obs.metadata.other["action_execution_time"] = time.perf_counter() - t
        return obs

    def actions(self) -> tuple[type[Action], ...]:
        return self.browser.actions

    def reset(self):
        try:
            self.browser.reset()
        except Exception as e:
            logger.warning(f"Failed to reset browser: {e}, recreate browser instance instead.")
            self.close()
            self.initialize()

    def close(self):
        try:
            self.browser.close()
        except Exception as e:
            logger.exception(f"Failed to close browser: {e}", stack_info=True)
