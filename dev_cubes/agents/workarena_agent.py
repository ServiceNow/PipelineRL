import logging
from typing import TYPE_CHECKING

from cube.core import Action, ActionSchema, Observation
from cube.task import STOP_ACTION
from litellm import Message
from termcolor import colored

from cube_harness.agent import Agent, AgentConfig, apply_description_overrides
from cube_harness.core import AgentOutput
from cube_harness.llm import LLMConfig, Prompt
from cube_harness.utils import parse_actions

if TYPE_CHECKING:
    from cube_harness.streamer import EventStreamer

logger = logging.getLogger(__name__)

def _action_to_bgym_string(action: Action) -> str:
    """Serialise a cube Action into a BrowserGym action string like 'click(bid="a51")'."""
    args_parts = []
    for key, value in action.arguments.items():
        args_parts.append(f"{key}={repr(value)}")
    return f"{action.name}({', '.join(args_parts)})"

class WorkArenaAgentConfig(AgentConfig):
    llm_config: LLMConfig
    can_finish: bool = True
    max_actions: int = 10
    safety_buffer: int = 32
    min_useful_tokens: int = 256
    system_prompt: str = """You are a web automation agent. Use the available browser tools to make progress on the task. Call at most one tool per turn. Only call tools that are provided to you. Use visible element ids when an action requires an element. Do not invent tool names or arguments. If the previous tool call returned an error, use that information to choose a corrected next action."""
    user_prompt: str = """# Instructions
Review the current state of the page and all other information to find the best possible next action to accomplish your goal. Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.
Note:
* [bid] is the unique alpha-numeric identifier at the beginning of lines for each element in the AXTree. Always use bid to refer to elements in your actions.
* You can only interact with visible elements. If the "visible" tag is not present, the element is not visible on the page.
* Some text field might have auto completion. To see it, you have to type a few characters and wait until next step.
* Make sure to use bid to identify elements when using commands.
* Interacting with combobox, dropdowns and auto-complete fields can be tricky, sometimes you need to use select_option, while other times you need to use fill or click and wait for the reaction of the page.

## Goal:
{workarena_goal}

## Current observation:
{axtree_txt}

## Focused element:
{focused_element_bid_or_None}

## Previous action error:
{last_action_error_if_any}

## Action history:
{actions_history}
"""

    @property
    def agent_name(self) -> str:
        return f"WorkArenaAgent-{self.llm_config.model_name}".replace("/", "_")

    def make(self, action_set: list[ActionSchema] | None = None, **kwargs) -> "WorkArenaAgent":
        return WorkArenaAgent(config=self, tools=action_set or [])


class WorkArenaAgent(Agent):
    name: str = "workarena_agent"
    description: str = "WorkArena agent."
    input_content_types: list[str] = ["image/png", "image/jpeg", "text/plain", "application/json"]
    output_content_types: list[str] = ["application/json"]

    def __init__(self, config: WorkArenaAgentConfig, tools: list[ActionSchema]):
        super().__init__(config)
        self.llm = config.llm_config.make()
        self.token_counter = config.llm_config.make_counter()
        self.tools: list[dict] = [tool.as_dict() for tool in tools]
        apply_description_overrides(self.tools, config.description_overrides)

        self.actions_history: str = ""
        self.goal: str = ""
        self._actions_cnt = 0
        self.last_action: list[Action] | None = None

        self.max_completion_tokens = config.llm_config.max_completion_tokens
        self.max_model_len = config.llm_config.max_model_len

    def attach_recorder(self, recorder: "EventStreamer") -> None:
        super().attach_recorder(recorder)
        self.llm.attach_recorder(recorder)

    def step(self, obs: Observation) -> AgentOutput:
        if self.max_actions_reached():
            logger.info("Max actions reached, issuing STOP action.")
            return AgentOutput(actions=[Action(id="stop", name=STOP_ACTION.name, arguments={})])
        
        if self._actions_cnt == 0:
            self._set_goal(obs)

        user_prompt = self._build_user_prompt(obs)
        messages = self._build_prompt_messages(user_prompt)

        prompt = Prompt(messages=messages, tools=self.tools)
        prompt_tokens = self.token_counter(messages=messages, tools=self.tools)

        logger.info(f"Prompt tokens (estimated): {prompt_tokens}")
        try:
            logger.debug(f"Prompt: {prompt}")
            call = self.llm.call(prompt, tag="act")
            logger.debug(f"LLM Response: {call.output}")
        except Exception as e:
            logger.exception(colored(f"Error getting LLM response: {e}. Prompt: {prompt}", "red"))
            raise e
        usage = call.usage
        logger.info(
            f"LLM usage - prompt: {usage.prompt_tokens}, completion: {usage.completion_tokens}, "
            f"cached: {usage.cached_tokens}, cache_created: {usage.cache_creation_tokens}, cost: ${usage.cost:.4f}"
        )
        llm_output = call.output
        action = parse_actions(llm_output)
        self.last_action = action
        self._actions_cnt += 1
        return AgentOutput(actions=action)

    def _build_prompt_messages(self, user_prompt: str) -> list[dict | Message]:
        messages: list[dict | Message] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages
    
    def _build_action_string(self, actions: list[Action], obs: Observation) -> str:
        action_strings = []
        for action in actions:
            action_str = _action_to_bgym_string(action)
            for content in obs.contents:
                if content.tool_call_id != None and content.tool_call_id == action.id:
                    action_strings.append(f"Action: {action_str}\nResult: {content.data}\n")
                    break
            else:
                action_strings.append(f"Action:: {action_str}\nResult: No result found for this action.")
        
        return "\n".join(action_strings)

    def _set_goal(self, obs: Observation) -> None:
        for content in obs.contents:
            if content.name is None and content.tool_call_id is None:
                self.goal = content.data
                break
    
    def _build_user_prompt(self, obs: Observation) -> str:
        axtree_txt = "none"
        focused_element_bid = "none"
        last_action_error = "none"

        for content in obs.contents:
            if content.tool_call_id != None:
                if content.data != "Success":
                    last_action_error = content.data
                    continue
            if content.name == 'axtree_txt':
                axtree_txt = content.data
            elif content.name == 'focused_element':
                focused_element_bid = content.data
        
        if self.last_action is not None:
            action_respose = self._build_action_string(self.last_action, obs)
            self.actions_history += action_respose

        user_prompt = self.config.user_prompt.format(
            workarena_goal=self.goal,
            axtree_txt=axtree_txt,
            focused_element_bid_or_None=focused_element_bid,
            last_action_error_if_any=last_action_error,
            actions_history=self.actions_history
        )
        return user_prompt

    def max_actions_reached(self) -> bool:
        return self._actions_cnt >= self.config.max_actions