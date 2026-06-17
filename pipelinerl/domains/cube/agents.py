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


class SimpleAgentConfig(AgentConfig):
    llm_config: LLMConfig
    can_finish: bool = True
    max_actions: int = 10
    system_prompt: str = """
You are an expert AI Agent trained to assist users with complex web tasks.
Your role is to understand the goal, perform actions until the goal is accomplished and respond in a helpful and accurate manner.
Keep your replies brief, concise, direct and on topic. Prioritize clarity and avoid over-elaboration.
Do not express emotions or opinions."""

    @property
    def agent_name(self) -> str:
        return f"SimpleAgent-{self.llm_config.model_name}".replace("/", "_")

    def make(self, action_set: list[ActionSchema] | None = None, **kwargs) -> "SimpleAgent":
        return SimpleAgent(config=self, tools=action_set or [])


class SimpleAgent(Agent):
    name: str = "simple_agent"
    description: str = "Simple agent loop."
    input_content_types: list[str] = ["image/png", "image/jpeg", "text/plain", "application/json"]
    output_content_types: list[str] = ["application/json"]

    def __init__(self, config: SimpleAgentConfig, tools: list[ActionSchema]):
        super().__init__(config)
        self.llm = config.llm_config.make()
        self.token_counter = config.llm_config.make_counter()
        self.tools: list[dict] = [tool.as_dict() for tool in tools]
        apply_description_overrides(self.tools, config.description_overrides)

        self.history: list[dict | Message] = []
        self._actions_cnt = 0

        self.max_completion_tokens = config.llm_config.max_completion_tokens
        self.max_model_len = config.llm_config.max_model_len

    def attach_recorder(self, recorder: "EventStreamer") -> None:
        super().attach_recorder(recorder)
        self.llm.attach_recorder(recorder)

    def step(self, obs: Observation) -> AgentOutput:
        if self.max_actions_reached():
            logger.info("Max actions reached, issuing STOP action.")
            return AgentOutput(actions=[Action(id="stop", name=STOP_ACTION.name, arguments={})])

        # reset in case it was modified in previous steps
        self.llm.config.max_completion_tokens = self.max_completion_tokens  
        
        self.history += obs.to_llm_messages()
        messages = self._build_prompt_messages()
        prompt = Prompt(messages=messages, tools=self.tools)
        prompt_tokens = self.token_counter(messages=messages)

        remaining = self.max_model_len - prompt_tokens
        if remaining < self.max_completion_tokens:
            logger.warning(
                "capping max_completion_tokens from %d to %d (prompt_len=%d, max_model_len=%d)",
                self.max_completion_tokens, remaining, prompt_tokens, self.max_model_len,
            )
            self.llm.config.max_completion_tokens = remaining

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
        self.history.append(llm_output)
        self._actions_cnt += 1
        return AgentOutput(actions=parse_actions(llm_output))

    def _build_prompt_messages(self) -> list[dict | Message]:
        messages: list[dict | Message] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.extend(self.history)
        return messages

    def max_actions_reached(self) -> bool:
        return self._actions_cnt >= self.config.max_actions