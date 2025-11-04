"""Base orchestrator interface for DeepResearcher domain."""
from abc import ABC, abstractmethod
from typing import Dict, List
import aiohttp
from tapeagents.llms.trainable import TrainableLLM
from tapeagents.core import LLMCall


class BaseOrchestrator(ABC):
    
    def __init__(self, cfg, llm: TrainableLLM, tool_registry):
        self.cfg = cfg
        self.llm = llm
        self.tool_registry = tool_registry
        self.max_turns = cfg.actor.get("max_turns", 20)
    
    @abstractmethod
    async def execute(
        self,
        question: str,
        session: aiohttp.ClientSession
    ) -> Dict:
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        pass
