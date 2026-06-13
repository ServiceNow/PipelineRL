from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pipelinerl.rollouts import RolloutRequest, RolloutResult


@dataclass
class RolloutWorkerContext:
    worker_id: int
    worker_name: str
    llm_router: Any | None = None
    extras: dict[str, Any] = field(default_factory=dict)

class RolloutDataset:
    context_extras: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> dict[str, Any]:
        raise NotImplementedError

    def affinity_key(self, item: dict[str, Any]) -> str | None:
        raise NotImplementedError


class RolloutWorker(ABC):
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.worker_name = str(config.get("worker_name", self.__class__.__name__))
        self.llm: dict[str, Any] | None = config.get("llm")
        self.llm_router: Any | None = None
        self.llm_tokenizer: Any | None = None
        self.ready = False

    def setup(self, context: RolloutWorkerContext) -> None:
        self.context = context
        self.worker_name = context.worker_name
        self.llm_router = context.llm_router
        if self.llm is not None:
            self.load_tokenizer()
        self.ready = True

    def load_tokenizer(self) -> None:
        if self.llm is None:
            return
        from pipelinerl.llm import TrainableLLM

        temp_llm = TrainableLLM(**self.llm)
        temp_llm.load_tokenizer()
        self.llm_tokenizer = temp_llm.tokenizer

    @abstractmethod
    def generate(self, request: RolloutRequest) -> RolloutResult:
        """Generate one rollout fragment/batch/episode."""

    def health(self) -> bool:
        return self.ready

    def close(self) -> None:
        """Best-effort cleanup."""
