from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pipelinerl.rollouts import RolloutRequest, RolloutResult


@dataclass
class RolloutWorkerContext:
    worker_id: int
    worker_name: str
    extras: dict[str, Any] = field(default_factory=dict)

class RolloutDataset:
    context_extras: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> dict[str, Any]:
        raise NotImplementedError


class RolloutWorker(ABC):
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.worker_name = str(config.get("worker_name", self.__class__.__name__))
        self.ready = False

    def setup(self, context: RolloutWorkerContext) -> None:
        self.context = context
        self.worker_name = context.worker_name
        self.ready = True

    @abstractmethod
    def generate(self, request: RolloutRequest) -> RolloutResult:
        """Generate one rollout fragment/batch/episode."""

    def health(self) -> bool:
        return self.ready

    def close(self) -> None:
        """Best-effort cleanup."""
