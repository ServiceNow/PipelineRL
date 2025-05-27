from dataclasses import dataclass
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

ModelClass: TypeAlias = Literal["causal-language-modeling", "seq2seq-language-modeling"]


class DataPartArgs(BaseModel):
    path: str
    files: list[str] = ["*.jsonl"]
    weight: float = 1.0
    model_config = ConfigDict(frozen=True)


class DataArgs(BaseModel):
    data_parts_train: list[DataPartArgs]
    data_parts_valid: list[DataPartArgs] | None = None
    data_parts_dev: list[DataPartArgs] | None = None
    model_config = ConfigDict(frozen=True)


@dataclass
class TrainingMetrics:
    epoch: int = 0
    passes: int = 0
    completed_steps: int = 0
    samples: int = 0
    tokens: int = 0
    samples_too_old_to_queue: int = 0
    samples_too_old_to_train: int = 0
    last_broadcasted_version: int = 0
    train_loss: float = 1e9
    eval_loss: float = 1e9
    dev_loss: float = 1e9
    grad_norm: float = 0.0
    best_eval_loss: float = 1e9
    best_completed_steps: int = 0
    lr: float = 0.0
    max_batch_len: int = 0
    min_batch_len: int = int(1e9)
    time_waiting_for_data: float = 0.0
