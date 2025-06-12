
from pydantic import BaseModel

from pipelinerl.finetune.data import MASKED_TOKEN_ID
from tapeagents.core import LLMCall, TrainingText
from tapeagents.llms.trainable import TrainableLLM

class BaseMetrics(BaseModel):
    reward: float
    success: bool
    no_error: bool
    no_answer: bool
    overflow: bool
    prompt_tokens: list[int]
    output_tokens: list[int]

class RolloutResult(BaseModel):
    training_texts: list[TrainingText]
    metrics: BaseMetrics
    latency: float
    # optional so fields that it can be filled later after RolloutResult is created
    model_version: int | None = None
    dataset_name: str | None = None
    group_id: str | None = None

