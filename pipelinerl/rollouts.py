from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Sequence
import numpy as np

class BaseMetrics(BaseModel):
    reward: float
    success: bool
    no_error: bool
    no_answer: bool
    # Allow domain-specific metrics (e.g., num_steps, num_python_calls, penalty)
    # to flow through RolloutResult model validation without being dropped.
    model_config = {"extra": "allow"}


class TrainingText(BaseModel):
    """
    Training text instance used to finetune a language model.

    Attributes:
        text (str): The full text of the training instance.
        n_predicted (int): The number of predicted tokens in the text.
        reward (float): The reward associated with the training instance. Defaults to 0.0.
        logprobs (List[float]): A list of log probabilities of the completion tokens from the assistant model.
        ref_logprobs (List[float]): A list of reference log probabilities of the completion tokens from the reference model.
        input_ids (List[int]): A list of token IDs representing the input text, including the prompt and the predicted tokens.
        labels (List[int]): A list of token IDs that are used as labels for training. The last n_predicted tokens are set to MASKED_TOKEN_ID.
        group_id (str, optional): ID of the group. It is used by the RL finetuning script to normalize rewards.
        finished (bool): Indicates whether the text is finished or not.
        prompt_tokens (int): The number of tokens in the prompt part of the text.
        output_tokens (int): The number of tokens in the output part of the text.
        visual_features (Optional[Dict[str, np.ndarray]]): Optional visual features for vision language models.
        metadata (dict): Additional metadata associated with the training text.
        prompt_text (str): Portion of the text that serves as the prompt (i.e., the text excluding the predicted tokens).
        output_text (str): Portion of the text that represents the predicted output (i.e., the last n_predicted tokens).
    """

    text: str
    n_predicted: int
    reward: float = 0.0
    logprobs: List[float] = Field(default_factory=list)
    ref_logprobs: List[float] = Field(default_factory=list)
    input_ids: List[int] = Field(default_factory=list)
    labels: List[int] = Field(default_factory=list)
    group_id: str | None = None
    finished: bool = False
    prompt_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)
    visual_features: Optional[Dict[str, np.ndarray]] = None  # For vision language models
    metadata: dict = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def prompt_text(self) -> str:
        return self.text[: -self.n_predicted]

    @property
    def output_text(self) -> str:
        return self.text[-self.n_predicted :]


class RolloutResult(BaseModel):
    training_texts: list[TrainingText]
    metrics: BaseMetrics
    latency: float
    # optional so fields that it can be filled later after RolloutResult is created
    model_version: int | None = None
    dataset_name: str | None = None
    group_id: str | None = None
    domain: str | None = None


@dataclass(frozen=True)
class TrainingTextSummary:
    prompt_tokens: list[int]
    output_tokens: list[int]
    overflow: bool
    num_turns: int


def apply_rollout_reward(training_texts: Sequence[TrainingText], reward: float) -> list[TrainingText]:
    texts = list(training_texts)
    for training_text in texts:
        training_text.reward = reward
    return texts


def rollout_has_overflow(training_texts: Sequence[TrainingText]) -> bool:
    return any(not training_text.finished for training_text in training_texts)


def summarize_training_texts(training_texts: Sequence[TrainingText]) -> TrainingTextSummary:
    texts = list(training_texts)
    return TrainingTextSummary(
        prompt_tokens=[training_text.prompt_tokens for training_text in texts],
        output_tokens=[training_text.output_tokens for training_text in texts],
        overflow=rollout_has_overflow(texts),
        num_turns=len(texts),
    )
