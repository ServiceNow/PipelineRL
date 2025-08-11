from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import numpy as np

class BaseMetrics(BaseModel):
    reward: float
    success: bool
    no_error: bool
    no_answer: bool


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
