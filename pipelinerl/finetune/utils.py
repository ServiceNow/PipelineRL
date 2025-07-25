import torch
from pydantic import BaseModel
import logging
from pipelinerl.finetune.context import get_accelerator
from pipelinerl.finetune.types import PipelineBatchEncoding

logger = logging.getLogger(__name__)


#TODO: why do we need VersionedTensors?
class VersionedTensors(BaseModel):
    tensors: dict
    model_version: int


def create_sentinel_batch(device, tokenizer=None, model_version=0) -> PipelineBatchEncoding:
    """
    Create a sentinel batch that matches the format expected by rl_step and works with sequence packing.
    The batch will have valid tokens for loss calculation but will be marked as sentinel to ensure zero loss contribution.
    """

    # get special tokens, defaulting to EOS token or generic IDs if not available
    eos_token_id = getattr(tokenizer, "eos_token_id", 2) if tokenizer else 2
    length = 8

    # create the minimal tensors needed
    input_ids = [eos_token_id] * length
    labels = [-100] * length
    attention_mask = [1] * length 
    position_ids = list(range(length))

    # Prepare fields for dummy values (only needed for reward, advantages, etc.)
    zeros = [0.0] * length
    ones = [1.0] * length

    sentinel_batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long).reshape(1, -1),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long).reshape(1, -1),
        "labels": torch.tensor(labels, dtype=torch.long).reshape(1, -1),
        "position_ids": torch.tensor(position_ids, dtype=torch.long).reshape(1, -1),
        "rewards": torch.tensor(zeros, dtype=torch.float).reshape(1, -1),
        "advantages": torch.tensor(zeros, dtype=torch.float).reshape(1, -1),
        "ref_logprobs": torch.tensor(zeros, dtype=torch.float).reshape(1, -1),
        "old_logprobs": torch.tensor(zeros, dtype=torch.float).reshape(1, -1),
        "group_tokens": torch.tensor(ones, dtype=torch.float).reshape(1, -1),
        "num_labels": torch.tensor(ones, dtype=torch.float).reshape(1, -1),
        "overflow": torch.tensor(zeros, dtype=torch.float).reshape(1, -1),
        "seq_boundaries": torch.tensor([0, length], dtype=torch.int)
    }

    # Add model_version and sentinel flag to match the expected format
    sentinel_batch["model_version"] = model_version
    sentinel_batch["sentinel"] = True
    sentinel_batch["is_packed"] = True 

    return PipelineBatchEncoding(**sentinel_batch)


def create_sentinel_example(n_tokens: int, tokenizer=None, model_version=0) -> dict:
    eos_token_id = tokenizer.eos_token_id # type: ignore
    example = {
        "input_ids": n_tokens * [eos_token_id],
        "attention_mask": n_tokens * [1],
        "labels": n_tokens * [-100],  # -100 for ignored labels in loss calculation
        "position_ids": list(range(n_tokens)),
        "rewards": n_tokens * [0.0],
        "advantages": n_tokens * [0.0],
        "ref_logprobs": n_tokens * [0.0],
        "old_logprobs": n_tokens * [0.0],
        "group_tokens": n_tokens * [1.0],
        "num_labels": n_tokens * [1.0], 
        "overflow": n_tokens * [0.0],
        "model_version": model_version,
    }
    return example
