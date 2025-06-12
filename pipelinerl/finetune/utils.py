import torch
from pydantic import BaseModel
import logging
from pipelinerl.finetune.context import get_accelerator

logger = logging.getLogger(__name__)


class VersionedTensors(BaseModel):
    tensors: dict
    model_version: int


def create_sentinel_batch(device, tokenizer=None, model_version=0) -> VersionedTensors:
    """
    Create a sentinel batch that matches the format expected by rl_step and works with sequence packing.
    The batch will have valid tokens for loss calculation but will be marked as sentinel to ensure zero loss contribution.
    """

    # get special tokens, defaulting to EOS token or generic IDs if not available
    eos_token_id = getattr(tokenizer, "eos_token_id", 2) if tokenizer else 2

    # for start token, try BOS first, fall back to EOS if BOS is None/not available
    bos_token_id = getattr(tokenizer, "bos_token_id", None) if tokenizer else None
    if bos_token_id is None:
        bos_token_id = eos_token_id  # Use EOS as start token if BOS not available

    # create the minimal tensors needed
    input_ids = [bos_token_id, eos_token_id]
    attention_mask = [1, 1]  # both tokens are attended to
    position_ids = [0, 1]  # valid positions for both tokens

    # Prepare fields for dummy values (only needed for reward, advantages, etc.)
    zeros = [0.0] * 2
    ones = [1.0] * 2

    sentinel_batch = {
        "input_ids": torch.tensor([bos_token_id, eos_token_id], dtype=torch.long).reshape(1, -1),
        "attention_mask": torch.tensor([1, 1], dtype=torch.long).reshape(1, -1),
        "labels": torch.tensor([-100, eos_token_id], dtype=torch.long).reshape(1, -1),
        "position_ids": torch.tensor([0, 1], dtype=torch.long).reshape(1, -1),
        "rewards": torch.tensor(zeros, dtype=torch.float).reshape(1, -1),
        "advantages": torch.tensor(zeros, dtype=torch.float).reshape(1, -1),
        "ref_logprobs": torch.tensor(zeros, dtype=torch.float).reshape(1, -1),
        "old_logprobs": torch.tensor(zeros, dtype=torch.float).reshape(1, -1),
        "group_tokens": torch.tensor(ones, dtype=torch.float).reshape(1, -1),
        "overflow": torch.tensor(zeros, dtype=torch.float).reshape(1, -1),
    }

    sentinel_batch = {
        k: (v.to(get_accelerator().device) if isinstance(v, torch.Tensor) else v) for k, v in sentinel_batch.items()
    }

    return VersionedTensors(tensors=sentinel_batch, model_version=model_version)
