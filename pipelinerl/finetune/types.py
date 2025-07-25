from dataclasses import dataclass
from typing import Any, Dict, List, Literal, TypeAlias, Union

import torch
from pydantic import BaseModel, ConfigDict, field_validator

import numpy as np

ModelClass: TypeAlias = Literal["causal-language-modeling", "seq2seq-language-modeling", "vision2seq-language-modeling"]


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
    time_waiting_for_data: float = 0.0


class PipelineBatchEncoding(BaseModel):
    """Pydantic model for batch encoding with automatic tensor conversion."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # All fields are tensors after validation
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    labels: torch.LongTensor
    position_ids: torch.LongTensor | None = None  # Required when seq_packing=True
    
    rewards: torch.FloatTensor
    advantages: torch.FloatTensor
    ref_logprobs: torch.FloatTensor
    old_logprobs: torch.FloatTensor
    group_tokens: torch.FloatTensor
    num_labels: torch.FloatTensor 
    overflow: torch.FloatTensor
    
    model_version: int
    sentinel: bool = False
    padding: int = 0 # Padding to make the batch size divisible by seq_parallel
    is_packed: bool = False 
    seq_boundaries: torch.IntTensor | None = None  # Required when seq_packing=True
    
    # Visual feature fields (optional, for multimodal models)
    pixel_values: torch.FloatTensor | None = None
    image_grid_thw: torch.LongTensor | None = None
    
    @field_validator('input_ids', 'attention_mask', 'labels', 'position_ids', 'image_grid_thw', mode='before')
    @classmethod
    def convert_to_long_tensor(cls, v: List[int] | torch.Tensor | None) -> torch.LongTensor | None:
        """Handle initialization of long tensors from different types."""
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return v.long()
        if isinstance(v, list) or isinstance(v, np.ndarray):
            return torch.tensor(v, dtype=torch.long)
        raise ValueError(f"Unsupported type for long tensor: {type(v)}")
    
    @field_validator('seq_boundaries', mode='before')
    @classmethod
    def convert_to_int_tensor(cls, v: List[int] | torch.Tensor | None) -> torch.IntTensor | None:
        """Convert lists to int tensors."""
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return v.int() # type: ignore
        return torch.tensor(v, dtype=torch.int)
    
    # TODO: am i needed?
    @field_validator('rewards', 'advantages', 'ref_logprobs', 'old_logprobs', 'group_tokens', 'num_labels', 'overflow', 'pixel_values', mode='before')
    @classmethod
    def convert_to_float_tensor(cls, v: List[float] | torch.Tensor | None) -> torch.FloatTensor | None:
        """Handle initialization of float tensors from different types."""
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return v.float()
        if isinstance(v, list) or isinstance(v, np.ndarray):
            return torch.tensor(v, dtype=torch.float)
        raise ValueError(f"Unsupported type for float tensor: {type(v)}")
    
    def to_device(self, device: Union[str, torch.device]) -> 'PipelineBatchEncoding':
        """Move all tensors to the specified device and return updated instance."""
        for field_name in self.model_fields:
            field_value = getattr(self, field_name)
            if isinstance(field_value, torch.Tensor):
                setattr(self, field_name, field_value.to(device))
        return self
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], **defaults) -> 'PipelineBatchEncoding':
        """Create from dictionary, filling in missing required fields with defaults."""
        # Merge defaults with data
        merged = {**defaults, **data}
        
        # Extract only known fields for the model
        model_fields = {}
        extra_fields = {}
        
        for key, value in merged.items():
            if key in cls.model_fields:
                model_fields[key] = value
            else:
                extra_fields[key] = value
        
        # Create instance with model fields
        instance = cls(**model_fields)
        
        # Add extra fields
        for key, value in extra_fields.items():
            instance.model_extra[key] = value
            
        return instance

    def make_slices(self, num_slices: int) -> list['PipelineBatchEncoding']:
        # Compute slice boundaries
        if self.position_ids is None or self.input_ids.shape[0] > 1:
            raise ValueError("Cannot a batch that is not properly packed")
        if self.input_ids.shape[1] < num_slices:
            raise ValueError(f"Cannot slice batch of size {self.input_ids.shape[1]} into {num_slices} slices")
        if self.input_ids.shape[1] % num_slices != 0:
            raise ValueError(f"Sequence length {self.input_ids.shape[1]} is not divisible by number of slices {num_slices}")
        bs = [i * len(self.input_ids[0]) // num_slices for i in range(num_slices + 1)]
        slices = []
        for i in range(num_slices):
            result = {
                # [1, L] tensors
                "input_ids": self.input_ids[:, bs[i]:bs[i + 1]],
                "attention_mask": self.attention_mask[:, bs[i]:bs[i + 1]],
                "labels": self.labels[:, bs[i]:bs[i + 1]],
                "position_ids": self.position_ids[:, bs[i]:bs[i + 1]] if self.position_ids is not None else None,
                "rewards": self.rewards[:, bs[i]:bs[i + 1]],
                "advantages": self.advantages[:, bs[i]:bs[i + 1]],
                "ref_logprobs": self.ref_logprobs[:, bs[i]:bs[i + 1]],
                "old_logprobs": self.old_logprobs[:, bs[i]:bs[i + 1]],
                "group_tokens": self.group_tokens[:, bs[i]:bs[i + 1]],
                "overflow": self.overflow[:, bs[i]:bs[i + 1]],
                "num_labels": self.num_labels[:, bs[i]:bs[i + 1]],
                # metadata
                "model_version": self.model_version,
                "sentinel": self.sentinel,
                "is_packed": self.is_packed,
                "padding": self.padding,
                "seq_boundaries": self.seq_boundaries,
                "pixel_values": self.pixel_values, 
                "image_grid_thw": self.image_grid_thw
            }
            slices.append(PipelineBatchEncoding(**result))
        return slices
        

