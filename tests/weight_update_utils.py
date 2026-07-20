"""Utility functions for weight update testing."""

import torch
from pipelinerl.finetune_loop import WeightUpdateRequest, ParameterInfo


def dtype_to_string(dtype: torch.dtype) -> str:
    """Convert torch dtype to string format expected by vLLM.

    Args:
        dtype: PyTorch dtype

    Returns:
        String representation (e.g., 'bfloat16', 'float32')
    """
    return str(dtype).replace("torch.", "")


def create_weight_update_request_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    version: int = 0,
) -> WeightUpdateRequest:
    """Create a WeightUpdateRequest from a model state dict.

    This helper function is useful for testing and for creating weight
    update requests from saved model checkpoints.

    Args:
        state_dict: Dictionary mapping parameter names to tensors
        version: Version number for this weight update

    Returns:
        WeightUpdateRequest object ready to be sent to workers

    Example:
        >>> state_dict = torch.load('model.pt')
        >>> request = create_weight_update_request_from_state_dict(state_dict, version=1)
        >>> # Send request to vLLM server via HTTP endpoint
    """
    parameters_info = []
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            parameters_info.append(
                ParameterInfo(
                    name=name,
                    shape=list(tensor.shape),
                    dtype=dtype_to_string(tensor.dtype),
                )
            )

    return WeightUpdateRequest(version=version, parameters_info=parameters_info)
