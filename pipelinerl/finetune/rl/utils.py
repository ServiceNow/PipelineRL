from typing import List, Optional

import numpy as np
import torch
from datasets import Dataset


def aggregate_rl_stats(rl_stats: dict, num_samples: int):
    avg_rl_stats: dict[str, float] = {}
    for k, v in rl_stats.items():
        if "min" in k:
            op = torch.min
        elif "max" in k:
            op = torch.max
        elif "loss" in k:
            op = torch.sum
        elif "sum" in k:
            op = torch.sum
        else:
            op = lambda x: torch.sum(x) / num_samples
        avg_rl_stats["rl/" + k] = op(torch.Tensor(v)).item()
    return avg_rl_stats


def mask_sum(values: torch.Tensor, mask: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
    """Compute sum of tensor with a masked values."""
    if axis is not None:
        return (values * mask).nan_to_num(0).sum(axis=axis)  # type: ignore
    else:
        return (values * mask).nan_to_num(0).sum()


def mask_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).nan_to_num(0).sum(axis=axis) / mask.sum(axis=axis)  # type: ignore
    else:
        return (values * mask).nan_to_num(0).sum() / mask.sum()


def mean_sum(values: torch.Tensor, masks: torch.Tensor, segments: list | None):
    """
    Compute mean-sum of values with masking, handling both packed and unpacked sequences.

    Args:
        values (torch.Tensor): Input tensor of values to aggregate
        masks (torch.Tensor): Boolean mask tensor indicating valid positions
        segments (list | None): List of (start, end) tuples for packed sequences, or None for unpacked

    Returns:
        torch.Tensor: Mean-sum of masked values, computed differently for packed vs unpacked sequences:
            - For packed (segments provided): Computes mean within each segment then sums across segments
            - For unpacked (no segments): Computes masked mean across all values then sums
    """
    is_sentinel_batch = values.shape[-1] == 1  # sentinel batch
    if segments and not is_sentinel_batch:
        # the values are seq packed, we drop the first dimension
        assert values.shape[0] == 1, "seq packed samples must have dimension 0 of 1"
        masked_sums = torch.stack([mask_sum(values[0, start:end], masks[0, start:end]) for start, end in segments])
        masked_counts = torch.stack([masks[0, start:end].sum() for start, end in segments])
        return (masked_sums / masked_counts).sum()
    else:
        return mask_mean(values, masks, -1).sum()


def sum_sum(values: torch.Tensor, masks: torch.Tensor, segments: list | None):
    """
    Compute sum-sum of values with masking, handling both packed and unpacked sequences.

    Args:
        values (torch.Tensor): Input tensor of values to aggregate
        masks (torch.Tensor): Boolean mask tensor indicating valid positions
        segments (list | None): List of (start, end) tuples for packed sequences, or None for unpacked

    Returns:
        torch.Tensor: Sum-sum of masked values, computed differently for packed vs unpacked sequences:
            - For packed (segments provided): Computes sum within each segment then sums across segments
            - For unpacked (no segments): Computes masked sum across all values
    """
    is_sentinel_batch = values.shape[-1] == 1  # sentinel batch
    if segments and not is_sentinel_batch:
        # the values are seq packed, we drop the first dimension
        assert values.shape[0] == 1, "seq packed samples must have dimension 0 of 1"
        masked_sums = torch.stack([mask_sum(values[0, start:end], masks[0, start:end]) for start, end in segments])
        return (masked_sums).sum()
    else:
        return mask_sum(values, masks)


def replace_dataset_column(dataset: Dataset, column_name: str, new_column: List[List[float]]) -> Dataset:
    """
    Replace a column in the dataset with a new column.
    """
    if column_name in dataset.features:
        dataset = dataset.map(remove_columns=[column_name])
    dataset = dataset.add_column(name=column_name, column=new_column)  # type: ignore

    return dataset
