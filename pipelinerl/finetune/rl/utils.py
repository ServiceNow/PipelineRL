from typing import List, Optional

import numpy as np
import torch
from datasets import Dataset

def get_avg_rl_stats(rl_stats: dict, num_samples: int):
    avg_rl_stats: dict[str, float] = {}
    for k, v in rl_stats.items():
        if "min" in k:
            op = torch.min
        elif "max" in k:
            op = torch.max
        elif k == "loss": # loss is already normalized
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
    is_sentinel_batch = (values.shape[-1] == 1) # sentinel batch
    if segments and not is_sentinel_batch:
        # the values are seq packed, we drop the first dimension
        assert values.shape[0] == 1, "seq packed samples must have dimension 0 of 1"
        masked_sums = torch.stack([
                mask_sum(values[0, start:end], masks[0, start:end])
                for start, end in segments
            ])  
        masked_counts = torch.stack([
                masks[0, start:end].sum()
                for start, end in segments
            ])  
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
    is_sentinel_batch = (values.shape[-1] == 1) # sentinel batch
    if segments and not is_sentinel_batch:
        # the values are seq packed, we drop the first dimension
        assert values.shape[0] == 1, "seq packed samples must have dimension 0 of 1"
        masked_sums = torch.stack([
                mask_sum(values[0, start:end], masks[0, start:end])
                for start, end in segments
            ])  
        return (masked_sums).sum()
    else:
        return mask_sum(values, masks)

def calculate_rewards_with_implicit_kl(row, reward_minus_kl_coef):
    """
    Calculate reward with implicit KL penalty.

    Args:
        row (dict): Dictionary containing reward and log probability data with keys:

            - reward: Base reward value
            - old_logprobs: Log probabilities from old policy
            - ref_logprobs: Reference log probabilities
        reward_minus_kl_coef (float): Coefficient for implicit KL penalty term

    Returns:
        (float): Reward value adjusted by implicit KL penalty, calculated as:
            reward - reward_minus_kl_coef * KL(ref||old)

        The KL divergence is approximated using the Schulman approximation:
            KL ≈ exp(log_ratio) - log_ratio - 1
            where log_ratio = ref_logprobs - old_logprobs
    """
    rewards = row["rewards"]
    old_logprobs = row["old_logprobs"]
    ref_logprobs = row["ref_logprobs"]
    log_ratio_ref_old = ref_logprobs - old_logprobs
    kl = (np.exp(log_ratio_ref_old) - log_ratio_ref_old - 1).sum()  # Schulman KL approx
    return [reward - reward_minus_kl_coef * kl for reward in rewards]


def calculate_advantage(row, divide_advantage_by_std):
    """
    Calculate advantage values for a row of data.

    Args:
        row (dict): Dictionary containing rewards and statistics with keys:

            - rewards: List of reward values
            - reward_mean: Mean reward value
            - reward_std: Standard deviation of rewards

    Returns:
       (list[float]): List of advantage values calculated as (reward - mean)/(std + eps)
            where eps=1e-4 is added for numerical stability
    """
    rewards = row["rewards"]
    mean = row["reward_mean"]
    std = row["reward_std"]
    if divide_advantage_by_std:
        advantages = [(reward - mean) / (np.nan_to_num(std) + 1e-4) for reward in rewards]
    else:
        advantages = [(reward - mean) for reward in rewards]
    return advantages


def replace_dataset_column(dataset: Dataset, column_name: str, new_column: List[List[float]]) -> Dataset:
    """
    Replace a column in the dataset with a new column.
    """
    if column_name in dataset.features:
        dataset = dataset.map(remove_columns=[column_name])
    dataset = dataset.add_column(name=column_name, column=new_column)  # type: ignore

    return dataset
