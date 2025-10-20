from typing import List, Optional

import numpy as np
import torch
from datasets import Dataset
from torch import distributed as dist

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
    """Compute mean of tensor with masked values, safely handling empty masks.

    Uses a clamped denominator so the result is 0 when the mask sums to 0,
    while keeping the computation connected to the graph via the numerator.
    """
    if axis is not None:
        num = (values * mask).nan_to_num(0).sum(axis=axis)  # type: ignore
        den = mask.sum(axis=axis).clamp(min=1).to(dtype=values.dtype)  # type: ignore
        return num / den
    else:
        num = (values * mask).nan_to_num(0).sum()
        den = mask.sum().clamp(min=1).to(dtype=values.dtype)
        return num / den


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



def per_segment_sums(
    segment_ids: torch.LongTensor,
    masks_shifted: torch.Tensor,
    log_ratio_new_old: torch.Tensor,
    advantages: torch.Tensor,
    seq_parallel_group=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Differentiable version of per-segment sums using scatter_add instead of bincount.

    Args:
        segment_ids: [1, L] integer segment id per token (packed batches). Should align with labels positions.
        masks_shifted: [1, L-1] boolean mask for valid (non -100) labels excluding first token.
        log_ratio_new_old: [1, L-1] tensor.
        advantages: [1, L-1] tensor.
        n_segments: total number of segments.
        seq_parallel_group: optional torch.distributed group for seq-parallel; if provided, results are all-reduced.

    Returns:
        (log_ratio_sum_per_seg, advantages_sum_per_seg, token_count_per_seg), each shaped [n_segments].
    """
    if segment_ids is None:
        raise ValueError("segment_ids must be provided for per-segment reductions")

    # Expect [1, L] -> align to shifted tensors [1, L-1]
    if segment_ids.dim() != 2 or segment_ids.shape[0] != 1:
        raise ValueError(f"Expected segment_ids of shape [1, L], got {tuple(segment_ids.shape)}")

    # Slice and unify device/dtypes
    # Always compute a consistent number of collectives across ranks to avoid NCCL deadlocks.
    # We cannot call seg.max() on empty tensors, so handle that path explicitly while still
    # participating in all necessary collectives.
    seg = segment_ids[:, 1:].contiguous().squeeze(0).to(dtype=torch.long, device=log_ratio_new_old.device)
    seg_is_empty = seg.numel() == 0

    # Determine n_segments. For distributed, we first all-reduce the local max (or -1 for empty)
    # so all ranks agree on a global n_segments. This preserves the collective call even for empty ranks.
    if seq_parallel_group is None or not dist.is_available() or not dist.is_initialized():
        if seg_is_empty:
            n_segments = 0
        else:
            n_segments = int(seg.max().to(torch.int64).item()) + 1
    else:
        local_max = torch.tensor(-1, dtype=torch.int64, device=log_ratio_new_old.device)
        if not seg_is_empty:
            local_max = seg.max().to(torch.int64)
        global_max = local_max.clone()
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=seq_parallel_group)
        n_segments = int(global_max.item()) + 1  # will be 0 if all ranks are empty

    mask = masks_shifted[:, :seg.numel()].contiguous().squeeze(0)
    lrn  = log_ratio_new_old[:, :seg.numel()].contiguous().squeeze(0)
    adv  = advantages[:, :seg.numel()].contiguous().squeeze(0)

    # Put everything on same device
    device = lrn.device
    seg  = seg.to(device=device)
    mask = mask.to(device=device, dtype=lrn.dtype)  # float mask is fine for weighting
    adv  = adv.to(device=device, dtype=lrn.dtype)
    # lrn already on device

    # Consider only VALID tokens before indexing
    # Important: this prevents out-of-bounds reads from indices you intended to ignore
    valid = (mask != 0)
    if valid.ndim != 1 or valid.shape[0] != seg.shape[0]:
        raise ValueError("Mask shape mismatch after alignment with segment_ids.")

    if (not seg_is_empty) and valid.any():
        seg_v  = seg[valid]                 # indices actually used
        w_v    = mask[valid]                # weights for counts
        lrn_v  = lrn[valid]
        adv_v  = adv[valid]

        # Range check BEFORE scatter to produce a clean error
        smin = int(seg_v.min())
        smax = int(seg_v.max())
        if smin < 0 or smax >= n_segments:
            raise IndexError(
                f"per_segment_sums_diff_safe: segment index out of bounds. "
                f"min(seg)={smin}, max(seg)={smax}, n_segments={n_segments}. "
                "Likely causes: (1) n_segments too small (compute after packing), "
                "(2) off-by-one when dropping the first token, "
                "(3) segment ids are global across workers but you passed local n_segments."
            )

        # Allocate outputs
        token_count = torch.zeros(n_segments, dtype=lrn.dtype, device=device)
        lrn_sum     = torch.zeros(n_segments, dtype=lrn.dtype, device=device)
        adv_sum     = torch.zeros(n_segments, dtype=lrn.dtype, device=device)

        # index_add_ is equivalent to scatter_add_ for 1D reductions and can be clearer
        token_count.index_add_(0, seg_v, w_v)
        lrn_sum.index_add_(0, seg_v, lrn_v * w_v)
        adv_sum.index_add_(0, seg_v, adv_v * w_v)
    else:
        # No valid tokens: return zeros, but keep graph connectivity so downstream
        # losses still "require_grad" (DeepSpeed/PyTorch expect this).
        # Create zero scalars tied to inputs; gradients will be zero but the graph remains.
        zero_from_lrn = (lrn * 0).sum()  # requires_grad if lrn requires_grad
        zero_from_adv = (adv * 0).sum()  # requires_grad if adv requires_grad
        token_count = torch.zeros(n_segments, dtype=lrn.dtype, device=device)
        # Broadcastable add keeps autograd connection even when n_segments > 0
        lrn_sum = torch.zeros(n_segments, dtype=lrn.dtype, device=device) + zero_from_lrn
        adv_sum = torch.zeros(n_segments, dtype=lrn.dtype, device=device) + zero_from_adv

    # Optional all-reduce across sequence-parallel group. Ensure all ranks perform the same
    # number of collectives, even when n_segments == 0 locally (e.g., sentinel micro-batches).
    if seq_parallel_group is not None and dist.is_available() and dist.is_initialized():
        from torch.distributed.nn.functional import all_reduce
        if n_segments == 0:
            # Perform three dummy all-reduces to match the non-empty path (token_count, lrn_sum, adv_sum)
            dummy = torch.zeros(1, dtype=lrn.dtype, device=device)
            _ = all_reduce(dummy, op=dist.ReduceOp.SUM, group=seq_parallel_group)
            _ = all_reduce(dummy, op=dist.ReduceOp.SUM, group=seq_parallel_group)
            _ = all_reduce(dummy, op=dist.ReduceOp.SUM, group=seq_parallel_group)
        else:
            token_count = all_reduce(token_count, op=dist.ReduceOp.SUM, group=seq_parallel_group)
            lrn_sum = all_reduce(lrn_sum, op=dist.ReduceOp.SUM, group=seq_parallel_group)
            adv_sum = all_reduce(adv_sum, op=dist.ReduceOp.SUM, group=seq_parallel_group)

    return lrn_sum, adv_sum, token_count
