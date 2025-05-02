import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Literal
from pydantic import BaseModel, Field

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import BatchEncoding, PreTrainedModel

from .utils import (
    calculate_advantage,
    calculate_rewards_with_implicit_kl,
    sum_sum,
    mean_sum,
    replace_dataset_column,
)

# FIXME: remove a warnings, but might be worth investigating
os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = logging.getLogger(__name__)

RL_DATA_COLUMNS = [
    "reward",
    "overflow",
    "group_tokens",
    "rewards",
    "advantages",
    "old_logprobs",
    "ref_logprobs",
    "overflow",
    "group_tokens",
]


class RLConfig(BaseModel):
    algo: str = Field(default="grpo", description="Algorithm to use for RL", choices=["grpo", "reinforce"])
    use_advantages: bool = Field(
        default=True,
        description="Use advantages instead of rewards to compute the loss",
    )
    epsilon: float = Field(default=0.2, description="Clip parameter for the ration of log probs")
    batch_size: int = Field(default=0, description="Batch size is required for normalization")
    reward_minus_kl_coef: float = Field(
        default=0.0,
        # https://arxiv.org/abs/2402.14740
        description="Implicit KL coefficient similar to the RLOO paper",
    )
    kl_coef: float = Field(
        default=0.1,
        description="KL penalty coefficient with reference policy",
    )
    final_kl_coef: float = Field(
        default=0.1,
        description="Final KL penalty coefficient value",
    )
    entropy_bonus: float = Field(
        default=0.0,
        description="Entropy bonus coefficient",
    )
    final_entropy_bonus: float = Field(
        default=0.0,
        description="Final entropy bonus value",
    ) 
    relu_log_p_weights: bool = Field(
        default=False,
        description="ReLU the weights before updating the model",
    )
    clamp_log_ratio_ref_new_value: float = Field(
        default=10,
        description="Clamp the log ratio ref new value",
    )
    overlong_filtering: bool = Field(
        default=False,
        description="Filter out sequence that do not have eos_token_id"
    )
    group_normalization: bool = Field(
        default=False,
        description="Divide the weight of each sequence by the (average) number of tokens in the group"
    )
    temperature: float = Field(
        default=1.0,
        description="Temperature for the training log probs",
    )
    rl_num_proc: int = Field(
        default=8,
        description="Number of processes to use for dataset map operations",
    )

def make_rl_data_callback(args, current_dir, rl_config, model):
    if rl_config:
        populate_rl_data_ = partial(
            populate_rl_data,
            config=rl_config,
        )
    else:
        populate_rl_data_ = None
    return populate_rl_data_


def linear_decay_coef(current_step: int, max_step: int, initial_coef: float, final_coef: float) -> float:
    """
    Linearly decay the coefficient from initial to final value over the course of training.

    Args:
        current_step (int): Current step in the training
        max_step (int): Maximum number of steps in the training
        initial_coef (float): Initial coefficient value
        final_coef (float): Final coefficient value
    
    Returns:
        float: Linearly decayed coefficient value
    
    """
    return initial_coef + (final_coef - initial_coef) * current_step / max_step


def rl_step(
    model: PreTrainedModel,
    batch: dict,
    current_step: int,
    max_step: int,
    config: RLConfig
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Perform a single RL step on the model using the given batch and config.
    Handles both packed and unpacked sequences.

    Args:
        model (PreTrainedModel): The model to train
        batch (dict): Batch of data containing rewards, advantages, masks, input_ids etc.
        current_step (int): Current training step
        max_step (int): Maximum number of training steps
        config (RLConfig): Configuration for the RL training

    Returns:
        tuple[torch.Tensor, dict[str, float]]: Loss tensor and metrics dictionary
    """
    # pre-compute masks
    masks = batch["labels"] != -100
    masks_shifted = masks[:, 1:]

    # if we have position_ids, we are packing
    is_packed = "position_ids" in batch
    if is_packed:
        position_ids = batch["position_ids"][0]
        # sequence boundary computation
        sequence_starts = torch.where(position_ids == 0)[0]
        seq_boundaries = torch.cat([sequence_starts, torch.tensor([position_ids.shape[0]], device=position_ids.device)])
        num_sequences = len(sequence_starts)

        # ensure we have valid sequence boundaries
        assert num_sequences > 0, "No sequences found in packed batch"
        assert seq_boundaries[-1] == position_ids.shape[0], "Sequence boundaries don't match input length"

        # pre-compute segment boundaries
        segments = list(zip(seq_boundaries[:-1], seq_boundaries[1:]))
    else:
        num_sequences = masks.shape[0]
        segments = None

    model_inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"],
    }
    if is_packed:
        model_inputs["position_ids"] = batch["position_ids"]

    outputs = model(**model_inputs)

    # compute log probs and entropy
    logits = outputs.logits[:, :-1, :]
    logits = logits / config.temperature
    logprobs = F.log_softmax(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * logprobs).sum(dim=-1)
    del logits, probs

    # get log probs for actual tokens
    new_logprobs = torch.gather(
        logprobs,
        dim=2,
        index=batch["input_ids"][:, 1:].unsqueeze(2)
    ).squeeze(2)
    assert torch.isfinite(new_logprobs).all(), f"new_logprobs is not finite: {new_logprobs}"
    del logprobs

    # get shifted values and compute ratios
    rewards = batch.pop("rewards")[:, 1:]
    advantages = batch.pop("advantages")[:, 1:]
    ref_logprobs = batch["ref_logprobs"][:, 1:]
    old_logprobs = batch["old_logprobs"][:, 1:]
    group_tokens = batch["group_tokens"][:, 1:]
    overflow = batch["overflow"][:, 1:]

    if config.group_normalization:
        tokens_weights = torch.ones_like(group_tokens) / group_tokens
    else:
        tokens_weights = torch.ones_like(group_tokens) / config.batch_size 
    
    if config.overlong_filtering:
        # filter out sequences that do not have eos_token_id
        overflow = torch.tensor(overflow, device=overflow.device)
        tokens_weights = tokens_weights * (1 - overflow)
    
    assert new_logprobs.shape == ref_logprobs.shape

    log_ratio_new_old = new_logprobs - old_logprobs
    ratio_new_old = torch.exp(log_ratio_new_old)
    log_ratio_ref_new = ref_logprobs - new_logprobs
    assert torch.isfinite(log_ratio_ref_new).all(), f"log_ratio_ref_new is not finite: {log_ratio_ref_new}"
    # compute weights and KL divergence
    log_p_weights = advantages if config.use_advantages else rewards
    if config.relu_log_p_weights:
        log_p_weights = torch.clamp(log_p_weights, min=0)

    clamp_log_ratio_ref_new_indicators = torch.abs(log_ratio_ref_new) > config.clamp_log_ratio_ref_new_value


    log_ratio_ref_new_clamp = torch.clamp(
        log_ratio_ref_new,
        min=-config.clamp_log_ratio_ref_new_value,
        max=config.clamp_log_ratio_ref_new_value,
    )

    approx_kl = torch.exp(log_ratio_ref_new_clamp) - log_ratio_ref_new_clamp - 1  # Schulman KL approx

    assert torch.isfinite(approx_kl).all(), f"approx_kl is not finite: {approx_kl}"
    entropy_bonus_coef = linear_decay_coef(current_step, max_step, config.entropy_bonus, config.final_entropy_bonus)
    kl_coef = linear_decay_coef(current_step, max_step, config.kl_coef, config.final_kl_coef)

    # compute algorithm-specific losses
    match config.algo:
        case "grpo":
            surr1 = ratio_new_old * log_p_weights
            clamped_ratio = torch.clamp(ratio_new_old, 1 - config.epsilon, 1 + config.epsilon)
            clamp_log_ratio_new_old_indicators = (clamped_ratio != ratio_new_old)
            surr2 = clamped_ratio * log_p_weights
            policy_loss = torch.min(surr1, surr2)
        case "reinforce":
            surr1 = torch.zeros_like(ratio_new_old)
            surr2 = torch.zeros_like(ratio_new_old)
            clamp_log_ratio_new_old_indicators = torch.zeros_like(ratio_new_old)
            policy_loss = new_logprobs * log_p_weights
        case _:
            raise ValueError(f"Unknown algorithm {config.algo}")

    # combine loss components
    loss = policy_loss - kl_coef * approx_kl + entropy_bonus_coef * entropy  # 1 x (BxL) x 1
    assert loss.shape == tokens_weights.shape, f"Loss shape {loss.shape} does not match example weights shape {tokens_weights.shape}"
    loss = loss * tokens_weights  # 1 x (BxL) x 1

    final_loss = -sum_sum(loss, masks_shifted, segments)

    # ensure loss is valid
    assert torch.isfinite(final_loss), f"Non-finite loss detected: {final_loss}"

    # All the stats are average then summed. They will be normalized by the number of sequences at the end of the step
    stats = {
        "loss": final_loss.item(),
        "max_loss": final_loss.item(),
        "min_loss": final_loss.item(),
        "reward": mean_sum(rewards, masks_shifted, segments).item(),
        "max_reward": rewards[masks_shifted].max().item(),
        "min_reward": rewards[masks_shifted].min().item(),
        "entropy": mean_sum(entropy, masks_shifted, segments).item(),
        "old_logprobs": mean_sum(old_logprobs, masks_shifted, segments).item(),
        "new_logprobs": mean_sum(new_logprobs, masks_shifted, segments).item(),
        "ref_logprobs": mean_sum(ref_logprobs, masks_shifted, segments).item(),
        "advantage": mean_sum(advantages, masks_shifted, segments).item(),
        "max_advantage": advantages[masks_shifted].max().item(),
        "min_advantage": advantages[masks_shifted].min().item(),
        "kl": mean_sum(approx_kl, masks_shifted, segments).item(),
        "max_kl": approx_kl[masks_shifted].max().item(),
        "min_kl": approx_kl[masks_shifted].min().item(),
        "policy_loss": mean_sum(policy_loss, masks_shifted, segments).item(),
        "surr1": mean_sum(surr1, masks_shifted, segments).item(),
        "surr2": mean_sum(surr2, masks_shifted, segments).item(),
        "ratio_new_old": mean_sum(ratio_new_old, masks_shifted, segments).item(),
        "ratio_ref_new": mean_sum(torch.exp(log_ratio_ref_new), masks_shifted, segments).item(),
        "ratio_ref_old": mean_sum(torch.exp(ref_logprobs - old_logprobs), masks_shifted, segments).item(),
        "clamp_log_ratio_ref_new_indicator": mean_sum(clamp_log_ratio_ref_new_indicators, masks_shifted, segments).item(),
        "clamp_log_ratio_new_old_indicator": mean_sum(clamp_log_ratio_new_old_indicators, masks_shifted, segments).item(),
        "num_nans": torch.isnan(loss).sum().item(),
        "token_weight": mean_sum(tokens_weights, masks_shifted, segments).item(),
        "kl_coef": num_sequences * kl_coef,
        "entropy_bonus_coef": num_sequences * entropy_bonus_coef,
    }

    return final_loss, stats


def _prepare_reward_columns(batch: dict[str, list], config: RLConfig) -> dict[str, list]:
    """
    First pass: compute per-example rewards and scalar rewards.
    
    Args:
        batch: dict-of-lists (HF 'batched' format)
        config: RLConfig instance
    
    Returns:
        batch with updated/added reward columns
    """
    rewards_out = []
    scalar_out = []
    
    if "rewards" not in batch and "reward" not in batch:
        raise ValueError("Missing both 'rewards' and 'reward' columns for RL preprocessing")
    
    seq_lengths = [len(ids) for ids in batch["input_ids"]]
    
    reward_minus_kl_coef = config.reward_minus_kl_coef
    
    for i, seq_len in enumerate(seq_lengths):
        if "rewards" in batch and len(batch["rewards"][i]) > 0:
            r_tok = batch["rewards"][i]
        else:
            scalar = float(batch["reward"][i])
            r_tok = [scalar] * seq_len
        
        if reward_minus_kl_coef > 0:
            old_lp = batch["old_logprobs"][i]
            ref_lp = batch["ref_logprobs"][i]
            
            if len(old_lp) != len(ref_lp):
                raise ValueError(f"old_logprobs length {len(old_lp)} doesn't match ref_logprobs length {len(ref_lp)}")
            
            old_lp_arr = np.asarray(old_lp, dtype=np.float32)
            ref_lp_arr = np.asarray(ref_lp, dtype=np.float32)
            log_ratio = ref_lp_arr - old_lp_arr
            kl = (np.exp(log_ratio) - log_ratio - 1).sum()
            
            r_tok = [float(r - reward_minus_kl_coef * kl) for r in r_tok]
        
        rewards_out.append(r_tok)
        scalar_out.append(float(np.mean(r_tok)))

    batch["rewards"] = rewards_out
    batch["reward"] = scalar_out
    return batch


def _finalise_rl_columns(
        batch: dict[str, list],
        indices: list[int],
        group_lookup: dict[str, np.ndarray],
        group_ids: np.ndarray,
        config: RLConfig,
        eos_token_id: int
    ) -> dict[str, list]:
    """
    Second pass: compute advantages and other per-token columns.
    
    Args:
        batch: dict-of-lists (HF 'batched' format)
        indices: indices provided by datasets.map with_indices=True
        group_lookup: dict with group-level stats (mean, std, avg_tok)
        group_ids: array with mapping from dataset indices to group indices
        config (RLConfig): Configuration for RL training
        eos_token_id: Token ID for end of sequence
    
    Returns:
        batch with all RL columns populated
    """
    out_adv, out_gt, out_ov, out_w = [], [], [], []
    
    # Pre-compute only what's needed without full array conversions
    seq_lengths = [len(ids) for ids in batch["input_ids"]]
    has_eos_list = [(eos_token_id in ids) for ids in batch["input_ids"]]
    
    rewards_arrays = [np.asarray(r, dtype=np.float32) for r in batch["rewards"]]
    
    batch_group_indices = group_ids[indices]
    
    for i, (old_lp, ref_lp, group_idx) in enumerate(zip(
        batch["old_logprobs"],
        batch["ref_logprobs"],
        batch_group_indices,
    )):
        L = seq_lengths[i]
        
        g_mean = float(group_lookup["mean"][group_idx])
        g_std = float(group_lookup["std"][group_idx])
        g_tok = float(group_lookup["avg_tok"][group_idx])
        
        r_tok_arr = rewards_arrays[i]
        adv = ((r_tok_arr - g_mean) / (g_std + 1e-4)).tolist()
        
        gt = [g_tok] * L
        
        pad_len = L - len(old_lp)
        if pad_len > 0:
            old_lp = [0.0] * pad_len + old_lp
            ref_lp = [0.0] * pad_len + ref_lp
        
        has_eos = has_eos_list[i]
        ov = [0.0 if has_eos else 1.0] * L
        w = [0.0] * L if (config.overlong_filtering and not has_eos) else [1.0] * L
        
        out_adv.append(adv)
        out_gt.append(gt)
        out_ov.append(ov)
        out_w.append(w)
        batch["old_logprobs"][i] = old_lp
        batch["ref_logprobs"][i] = ref_lp

    batch["advantages"] = out_adv
    batch["group_tokens"] = out_gt
    batch["overflow"] = out_ov
    batch["example_weight"] = out_w
    return batch


def update_rewards_and_advantages(dataset: Dataset, eos_token_id: int, config: RLConfig) -> Dataset:
    """
    Updates the advantages column in the given dataset based on reward statistics.
    Uses vectorized operations and parallel processing where possible.

    Args:
        dataset (Dataset): The input dataset containing rewards and placeholder advantages.
        eos_token_id (int): Token ID for end of sequence
        config (RLConfig): Configuration for RL training

    Returns:
        Dataset: The updated dataset with the updated advantages column.
    """
    logger.info("Computing rewards (pass 1/2)")
    _NUM_PROC = max(1, min(config.rl_num_proc, (os.cpu_count() or 1)))

    dataset = dataset.map(
        partial(_prepare_reward_columns, config=config),
        batched=True,
        num_proc=_NUM_PROC,
        writer_batch_size=1000,
        desc="prepare rewards",
    )

    logger.info("Computing group statistics")
    group_ids_raw = np.asarray(dataset["group_id"])
    scalar_r = np.asarray(dataset["reward"], dtype=np.float32)
    token_lens = np.array([len(x) for x in dataset["input_ids"]], dtype=np.float32)

    _, group_indices = np.unique(group_ids_raw, return_inverse=True)
    
    # Compute stats per group
    counts = np.bincount(group_indices)
    reward_sum = np.bincount(group_indices, weights=scalar_r)
    reward_sq_sum = np.bincount(group_indices, weights=scalar_r**2)
    token_sum = np.bincount(group_indices, weights=token_lens)

    if (counts == 0).any():
        raise ValueError("Encountered empty group when computing RL statistics")

    group_means = (reward_sum / counts).astype(np.float32)
    group_vars = (reward_sq_sum / counts - group_means**2).astype(np.float32)
    group_stds = np.sqrt(np.maximum(0.0, group_vars)).astype(np.float32)
    group_avg_tokens = (token_sum / counts).astype(np.float32)

    logger.info("Computing advantages and final columns (pass 2/2)")
    
    # Create a lookup dict with group-level stats
    group_lookup = {
        "mean": group_means,
        "std": group_stds,
        "avg_tok": group_avg_tokens
    }
    
    dataset = dataset.map(
        partial(_finalise_rl_columns, 
                group_lookup=group_lookup, 
                group_ids=group_indices,
                config=config, 
                eos_token_id=eos_token_id),
        batched=True,
        with_indices=True,
        num_proc=_NUM_PROC,
        writer_batch_size=1000,
        desc="finalise RL columns",
    )

    return dataset


def populate_rl_data(dataset: Dataset, eos_token_id: int, config: RLConfig) -> Dataset:
    """
    Populates a dataset with reinforcement learning specific data columns.

    Args:
        dataset (Dataset): The input dataset to populate with RL data
        columns (list[str]): List of column names to include in the dataset
        collate_fn (Callable): Function to collate/batch the data
        config (RLConfig): Configuration object containing RL training parameters

    Returns:
        Dataset: The dataset populated with RL-specific columns including rewards and advantages
    """

    logger.debug("Populate RL Data")

    dataset = update_rewards_and_advantages(dataset, eos_token_id, config)

    logger.debug("Finish Populate RL Data")
    return dataset


def prepare_rl_fields(
    encoding: BatchEncoding,
    reward: float,
    old_logprobs: list[float],
    ref_logprobs: list[float],
) -> BatchEncoding:
    """
    Convert reward per agent step to reward per token and add returns and advantages placeholders
    """
    target_tokens = [token for token in encoding["labels"] if token != -100]
    assert len(target_tokens) == len(
        old_logprobs
    ), f"Target tokens: {len(target_tokens)}, old logprobs: {len(old_logprobs)}"

    full_len = len(encoding["labels"])
    encoding["rewards"] = [reward] * full_len
    encoding["advantages"] = [0.0] * full_len
    encoding["old_logprobs"] = [0.0] * (full_len - len(old_logprobs)) + old_logprobs
    encoding["ref_logprobs"] = [0.0] * (full_len - len(ref_logprobs)) + ref_logprobs
    encoding["overflow"] = [0.0] * full_len
    encoding["group_tokens"] = [0.0] * full_len
    encoding["example_weight"] = [0.0] * full_len

    return encoding
