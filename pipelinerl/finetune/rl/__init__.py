import logging
import os
from functools import partial
from typing import Any
from pydantic import BaseModel, Field

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import PreTrainedModel
from pipelinerl.finetune.types import PipelineBatchEncoding
from pipelinerl.finetune.rl.utils import per_segment_sums

from .utils import (
    sum_sum,
    mean_sum,
    replace_dataset_column,
)

# FIXME: remove a warnings, but might be worth investigating
os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = logging.getLogger(__name__)

RL_DATA_COLUMNS = [
    "overflow",
    "group_tokens",
    "num_labels",
    "loss_weight_scale",
    "rewards",
    "advantages",
    "old_logprobs",
    "ref_logprobs",
]


class RLConfig(BaseModel):
    policy_loss: str = Field(
        default="ppo",
        description="Policy Loss to use for RL",
        choices=["ppo", "reinforce", "gspo"],
    )
    use_advantages: bool = Field(
        default=True,
        description="Use advantages instead of rewards to compute the loss",
    )
    epsilon_low: float = Field(default=0.2, description="Lower clip parameter for ratio of log probs")
    epsilon_high: float = Field(default=0.2, description="Upper clip parameter for ratio of log probs")
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
    divide_advantage_by_std: bool = Field(
        default=True,
        description="Normalize the advantage by the standard deviation",
    )
    overlong_filtering: bool = Field(default=False, description="Filter out sequence that do not have eos_token_id")
    group_normalization: bool = Field(
        default=False,
        description="Divide the weight of each sequence by the (average) number of tokens in the group",
    )
    temperature: float = Field(
        default=1.0,
        description="Temperature for the training log probs",
    )
    filter_zero_advantage_groups: bool = Field(
        default=False,
        description="Filter out groups where all advantages are zero during preprocessing",
    )
    step_reward_advantages: bool = Field(
        default=False,
        description="Compute leave-one-out advantages by (group_id, step_index), allowing different rewards per rollout step",
    )
    pad_step_rewards_for_advantage: bool = Field(
        default=False,
        description="When step_reward_advantages is enabled, pad each rollout's reward sequence with its final reward for advantage statistics only",
    )
    batch_size_normalization: str = Field(
        default="training_texts",
        description="Loss denominator unit: training_texts counts every trainable call; rollouts groups sub-calls from the same rollout attempt",
    )
    value_loss_coef: float = Field(
        default=0.0,
        description="Coefficient for the value loss in the final loss",
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
    batch: PipelineBatchEncoding,
    current_step: int,
    max_step: int,
    config: RLConfig,
    seq_parallel_group=None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Perform a single RL step on the model using the given batch and config.
    Handles both packed and unpacked sequences.

    Args:
        model (PreTrainedModel): The model to train
        batch (PipelineBatchEncoding): Batch of data containing rewards, advantages, masks, input_ids etc.
        current_step (int): Current training step
        max_step (int): Maximum number of training steps
        config (RLConfig): Configuration for the RL training

    Returns:
        tuple[torch.Tensor, dict[str, float]]: Loss tensor and metrics dictionary
    """
    # pre-compute masks
    masks = batch.labels != -100
    masks_shifted = masks[:, 1:]

    has_value_head = hasattr(model, 'value_head')

    # if we have position_ids, we are packing
    if batch.is_packed:
        position_ids = batch.position_ids[0]
        is_sequence_start = position_ids == 0
        # For computing the loss we will consider the first token the beginning of the sequence,
        # even if currently we are in the middle of a sequence.
        is_sequence_start[0] = True 
        sequence_starts = torch.where(is_sequence_start)[0]
        seq_boundaries = torch.cat(
            [
                sequence_starts,
                torch.tensor([position_ids.shape[0]], device=position_ids.device),
            ]
        )
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
        "input_ids": batch.input_ids,
        "attention_mask": batch.attention_mask,
        "labels": batch.labels,
    }
    if batch.is_packed:
        model_inputs["position_ids"] = batch.position_ids
    
    # Add visual features if present (for multimodal models)
    if hasattr(batch, 'pixel_values') and batch.pixel_values is not None:
        model_inputs["pixel_values"] = batch.pixel_values
    if hasattr(batch, 'image_grid_thw') and batch.image_grid_thw is not None:
        model_inputs["image_grid_thw"] = batch.image_grid_thw #torch.tensor(.reshape((1, 3))
    
    outputs = model(**model_inputs)

    # compute log probs for actual tokens without materializing full logprobs unless needed
    logits = outputs.logits[:, :-1, :]
    logits = logits / config.temperature
    next_token_ids = batch.input_ids[:, 1:].unsqueeze(2)
    selected_logits = torch.gather(logits, dim=2, index=next_token_ids).squeeze(2)
    log_norm = torch.logsumexp(logits, dim=-1)
    new_logprobs = selected_logits - log_norm
    assert torch.isfinite(new_logprobs).all(), f"new_logprobs is not finite: {new_logprobs}"

    use_entropy_loss = config.entropy_bonus != 0.0 or config.final_entropy_bonus != 0.0
    if use_entropy_loss:
        logprobs = logits - log_norm.unsqueeze(-1)
        probs = torch.exp(logprobs)
        entropy = -(probs * logprobs).sum(dim=-1)
        del logprobs, probs
    else:
        # Keep exact entropy stats without allocating full-vocab softmax/log-softmax tensors.
        entropy = torch.zeros_like(log_norm)
        detached_logits = logits.detach()
        detached_log_norm = log_norm.detach()
        log_norm_unsqueezed = detached_log_norm.unsqueeze(-1)
        vocab_chunk_size = 4096
        with torch.no_grad():
            for start in range(0, detached_logits.shape[-1], vocab_chunk_size):
                chunk = detached_logits[..., start:start + vocab_chunk_size]
                chunk_logprobs = chunk - log_norm_unsqueezed
                chunk_probs = torch.exp(chunk_logprobs)
                entropy -= (chunk_probs * chunk_logprobs).sum(dim=-1)

    del logits, selected_logits, log_norm

    # get shifted values and compute ratios
    rewards = batch.rewards[:, 1:]
    ref_logprobs = batch.ref_logprobs[:, 1:]
    old_logprobs = batch.old_logprobs[:, 1:]
    group_tokens = batch.group_tokens[:, 1:]
    num_labels_in_seq = batch.num_labels[:, 1:] # sequence dependent normalization
    overflow = batch.overflow[:, 1:]

    if config.group_normalization:
        # assert that group_tokens is not zero
        assert (group_tokens > 0).all(), "group_tokens must be greater than zero for group normalization"
        tokens_weights = torch.ones_like(group_tokens) / group_tokens
    else:
        tokens_weights = torch.ones_like(group_tokens) / config.batch_size

    if config.overlong_filtering:
        # filter out sequences that do not have eos_token_id
        overflow = torch.tensor(overflow, device=overflow.device)
        tokens_weights = tokens_weights * (1 - overflow)
    if batch.loss_weight_scale is not None:
        # Stepwise domains can emit multiple trainable texts for one rollout.
        # This scale preserves the configured loss denominator after packing.
        tokens_weights = tokens_weights * batch.loss_weight_scale[:, 1:].to(tokens_weights.device)

    assert new_logprobs.shape == ref_logprobs.shape

    log_ratio_new_old = new_logprobs - old_logprobs
    abs_log_ratio_new_old = torch.abs(log_ratio_new_old)
    ratio_new_old = torch.exp(log_ratio_new_old)
    log_ratio_ref_new = ref_logprobs - new_logprobs
    assert torch.isfinite(log_ratio_ref_new).all(), f"log_ratio_ref_new is not finite: {log_ratio_ref_new}"

    if has_value_head:
        # Get value predictions if available
        value_predictions = outputs.value[:, :-1] # no target for the last token 
        # Compute value-based advantages: A(s,a) = MC_return - V(s)
        # where MC_return is the Monte Carlo return (rewards) and V(s) is the value prediction
        #FIXME: if this works better it should be a config
        #advantages = rewards - torch.clamp(value_predictions, 0, 1)
        advantages = rewards - value_predictions
    else:
        advantages = batch.advantages[:, 1:]

    log_p_weights = advantages.detach() if config.use_advantages else rewards
    if config.relu_log_p_weights:
        log_p_weights = torch.clamp(log_p_weights, min=0)

    clamp_log_ratio_ref_new_indicators = torch.abs(log_ratio_ref_new) > config.clamp_log_ratio_ref_new_value

    log_ratio_ref_new_clamp = torch.clamp(
        log_ratio_ref_new,
        min=-config.clamp_log_ratio_ref_new_value,
        max=config.clamp_log_ratio_ref_new_value,
    )

    approx_kl = torch.exp(log_ratio_ref_new_clamp) - log_ratio_ref_new_clamp - 1  # Schulman KL approx
    approx_kl_new_old = torch.exp(log_ratio_new_old) - log_ratio_new_old - 1  # Schulman KL approx

    assert torch.isfinite(approx_kl).all(), f"approx_kl is not finite: {approx_kl}"
    entropy_bonus_coef = linear_decay_coef(current_step, max_step, config.entropy_bonus, config.final_entropy_bonus)
    kl_coef = linear_decay_coef(current_step, max_step, config.kl_coef, config.final_kl_coef)

    # compute algorithm-specific losses
    policy_loss_total = None
    match config.policy_loss:
        case "ppo":
            surr1 = ratio_new_old * log_p_weights
            clamped_ratio = torch.clamp(ratio_new_old, 1 - config.epsilon_low, 1 + config.epsilon_high)
            clamp_log_ratio_new_old_indicators = clamped_ratio != ratio_new_old
            surr2 = clamped_ratio * log_p_weights
            policy_loss = torch.min(surr1, surr2)
        case "reinforce":
            surr1 = torch.zeros_like(ratio_new_old)
            surr2 = torch.zeros_like(ratio_new_old)
            clamp_log_ratio_new_old_indicators = ratio_new_old > 1 + config.epsilon_high
            ratio_new_old = torch.clamp(ratio_new_old, 0, 1 + config.epsilon_high)
            policy_loss = new_logprobs * log_p_weights * ratio_new_old.detach()
        case "gspo":
            if segments is None:
                raise ValueError("GSPO loss requires packed sequences with segments")
            lrn_sum, adv_sum, tok_count = per_segment_sums(
                batch.segment_ids,
                masks_shifted,
                log_ratio_new_old,
                advantages,
                seq_parallel_group=seq_parallel_group,
            )
            group_ratio_new_old = torch.exp(lrn_sum / tok_count.clamp(min=1e-6)).unsqueeze(1).unsqueeze(2)
            group_advantages_t = (adv_sum / tok_count.clamp(min=1e-6)).unsqueeze(1).unsqueeze(2).detach()
            zero_weights = torch.zeros_like(tokens_weights)
            weight_sum, _, _ = per_segment_sums(
                batch.segment_ids,
                masks_shifted,
                tokens_weights,
                zero_weights,
                seq_parallel_group=seq_parallel_group,
            )
            valid_mask = (tok_count > 0) & (weight_sum > 0)
            valid_mask_3d = valid_mask.unsqueeze(1).unsqueeze(2)
            surr1 = group_ratio_new_old * group_advantages_t
            clamped_group_ratio = torch.clamp(group_ratio_new_old, 1 - config.epsilon_low, 1 + config.epsilon_high)
            clamp_log_ratio_new_old_indicators = (clamped_group_ratio != group_ratio_new_old) & valid_mask_3d
            surr2 = clamped_group_ratio * group_advantages_t
            # Length-proportional weighting is intentional: longer sequences carry more
            # gradient, which complements difficulty-aware penalty (DAP) — DAP reduces the
            # length penalty for successful hard rollouts, and length-proportional weights
            # amplify that signal into the update. Uniform weighting hurts training dynamics.
            sequence_weights = weight_sum.unsqueeze(1).unsqueeze(2)
            if batch.sentinel or surr1.numel() == 0:
                policy_loss_total = new_logprobs[..., :1].sum() * 0.0
            else:
                mask_float = valid_mask_3d.to(dtype=surr1.dtype)
                min_terms = torch.min(surr1, surr2) * mask_float * sequence_weights
                policy_loss_total = -min_terms.sum()
            expanded_indicators = torch.zeros_like(masks_shifted, dtype=torch.float)
            for (start, end), val in zip(segments, clamp_log_ratio_new_old_indicators.flatten()):
                expanded_indicators[0, start:end] = float(val)
            clamp_log_ratio_new_old_indicators = expanded_indicators
        case _:
            raise ValueError(f"Unknown algorithm {config.policy_loss}")

    # combine loss components
    if config.policy_loss != "gspo":
        if use_entropy_loss:
            loss = policy_loss - kl_coef * approx_kl + entropy_bonus_coef * entropy
        else:
            loss = policy_loss - kl_coef * approx_kl
        assert loss.shape == tokens_weights.shape, (
            f"Loss shape {loss.shape} does not match example weights shape {tokens_weights.shape}"
        )
        loss = loss * tokens_weights  # 1 x (BxL) x 1

        policy_loss_total = -sum_sum(loss, masks_shifted, segments)

    if has_value_head:
        # Get the value predictions
        values = outputs.value
        # Use the already extracted and shifted rewards as value labels
        value_labels = rewards  # This is already shifted (from line 216)
        values = values[:, :-1]
        values_labels = value_labels
        assert values.shape == tokens_weights.shape, (
            f"Values shape {values.shape} does not match example weights shape {tokens_weights.shape}"
        )
        value_loss = 0.5 * torch.square(values - values_labels) * tokens_weights
        value_loss = sum_sum(value_loss, masks_shifted, segments) 
        
        # Combine policy loss and value loss
        final_loss = policy_loss_total + config.value_loss_coef * value_loss
    else:
        final_loss = policy_loss_total

    # ensure loss is valid
    assert torch.isfinite(final_loss), f"Non-finite loss detected: {final_loss}"

    if int(masks_shifted.sum().item()) == 0:
        stats_no_labels = {
            "input_size": float(batch.input_ids.numel()),
        }
        return final_loss, stats_no_labels

    # Metric aggregation behavior:
    # 1. loss: pre-multiplied by token_weights, reported as sum
    # 2. min/max values: computed across entire batch
    # 3. other statistics: averaged per sequence, then averaged across batch
    stats = {
        "loss": final_loss.item(),
        "max_loss": final_loss.item(),
        "min_loss": final_loss.item(),
        "reward": sum_sum(rewards / num_labels_in_seq, masks_shifted, segments).item(),
        "max_reward": rewards[masks_shifted].max().item(),
        "min_reward": rewards[masks_shifted].min().item(),
        "entropy": sum_sum(entropy / num_labels_in_seq, masks_shifted, segments).item(),
        "old_logprobs": sum_sum(old_logprobs / num_labels_in_seq, masks_shifted, segments).item(),
        "new_logprobs": sum_sum(new_logprobs / num_labels_in_seq, masks_shifted, segments).item(),
        "ref_logprobs": sum_sum(ref_logprobs / num_labels_in_seq, masks_shifted, segments).item(),
        "advantage": sum_sum(advantages / num_labels_in_seq, masks_shifted, segments).item(),
        "max_advantage": advantages[masks_shifted].max().item(),
        "min_advantage": advantages[masks_shifted].min().item(),
        "kl": sum_sum(approx_kl / num_labels_in_seq, masks_shifted, segments).item(),
        "kl_new_old": sum_sum(approx_kl_new_old / num_labels_in_seq, masks_shifted, segments).item(),
        "mean_abs_log_ratio_new_old": sum_sum(
            abs_log_ratio_new_old / num_labels_in_seq, masks_shifted, segments
        ).item(),
        "max_kl": approx_kl[masks_shifted].max().item(),
        "min_kl": approx_kl[masks_shifted].min().item(),
        "ratio_new_old": sum_sum(ratio_new_old / num_labels_in_seq, masks_shifted, segments).item(),
        "ratio_new_old_sum": sum_sum(ratio_new_old, masks_shifted, segments).item(),
        "ratio_new_old_squared_sum": sum_sum(  # useful to estimate the ESS
            ratio_new_old * ratio_new_old, masks_shifted, segments
        ).item(),
        "ratio_ref_new": sum_sum(torch.exp(log_ratio_ref_new) / num_labels_in_seq, masks_shifted, segments).item(),
        "ratio_ref_old": sum_sum(torch.exp(ref_logprobs - old_logprobs) / num_labels_in_seq, masks_shifted, segments).item(),
        "clamp_log_ratio_ref_new_indicator": sum_sum(
            clamp_log_ratio_ref_new_indicators / num_labels_in_seq, masks_shifted, segments
        ).item(),
        "clamp_log_ratio_new_old_indicator": sum_sum(
            clamp_log_ratio_new_old_indicators / num_labels_in_seq, masks_shifted, segments
        ).item(),
        "token_weight": sum_sum(tokens_weights / num_labels_in_seq, masks_shifted, segments).item(),
        "max_token_weight": tokens_weights[masks_shifted].max().item(),
        "min_token_weight": tokens_weights[masks_shifted].min().item(),
        "rollout_weight_scale": (
            sum_sum(batch.loss_weight_scale[:, 1:] / num_labels_in_seq, masks_shifted, segments).item()
            if batch.loss_weight_scale is not None
            else 1.0
        ),
        "kl_coef": num_sequences * kl_coef,
        "entropy_bonus_coef": num_sequences * entropy_bonus_coef,
        "num_output_tokens_sum": masks_shifted.sum().item(),
        "input_size": batch.input_ids.numel(), 
    }

    if has_value_head:
        stats["value_mean"] = sum_sum(value_predictions / num_labels_in_seq, masks_shifted, segments).item()
        stats["value_max"] = value_predictions[masks_shifted].max().item() if masks_shifted.any() else 0.0
        stats["value_min"] = value_predictions[masks_shifted].min().item() if masks_shifted.any() else 0.0
        stats["value_loss"] = value_loss.item()
        stats["value_mse"] = sum_sum(
            torch.square(value_predictions - value_labels) / num_labels_in_seq, masks_shifted, segments
        ).item()

    return final_loss, stats


def _compute_step_reward_advantages(
    df_init: pd.DataFrame, df_stats: pd.DataFrame, config: RLConfig
) -> pd.DataFrame:
    """Leave-one-out advantages for per-step rewards (rare; privacy_hopqa only today).

    Unlike the default one-reward-per-rollout estimator, every trainable step
    carries its own reward and the LOO baseline is computed within an "advantage
    bucket" instead of across the whole group. The bucket is the global step
    index ("step:{i}") by default; domains that attach step_advantage_group
    metadata (privacy_hopqa groups by (hop, stage)) instead get buckets aligned
    by position inside each semantic segment, so the k-th step of a segment is
    only compared with other k-th steps. With pad_step_rewards_for_advantage,
    shorter rollouts contribute virtual padding rows (advantage_scale=0) that
    enter the baseline but produce no gradient, so stopping early is not
    rewarded. Returns columns (group_id, rollout_index, step_index, group_tokens,
    advantages) — the same shape the default path produces.

    df_stats must already carry num_tokens, step_reward, and padding_reward.
    """
    # group_tokens baseline = mean trainable tokens per rollout in the group.
    df_rollouts = (
        df_stats.groupby(["group_id", "rollout_index"])
        .agg(rollout_tokens=("num_tokens", "sum"))
        .reset_index()
    )
    df_group_tokens = (
        df_rollouts.groupby("group_id")
        .agg(group_tokens=("rollout_tokens", "mean"))
        .reset_index()
    )
    df_reward_stats = df_stats[
        ["group_id", "rollout_index", "step_index", "rewards", "step_reward", "padding_reward"]
    ].copy()
    df_reward_stats["is_padding"] = False
    # Default bucket: each step is compared only against the same global step index.
    df_reward_stats["advantage_bucket"] = df_reward_stats["step_index"].map(lambda value: f"step:{int(value)}")
    df_reward_stats["advantage_scale"] = 1.0

    # Optional per-step metadata lets a domain define its own (custom) buckets.
    if "metadata" in df_init.columns:
        metadata = df_init["metadata"].apply(lambda value: value if isinstance(value, dict) else {})
        df_reward_stats["step_advantage_group"] = metadata.apply(lambda value: value.get("step_advantage_group"))
        df_reward_stats["step_advantage_local_index"] = metadata.apply(
            lambda value: value.get("step_advantage_local_index")
        )
        df_reward_stats["step_advantage_segment_length"] = metadata.apply(
            lambda value: value.get("step_advantage_segment_length")
        )
        df_reward_stats["step_advantage_padding_value"] = metadata.apply(
            lambda value: value.get("step_advantage_padding_value")
        )
    else:
        df_reward_stats["step_advantage_group"] = None
        df_reward_stats["step_advantage_local_index"] = None
        df_reward_stats["step_advantage_segment_length"] = None
        df_reward_stats["step_advantage_padding_value"] = None

    custom_mask = df_reward_stats["step_advantage_group"].notna()
    padded_rows = []
    if custom_mask.any():
        # Custom buckets: compare steps within a semantic segment (e.g. a hop's
        # planning stage), aligned to the segment END so segments of different
        # lengths still line up — last step with last step, second-to-last, etc.
        custom = df_reward_stats[custom_mask].copy()
        custom["_row_id"] = custom.index
        max_lengths = (
            custom.groupby(["group_id", "step_advantage_group"])
            .agg(max_segment_length=("step_advantage_segment_length", "max"))
            .reset_index()
        )
        custom = pd.merge(custom, max_lengths, on=["group_id", "step_advantage_group"], how="left")
        aligned_position = (
            custom["max_segment_length"].astype(int)
            - custom["step_advantage_segment_length"].astype(int)
            + custom["step_advantage_local_index"].astype(int)
        )
        custom["advantage_bucket"] = (
            custom["step_advantage_group"].astype(str)
            + ":pos:"
            + aligned_position.astype(int).astype(str)
        )
        df_reward_stats.loc[custom["_row_id"], "advantage_bucket"] = custom["advantage_bucket"].to_numpy()

        if config.pad_step_rewards_for_advantage:
            # Pad each short segment up to the group's max length so every aligned
            # position has the same population; scale 0 keeps padding out of the loss.
            segment_rows = (
                custom.sort_values(["step_index", "step_advantage_local_index"])
                .groupby(["group_id", "rollout_index", "step_advantage_group"], as_index=False)
                .first()
            )
            for row in segment_rows.itertuples(index=False):
                segment_length = int(row.step_advantage_segment_length)
                max_segment_length = int(row.max_segment_length)
                pad_value = (
                    float(row.step_advantage_padding_value)
                    if pd.notna(row.step_advantage_padding_value)
                    else float(row.step_reward)
                )
                for padded_position in range(max_segment_length - segment_length):
                    padded_rows.append(
                        {
                            "group_id": row.group_id,
                            "rollout_index": row.rollout_index,
                            "step_index": -1,
                            "rewards": [],
                            "step_reward": pad_value,
                            "padding_reward": pad_value,
                            "is_padding": True,
                            "advantage_bucket": f"{row.step_advantage_group}:pos:{padded_position}",
                            "step_advantage_group": row.step_advantage_group,
                            "step_advantage_local_index": None,
                            "step_advantage_segment_length": segment_length,
                            "step_advantage_padding_value": pad_value,
                            "advantage_scale": 0.0,
                        }
                    )

    default_mask = df_reward_stats["step_advantage_group"].isna()
    if config.pad_step_rewards_for_advantage and default_mask.any():
        # Same idea for globally-indexed steps: pad each rollout up to the group's
        # max step so every "step:{i}" bucket sees every rollout.
        max_step_by_group = df_stats.groupby("group_id")["step_index"].max().to_dict()
        last_steps = (
            df_reward_stats[default_mask]
            .sort_values("step_index")
            .groupby(["group_id", "rollout_index"], as_index=False)
            .tail(1)
        )
        for row in last_steps.itertuples(index=False):
            max_step = int(max_step_by_group[row.group_id])
            last_step = int(row.step_index)
            pad_value = getattr(row, "padding_reward", row.step_reward)
            for padded_step in range(last_step + 1, max_step + 1):
                padded_rows.append(
                    {
                        "group_id": row.group_id,
                        "rollout_index": row.rollout_index,
                        "step_index": padded_step,
                        "rewards": [],
                        "step_reward": pad_value,
                        "padding_reward": pad_value,
                        "is_padding": True,
                        "advantage_bucket": f"step:{padded_step}",
                        "step_advantage_group": None,
                        "step_advantage_local_index": None,
                        "step_advantage_segment_length": None,
                        "step_advantage_padding_value": None,
                        "advantage_scale": 0.0,
                    }
                )
    if padded_rows:
        df_reward_stats = pd.concat([df_reward_stats, pd.DataFrame(padded_rows)], ignore_index=True)

    # Baseline stats per bucket (padding rows included so the mean is stable).
    df_grouped = (
        df_reward_stats.groupby(["group_id", "advantage_bucket"])
        .agg(
            step_reward_sum=("step_reward", "sum"),
            step_reward_count=("step_reward", "count"),
            step_reward_std=("step_reward", "std"),
        )
        .reset_index()
    )
    assert df_group_tokens.columns.tolist() == ["group_id", "group_tokens"]
    assert df_grouped.columns.tolist() == [
        "group_id",
        "advantage_bucket",
        "step_reward_sum",
        "step_reward_count",
        "step_reward_std",
    ]
    # Only real (non-padding) rows receive advantages; padding only shaped the baseline.
    df_real_reward_stats = df_reward_stats[~df_reward_stats["is_padding"]].copy()
    df_advantages = pd.merge(
        df_real_reward_stats[
            ["group_id", "rollout_index", "step_index", "rewards", "step_reward", "advantage_bucket", "advantage_scale"]
        ],
        df_grouped,
        on=["group_id", "advantage_bucket"],
        how="left"
    )
    df_advantages = pd.merge(df_advantages, df_group_tokens, on="group_id", how="left")

    def calculate_advantages(row):
        rewards = row["rewards"]
        group_sum = row["step_reward_sum"]
        group_count = row["step_reward_count"]
        current_reward = row["step_reward"]
        # Leave-one-out baseline within the bucket.
        if group_count > 1:
            loo_mean = (group_sum - current_reward) / (group_count - 1)
        else:
            loo_mean = current_reward
        std = row["step_reward_std"]
        if config.divide_advantage_by_std:
            advantages = [(r - loo_mean) / (np.nan_to_num(std) + 1e-4) for r in rewards]
        else:
            advantages = [(r - loo_mean) for r in rewards]
        scale = float(row.get("advantage_scale", 1.0))
        return [advantage * scale for advantage in advantages]

    df_advantages["advantages"] = df_advantages.apply(calculate_advantages, axis=1)
    return df_advantages.drop(
        columns=[
            "rewards",
            "step_reward",
            "advantage_bucket",
            "advantage_scale",
            "step_reward_sum",
            "step_reward_count",
            "step_reward_std",
        ]
    )


def populate_rl_data(dataset: list[dict[str, Any]], eos_token_id: int, config: RLConfig) -> list[dict[str, Any]]:
    """Populate RL columns (advantages, overflow, num_labels) using a leave-one-out baseline."""
    df_init = pd.DataFrame(dataset)
    assert isinstance(df_init, pd.DataFrame)

    # Step 1: per-(group, rollout, step) stats shared by both advantage modes.
    df_stats = df_init[["group_id", "rollout_index", "step_index", "rewards"]].copy()
    df_stats["num_tokens"] = df_init["input_ids"].apply(len)

    # Step 2: advantages. step_reward_advantages is a rare opt-in (privacy_hopqa)
    # that rewards each step; that logic lives in its own helper so the default
    # one-reward-per-rollout path below stays close to upstream.
    if config.step_reward_advantages:
        df_stats["step_reward"] = df_stats["rewards"].apply(lambda x: x[0])
        # Optional per-rollout padding override (privacy_hopqa sets this to the max
        # prefix reward so error-terminated rollouts pad to last achieved progress
        # instead of the zero'd error step). Falls back to step_reward.
        if "padding_reward" in df_init.columns:
            df_stats["padding_reward"] = df_init["padding_reward"].fillna(df_stats["step_reward"])
        else:
            df_stats["padding_reward"] = df_stats["step_reward"]
        df_advantages = _compute_step_reward_advantages(df_init, df_stats, config)
    else:
        # One reward per rollout (all steps share it); LOO baseline over the group.
        df_stats["rollout_reward"] = df_stats["rewards"].apply(lambda x: x[0])
        assert df_stats.groupby(["group_id", "rollout_index"])["rollout_reward"].nunique().max() == 1
        df_rollout_stats = df_stats[df_stats["step_index"] == 0].drop(columns=["step_index"])
        df_grouped = (
            df_rollout_stats.groupby("group_id")
            .agg(
                rollout_reward_sum=("rollout_reward", "sum"),
                rollout_reward_count=("rollout_reward", "count"),
                rollout_reward_std=("rollout_reward", "std"),
                group_tokens=("num_tokens", "mean"),
            )
            .reset_index()
        )
        df_advantages = pd.merge(
            df_init[["group_id", "rollout_index", "step_index", "rewards"]],
            df_grouped,
            on="group_id",
            how="left"
        )
        assert len(df_advantages) == len(df_init)

        def calculate_advantages(row):
            rewards = row["rewards"]
            group_sum = row["rollout_reward_sum"]
            group_count = row["rollout_reward_count"]
            current_reward = rewards[0]
            if group_count > 1:
                loo_mean = (group_sum - current_reward) / (group_count - 1)
            else:
                loo_mean = current_reward
            std = row["rollout_reward_std"]
            if config.divide_advantage_by_std:
                return [(r - loo_mean) / (np.nan_to_num(std) + 1e-4) for r in rewards]
            return [(r - loo_mean) for r in rewards]

        df_advantages["advantages"] = df_advantages.apply(calculate_advantages, axis=1)
        df_advantages = df_advantages.drop(
            columns=["rewards", "rollout_reward_sum", "rollout_reward_count", "rollout_reward_std"]
        )
    assert len(df_advantages) == len(df_init)
    expected_advantage_columns = [
        "group_id",
        "rollout_index",
        "step_index",
        "group_tokens",
        "advantages",
    ]
    missing_advantage_columns = [
        column for column in expected_advantage_columns if column not in df_advantages.columns
    ]
    if missing_advantage_columns:
        raise ValueError(
            "RL advantage table missing columns "
            f"{missing_advantage_columns}; present columns={df_advantages.columns.tolist()}"
        )
    df_advantages = df_advantages[expected_advantage_columns]

    # Step 3: bring advantages and group level stats back to the main df
    drop_columns = [column for column in ["advantages", "group_tokens"] if column in df_init.columns]
    df = df_init.drop(columns=drop_columns)
    df = pd.merge(df, df_advantages, on=["group_id", "rollout_index", "step_index"], how="left")
    # Debug print lengths of all dataframes
    assert len(df) == len(df_init)

    # Step 4: make token-level overflow and mean group length information
    def _overflow_from_finish_reason(row):
        length = len(row["overflow"])
        finish_reason = row.get("finish_reason")
        if isinstance(finish_reason, str):
            finish_reason = finish_reason.strip().lower()
            if finish_reason == "length":
                return [1.0] * length
            if finish_reason in {"stop", "content_filter"}:
                return [0.0] * length
        if row.get("finished"):
            return [0.0] * length
        return [0.0] * length if eos_token_id in row["input_ids"] else [1.0] * length

    df["overflow"] = df.apply(_overflow_from_finish_reason, axis=1)
    df["group_tokens"] = df.apply(lambda row: [row["group_tokens"]] * len(row["input_ids"]), axis=1)
    df["num_labels"] = df.apply(
        lambda row: [sum(1 for label in row["labels"] if label != -100)] * len(row["input_ids"]), axis=1
    )
    loss_weight_scale = 1.0
    if config.batch_size_normalization == "rollouts":
        # A rollout attempt may produce several trainable traces. Scaling by
        # trainable_texts / rollouts makes each attempt count once in the loss.
        rollout_count = len(df[["group_id", "rollout_index"]].drop_duplicates())
        if rollout_count > 0:
            loss_weight_scale = len(df) / rollout_count
    elif config.batch_size_normalization != "training_texts":
        raise ValueError(f"unknown RL batch_size_normalization: {config.batch_size_normalization}")
    df["loss_weight_scale"] = df["input_ids"].apply(lambda input_ids: [loss_weight_scale] * len(input_ids))

    # Step 5: move the results back to the dataset
    advantages_list = df["advantages"].tolist()
    group_tokens_list = df["group_tokens"].tolist()
    overflow_list = df["overflow"].tolist()
    num_labels_list = df["num_labels"].tolist()
    loss_weight_scale_list = df["loss_weight_scale"].tolist()
    for i, entry in enumerate(dataset):
        entry["advantages"] = advantages_list[i]
        entry["group_tokens"] = group_tokens_list[i]
        entry["overflow"] = overflow_list[i]
        entry["num_labels"] = num_labels_list[i]
        entry["loss_weight_scale"] = loss_weight_scale_list[i]
    return dataset


def prepare_rl_fields(
    encoding: dict[str, Any],
    reward: float,
    old_logprobs: list[float],
    ref_logprobs: list[float],
) -> dict[str, Any]:
    """
    Convert reward per agent step to reward per token and add returns and advantages placeholders
    """
    target_tokens = [token for token in encoding["labels"] if token != -100]
    assert len(target_tokens) == len(old_logprobs), (
        f"Target tokens: {len(target_tokens)}, old logprobs: {len(old_logprobs)}"
    )

    encoding["rewards"] = [reward] * len(encoding["labels"])
    encoding["advantages"] = [0.0] * len(encoding["labels"])  # place holder
    encoding["old_logprobs"] = [0] * (len(encoding["labels"]) - len(old_logprobs)) + old_logprobs
    encoding["ref_logprobs"] = [0] * (len(encoding["labels"]) - len(ref_logprobs)) + ref_logprobs
    encoding["overflow"] = [0] * len(encoding["labels"])  # place holder
    encoding["group_tokens"] = [0] * len(encoding["labels"])  # place holder
    encoding["num_labels"] = [1 if label != -100 else 0 for label in encoding["labels"]]  # count only output tokens
    encoding["loss_weight_scale"] = [1.0] * len(encoding["labels"])  # place holder
    return encoding
