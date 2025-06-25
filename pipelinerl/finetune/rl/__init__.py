import logging
import os
from functools import partial
from pydantic import BaseModel, Field

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import BatchEncoding, PreTrainedModel

from .utils import (
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
]


class RLConfig(BaseModel):
    algo: str = Field(
        default="grpo",
        description="Algorithm to use for RL",
        choices=["grpo", "reinforce"],
    )
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
    config: RLConfig,
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
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"],
    }
    if is_packed:
        model_inputs["position_ids"] = batch["position_ids"]
    
    # Add visual features if present (for multimodal models)
    if "pixel_values" in batch and batch["pixel_values"] is not None:
        model_inputs["pixel_values"] = batch["pixel_values"]
    if "image_grid_thw" in batch and batch["image_grid_thw"] is not None:
        model_inputs["image_grid_thw"] = batch["image_grid_thw"].reshape((1, 3))

    outputs = model(**model_inputs)

    # compute log probs and entropy
    logits = outputs.logits[:, :-1, :]
    logits = logits / config.temperature
    logprobs = F.log_softmax(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * logprobs).sum(dim=-1)
    del logits, probs

    # get log probs for actual tokens
    new_logprobs = torch.gather(logprobs, dim=2, index=batch["input_ids"][:, 1:].unsqueeze(2)).squeeze(2)
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
        # assert that group_tokens is not zero
        assert (group_tokens > 0).all(), "group_tokens must be greater than zero for group normalization"
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
            clamp_log_ratio_new_old_indicators = clamped_ratio != ratio_new_old
            surr2 = clamped_ratio * log_p_weights
            policy_loss = torch.min(surr1, surr2)
        case "reinforce":
            surr1 = torch.zeros_like(ratio_new_old)
            surr2 = torch.zeros_like(ratio_new_old)
            clamp_log_ratio_new_old_indicators = ratio_new_old > 1 + config.epsilon
            ratio_new_old = torch.clamp(ratio_new_old, 0, 1 + config.epsilon)
            policy_loss = new_logprobs * log_p_weights * ratio_new_old.detach()
        case _:
            raise ValueError(f"Unknown algorithm {config.algo}")

    # combine loss components
    loss = policy_loss - kl_coef * approx_kl + entropy_bonus_coef * entropy  # 1 x (BxL) x 1
    assert loss.shape == tokens_weights.shape, (
        f"Loss shape {loss.shape} does not match example weights shape {tokens_weights.shape}"
    )
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
        "ratio_new_old_sum": sum_sum(ratio_new_old, masks_shifted, segments).item(),
        "ratio_new_old_squared_sum": sum_sum(  # useful to estimate the ESS
            ratio_new_old * ratio_new_old, masks_shifted, segments
        ).item(),
        "ratio_ref_new": mean_sum(torch.exp(log_ratio_ref_new), masks_shifted, segments).item(),
        "ratio_ref_old": mean_sum(torch.exp(ref_logprobs - old_logprobs), masks_shifted, segments).item(),
        "clamp_log_ratio_ref_new_indicator": mean_sum(
            clamp_log_ratio_ref_new_indicators, masks_shifted, segments
        ).item(),
        "clamp_log_ratio_new_old_indicator": mean_sum(
            clamp_log_ratio_new_old_indicators, masks_shifted, segments
        ).item(),
        "num_nans": torch.isnan(loss).sum().item(),
        "token_weight": mean_sum(tokens_weights, masks_shifted, segments).item(),
        "max_token_weight": tokens_weights[masks_shifted].max().item(),
        "min_token_weight": tokens_weights[masks_shifted].min().item(),
        "kl_coef": num_sequences * kl_coef,
        "entropy_bonus_coef": num_sequences * entropy_bonus_coef,
        "num_output_tokens_sum": masks_shifted.sum().item(),
    }

    return final_loss, stats


def populate_rl_data(dataset: list[BatchEncoding], eos_token_id: int, config: RLConfig) -> list[BatchEncoding]:
    """
    Populates a dataset with reinforcement learning specific data columns including
    rewards, advantages, and token weights.

    Args:
        dataset (Dataset): The input dataset to populate with RL data
        eos_token_id (int): End of sequence token ID
        config (RLConfig): Configuration object containing RL training parameters

    Returns:
        Dataset: The dataset populated with RL-specific columns
    """
    # Convert to pandas for processing
    df_init = pd.DataFrame(dataset)
    assert isinstance(df_init, pd.DataFrame)

    # Step 1: calculate group-level statistics
    df_stats = df_init[["group_id", "rollout_index", "step_index"]].copy()
    df_stats["num_tokens"] = df_init["input_ids"].apply(lambda x: len(x))
    # We assume that rewards for all tokens are the same
    df_stats["rollout_reward"] = df_init["rewards"].apply(lambda x: x[0])
    # Check that the reward is the same for each step in the rollout
    assert df_stats.groupby(["group_id", "rollout_index"])["rollout_reward"].nunique().max() == 1
    # Only keep step_index == 0
    df_stats = df_stats[df_stats["step_index"] == 0].drop(columns=["step_index"])
    df_grouped = (
        df_stats.groupby("group_id")
        .agg(
            rollout_reward_mean=("rollout_reward", "mean"),
            rollout_reward_std=("rollout_reward", "std"),
            group_tokens=("num_tokens", "mean"), 
        )
        .reset_index()
    )
    assert df_grouped.columns.tolist() == ["group_id", "rollout_reward_mean", "rollout_reward_std", "group_tokens"]

    # Step 2: calculate advantages for each sample
    df_advantages = pd.merge(
        df_init[["group_id", "rollout_index", "step_index", "rewards"]],
        df_grouped,
        on="group_id",
        how="left"
    )
    assert len(df_advantages) == len(df_init)
    def calculate_advantages(row):
        rewards = row["rewards"]
        mean = row["rollout_reward_mean"]
        std = row["rollout_reward_std"]
        if config.divide_advantage_by_std:
            advantages = [(reward - mean) / (np.nan_to_num(std) + 1e-4) for reward in rewards]
        else:
            advantages = [(reward - mean) for reward in rewards]
        return advantages
    df_advantages["advantages"] = df_advantages.apply(
        calculate_advantages,
        axis=1,
    )
    df_advantages = df_advantages.drop(columns=["rewards", "rollout_reward_mean", "rollout_reward_std"])
    assert df_advantages.columns.tolist() == ["group_id", "rollout_index", "step_index", "group_tokens", "advantages"]

    # Step 3: bring advantages and group level stats back to the main df
    df = df_init.drop(columns=["advantages", "group_tokens"])
    df = pd.merge(df, df_advantages, on=["group_id", "rollout_index", "step_index"], how="left")
    # Debug print lengths of all dataframes
    assert len(df) == len(df_init)

    # Step 4: make token-level overflow and mean group length information
    df["overflow"] = df.apply(
        lambda row: [0.0] * len(row["overflow"]) if eos_token_id in row["input_ids"] else [1.0] * len(row["overflow"]),
        axis=1,
    )
    df["group_tokens"] = df.apply(lambda row: [row["group_tokens"]] * len(row["input_ids"]), axis=1)

    # Step 5: move the results back to the dataset
    advantages_list = df["advantages"].tolist()
    group_tokens_list = df["group_tokens"].tolist()
    overflow_list = df["overflow"].tolist()
    for i, entry in enumerate(dataset):
        entry["advantages"] = advantages_list[i]
        entry["group_tokens"] = group_tokens_list[i]
        entry["overflow"] = overflow_list[i]
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
    assert len(target_tokens) == len(old_logprobs), (
        f"Target tokens: {len(target_tokens)}, old logprobs: {len(old_logprobs)}"
    )

    encoding["rewards"] = [reward] * len(encoding["labels"])
    encoding["advantages"] = [0.0] * len(encoding["labels"])  # place holder
    encoding["old_logprobs"] = [0] * (len(encoding["labels"]) - len(old_logprobs)) + old_logprobs
    encoding["ref_logprobs"] = [0] * (len(encoding["labels"]) - len(ref_logprobs)) + ref_logprobs
    encoding["overflow"] = [0] * len(encoding["labels"])  # place holder
    encoding["group_tokens"] = [0] * len(encoding["labels"])  # place holder
    return encoding
