import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Literal
from pydantic import BaseModel, Field

import numpy as np
import pandas as pd
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
        "kl_coef": num_sequences * kl_coef,
        "entropy_bonus_coef": num_sequences * entropy_bonus_coef,
        "num_output_tokens_sum": masks_shifted.sum().item(),
    }

    return final_loss, stats


def populate_rl_data(dataset: Dataset, eos_token_id: int, config: RLConfig) -> Dataset:
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
    logger.debug("Populate RL Data")

    # Convert to pandas for processing
    df = dataset.to_pandas()

    # Update rewards with implicit KL if needed
    if config.reward_minus_kl_coef > 0:
        logger.info("Updating Reward with Implicit KL")
        calculate_rewards_with_implicit_kl_ = partial(
            calculate_rewards_with_implicit_kl,
            reward_minus_kl_coef=config.reward_minus_kl_coef,
        )
        df["rewards"] = df.apply(calculate_rewards_with_implicit_kl_, axis=1)
        df["reward"] = df["rewards"].apply(lambda x: np.mean(x))

    # Combined groupby for both reward statistics and token calculations
    grouped = (
        df.groupby("group_id")
        .agg(
            reward_mean=("reward", "mean"),
            reward_std=("reward", "std"),
            new_group_tokens=(
                "input_ids",
                lambda x: sum(len(tokens) for tokens in x) / len(x),
            ),
        )
        .reset_index()
    )

    # Single merge to bring all statistics back
    df = pd.merge(df, grouped, on="group_id", how="left")

    # Calculate advantages
    df["advantages"] = df.apply(
        calculate_advantage,
        axis=1,
        divide_advantage_by_std=config.divide_advantage_by_std,
    )

    # Handle overflow
    df["overflow"] = df.apply(
        lambda row: [0.0] * len(row["overflow"]) if eos_token_id in row["input_ids"] else [1.0] * len(row["overflow"]),
        axis=1,
    )

    # Broadcast group tokens
    df["new_group_tokens"] = df.apply(lambda row: [row["new_group_tokens"]] * len(row["input_ids"]), axis=1)

    # Replace columns in the dataset
    dataset = replace_dataset_column(dataset, "advantages", df["advantages"].tolist())
    dataset = replace_dataset_column(dataset, "group_tokens", df["new_group_tokens"].tolist())
    dataset = replace_dataset_column(dataset, "overflow", df["overflow"].tolist())

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
