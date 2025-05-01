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
    "example_weight",
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
    aggregate_loss: Literal["mean", "sum"] = Field(
        default="sum",
        description="How to aggregate the loss within a batch (when batch size is 1, there is no difference)",
    )
    overlong_filtering: bool = Field(
        default=False,
        description="Filter out sequence that do not have eos_token_id"
    )
    temperature: float = Field(
        default=1.0,
        description="Temperature for the training log probs",
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
    examples_weights = batch["example_weight"][:, 1:]
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
    assert loss.shape == examples_weights.shape, f"Loss shape {loss.shape} does not match example weights shape {examples_weights.shape}"
    loss = loss * examples_weights  # 1 x (BxL) x 1

    if config.aggregate_loss == "mean":
        final_loss = -mean_sum(loss, masks_shifted, segments)
    elif config.aggregate_loss == "sum":
        final_loss = -sum_sum(loss, masks_shifted, segments)
    else:
        raise ValueError(f"{config.aggregate_loss} is not defined")

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
        "example_weight": mean_sum(examples_weights, masks_shifted, segments).item(),
        "kl_coef": num_sequences * kl_coef,
        "entropy_bonus_coef": num_sequences * entropy_bonus_coef,
    }

    return final_loss, stats


def update_rewards_and_advantages(dataset: Dataset, eos_token_id: int, config: RLConfig) -> Dataset:
    """
    Updates the advantages column in the given dataset based on reward statistics.

    Args:
        dataset (Dataset): The input dataset containing rewards and placeholder advantages.

    Returns:
        Dataset: The updated dataset with the updated advantages column.

    """
    processed_items: list[dict] = []
    group_ids: list[int] = []
    rewards_scalar: list[float] = []
    token_lens: list[int] = []

    for item in dataset:
        example = item.copy()

        if "old_logprobs" not in example:
            example["old_logprobs"] = example.get("logprobs", [])
        if "ref_logprobs" not in example:
            example["ref_logprobs"] = example.get("ref_logprobs", [])

        reward_scalar = float(example.get("reward", 0.0))
        rewards_list = example.get("rewards", [reward_scalar] * len(example["input_ids"]))

        if config.reward_minus_kl_coef > 0:
            item_for_kl = {
                "rewards": rewards_list,
                "old_logprobs": example["old_logprobs"],
                "ref_logprobs": example["ref_logprobs"],
            }
            rewards_list = calculate_rewards_with_implicit_kl(
                item_for_kl, reward_minus_kl_coef=config.reward_minus_kl_coef
            )
            reward_scalar = float(np.mean(rewards_list)) if rewards_list else 0.0

        example["rewards"] = rewards_list
        example["reward"] = reward_scalar

        processed_items.append(example)
        group_ids.append(example["group_id"])
        rewards_scalar.append(reward_scalar)
        token_lens.append(len(example["input_ids"]))

    group_ids_np = np.asarray(group_ids)
    rewards_np = np.asarray(rewards_scalar, dtype=np.float32)
    token_lens_np = np.asarray(token_lens, dtype=np.float32)

    _, inverse_inds = np.unique(group_ids_np, return_inverse=True)
    counts = np.bincount(inverse_inds)
    reward_sum = np.bincount(inverse_inds, weights=rewards_np)
    reward_sq_sum = np.bincount(inverse_inds, weights=rewards_np ** 2)
    token_sum = np.bincount(inverse_inds, weights=token_lens_np)

    reward_mean_per_group = reward_sum / np.maximum(counts, 1)
    reward_var_per_group = reward_sq_sum / np.maximum(counts, 1) - reward_mean_per_group ** 2
    reward_std_per_group = np.sqrt(np.maximum(0.0, reward_var_per_group))
    avg_tokens_per_group = token_sum / np.maximum(counts, 1)

    reward_mean_arr = reward_mean_per_group[inverse_inds]
    reward_std_arr = reward_std_per_group[inverse_inds]
    avg_tokens_arr = avg_tokens_per_group[inverse_inds]

    for idx, example in enumerate(processed_items):
        advantages = calculate_advantage(
            {
                "rewards": example["rewards"],
                "reward_mean": float(reward_mean_arr[idx]),
                "reward_std": float(reward_std_arr[idx]),
            }
        )
        example["advantages"] = advantages

        example["group_tokens"] = [float(avg_tokens_arr[idx])] * len(example["input_ids"])

        full_len = len(example["input_ids"])
        old_logs = example["old_logprobs"]
        ref_logs = example["ref_logprobs"]
        example["old_logprobs"] = [0.0] * (full_len - len(old_logs)) + old_logs
        example["ref_logprobs"] = [0.0] * (full_len - len(ref_logs)) + ref_logs

        has_eos = eos_token_id in example["input_ids"]
        example["overflow"] = [0.0 if has_eos else 1.0] * full_len

        if config.overlong_filtering and not has_eos:
            example["example_weight"] = [0.0] * full_len
        else:
            example["example_weight"] = [1.0] * full_len

    return Dataset.from_list(processed_items)


def assign_example_weights(dataset: Dataset, eos_token_id: int, config: RLConfig) -> Dataset:
    """
    Ensure ``example_weight`` and ``overflow`` columns exist. If they are
    already present the dataset is returned unchanged; otherwise the
    columns are created in a single pass without using pandas.
    """
    if "example_weight" in dataset.column_names and "overflow" in dataset.column_names:
        return dataset

    processed: list[dict] = []
    for item in dataset:
        ex = item.copy()
        has_eos = eos_token_id in ex["input_ids"]
        full_len = len(ex["input_ids"])

        ex["overflow"] = [0.0 if has_eos else 1.0] * full_len
        if config.overlong_filtering and not has_eos:
            ex["example_weight"] = [0.0] * full_len
        else:
            ex["example_weight"] = [1.0] * full_len
        processed.append(ex)

    return Dataset.from_list(processed)


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
    dataset = assign_example_weights(dataset, eos_token_id, config)

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
