import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, TYPE_CHECKING
from pydantic import BaseModel, Field

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from pipelinerl.finetune.types import PipelineBatchEncoding
from pipelinerl.finetune.rl.utils import per_segment_sums

if TYPE_CHECKING:
    from transformers import PreTrainedModel
else:
    PreTrainedModel = Any

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
    rollout_level_loo: bool = Field(
        default=False,
        description="Use rollout-level leave-one-out baseline instead of per-step-index baseline",
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
    value_loss_coef: float = Field(
        default=0.0,
        description="Coefficient for the value loss in the final loss",
    )
    multi_turn_credit: bool = Field(
        default=False,
        description="Use turn-end value predictions to decompose rollout-level advantages",
    )
    frozen_probe_credit: Literal["off", "shadow", "live"] = Field(
        default="off",
        description="Frozen-probe credit mode: off, shadow, or live",
    )
    frozen_probe_path: str | None = Field(
        default=None,
        description="Path to a frozen rollout-success probe artifact",
    )


@dataclass
class FrozenProbeRuntime:
    layer: torch.nn.Module
    w_prime: torch.Tensor
    b_prime: float
    _device_w_prime: torch.Tensor | None = None

    def weights_for(self, device: torch.device) -> torch.Tensor:
        if self._device_w_prime is None or self._device_w_prime.device != device:
            self._device_w_prime = self.w_prime.to(device=device)
        return self._device_w_prime


def _model_hidden_size(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        text_config = getattr(config, "text_config", None)
        hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("frozen probe requires model config.hidden_size or config.text_config.hidden_size")
    return int(hidden_size)


def load_frozen_probe_runtime(path: str, model: torch.nn.Module) -> FrozenProbeRuntime:
    from pipelinerl.domains.terminal.turn_probes import find_decoder_layers

    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("version") != 1:
        raise ValueError("frozen probe artifact must have version == 1")
    if artifact.get("target") != "rollout_success":
        raise ValueError("frozen probe artifact target must be rollout_success")

    layer_index_value = artifact.get("layer_index")
    if layer_index_value is None:
        raise ValueError("frozen probe artifact layer_index must be 31")
    layer_index = int(layer_index_value)
    if layer_index != 31:
        raise ValueError("frozen probe artifact layer_index must be 31")

    w_prime = artifact.get("w_prime")
    if not isinstance(w_prime, torch.Tensor):
        raise ValueError("frozen probe artifact w_prime must be a tensor")
    hidden_size = _model_hidden_size(model)
    if w_prime.shape != (hidden_size,) or w_prime.dtype != torch.float32 or w_prime.device.type != "cpu":
        raise ValueError(
            "frozen probe artifact w_prime must be a fp32 cpu tensor shaped "
            f"[{hidden_size}], got shape={tuple(w_prime.shape)} dtype={w_prime.dtype} device={w_prime.device}"
        )

    b_prime = artifact.get("b_prime")
    if not isinstance(b_prime, (float, int)):
        raise ValueError("frozen probe artifact b_prime must be a float")

    layers = find_decoder_layers(model)
    if layer_index >= len(layers):
        raise ValueError(f"frozen probe layer_index {layer_index} outside model with {len(layers)} layers")

    return FrozenProbeRuntime(layer=layers[layer_index], w_prime=w_prime.contiguous(), b_prime=float(b_prime))


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


def _segment_bound(value: Any) -> int:
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def _turn_boundary_indices(
    segments: list[tuple[Any, Any]], masks_shifted: torch.Tensor, offset_index: int
) -> torch.LongTensor:
    if masks_shifted.dim() != 2 or masks_shifted.shape[0] != 1:
        raise ValueError(f"Expected masks_shifted shaped [1, L], got {tuple(masks_shifted.shape)}")

    indices: list[int] = []
    max_length = masks_shifted.shape[1]
    for start, end in segments:
        start_i = _segment_bound(start)
        end_i = min(_segment_bound(end), max_length)
        if start_i >= end_i:
            continue
        segment_mask = masks_shifted[0, start_i:end_i].bool()
        valid_offsets = torch.nonzero(segment_mask, as_tuple=False).flatten()
        if valid_offsets.numel() == 0:
            continue
        indices.append(start_i + int(valid_offsets[offset_index].item()))
    return torch.tensor(indices, dtype=torch.long, device=masks_shifted.device)


def turn_start_indices(segments: list[tuple[Any, Any]], masks_shifted: torch.Tensor) -> torch.LongTensor:
    return _turn_boundary_indices(segments, masks_shifted, 0)


def turn_end_indices(segments: list[tuple[Any, Any]], masks_shifted: torch.Tensor) -> torch.LongTensor:
    return _turn_boundary_indices(segments, masks_shifted, -1)


def _forward_with_frozen_probe_capture(
    model: PreTrainedModel,
    model_inputs: dict[str, Any],
    frozen_probe: FrozenProbeRuntime | None,
):
    if frozen_probe is None:
        return model(**model_inputs), None

    captured_hidden = None

    def capture_hidden(_module, _inputs, output):
        nonlocal captured_hidden
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        captured_hidden = hidden.detach()

    hook = frozen_probe.layer.register_forward_hook(capture_hidden)
    try:
        outputs = model(**model_inputs)
    finally:
        hook.remove()

    if captured_hidden is None:
        raise RuntimeError("Frozen-probe hidden-state hook did not fire")
    return outputs, captured_hidden.float()


def multi_turn_credit_advantages(
    segments: list[tuple[Any, Any]],
    masks_shifted: torch.Tensor,
    value_predictions: torch.Tensor,
    centered_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if value_predictions.shape != centered_targets.shape or value_predictions.shape != masks_shifted.shape:
        raise ValueError(
            "value_predictions, centered_targets, and masks_shifted must have matching shapes; "
            f"got {tuple(value_predictions.shape)}, {tuple(centered_targets.shape)}, {tuple(masks_shifted.shape)}"
        )

    turn_indices = turn_end_indices(segments, masks_shifted)
    expanded_advantages = torch.zeros_like(centered_targets)
    turn_values = value_predictions[0, turn_indices]
    turn_targets = centered_targets[0, turn_indices]
    turn_advantages = turn_targets - turn_values.detach()

    turn_idx = 0
    max_length = masks_shifted.shape[1]
    for start, end in segments:
        start_i = _segment_bound(start)
        end_i = min(_segment_bound(end), max_length)
        if start_i >= end_i:
            continue
        segment_mask = masks_shifted[0, start_i:end_i].bool()
        if not segment_mask.any():
            continue
        expanded_advantages[0, start_i:end_i] = torch.where(
            segment_mask,
            turn_advantages[turn_idx],
            expanded_advantages[0, start_i:end_i],
        )
        turn_idx += 1

    return expanded_advantages, turn_indices, turn_values, turn_targets, turn_advantages


def frozen_probe_advantages(
    segments: list[tuple[Any, Any]],
    masks_shifted: torch.Tensor,
    hidden_states: torch.Tensor,
    rewards: torch.Tensor,
    frozen_probe: FrozenProbeRuntime,
) -> tuple[torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if hidden_states.dim() != 3 or hidden_states.shape[0] != 1:
        raise ValueError(f"Expected hidden_states shaped [1, L, H], got {tuple(hidden_states.shape)}")
    if rewards.shape != masks_shifted.shape:
        raise ValueError(f"rewards and masks_shifted must have matching shapes, got {tuple(rewards.shape)} and {tuple(masks_shifted.shape)}")

    turn_indices = turn_start_indices(segments, masks_shifted)
    expanded_advantages = torch.zeros_like(rewards)
    if turn_indices.numel() == 0:
        empty = torch.empty(0, dtype=torch.float32, device=hidden_states.device)
        return expanded_advantages, turn_indices, empty, empty, empty

    h = hidden_states[0, turn_indices, :].float()
    w_prime = frozen_probe.weights_for(h.device)
    logits = (h * w_prime).sum(dim=-1) + frozen_probe.b_prime
    turn_probs = torch.sigmoid(logits)
    turn_targets = (rewards[0, turn_indices] >= 0.999).to(dtype=turn_probs.dtype)
    turn_advantages = turn_targets - turn_probs

    turn_idx = 0
    max_length = masks_shifted.shape[1]
    for start, end in segments:
        start_i = _segment_bound(start)
        end_i = min(_segment_bound(end), max_length)
        if start_i >= end_i:
            continue
        segment_mask = masks_shifted[0, start_i:end_i].bool()
        if not segment_mask.any():
            continue
        segment_rewards = rewards[0, start_i:end_i][segment_mask]
        if not torch.all(segment_rewards == segment_rewards[0]):
            raise ValueError("frozen probe credit requires constant rewards within each segment")
        expanded_advantages[0, start_i:end_i] = torch.where(
            segment_mask,
            turn_advantages[turn_idx].to(dtype=expanded_advantages.dtype),
            expanded_advantages[0, start_i:end_i],
        )
        turn_idx += 1

    return expanded_advantages, turn_indices, turn_probs, turn_targets, turn_advantages


def rl_step(
    model: PreTrainedModel,
    batch: PipelineBatchEncoding,
    current_step: int,
    max_step: int,
    config: RLConfig,
    seq_parallel_group=None,
    frozen_probe: FrozenProbeRuntime | None = None,
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

    if config.multi_turn_credit:
        if not has_value_head:
            raise ValueError("multi_turn_credit requires a value head")
        if config.policy_loss != "gspo":
            raise ValueError("multi_turn_credit requires policy_loss='gspo'")
        if segments is None:
            raise ValueError("multi_turn_credit requires packed sequences with segments")

    if config.frozen_probe_credit not in {"off", "shadow", "live"}:
        raise ValueError("frozen_probe_credit must be one of off, shadow, live")
    if config.frozen_probe_credit != "off":
        if frozen_probe is None:
            raise ValueError("frozen_probe_credit requires frozen_probe_path")
        if config.policy_loss != "gspo":
            raise ValueError("frozen_probe_credit requires policy_loss='gspo'")
        if segments is None:
            raise ValueError("frozen_probe_credit requires packed sequences with segments")

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
    
    active_frozen_probe = frozen_probe if config.frozen_probe_credit != "off" else None
    outputs, _frozen_probe_hidden = _forward_with_frozen_probe_capture(model, model_inputs, active_frozen_probe)

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

    assert new_logprobs.shape == ref_logprobs.shape

    log_ratio_new_old = new_logprobs - old_logprobs
    abs_log_ratio_new_old = torch.abs(log_ratio_new_old)
    ratio_new_old = torch.exp(log_ratio_new_old)
    log_ratio_ref_new = ref_logprobs - new_logprobs
    assert torch.isfinite(log_ratio_ref_new).all(), f"log_ratio_ref_new is not finite: {log_ratio_ref_new}"

    frozen_probe_expanded_advantages = None
    frozen_probe_turn_probs = None
    frozen_probe_turn_targets = None
    frozen_probe_turn_advantages = None
    if config.frozen_probe_credit != "off":
        assert frozen_probe is not None
        assert segments is not None
        if _frozen_probe_hidden is None:
            raise RuntimeError("Frozen-probe hidden capture missing despite loaded probe")
        (
            frozen_probe_expanded_advantages,
            _frozen_probe_turn_idx,
            frozen_probe_turn_probs,
            frozen_probe_turn_targets,
            frozen_probe_turn_advantages,
        ) = frozen_probe_advantages(segments, masks_shifted, _frozen_probe_hidden, rewards, frozen_probe)

    turn_end_idx = None
    turn_values = None
    turn_value_targets = None
    turn_advantages = None
    if has_value_head:
        value_predictions = outputs.value[:, :-1]  # no target for the last token
        if config.policy_loss == "gspo":
            assert segments is not None
            centered_targets = batch.advantages[:, 1:]
            (
                residual_advantages,
                turn_end_idx,
                turn_values,
                turn_value_targets,
                turn_advantages,
            ) = multi_turn_credit_advantages(segments, masks_shifted, value_predictions, centered_targets)
            advantages = residual_advantages if config.multi_turn_credit else centered_targets
        else:
            # Legacy value-head mode: replace precomputed advantages with raw reward residuals.
            advantages = rewards - value_predictions
    else:
        value_predictions = None
        advantages = batch.advantages[:, 1:]

    if config.frozen_probe_credit == "live":
        assert frozen_probe_expanded_advantages is not None
        advantages = frozen_probe_expanded_advantages

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
        assert value_predictions is not None
        values = value_predictions
        assert values.shape == tokens_weights.shape, (
            f"Values shape {values.shape} does not match example weights shape {tokens_weights.shape}"
        )
        if config.policy_loss == "gspo":
            assert turn_end_idx is not None
            assert turn_values is not None
            assert turn_value_targets is not None
            if turn_end_idx.numel() == 0:
                value_loss = (values * 0).sum()
            else:
                turn_weights = tokens_weights[0, turn_end_idx]
                value_loss = 0.5 * torch.square(turn_values - turn_value_targets) * turn_weights
                value_loss = value_loss.sum()
        else:
            value_loss = 0.5 * torch.square(values - rewards) * tokens_weights
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
        "kl_coef": num_sequences * kl_coef,
        "entropy_bonus_coef": num_sequences * entropy_bonus_coef,
        "num_output_tokens_sum": masks_shifted.sum().item(),
        "input_size": batch.input_ids.numel(), 
    }

    if frozen_probe_turn_probs is not None:
        assert frozen_probe_turn_targets is not None
        assert frozen_probe_turn_advantages is not None
        stats["frozen_probe/p_mean"] = frozen_probe_turn_probs.mean().item() if frozen_probe_turn_probs.numel() else 0.0
        success_mask = frozen_probe_turn_targets == 1
        fail_mask = frozen_probe_turn_targets == 0
        stats["frozen_probe/p_at_success"] = (
            frozen_probe_turn_probs[success_mask].mean().item() if success_mask.any() else 0.0
        )
        stats["frozen_probe/p_at_fail"] = (
            frozen_probe_turn_probs[fail_mask].mean().item() if fail_mask.any() else 0.0
        )
        stats["frozen_probe/A_mean"] = frozen_probe_turn_advantages.mean().item() if frozen_probe_turn_advantages.numel() else 0.0
        stats["frozen_probe/A_min"] = frozen_probe_turn_advantages.min().item() if frozen_probe_turn_advantages.numel() else 0.0
        stats["frozen_probe/A_max"] = frozen_probe_turn_advantages.max().item() if frozen_probe_turn_advantages.numel() else 0.0

    if has_value_head:
        assert value_predictions is not None
        stats["value_mean"] = sum_sum(value_predictions / num_labels_in_seq, masks_shifted, segments).item()
        stats["value_max"] = value_predictions[masks_shifted].max().item() if masks_shifted.any() else 0.0
        stats["value_min"] = value_predictions[masks_shifted].min().item() if masks_shifted.any() else 0.0
        stats["value_loss"] = value_loss.item()
        if config.policy_loss == "gspo":
            assert turn_values is not None
            assert turn_value_targets is not None
            assert turn_advantages is not None
            if turn_values.numel() == 0:
                stats["value_mse"] = 0.0
                if config.multi_turn_credit:
                    stats["multi_turn_advantage_mean"] = 0.0
                    stats["multi_turn_advantage_min"] = 0.0
                    stats["multi_turn_advantage_max"] = 0.0
            else:
                residuals = torch.square(turn_values - turn_value_targets)
                stats["value_mse"] = residuals.mean().item()
                if config.multi_turn_credit:
                    stats["multi_turn_advantage_mean"] = turn_advantages.mean().item()
                    stats["multi_turn_advantage_min"] = turn_advantages.min().item()
                    stats["multi_turn_advantage_max"] = turn_advantages.max().item()
        else:
            stats["value_mse"] = sum_sum(
                torch.square(value_predictions - rewards) / num_labels_in_seq, masks_shifted, segments
            ).item()

    return final_loss, stats


def populate_rl_data(dataset: list[dict[str, Any]], eos_token_id: int, config: RLConfig) -> list[dict[str, Any]]:
    """Populate RL-specific columns (advantages, overflow, num_labels) using a leave-one-out baseline."""
    # Convert to pandas for processing
    df_init = pd.DataFrame(dataset)
    assert isinstance(df_init, pd.DataFrame)

    # Step 1: calculate rollout-level token statistics and step-level reward statistics
    df_stats = df_init[["group_id", "rollout_index", "step_index", "rewards"]].copy()
    df_stats["num_tokens"] = df_init["input_ids"].apply(len)
    df_stats["step_reward"] = df_stats["rewards"].apply(lambda rewards: rewards[0])
    df_rollouts = (
        df_stats.groupby(["group_id", "rollout_index"])
        .agg(
            rollout_tokens=("num_tokens", "sum"),
        )
        .reset_index()
    )
    df_group_tokens = (
        df_rollouts.groupby("group_id")
        .agg(
            group_tokens=("rollout_tokens", "mean"),
        )
        .reset_index()
    )
    assert df_group_tokens.columns.tolist() == [
        "group_id",
        "group_tokens",
    ]

    # Step 2: calculate advantages for each sample
    if config.rollout_level_loo:
        current_reward_keys = ["group_id", "rollout_index"]
        baseline_group_keys = ["group_id"]
        df_current_rewards = (
            df_stats.groupby(current_reward_keys)
            .agg(current_reward=("step_reward", "mean"))
            .reset_index()
        )
    else:
        current_reward_keys = ["group_id", "rollout_index", "step_index"]
        baseline_group_keys = ["group_id", "step_index"]
        df_current_rewards = df_stats[[*current_reward_keys, "step_reward"]].rename(
            columns={"step_reward": "current_reward"}
        )

    df_grouped = (
        df_current_rewards.groupby(baseline_group_keys)
        .agg(
            current_reward_sum=("current_reward", "sum"),
            current_reward_count=("current_reward", "count"),
            current_reward_std=("current_reward", "std"),
        )
        .reset_index()
    )
    df_advantages = pd.merge(
        df_stats[["group_id", "rollout_index", "step_index", "rewards"]],
        df_current_rewards,
        on=current_reward_keys,
        how="left",
    )
    df_advantages = pd.merge(df_advantages, df_grouped, on=baseline_group_keys, how="left")
    df_advantages = pd.merge(df_advantages, df_group_tokens, on="group_id", how="left")
    assert len(df_advantages) == len(df_init)

    def calculate_advantages(row):
        rewards = row["rewards"]
        group_count = row["current_reward_count"]
        current_reward = row["current_reward"]
        if group_count > 1:
            loo_mean = (row["current_reward_sum"] - current_reward) / (group_count - 1)
        elif config.rollout_level_loo:
            return [0.0 for _ in rewards]
        else:
            loo_mean = current_reward
        std = row["current_reward_std"]
        if config.divide_advantage_by_std:
            return [(r - loo_mean) / (np.nan_to_num(std) + 1e-4) for r in rewards]
        return [(r - loo_mean) for r in rewards]

    df_advantages["advantages"] = df_advantages.apply(calculate_advantages, axis=1)
    df_advantages = df_advantages.drop(
        columns=["rewards", "current_reward", "current_reward_sum", "current_reward_count", "current_reward_std"]
    )
    assert df_advantages.columns.tolist() == [
        "group_id",
        "rollout_index",
        "step_index",
        "group_tokens",
        "advantages",
    ]

    # Step 3: bring advantages and group level stats back to the main df
    df = df_init.drop(columns=["advantages", "group_tokens"])
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

    # Step 5: move the results back to the dataset
    advantages_list = df["advantages"].tolist()
    group_tokens_list = df["group_tokens"].tolist()
    overflow_list = df["overflow"].tolist()
    num_labels_list = df["num_labels"].tolist()
    for i, entry in enumerate(dataset):
        entry["advantages"] = advantages_list[i]
        entry["group_tokens"] = group_tokens_list[i]
        entry["overflow"] = overflow_list[i]
        entry["num_labels"] = num_labels_list[i]
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
    return encoding
