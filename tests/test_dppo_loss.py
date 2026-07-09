from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pipelinerl.finetune.rl import (
    RLConfig,
    compute_binary_tv_divergence,
    compute_dppo_mask,
    rl_step,
)
from pipelinerl.finetune.types import PipelineBatchEncoding


class _FixedPolicy(nn.Module):
    def __init__(self, sampled_token_probs: list[float]) -> None:
        super().__init__()
        probs = torch.tensor(sampled_token_probs, dtype=torch.float32)
        token_logits = torch.stack([torch.zeros_like(probs), torch.logit(probs)], dim=-1)
        final_position = torch.zeros(1, 2, dtype=torch.float32)
        self.logits = nn.Parameter(torch.cat([token_logits, final_position], dim=0))

    def forward(self, input_ids, attention_mask=None, labels=None, position_ids=None):
        return SimpleNamespace(logits=self.logits.unsqueeze(0))


def _packed_batch(
    advantages: list[float],
    behavior_probs: list[float],
) -> PipelineBatchEncoding:
    n_tokens = len(advantages)
    sequence_length = n_tokens + 1
    return PipelineBatchEncoding(
        input_ids=torch.tensor([[0, *([1] * n_tokens)]], dtype=torch.long),
        attention_mask=torch.ones(1, sequence_length, dtype=torch.long),
        labels=torch.tensor([[-100, *([1] * n_tokens)]], dtype=torch.long),
        position_ids=torch.arange(sequence_length, dtype=torch.long).unsqueeze(0),
        segment_ids=torch.zeros(1, sequence_length, dtype=torch.long),
        rewards=torch.zeros(1, sequence_length),
        advantages=torch.tensor([[0.0, *advantages]], dtype=torch.float32),
        ref_logprobs=torch.zeros(1, sequence_length),
        old_logprobs=torch.tensor(
            [[0.0, *torch.log(torch.tensor(behavior_probs, dtype=torch.float32)).tolist()]],
            dtype=torch.float32,
        ),
        group_tokens=torch.ones(1, sequence_length),
        num_labels=torch.full((1, sequence_length), float(n_tokens)),
        overflow=torch.zeros(1, sequence_length),
        model_version=0,
        is_packed=True,
    )


def _dppo_config(**kwargs) -> RLConfig:
    return RLConfig(
        policy_loss="dppo",
        batch_size=1,
        kl_coef=0.0,
        final_kl_coef=0.0,
        **kwargs,
    )


def test_binary_tv_divergence_uses_sampled_token_bernoulli_probabilities() -> None:
    behavior = torch.log(torch.tensor([[0.5, 0.4, 0.2]]))
    policy = torch.log(torch.tensor([[0.8, 0.1, 0.9]]))
    response_mask = torch.tensor([[True, False, True]])

    divergence = compute_binary_tv_divergence(behavior, policy, response_mask)

    torch.testing.assert_close(divergence, torch.tensor([[0.3, 0.0, 0.7]]))


def test_dppo_mask_blocks_only_outside_updates_moving_farther_away() -> None:
    behavior_probs = torch.full((1, 6), 0.5)
    policy_probs = torch.tensor([[0.8, 0.2, 0.8, 0.2, 0.8, 0.8]])
    behavior = torch.log(behavior_probs)
    policy = torch.log(policy_probs)
    ratio = policy_probs / behavior_probs
    weights = torch.tensor([[1.0, 1.0, -1.0, -1.0, 0.0, 1.0]])
    response_mask = torch.tensor([[True, True, True, True, True, False]])

    mask, divergence = compute_dppo_mask(
        policy,
        behavior,
        weights,
        ratio,
        response_mask,
        divergence_threshold=0.1,
    )

    torch.testing.assert_close(mask, torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0, 0.0]]))
    torch.testing.assert_close(divergence, torch.tensor([[0.3, 0.3, 0.3, 0.3, 0.3, 0.0]]))


def test_dppo_mask_keeps_exact_threshold() -> None:
    behavior = torch.log(torch.tensor([[0.5]]))
    policy = torch.log(torch.tensor([[0.8]]))
    ratio = torch.exp(policy - behavior)
    response_mask = torch.tensor([[True]])
    divergence = compute_binary_tv_divergence(behavior, policy, response_mask)

    mask, measured = compute_dppo_mask(
        policy,
        behavior,
        torch.tensor([[1.0]]),
        ratio,
        response_mask,
        divergence_threshold=float(divergence.item()),
    )

    torch.testing.assert_close(mask, torch.ones_like(mask))
    torch.testing.assert_close(measured, divergence)


def test_dppo_rl_step_matches_masked_surrogate_and_gradients() -> None:
    policy_probs = [0.8, 0.2, 0.2, 0.8, 0.55, 0.8]
    behavior_probs = [0.5] * len(policy_probs)
    advantages = [1.0, 1.0, -1.0, -1.0, 1.0, 0.0]
    model = _FixedPolicy(policy_probs)

    loss, stats = rl_step(
        model,
        _packed_batch(advantages, behavior_probs),
        current_step=0,
        max_step=1,
        config=_dppo_config(),
    )
    loss.backward()

    mask = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    ratio = torch.tensor(policy_probs) / torch.tensor(behavior_probs)
    expected_loss = -(mask * ratio * torch.tensor(advantages)).sum()
    torch.testing.assert_close(loss.detach(), expected_loss)

    gradient_norms = model.logits.grad[:-1].norm(dim=-1)
    torch.testing.assert_close(gradient_norms[[0, 2, 5]], torch.zeros(3))
    assert torch.all(gradient_norms[[1, 3, 4]] > 0)

    assert stats["dppo_mask_frac_kept"] == pytest.approx(4 / 6)
    assert stats["clamp_log_ratio_new_old_indicator"] == pytest.approx(2 / 6)
    assert stats["dppo_binary_tv_mean"] == pytest.approx((0.3 * 5 + 0.05) / 6)
    assert stats["dppo_binary_tv_max"] == pytest.approx(0.3)


def test_dppo_relu_weights_disable_negative_direction_gate() -> None:
    model = _FixedPolicy([0.2])

    loss, stats = rl_step(
        model,
        _packed_batch([-1.0], [0.5]),
        current_step=0,
        max_step=1,
        config=_dppo_config(relu_log_p_weights=True),
    )

    torch.testing.assert_close(loss.detach(), torch.tensor(0.0))
    assert stats["dppo_mask_frac_kept"] == pytest.approx(1.0)
    assert stats["clamp_log_ratio_new_old_indicator"] == pytest.approx(0.0)


def test_dppo_config_rejects_unsupported_divergence_and_threshold() -> None:
    config = _dppo_config()
    assert config.dppo_divergence_type == "binary_tv"
    assert config.dppo_divergence_threshold == 0.1

    with pytest.raises(ValueError):
        _dppo_config(dppo_divergence_type="tv")
    with pytest.raises(ValueError):
        _dppo_config(dppo_divergence_threshold=0.0)
    with pytest.raises(ValueError):
        _dppo_config(dppo_divergence_threshold=1.1)
