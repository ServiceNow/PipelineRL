import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pipelinerl.domains.tau2.client import Tau2RunResponse
from pipelinerl.domains.tau2.rollouts import build_tau2_training_text
from pipelinerl.finetune.data import collate_packed
from pipelinerl.finetune.rl import (
    RLConfig,
    compute_binary_tv_divergence,
    compute_dppo_mask,
    prepare_rl_fields,
    rl_step,
)
from pipelinerl.finetune.types import PipelineBatchEncoding


_CURRENT_HEAD_SNAPSHOTS = {
    "ppo": {
        "loss": -0.3000004291534424,
        "gradient": [
            [0.0, 0.0],
            [0.3199999928474426, -0.3199999928474426],
            [0.0, 0.0],
            [-0.31999993324279785, 0.3200000524520874],
            [0.4950000047683716, -0.4950000047683716],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        "stats": {
            "clamp_log_ratio_new_old_indicator": 0.8333333730697632,
            "kl_new_old": 0.17121002078056335,
            "ratio_new_old": 1.116666555404663,
            "num_output_tokens_sum": 6,
            "token_weight": 1.0,
        },
    },
    "reinforce": {
        "loss": 0.6576206088066101,
        "gradient": [
            [0.23999997973442078, -0.24000006914138794],
            [0.3199999928474426, -0.3199999928474426],
            [-0.3199999928474426, 0.3199999928474426],
            [-0.23999997973442078, 0.24000006914138794],
            [0.4950000047683716, -0.4950000047683716],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        "stats": {
            "clamp_log_ratio_new_old_indicator": 0.5,
            "kl_new_old": 0.17121002078056335,
            "ratio_new_old": 0.9166666269302368,
            "num_output_tokens_sum": 6,
            "token_weight": 1.0,
        },
    },
    "gspo": {
        "loss": -0.9469174146652222,
        "gradient": [
            [0.031563907861709595, -0.03156392276287079],
            [0.12625566124916077, -0.12625566124916077],
            [0.12625566124916077, -0.12625566124916077],
            [0.031563907861709595, -0.03156392276287079],
            [0.07101880759000778, -0.07101880759000778],
            [0.031563907861709595, -0.03156392276287079],
            [0.0, 0.0],
        ],
        "stats": {
            "clamp_log_ratio_new_old_indicator": 0.0,
            "kl_new_old": 0.17121002078056335,
            "ratio_new_old": 1.116666555404663,
            "num_output_tokens_sum": 6,
            "token_weight": 1.0,
        },
    },
}


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


def _loss_config(policy_loss: str, **kwargs) -> RLConfig:
    return RLConfig(
        policy_loss=policy_loss,
        batch_size=1,
        kl_coef=0.0,
        final_kl_coef=0.0,
        **kwargs,
    )


@pytest.mark.parametrize("policy_loss", ["ppo", "reinforce", "gspo"])
def test_existing_policy_losses_match_pre_dppo_current_head_snapshot(policy_loss: str) -> None:
    model = _FixedPolicy([0.8, 0.2, 0.2, 0.8, 0.55, 0.8])

    loss, stats = rl_step(
        model,
        _packed_batch(
            advantages=[1.0, 1.0, -1.0, -1.0, 1.0, 0.0],
            behavior_probs=[0.5] * 6,
        ),
        current_step=0,
        max_step=1,
        config=_loss_config(policy_loss),
    )
    loss.backward()

    expected = _CURRENT_HEAD_SNAPSHOTS[policy_loss]
    torch.testing.assert_close(loss.detach(), torch.tensor(expected["loss"]))
    torch.testing.assert_close(model.logits.grad, torch.tensor(expected["gradient"]))
    for key, value in expected["stats"].items():
        assert stats[key] == pytest.approx(value)


class _TinyTokenizer:
    padding_side = "right"

    def decode(self, input_ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return " ".join(str(token_id) for token_id in input_ids)

    def get_vocab(self):
        return {"zero": 0, "one": 1}


def _mixed_version_training_text():
    endpoint = "http://actor-0:8000/v1"
    behavior_probs = [0.5, 0.4, 0.5, 0.8]
    response = Tau2RunResponse(
        reward=1.0,
        responses_create_params={
            "input": [
                {"role": "system", "content": "policy"},
                {"role": "assistant", "content": "How can I help?"},
                {"role": "user", "content": "request"},
            ]
        },
        response={
            "output": [
                {
                    "type": "function_call",
                    "prompt_token_ids": [0],
                    "generation_token_ids": [1, 1],
                    "generation_log_probs": [
                        math.log(behavior_probs[0]),
                        math.log(behavior_probs[1]),
                    ],
                    "model_version_start": 8,
                    "model_version_end": 10,
                    "policy_endpoint": endpoint,
                },
                {
                    "type": "function_call",
                    "prompt_token_ids": [0, 1, 1, 0],
                    "generation_token_ids": [1, 1],
                    "generation_log_probs": [
                        math.log(behavior_probs[2]),
                        math.log(behavior_probs[3]),
                    ],
                    "model_version_start": 10,
                    "model_version_end": 10,
                    "policy_endpoint": endpoint,
                },
            ]
        },
        result={"termination_reason": "user_stop"},
        num_agent_calls=3,
    )
    return build_tau2_training_text(
        response,
        _TinyTokenizer(),
        expected_policy_endpoint=endpoint,
        max_sequence_length=16,
        shared_memory_entry_size=1_000_000,
    )


def _mixed_version_batch():
    text = _mixed_version_training_text()
    entry = prepare_rl_fields(
        {"input_ids": text.input_ids, "labels": text.labels},
        reward=text.reward,
        old_logprobs=text.logprobs,
        ref_logprobs=text.ref_logprobs,
    )
    n_labels = sum(label != -100 for label in text.labels)
    entry["advantages"] = [1.0] * len(text.input_ids)
    entry["num_labels"] = [float(n_labels)] * len(text.input_ids)
    entry["model_version"] = text.metadata["model_version"]
    batch = collate_packed([entry], tokenizer=_TinyTokenizer(), seq_parallel=1)
    return text, batch


def _run_mixed_version_loss(policy_loss: str):
    model = _FixedPolicy([0.8, 0.2, 0.5, 0.55, 0.6])
    text, batch = _mixed_version_batch()
    loss, stats = rl_step(
        model,
        batch,
        current_step=0,
        max_step=1,
        config=_loss_config(policy_loss),
    )
    loss.backward()
    return text, batch, loss.detach(), model.logits.grad.detach().clone(), stats


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


def test_dppo_relu_weights_disable_negative_direction_gate() -> None:
    model = _FixedPolicy([0.2])

    loss, stats = rl_step(
        model,
        _packed_batch([-1.0], [0.5]),
        current_step=0,
        max_step=1,
        config=_loss_config("dppo", relu_log_p_weights=True),
    )

    torch.testing.assert_close(loss.detach(), torch.tensor(0.0))
    assert stats["dppo_mask_frac_kept"] == pytest.approx(1.0)
    assert stats["clamp_log_ratio_new_old_indicator"] == pytest.approx(0.0)


def test_dppo_config_is_default_off_and_rejects_unsupported_values() -> None:
    assert RLConfig().policy_loss == "ppo"

    config = _loss_config("dppo")
    assert config.dppo_divergence_type == "binary_tv"
    assert config.dppo_divergence_threshold == 0.1

    with pytest.raises(ValueError):
        _loss_config("dppo", dppo_divergence_type="tv")
    with pytest.raises(ValueError):
        _loss_config("dppo", dppo_divergence_threshold=0.0)
    with pytest.raises(ValueError):
        _loss_config("dppo", dppo_divergence_threshold=1.1)


def _analytic_gradient(
    policy_probs: list[float],
    logprob_coefficients: list[float],
) -> torch.Tensor:
    rows = []
    for probability, coefficient in zip(policy_probs, logprob_coefficients):
        complement = 1.0 - probability
        rows.append([-coefficient * complement, coefficient * complement])
    return torch.tensor([*rows, [0.0, 0.0]], dtype=torch.float32)


def test_mixed_version_tau2_gspo_dppo_gate_writes_comparison(
    tmp_path: Path,
) -> None:
    gspo_text, gspo_batch, gspo_loss, gspo_gradient, gspo_stats = _run_mixed_version_loss("gspo")
    dppo_text, dppo_batch, dppo_loss, dppo_gradient, dppo_stats = _run_mixed_version_loss("dppo")

    expected_calls = [
        {
            "call_index": 0,
            "version_start": 8,
            "version_end": 10,
            "endpoint": "http://actor-0:8000/v1",
            "token_start": 1,
            "token_end": 3,
        },
        {
            "call_index": 1,
            "version_start": 10,
            "version_end": 10,
            "endpoint": "http://actor-0:8000/v1",
            "token_start": 4,
            "token_end": 6,
        },
    ]
    assert gspo_text.metadata["model_calls"] == expected_calls
    assert dppo_text.metadata["model_calls"] == expected_calls
    assert gspo_text.metadata["model_version"] == 8
    assert gspo_text.metadata["model_version_min"] == 8
    assert gspo_text.metadata["model_version_max"] == 10
    assert gspo_text.metadata["model_version_spread"] == 2
    assert gspo_batch.model_version == dppo_batch.model_version == 8

    behavior_probs = torch.tensor([0.5, 0.4, 0.5, 0.8])
    policy_probs = torch.tensor([0.8, 0.2, 0.55, 0.6])
    labeled_mask = gspo_batch.labels != -100
    torch.testing.assert_close(
        torch.exp(gspo_batch.old_logprobs[labeled_mask]),
        behavior_probs,
    )
    assert torch.equal(gspo_batch.input_ids, dppo_batch.input_ids)
    assert torch.equal(gspo_batch.labels, dppo_batch.labels)

    group_ratio = float(torch.prod(policy_probs / behavior_probs).pow(0.25))
    expected_gspo_gradient = _analytic_gradient(
        [0.8, 0.2, 0.5, 0.55, 0.6],
        [-group_ratio, -group_ratio, 0.0, -group_ratio, -group_ratio],
    )
    expected_dppo_gradient = _analytic_gradient(
        [0.8, 0.2, 0.5, 0.55, 0.6],
        [0.0, -0.5, 0.0, -1.1, -0.75],
    )

    torch.testing.assert_close(gspo_loss, torch.tensor(-4.0 * group_ratio))
    torch.testing.assert_close(dppo_loss, torch.tensor(-2.35))
    torch.testing.assert_close(gspo_gradient, expected_gspo_gradient)
    torch.testing.assert_close(dppo_gradient, expected_dppo_gradient)
    assert torch.isfinite(gspo_gradient).all()
    assert torch.isfinite(dppo_gradient).all()
    assert gspo_stats["clamp_log_ratio_new_old_indicator"] == pytest.approx(0.0)
    assert dppo_stats["clamp_log_ratio_new_old_indicator"] == pytest.approx(0.25)
    assert dppo_stats["dppo_mask_frac_kept"] == pytest.approx(0.75)
    assert dppo_stats["dppo_binary_tv_mean"] == pytest.approx(0.1875)
    assert dppo_stats["dppo_binary_tv_max"] == pytest.approx(0.3)

    labeled_prediction_rows = [0, 1, 3, 4]
    gspo_active = sum(
        gspo_gradient[index].norm().item() > 1e-7
        for index in labeled_prediction_rows
    )
    dppo_active = sum(
        dppo_gradient[index].norm().item() > 1e-7
        for index in labeled_prediction_rows
    )
    comparison = {
        "behavior_probabilities": [0.5, 0.4, 0.5, 0.8],
        "call_versions": [[8, 10], [10, 10]],
        "dppo": {
            "active_labeled_gradients": dppo_active,
            "binary_tv_max": round(dppo_stats["dppo_binary_tv_max"], 6),
            "binary_tv_mean": round(dppo_stats["dppo_binary_tv_mean"], 6),
            "blocked_fraction": round(dppo_stats["clamp_log_ratio_new_old_indicator"], 6),
            "gradient_l2": round(dppo_gradient.norm().item(), 6),
            "loss": round(dppo_loss.item(), 6),
            "mask_kept_fraction": round(dppo_stats["dppo_mask_frac_kept"], 6),
        },
        "gspo": {
            "active_labeled_gradients": gspo_active,
            "clip_fraction": round(gspo_stats["clamp_log_ratio_new_old_indicator"], 6),
            "gradient_l2": round(gspo_gradient.norm().item(), 6),
            "loss": round(gspo_loss.item(), 6),
        },
        "policy_probabilities": [0.8, 0.2, 0.55, 0.6],
        "rollout_version": 8,
    }
    assert gspo_active == 4
    assert dppo_active == 3

    comparison_path = tmp_path / "tau2_mixed_version_loss.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    assert json.loads(comparison_path.read_text()) == comparison
