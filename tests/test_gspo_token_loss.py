from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pipelinerl.finetune.rl import RLConfig, rl_step
from pipelinerl.finetune.types import PipelineBatchEncoding
from pipelinerl.finetune.utils import create_sentinel_batch
from pipelinerl.finetune_loop import _aggregate_step_rl_metrics


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
    *,
    group_tokens: float = 1.0,
    overflow: bool = False,
    packed: bool = True,
) -> PipelineBatchEncoding:
    n_tokens = len(advantages)
    sequence_length = n_tokens + 1
    return PipelineBatchEncoding(
        input_ids=torch.tensor([[0, *([1] * n_tokens)]], dtype=torch.long),
        attention_mask=torch.ones(1, sequence_length, dtype=torch.long),
        labels=torch.tensor([[-100, *([1] * n_tokens)]], dtype=torch.long),
        position_ids=(
            torch.arange(sequence_length, dtype=torch.long).unsqueeze(0)
            if packed
            else None
        ),
        segment_ids=(
            torch.zeros(1, sequence_length, dtype=torch.long)
            if packed
            else None
        ),
        rewards=torch.zeros(1, sequence_length),
        advantages=torch.tensor([[0.0, *advantages]], dtype=torch.float32),
        ref_logprobs=torch.zeros(1, sequence_length),
        old_logprobs=torch.tensor(
            [[0.0, *torch.log(torch.tensor(behavior_probs, dtype=torch.float32)).tolist()]],
            dtype=torch.float32,
        ),
        group_tokens=torch.full((1, sequence_length), group_tokens),
        num_labels=torch.full((1, sequence_length), float(n_tokens)),
        overflow=torch.full((1, sequence_length), float(overflow)),
        model_version=0,
        is_packed=packed,
    )


def _two_segment_batch() -> PipelineBatchEncoding:
    return PipelineBatchEncoding(
        input_ids=torch.tensor([[0, 1, 1, 0, 1, 1]], dtype=torch.long),
        attention_mask=torch.ones(1, 6, dtype=torch.long),
        labels=torch.tensor([[-100, 1, 1, -100, 1, 1]], dtype=torch.long),
        position_ids=torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long),
        segment_ids=torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long),
        rewards=torch.zeros(1, 6),
        advantages=torch.tensor([[0.0, 1.0, 1.0, 0.0, -1.0, -1.0]]),
        ref_logprobs=torch.zeros(1, 6),
        old_logprobs=torch.tensor(
            [[0.0, *torch.log(torch.full((5,), 0.5)).tolist()]],
            dtype=torch.float32,
        ),
        group_tokens=torch.ones(1, 6),
        num_labels=torch.tensor([[1.0, 2.0, 2.0, 1.0, 2.0, 2.0]]),
        overflow=torch.zeros(1, 6),
        model_version=0,
        is_packed=True,
    )


def _config(policy_loss: str, **kwargs) -> RLConfig:
    kwargs.setdefault("kl_coef", 0.0)
    kwargs.setdefault("final_kl_coef", 0.0)
    return RLConfig(
        policy_loss=policy_loss,
        batch_size=1,
        **kwargs,
    )


def _run(model: _FixedPolicy, batch: PipelineBatchEncoding, config: RLConfig):
    loss, stats = rl_step(model, batch, current_step=0, max_step=1, config=config)
    loss.backward()
    assert model.logits.grad is not None
    return loss.detach(), model.logits.grad.detach().clone(), stats


def _loss_and_grad(
    policy_loss: str,
    policy_probs: list[float],
    behavior_probs: list[float],
    advantages: list[float],
    **config_kwargs,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    model = _FixedPolicy(policy_probs)
    return _run(
        model,
        _packed_batch(
            advantages,
            behavior_probs,
            group_tokens=config_kwargs.pop("group_tokens", 1.0),
            overflow=config_kwargs.pop("overflow", False),
        ),
        _config(policy_loss, **config_kwargs),
    )


@pytest.mark.parametrize(
    ("policy_probs", "behavior_probs", "advantages", "config_kwargs"),
    [
        pytest.param(
            [0.55, 0.55, 0.55],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
            {},
            id="unclipped",
        ),
        pytest.param(
            [0.7, 0.7, 0.7],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
            {},
            id="upper-clipped-positive",
        ),
        pytest.param(
            [0.3, 0.3, 0.3],
            [0.5, 0.5, 0.5],
            [-1.0, -1.0, -1.0],
            {},
            id="lower-clipped-negative",
        ),
        pytest.param(
            [0.6, 0.6, 0.6],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
            {},
            id="exact-upper-boundary",
        ),
        pytest.param(
            [0.55, 0.55, 0.55],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
            {"group_normalization": True, "group_tokens": 3.0},
            id="group-normalization",
        ),
        pytest.param(
            [0.55, 0.55, 0.55],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
            {"overlong_filtering": True, "overflow": True},
            id="overlong-zero-weight",
        ),
    ],
)
def test_gspo_token_matches_gspo_loss_and_gradients(
    policy_probs: list[float],
    behavior_probs: list[float],
    advantages: list[float],
    config_kwargs: dict,
) -> None:
    gspo_loss, gspo_grad, _ = _loss_and_grad(
        "gspo",
        policy_probs,
        behavior_probs,
        advantages,
        **config_kwargs,
    )
    token_loss, token_grad, _ = _loss_and_grad(
        "gspo_token",
        policy_probs,
        behavior_probs,
        advantages,
        **config_kwargs,
    )

    torch.testing.assert_close(token_loss, gspo_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(token_grad, gspo_grad, rtol=1e-5, atol=1e-7)


def test_gspo_token_matches_gspo_across_packed_segments() -> None:
    gspo_model = _FixedPolicy([0.55, 0.55, 0.5, 0.45, 0.45])
    token_model = _FixedPolicy([0.55, 0.55, 0.5, 0.45, 0.45])

    gspo_loss, gspo_grad, _ = _run(
        gspo_model,
        _two_segment_batch(),
        _config("gspo"),
    )
    token_loss, token_grad, _ = _run(
        token_model,
        _two_segment_batch(),
        _config("gspo_token"),
    )

    torch.testing.assert_close(token_loss, gspo_loss)
    torch.testing.assert_close(token_grad, gspo_grad)


def test_gspo_token_uses_gspo_custom_reduction() -> None:
    zero_kl_loss, zero_kl_grad, _ = _loss_and_grad(
        "gspo_token", [0.55, 0.55], [0.5, 0.5], [1.0, 1.0]
    )
    with_kl_loss, with_kl_grad, _ = _loss_and_grad(
        "gspo_token", [0.55, 0.55], [0.5, 0.5], [1.0, 1.0],
        kl_coef=1.0, final_kl_coef=1.0,
    )

    torch.testing.assert_close(with_kl_loss, zero_kl_loss)
    torch.testing.assert_close(with_kl_grad, zero_kl_grad)


def test_gspo_token_nonuniform_advantage_has_no_cross_token_gradient_leak() -> None:
    _, base_grad, _ = _loss_and_grad(
        "gspo_token",
        [0.55, 0.55, 0.55],
        [0.5, 0.5, 0.5],
        [-1.0, -0.5, -0.25],
    )
    _, changed_grad, _ = _loss_and_grad(
        "gspo_token",
        [0.55, 0.55, 0.55],
        [0.5, 0.5, 0.5],
        [-2.0, -0.5, -0.25],
    )

    gradient_delta = changed_grad - base_grad
    assert gradient_delta[0].abs().sum() > 0
    torch.testing.assert_close(gradient_delta[1:], torch.zeros_like(gradient_delta[1:]))


def test_gspo_token_confident_protection_is_negative_advantage_only() -> None:
    policy_probs = [0.98, 0.94, 0.98]
    behavior_probs = [0.99, 0.949, 0.99]
    advantages = [-1.0, -1.0, 1.0]
    _, unprotected_grad, unprotected_stats = _loss_and_grad(
        "gspo_token",
        policy_probs,
        behavior_probs,
        advantages,
    )
    _, protected_grad, protected_stats = _loss_and_grad(
        "gspo_token",
        policy_probs,
        behavior_probs,
        advantages,
        gspo_token_protect_confident=True,
        gspo_token_confidence_threshold=0.95,
    )

    torch.testing.assert_close(protected_grad[0], torch.zeros_like(protected_grad[0]))
    assert unprotected_grad[0].abs().sum() > 0
    torch.testing.assert_close(protected_grad[1:], unprotected_grad[1:])
    assert unprotected_stats["gspo_token_protected_tokens_sum"] == 0
    assert protected_stats["gspo_token_protected_tokens_sum"] == 1
    assert protected_stats["gspo_token_response_tokens_sum"] == 3


def test_gspo_token_uses_raw_advantages_independent_of_use_advantages() -> None:
    with_adv_loss, with_adv_grad, _ = _loss_and_grad(
        "gspo_token",
        [0.55, 0.55],
        [0.5, 0.5],
        [-1.0, -1.0],
        use_advantages=True,
    )
    reward_mode_loss, reward_mode_grad, _ = _loss_and_grad(
        "gspo_token",
        [0.55, 0.55],
        [0.5, 0.5],
        [-1.0, -1.0],
        use_advantages=False,
    )

    torch.testing.assert_close(reward_mode_loss, with_adv_loss)
    torch.testing.assert_close(reward_mode_grad, with_adv_grad)


def test_gspo_token_config_validation() -> None:
    config = _config("gspo_token")
    assert config.gspo_token_protect_confident is False
    assert config.gspo_token_confidence_threshold == 0.95

    with pytest.raises(ValueError, match="relu_log_p_weights"):
        _config("gspo_token", relu_log_p_weights=True)
    with pytest.raises(ValueError, match="requires policy_loss='gspo_token'"):
        _config("gspo", gspo_token_protect_confident=True)
    with pytest.raises(ValueError):
        _config("gspo_token", gspo_token_confidence_threshold=0.5)
    with pytest.raises(ValueError):
        _config("gspo_token", gspo_token_confidence_threshold=1.01)


def test_gspo_token_requires_packed_segments() -> None:
    model = _FixedPolicy([0.55])
    with pytest.raises(ValueError, match="requires packed sequences"):
        rl_step(
            model,
            _packed_batch([1.0], [0.5], packed=False),
            current_step=0,
            max_step=1,
            config=_config("gspo_token"),
        )


def test_gspo_token_sentinel_returns_connected_zero_loss() -> None:
    model = _FixedPolicy([0.5] * 7)
    loss, stats = rl_step(
        model,
        create_sentinel_batch(
            device="cpu",
            tokenizer=SimpleNamespace(eos_token_id=1),
        ),
        current_step=0,
        max_step=1,
        config=_config("gspo_token"),
    )
    loss.backward()

    torch.testing.assert_close(loss.detach(), torch.tensor(0.0))
    assert model.logits.grad is not None
    torch.testing.assert_close(model.logits.grad, torch.zeros_like(model.logits.grad))
    assert stats == {"input_size": 8.0}


def test_gspo_token_sequence_parallel_reductions_match_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_loss, local_grad, _ = _loss_and_grad(
        "gspo_token",
        [0.55, 0.55],
        [0.5, 0.5],
        [1.0, 1.0],
    )

    import torch.distributed.nn.functional as dist_nn

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, **kwargs: None)
    monkeypatch.setattr(dist_nn, "all_reduce", lambda tensor, **kwargs: tensor)
    model = _FixedPolicy([0.55, 0.55])
    distributed_loss, _ = rl_step(
        model,
        _packed_batch([1.0, 1.0], [0.5, 0.5]),
        current_step=0,
        max_step=1,
        config=_config("gspo_token"),
        seq_parallel_group=object(),
    )
    distributed_loss.backward()

    torch.testing.assert_close(distributed_loss.detach(), local_loss)
    assert model.logits.grad is not None
    torch.testing.assert_close(model.logits.grad, local_grad)


def test_gspo_token_global_protected_fraction_uses_summed_token_counts() -> None:
    metrics = _aggregate_step_rl_metrics(
        {
            "gspo_token_protected_tokens_sum": [1.0, 2.0],
            "gspo_token_response_tokens_sum": [2.0, 8.0],
        },
        num_samples=2,
    )

    assert metrics["rl/gspo_token_protected_tokens_sum"] == 3.0
    assert metrics["rl/gspo_token_response_tokens_sum"] == 10.0
    assert metrics["rl/gspo_token_protected_frac"] == pytest.approx(0.3)

    empty = _aggregate_step_rl_metrics(
        {
            "gspo_token_protected_tokens_sum": [0.0],
            "gspo_token_response_tokens_sum": [0.0],
        },
        num_samples=1,
    )
    assert empty["rl/gspo_token_protected_frac"] == 0.0
