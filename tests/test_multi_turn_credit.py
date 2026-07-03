from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pipelinerl.finetune.rl import (
    FrozenProbeRuntime,
    RLConfig,
    _forward_with_frozen_probe_capture,
    load_frozen_probe_runtime,
    multi_turn_credit_advantages,
    rl_step,
    turn_end_indices,
    turn_start_indices,
)
from pipelinerl.finetune.types import PipelineBatchEncoding
from pipelinerl.finetune.value_model import AutoModelForCausalLMWithValueHead, ValueHead


def test_turn_indices_select_labeled_boundaries_per_segment() -> None:
    segments = [
        (torch.tensor(0), torch.tensor(3)),
        (torch.tensor(3), torch.tensor(5)),
        (torch.tensor(5), torch.tensor(6)),
    ]
    masks_shifted = torch.tensor([[False, True, True, False, True, False]])

    start_indices = turn_start_indices(segments, masks_shifted)
    end_indices = turn_end_indices(segments, masks_shifted)

    assert start_indices.tolist() == [1, 4]
    assert end_indices.tolist() == [2, 4]


def test_multi_turn_credit_advantages_expands_turn_residuals_per_segment() -> None:
    segments = [(0, 3), (3, 5)]
    masks_shifted = torch.tensor([[True, True, False, True, True]])
    value_predictions = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    centered_targets = torch.tensor([[1.0, 1.0, 1.0, -0.5, -0.5]])

    expanded, turn_indices, turn_values, turn_targets, turn_advantages = multi_turn_credit_advantages(
        segments,
        masks_shifted,
        value_predictions,
        centered_targets,
    )

    assert turn_indices.tolist() == [1, 4]
    torch.testing.assert_close(turn_values, torch.tensor([0.2, 0.5]))
    torch.testing.assert_close(turn_targets, torch.tensor([1.0, -0.5]))
    torch.testing.assert_close(turn_advantages, torch.tensor([0.8, -1.0]))
    torch.testing.assert_close(expanded, torch.tensor([[0.8, 0.8, 0.0, -1.0, -1.0]]))


def _packed_batch() -> PipelineBatchEncoding:
    return PipelineBatchEncoding(
        input_ids=torch.tensor([[0, 1, 2]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        labels=torch.tensor([[-100, 1, 2]]),
        position_ids=torch.tensor([[0, 1, 2]]),
        segment_ids=torch.tensor([[0, 0, 0]]),
        rewards=torch.zeros(1, 3),
        advantages=torch.zeros(1, 3),
        ref_logprobs=torch.zeros(1, 3),
        old_logprobs=torch.zeros(1, 3),
        group_tokens=torch.ones(1, 3),
        num_labels=torch.ones(1, 3),
        overflow=torch.zeros(1, 3),
        model_version=0,
        is_packed=True,
    )


class _CountingLayer(nn.Module):
    def __init__(self, hidden_size: int, *, output_bfloat16: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.output_bfloat16 = output_bfloat16
        self.register_calls = 0

    def register_forward_hook(self, hook, *args, **kwargs):
        self.register_calls += 1
        return super().register_forward_hook(hook, *args, **kwargs)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear(hidden_states)
        return hidden_states.to(torch.bfloat16) if self.output_bfloat16 else hidden_states


class _TinyProbePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embed = nn.Embedding(8, 4)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [_CountingLayer(4, output_bfloat16=i == 31) for i in range(32)]
        )
        self.lm_head = nn.Linear(4, 8)

    def forward(self, input_ids, attention_mask=None, labels=None, position_ids=None):
        hidden_states = self.embed(input_ids)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return SimpleNamespace(logits=self.lm_head(hidden_states.float()))


def test_load_frozen_probe_runtime_validates_artifact_and_resolves_layer(tmp_path) -> None:
    model = _TinyProbePolicy()
    artifact_path = tmp_path / "probe.pt"
    torch.save(
        {
            "version": 1,
            "target": "rollout_success",
            "layer_index": 31,
            "w_prime": torch.ones(4, dtype=torch.float32),
            "b_prime": 0.25,
        },
        artifact_path,
    )

    runtime = load_frozen_probe_runtime(str(artifact_path), model)

    assert runtime.layer is model.model.layers[31]
    torch.testing.assert_close(runtime.w_prime, torch.ones(4))
    assert runtime.b_prime == 0.25

    torch.save(
        {
            "version": 1,
            "target": "rollout_success",
            "layer_index": 31,
            "w_prime": torch.ones(3, dtype=torch.float32),
            "b_prime": 0.25,
        },
        artifact_path,
    )
    with pytest.raises(ValueError, match="w_prime"):
        load_frozen_probe_runtime(str(artifact_path), model)


def test_frozen_probe_forward_capture_detaches_upcasts_and_removes_hook() -> None:
    model = _TinyProbePolicy()
    runtime = FrozenProbeRuntime(
        layer=model.model.layers[31],
        w_prime=torch.ones(4, dtype=torch.float32),
        b_prime=0.0,
    )

    outputs, hidden = _forward_with_frozen_probe_capture(
        model,
        {"input_ids": torch.tensor([[0, 1, 2]]), "attention_mask": torch.ones(1, 3), "labels": None},
        runtime,
    )

    assert outputs.logits.shape == (1, 3, 8)
    assert hidden is not None
    assert hidden.shape == (1, 3, 4)
    assert hidden.dtype == torch.float32
    assert not hidden.requires_grad
    assert model.model.layers[31].register_calls == 1
    assert len(model.model.layers[31]._forward_hooks) == 0


def test_rl_step_without_frozen_probe_does_not_register_probe_hook() -> None:
    model = _TinyProbePolicy()

    loss, stats = rl_step(
        model,
        _packed_batch(),
        current_step=0,
        max_step=1,
        config=RLConfig(policy_loss="gspo", batch_size=1),
    )

    assert torch.isfinite(loss)
    assert "loss" in stats
    assert sum(layer.register_calls for layer in model.model.layers) == 0


def test_multi_turn_credit_requires_value_head() -> None:
    with pytest.raises(ValueError, match="requires a value head"):
        rl_step(
            nn.Module(),
            _packed_batch(),
            current_step=0,
            max_step=1,
            config=RLConfig(policy_loss="gspo", batch_size=1, multi_turn_credit=True),
        )


def test_multi_turn_credit_requires_gspo() -> None:
    model = nn.Module()
    model.value_head = nn.Linear(1, 1)

    with pytest.raises(ValueError, match="requires policy_loss='gspo'"):
        rl_step(
            model,
            _packed_batch(),
            current_step=0,
            max_step=1,
            config=RLConfig(policy_loss="ppo", batch_size=1, multi_turn_credit=True),
        )


def test_value_head_initialization_does_not_mutate_global_rng_state() -> None:
    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()

    head = ValueHead(4)

    after = torch.random.get_rng_state()
    assert torch.equal(after, before)

    second_head = ValueHead(4)
    torch.testing.assert_close(head.output.weight, second_head.output.weight)
    torch.testing.assert_close(head.output.bias, second_head.output.bias)


class _TinyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.main_input_name = "input_ids"
        self.embed = nn.Embedding(8, 4)
        self.model = nn.Module()
        self.model.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8)
        self.seen_output_hidden_states = None

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        self.seen_output_hidden_states = output_hidden_states
        hidden_states = self.embed(input_ids)
        hidden_states = self.model.norm(hidden_states)
        return SimpleNamespace(
            loss=None,
            logits=self.lm_head(hidden_states),
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


class _TinyCompositeCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=4))
        self.main_input_name = "input_ids"
        self.embed = nn.Embedding(8, 4)
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8)
        self.seen_output_hidden_states = None

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        self.seen_output_hidden_states = output_hidden_states
        hidden_states = self.embed(input_ids)
        hidden_states = self.model.language_model.norm(hidden_states)
        return SimpleNamespace(
            loss=None,
            logits=self.lm_head(hidden_states),
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


def test_value_head_wrapper_captures_final_hidden_state_without_all_hidden_states() -> None:
    backbone = _TinyCausalLM()
    model = AutoModelForCausalLMWithValueHead(backbone)

    outputs = model(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3))

    assert backbone.seen_output_hidden_states is False
    assert outputs.value.shape == (1, 3)


def test_value_head_wrapper_supports_composite_config_and_language_model_norm() -> None:
    backbone = _TinyCompositeCausalLM()
    model = AutoModelForCausalLMWithValueHead(backbone)

    outputs = model(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3))

    assert not hasattr(backbone.config, "hidden_size")
    assert model.value_head.output.in_features == 4
    assert backbone.seen_output_hidden_states is False
    assert outputs.value.shape == (1, 3)


def test_value_head_wrapper_does_not_backprop_value_loss_into_backbone() -> None:
    backbone = _TinyCausalLM()
    model = AutoModelForCausalLMWithValueHead(backbone)

    outputs = model(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3))
    outputs.value.sum().backward()

    assert model.value_head.output.weight.grad is not None
    assert backbone.embed.weight.grad is None
    assert backbone.model.norm.weight.grad is None
    assert backbone.lm_head.weight.grad is None
