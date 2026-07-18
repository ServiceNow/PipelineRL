import asyncio
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from pydantic import ValidationError

from pipelinerl.actor import apply_group_boundary_policy, stamp_rollout_metadata
from pipelinerl.domains.tau2.client import Tau2RunResponse
from pipelinerl.domains.tau2.rollouts import (
    Tau2TrajectoryError,
    build_tau2_training_text,
    generate_tau2_rollout,
    serialized_training_text_size,
)
from pipelinerl.finetune.rl import RLConfig, prepare_rl_fields, rl_step
from pipelinerl.finetune.types import PipelineBatchEncoding
from pipelinerl.preprocess import batch_annotate_traces_with_ref_logprobs
from pipelinerl.rollouts import BaseMetrics, RolloutResult


class _Vocabulary:
    def values(self):
        return range(65_536)


class _Tokenizer:
    def decode(self, input_ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return " ".join(str(token_id) for token_id in input_ids)

    def get_vocab(self):
        return _Vocabulary()


POLICY_ENDPOINT = "http://actor-0:8000/v1"


def _policy_item(
    prompt,
    generation,
    logprobs,
    *,
    version_start=8,
    version_end=8,
    endpoint=POLICY_ENDPOINT,
):
    return {
        "type": "function_call",
        "prompt_token_ids": prompt,
        "generation_token_ids": generation,
        "generation_log_probs": logprobs,
        "model_version_start": version_start,
        "model_version_end": version_end,
        "policy_endpoint": endpoint,
    }


def _run_response(
    *,
    output=None,
    response_input=None,
    reward=0.75,
    termination_reason="user_stop",
    num_agent_calls=3,
):
    if output is None:
        output = [
            _policy_item([10, 11], [20, 21], [-0.1, -0.2]),
            {"type": "message", "role": "user", "content": "next"},
            _policy_item([10, 11, 20, 21, 30], [40], [-0.3]),
        ]
    if response_input is None:
        response_input = [
            {"role": "system", "content": "policy"},
            {"role": "assistant", "content": "How can I help?"},
            {"role": "user", "content": "request"},
        ]
    return Tau2RunResponse(
        reward=reward,
        responses_create_params={"input": response_input},
        response={"output": output},
        result={"termination_reason": termination_reason},
        num_agent_calls=num_agent_calls,
    )


def _build(response=None, **kwargs):
    return build_tau2_training_text(
        response or _run_response(),
        _Tokenizer(),
        expected_policy_endpoint=kwargs.pop("expected_policy_endpoint", POLICY_ENDPOINT),
        max_sequence_length=kwargs.pop("max_sequence_length", 128),
        shared_memory_entry_size=kwargs.pop("shared_memory_entry_size", 10_000_000),
        **kwargs,
    )


def test_builds_prefix_contiguous_all_turn_training_text():
    text = _build()

    assert text.input_ids == [10, 11, 20, 21, 30, 40]
    assert text.labels == [-100, -100, 20, 21, -100, 40]
    assert text.logprobs == [0.0, 0.0, -0.1, -0.2, 0.0, -0.3]
    assert text.ref_logprobs == text.logprobs
    assert text.reward == 0.75
    assert text.prompt_tokens == 3
    assert text.output_tokens == 3
    assert text.finished is True
    assert text.metadata == {
        "num_policy_calls": 2,
        "termination_reason": "user_stop",
        "model_version": 8,
        "model_version_min": 8,
        "model_version_max": 8,
        "model_version_spread": 0,
        "model_calls": [
            {
                "call_index": 0,
                "version_start": 8,
                "version_end": 8,
                "endpoint": POLICY_ENDPOINT,
                "token_start": 2,
                "token_end": 4,
            },
            {
                "call_index": 1,
                "version_start": 8,
                "version_end": 8,
                "endpoint": POLICY_ENDPOINT,
                "token_start": 5,
                "token_end": 6,
            },
        ],
    }


def test_mixed_call_boundaries_preserve_ordered_provenance_and_oldest_version():
    response = _run_response(
        output=[
            _policy_item(
                [10, 11],
                [20, 21],
                [-0.1, -0.2],
                version_start=8,
                version_end=10,
            ),
            _policy_item(
                [10, 11, 20, 21, 30],
                [40],
                [-0.3],
                version_start=10,
                version_end=10,
            ),
        ]
    )

    text = _build(response)

    assert text.metadata["model_version"] == 8
    assert text.metadata["model_version_min"] == 8
    assert text.metadata["model_version_max"] == 10
    assert text.metadata["model_version_spread"] == 2
    assert text.metadata["model_calls"] == [
        {
            "call_index": 0,
            "version_start": 8,
            "version_end": 10,
            "endpoint": POLICY_ENDPOINT,
            "token_start": 2,
            "token_end": 4,
        },
        {
            "call_index": 1,
            "version_start": 10,
            "version_end": 10,
            "endpoint": POLICY_ENDPOINT,
            "token_start": 5,
            "token_end": 6,
        },
    ]


def test_rejects_missing_provenance_and_endpoint_affinity_drift():
    missing = _policy_item([10], [20], [-0.1])
    missing.pop("model_version_end")
    response = _run_response(output=[missing], num_agent_calls=2)
    with pytest.raises(Tau2TrajectoryError, match="incomplete capture") as exc_info:
        _build(response)
    assert exc_info.value.reason == "incomplete_policy_capture"

    response = _run_response(
        output=[
            _policy_item(
                [10],
                [20],
                [-0.1],
                endpoint="http://actor-wrong:8000/v1",
            )
        ],
        num_agent_calls=2,
    )
    with pytest.raises(Tau2TrajectoryError, match="does not match") as exc_info:
        _build(response)
    assert exc_info.value.reason == "policy_endpoint_mismatch"


def test_rejects_labeled_seeded_assistant_input():
    seeded = {
        "role": "assistant",
        "content": "How can I help?",
        "prompt_token_ids": [1],
        "generation_token_ids": [2],
        "generation_log_probs": [-0.1],
    }
    response = _run_response(
        response_input=[seeded],
        output=[_policy_item([10], [20], [-0.1])],
        num_agent_calls=2,
    )

    with pytest.raises(Tau2TrajectoryError, match="Seeded Tau2 assistant"):
        _build(response)


def test_rejects_non_prefix_turn_and_context_overflow():
    response = _run_response(
        output=[
            _policy_item([10, 11], [20], [-0.1]),
            _policy_item([10, 99, 20, 30], [40], [-0.2]),
        ]
    )
    with pytest.raises(Tau2TrajectoryError, match="not an exact prefix extension"):
        _build(response)

    with pytest.raises(Tau2TrajectoryError, match="exceeds sequence limit"):
        _build(max_sequence_length=5)


def test_rejects_out_of_vocabulary_token_ids_before_decode():
    response = _run_response(
        output=[_policy_item([10], [70_000], [-0.1])],
        num_agent_calls=2,
    )

    with pytest.raises(Tau2TrajectoryError, match="out-of-vocabulary") as exc_info:
        _build(response)

    assert exc_info.value.reason == "oov_token_ids"


def test_max_steps_is_recorded_as_incomplete():
    text = _build(_run_response(termination_reason="max_steps"))
    assert text.finished is False


def test_real_length_serialization_fits_default_actor_cap_and_rejects_oversize():
    prompt_length = 49_152
    generation_length = 16_384
    prompt = list(range(prompt_length))
    generation = list(range(prompt_length, prompt_length + generation_length))
    logprobs = [-(index + 1) / 100_000 for index in range(generation_length)]
    response = _run_response(
        output=[_policy_item(prompt, generation, logprobs)],
        num_agent_calls=2,
    )
    actor_entry_cap = int(
        OmegaConf.load(Path(__file__).parents[1] / "conf" / "base.yaml").actor.shared_memory_entry_size
    )

    text = _build(
        response,
        max_sequence_length=prompt_length + generation_length,
        shared_memory_entry_size=actor_entry_cap,
    )
    serialized_size = serialized_training_text_size(text)

    assert len(text.input_ids) == 65_536
    rollout_result = RolloutResult(
        training_texts=[text],
        metrics=BaseMetrics(
            reward=text.reward,
            success=False,
            no_error=True,
            no_answer=False,
        ),
        latency=1.0,
    )
    result_entry_size = len(pickle.dumps([rollout_result]))
    serialized_result = pickle.dumps(rollout_result)
    group_entry_size = len(
        pickle.dumps(
            [pickle.loads(serialized_result) for _ in range(16)]
        )
    )
    assert serialized_size < actor_entry_cap
    assert result_entry_size < actor_entry_cap < group_entry_size
    assert group_entry_size > result_entry_size * 15
    assert 32_000_000 < group_entry_size < 34_000_000
    with pytest.raises(Tau2TrajectoryError, match="serialized size"):
        _build(
            response,
            max_sequence_length=prompt_length + generation_length,
            shared_memory_entry_size=serialized_size - 1,
        )


def test_prepare_rl_fields_preserves_existing_suffix_bytes():
    encoding = {
        "input_ids": [10, 11, 20, 21],
        "labels": [-100, -100, 20, 21],
    }
    actual = prepare_rl_fields(
        dict(encoding),
        reward=0.5,
        old_logprobs=[-0.1, -0.2],
        ref_logprobs=[-0.3, -0.4],
    )
    expected = {
        **encoding,
        "rewards": [0.5, 0.5, 0.5, 0.5],
        "advantages": [0.0, 0.0, 0.0, 0.0],
        "old_logprobs": [0, 0, -0.1, -0.2],
        "ref_logprobs": [0, 0, -0.3, -0.4],
        "overflow": [0, 0, 0, 0],
        "group_tokens": [0, 0, 0, 0],
        "num_labels": [0, 0, 1, 1],
    }

    assert pickle.dumps(actual) == pickle.dumps(expected)


def test_prepare_rl_fields_preserves_full_interleaved_alignment():
    encoding = {
        "input_ids": [10, 20, 30, 40],
        "labels": [-100, 20, -100, 40],
    }
    old_logprobs = [0.0, -0.1, 0.0, -0.2]
    ref_logprobs = [0.0, -0.3, 0.0, -0.4]

    actual = prepare_rl_fields(dict(encoding), 1.0, old_logprobs, ref_logprobs)

    assert actual["old_logprobs"] == old_logprobs
    assert actual["ref_logprobs"] == ref_logprobs


class _ReferenceLLM:
    def __init__(self):
        self.call = None

    def get_batch_logprobs_token_ids(self, prompt_token_ids, completion_token_ids):
        self.call = (prompt_token_ids, completion_token_ids)
        return [
            {"content": [{"logprob": value} for value in [-1.1, -1.2, -1.3, -1.4, -1.5]]},
            {"content": [{"logprob": value} for value in [-2.1, -2.2]]},
        ]


def test_reference_scoring_uses_full_sequence_and_masks_fillers():
    full = {
        "input_ids": [10, 11, 12, 13, 14, 15],
        "labels": [-100, -100, 12, -100, 14, 15],
        "logprobs": [0.0, 0.0, -0.1, 0.0, -0.2, -0.3],
    }
    suffix = {
        "input_ids": [1, 2, 3, 4],
        "labels": [-100, -100, 3, 4],
        "logprobs": [-0.4, -0.5],
    }
    llm = _ReferenceLLM()

    batch_annotate_traces_with_ref_logprobs(llm, [full, suffix])

    assert llm.call == (
        [[10], [1, 2]],
        [[11, 12, 13, 14, 15], [3, 4]],
    )
    assert full["ref_logprobs"] == [0.0, 0.0, -1.2, 0.0, -1.4, -1.5]
    assert suffix["ref_logprobs"] == [-2.1, -2.2]


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        values = torch.arange(48, dtype=torch.float32).reshape(1, 6, 8) / 20
        self.logits = torch.nn.Parameter(values)

    def forward(self, **kwargs):
        return SimpleNamespace(logits=self.logits)


def _loss_and_gradient(old_logprobs, ref_logprobs):
    model = _TinyModel()
    batch = PipelineBatchEncoding(
        input_ids=[[0, 1, 2, 3, 4, 5]],
        attention_mask=[[1, 1, 1, 1, 1, 1]],
        labels=[[-100, 1, -100, 3, -100, 5]],
        rewards=[[0.0, 1.0, 0.0, -0.5, 0.0, 0.75]],
        advantages=[[0.0, 0.5, 0.0, -0.25, 0.0, 0.75]],
        ref_logprobs=[ref_logprobs],
        old_logprobs=[old_logprobs],
        group_tokens=[[0.0] * 6],
        num_labels=[[3.0] * 6],
        overflow=[[0.0] * 6],
        model_version=0,
    )
    config = RLConfig(
        policy_loss="reinforce",
        batch_size=1,
        kl_coef=0.2,
        final_kl_coef=0.2,
    )
    loss, _ = rl_step(model, batch, current_step=0, max_step=1, config=config)
    loss.backward()
    return loss.detach(), model.logits.grad.detach().clone()


def test_masked_logprob_fillers_do_not_affect_loss_or_gradients():
    base_old = [0.0, -2.0, 0.0, -2.5, 0.0, -3.0]
    base_ref = [0.0, -2.1, 0.0, -2.6, 0.0, -3.1]
    changed_old = [7.0, -2.0, -4.0, -2.5, 5.0, -3.0]
    changed_ref = [-6.0, -2.1, 3.0, -2.6, -5.0, -3.1]

    base_loss, base_gradient = _loss_and_gradient(base_old, base_ref)
    changed_loss, changed_gradient = _loss_and_gradient(changed_old, changed_ref)

    torch.testing.assert_close(changed_loss, base_loss, rtol=0, atol=0)
    torch.testing.assert_close(changed_gradient, base_gradient, rtol=0, atol=0)


class _GymClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def run(self, policy_base_url, problem, session):
        self.calls.append((policy_base_url, problem, session))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class _Tau2LLM:
    def __init__(self):
        self.tokenizer = _Tokenizer()

    def get_base_url(self):
        return "http://actor-0:8000"

    def load_tokenizer(self):
        return None


def test_tau2_rollout_wrapper_preserves_audit_and_returns_typed_boundary_failure(monkeypatch):
    good_response = _run_response(
        output=[
            _policy_item(
                [10, 11],
                [20],
                [-0.1],
                version_start=8,
                version_end=10,
            )
        ],
        num_agent_calls=2,
        reward=0.75,
    )
    good_response.result.update(
        {
            "reward_info": {"action_checks": [{"passed": True}]},
            "messages": [
                {
                    "turn_idx": 2,
                    "tool_calls": [
                        {
                            "name": "lookup",
                            "arguments": {"id": "123"},
                        }
                    ],
                }
            ],
        }
    )
    client = _GymClient(good_response)
    monkeypatch.setattr(
        "pipelinerl.domains.tau2.rollouts._get_tau2_gym_client",
        lambda cfg: client,
    )
    cfg = OmegaConf.create(
        {
            "finetune": {"seq_length": 128},
            "actor": {"shared_memory_entry_size": 10_000_000},
        }
    )
    problem = {
        "task_id": "task-1",
        "dataset": "telecom",
        "domain": "tau2",
    }
    session = object()

    result = asyncio.run(generate_tau2_rollout(cfg, _Tau2LLM(), problem, session))

    assert result.model_version == 8
    assert result.atomic_group is True
    assert result.training_texts[0].metadata["model_version"] == 8
    assert result.metrics.reward == 0.75
    assert result.audit["reward"] == 0.75
    assert result.audit["submitted"] is True
    assert result.audit["verifier_result"] == {"action_checks": [{"passed": True}]}
    assert result.audit["actions"] == [
        {
            "turn_idx": 2,
            "tool_calls": [
                {
                    "name": "lookup",
                    "arguments": {"id": "123"},
                }
            ],
        }
    ]
    assert result.audit["model_calls"][0] == {
        "call_index": 0,
        "version_start": 8,
        "version_end": 10,
        "endpoint": POLICY_ENDPOINT,
        "token_start": 2,
        "token_end": 3,
    }
    assert client.calls == [(POLICY_ENDPOINT, problem, session)]

    client.response = _run_response(
        output=[_policy_item([10], [20], [-0.1])],
        num_agent_calls=2,
        termination_reason="context_length",
    )
    incomplete = asyncio.run(generate_tau2_rollout(cfg, _Tau2LLM(), problem, session))

    assert incomplete.audit["submitted"] is False
    assert incomplete.training_texts[0].finished is False

    bad_item = _policy_item([10], [20], [-0.1])
    bad_item.pop("model_version_start")
    bad_response = _run_response(output=[bad_item], num_agent_calls=2, reward=0.75)
    client.response = bad_response

    dropped = asyncio.run(generate_tau2_rollout(cfg, _Tau2LLM(), problem, session))

    assert dropped.training_texts == []
    assert dropped.metrics.reward == 0.75
    assert dropped.metrics.boundary_failure is True
    assert dropped.audit["boundary_failure"]["reason"] == "incomplete_policy_capture"

    oov_response = _run_response(
        output=[_policy_item([10], [70_000], [-0.1])],
        num_agent_calls=2,
        reward=0.75,
    )
    oov_response.result.update(good_response.result)
    client.response = oov_response

    oov = asyncio.run(generate_tau2_rollout(cfg, _Tau2LLM(), problem, session))

    assert oov.training_texts == []
    assert oov.metrics.reward == 0.75
    assert oov.audit["reward"] == 0.75
    assert oov.audit["verifier_result"] == {"action_checks": [{"passed": True}]}
    assert oov.audit["actions"] == result.audit["actions"]
    assert oov.audit["boundary_failure"]["reason"] == "oov_token_ids"

    stamp_rollout_metadata(result, "actor", 9, 0, 10)
    stamp_rollout_metadata(oov, "actor", 9, 1, 10)
    oov_counters = {}
    assert apply_group_boundary_policy([oov, result], oov_counters) == "oov_token_ids"
    assert oov_counters == {"oov_token_ids": 1}
    assert all(
        rollout.audit["entered_training"] is False
        for rollout in [oov, result]
    )

    with pytest.raises(ValidationError) as exc_info:
        Tau2RunResponse.model_validate({})
    client.response = exc_info.value

    malformed = asyncio.run(generate_tau2_rollout(cfg, _Tau2LLM(), problem, session))

    assert malformed.training_texts == []
    assert malformed.metrics.reward == 0.0
    assert malformed.metrics.boundary_failure is True
    assert malformed.audit["reward_available"] is False
    assert malformed.audit["submitted"] is None
    assert malformed.audit["boundary_failure"]["reason"] == "malformed_gym_response"
    assert malformed.audit["boundary_failure"]["validation_errors"]

    client.response = RuntimeError("transport failure")
    with pytest.raises(RuntimeError, match="transport failure"):
        asyncio.run(generate_tau2_rollout(cfg, _Tau2LLM(), problem, session))

    assert dropped.audit["entered_training"] is False
