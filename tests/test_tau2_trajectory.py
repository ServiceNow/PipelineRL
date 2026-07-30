import asyncio
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from pydantic import ValidationError

from pipelinerl.actor import apply_group_boundary_policy, stamp_rollout_metadata
from pipelinerl.domains.tau2.client import (
    Tau2BoundaryFailure,
    Tau2BoundaryResponse,
    Tau2RunResponse,
)
from pipelinerl.domains.tau2.rollouts import (
    Tau2TerminationContractError,
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
    assert text.n_predicted == 0
    assert text.prompt_text == ""
    assert text.output_text == text.text
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


def _rollout_config():
    return OmegaConf.create(
        {
            "finetune": {"seq_length": 128},
            "actor": {"shared_memory_entry_size": 10_000_000},
        }
    )


def _rollout_problem():
    return {
        "task_id": "task-1",
        "dataset": "telecom",
        "domain": "tau2",
    }


def _generate(monkeypatch, response):
    client = _GymClient(response)
    monkeypatch.setattr(
        "pipelinerl.domains.tau2.rollouts._get_tau2_gym_client",
        lambda cfg: client,
    )
    result = asyncio.run(
        generate_tau2_rollout(
            _rollout_config(),
            _Tau2LLM(),
            _rollout_problem(),
            object(),
        )
    )
    return result, client


def _judge_boundary_exception():
    marker = {
        "reason": "judge_reward_invalid",
        "producer_stage": "judge",
        "detail_code": "invalid_json",
        "attempt_count": 2,
        "diagnostic": "judge output failed strict assertion validation",
    }
    boundary = Tau2BoundaryResponse(
        schema_version=1,
        outcome="boundary_failure",
        config={},
        task={"id": "task-1"},
        seed=7,
        evaluation_type="all",
        save_dir=None,
        user_voice_settings=None,
        user_persona_config=None,
        verbose_logs=False,
        audio_debug=False,
        audio_taps=False,
        auto_review=False,
        review_mode="full",
        hallucination_feedback=None,
        responses_create_params={"input": [{"role": "user", "content": "request"}]},
        response={
            "output": [_policy_item([10], [20], [-0.1])],
        },
        result={
            "termination_reason": "agent_stop",
            "messages": [
                {
                    "role": "assistant",
                    "turn_idx": 2,
                    "tool_calls": [{"name": "lookup", "arguments": {"id": "123"}}],
                }
            ],
            "reward_info": {"info": {"nl": {"pipelinerl_judge_reward_invalid": marker}}},
        },
        auxiliary_model_calls=[
            {
                "role": "judge",
                "call_index": 1,
                "attempt_index": attempt_index,
                "requested_model_alias": "qwen-judge",
                "response_model_alias": "qwen-user",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "latency_s": 0.25,
                "finish_reason": "stop",
                "completion_budget": budget,
                "output_character_count": 8,
                "output_sha256": "a" * 64,
                "failure_code": "invalid_json",
            }
            for attempt_index, budget in ((1, 64), (2, 128))
        ],
        judge_sampling={
            "requested_model_alias": "qwen-judge",
            "enable_thinking": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "seed": 11,
            "initial_max_tokens": 64,
            "retry_max_tokens": 128,
            "auxiliary_model_timeout_s": 30.0,
        },
        duration=1.0,
        num_steps=2,
        num_agent_calls=2,
        reason="judge_reward_invalid",
        producer_stage="judge",
        detail_code="invalid_json",
        attempt_count=2,
        diagnostic="judge output failed strict assertion validation",
    )
    return Tau2BoundaryFailure(boundary)


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
    assert result.audit["response_tokens"] == 1
    assert result.audit["labeled_tokens"] == 1
    assert client.calls == [(POLICY_ENDPOINT, problem, session)]

    client.response = _run_response(
        output=[_policy_item([10], [20], [-0.1])],
        num_agent_calls=2,
        termination_reason="context_window_exceeded",
    )
    incomplete = asyncio.run(generate_tau2_rollout(cfg, _Tau2LLM(), problem, session))

    assert incomplete.audit["submitted"] is False
    assert incomplete.training_texts[0].finished is False
    assert incomplete.training_texts[0].reward == 0.0
    assert incomplete.metrics.reward == 0.0
    assert incomplete.audit["evaluator_reward"] == 0.75
    assert incomplete.audit["training_reward"] == 0.0

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
    assert malformed.metrics is None
    assert malformed.audit["reward_available"] is False
    assert malformed.audit["submitted"] is None
    assert malformed.audit["boundary_failure"]["reason"] == "malformed_gym_response"
    assert malformed.audit["boundary_failure"]["validation_errors"]

    client.response = RuntimeError("transport failure")
    with pytest.raises(RuntimeError, match="transport failure"):
        asyncio.run(generate_tau2_rollout(cfg, _Tau2LLM(), problem, session))

    assert dropped.audit["entered_training"] is False


@pytest.mark.parametrize(
    ("termination_reason", "expected_disposition"),
    [
        ("user_stop", "train_evaluator_reward"),
        ("agent_stop", "train_evaluator_reward"),
        ("max_steps", "train_zero"),
        ("context_window_exceeded", "train_zero"),
        ("empty_tool_calls_and_content", "train_zero"),
        ("agent_error", "train_zero"),
        ("too_many_errors", "train_zero"),
        ("user_error", "drop_group"),
        ("empty_user_message", "drop_group"),
        ("timeout", "fatal"),
        ("infrastructure_error", "fatal"),
        ("unexpected_error", "fatal"),
    ],
)
def test_exact_tau2_termination_disposition_table(
    monkeypatch,
    termination_reason,
    expected_disposition,
):
    evaluator_reward = (
        0.0 if termination_reason == "empty_user_message" else 0.37
    )
    response = _run_response(
        reward=evaluator_reward,
        termination_reason=termination_reason,
    )
    response.result["messages"] = []
    if termination_reason == "too_many_errors":
        response.result["messages"] = [
            {
                "id": "tool-1",
                "role": "tool",
                "content": "failed",
                "requestor": "assistant",
                "error": True,
                "turn_idx": 3,
            }
        ]

    if expected_disposition == "fatal":
        with pytest.raises(Tau2TerminationContractError, match=termination_reason):
            _generate(monkeypatch, response)
        return

    result, _ = _generate(monkeypatch, response)
    assert result.audit["termination_reason"] == termination_reason
    assert result.audit["reward"] == evaluator_reward
    assert result.audit["evaluator_reward"] == evaluator_reward
    assert result.audit["labelled_policy_tokens"] == 3

    if expected_disposition == "drop_group":
        expected_reason = f"termination_{termination_reason}"
        assert result.metrics is None
        assert result.training_texts == []
        assert result.audit["training_reward"] is None
        assert result.audit["reward_available"] is False
        assert result.audit["submitted"] is False
        assert result.audit["finished"] is False
        assert result.audit["disposition"] == "drop_group"
        stamp_rollout_metadata(result, "actor", 1, 0, 8)
        counters = {}
        assert apply_group_boundary_policy([result], counters) == expected_reason
        assert counters == {expected_reason: 1}
        return

    assert result.metrics is not None
    assert result.audit["disposition"] == expected_disposition
    if expected_disposition == "train_evaluator_reward":
        assert result.training_texts[0].reward == evaluator_reward
        assert result.training_texts[0].finished is True
        assert result.metrics.reward == evaluator_reward
        assert result.audit["training_reward"] == evaluator_reward
        assert result.audit["submitted"] is True
        assert result.audit["finished"] is True
    else:
        assert result.training_texts[0].reward == 0.0
        assert result.training_texts[0].finished is False
        assert result.metrics.reward == 0.0
        assert result.metrics.success is False
        assert result.audit["training_reward"] == 0.0
        assert result.audit["submitted"] is False
        assert result.audit["finished"] is False


@pytest.mark.parametrize(
    "termination_reason",
    ["empty_tool_calls_and_content", "agent_error"],
)
def test_conditional_termination_without_labeled_span_fires_group_counter(
    monkeypatch,
    termination_reason,
):
    response = _run_response(
        output=[],
        num_agent_calls=1,
        reward=0.9,
        termination_reason=termination_reason,
    )
    response.result["messages"] = []

    result, _ = _generate(monkeypatch, response)

    expected_reason = f"termination_{termination_reason}_no_labeled_span"
    assert result.metrics is None
    assert result.training_texts == []
    assert result.audit["labelled_policy_tokens"] == 0
    assert result.audit["evaluator_reward"] == 0.9
    assert result.audit["training_reward"] is None
    assert result.audit["submitted"] is False
    assert result.audit["finished"] is False
    stamp_rollout_metadata(result, "actor", 2, 0, 8)
    counters = {}
    assert apply_group_boundary_policy([result], counters) == expected_reason
    assert counters == {expected_reason: 1}


def test_too_many_errors_accepts_assistant_only_nested_tool_results(monkeypatch):
    response = _run_response(
        reward=0.8,
        termination_reason="too_many_errors",
    )
    response.result["messages"] = [
        {
            "role": "tool",
            "tool_messages": [
                {
                    "id": "tool-1",
                    "role": "tool",
                    "content": "failed",
                    "requestor": "assistant",
                    "error": True,
                    "turn_idx": 4,
                },
                {
                    "id": "tool-2",
                    "role": "tool",
                    "content": "ok",
                    "requestor": "assistant",
                    "error": False,
                    "turn_idx": 4,
                },
            ],
        }
    ]

    result, _ = _generate(monkeypatch, response)

    assert result.metrics is not None
    assert result.metrics.reward == 0.0
    assert result.audit["disposition"] == "train_zero"
    assert result.audit["tool_error_evidence"] == {
        "tool_results_total": 2,
        "tool_errors_total": 1,
        "assistant_tool_errors": 1,
        "user_tool_errors": 0,
        "unattributed_tool_errors": 0,
        "malformed_tool_results": 0,
        "error_results": [
            {
                "id": "tool-1",
                "turn_idx": 4,
                "requestor": "assistant",
            }
        ],
    }


@pytest.mark.parametrize(
    ("messages", "expected_counts"),
    [
        (
            [
                {
                    "id": "tool-user",
                    "role": "tool",
                    "requestor": "user",
                    "error": True,
                }
            ],
            {"user_tool_errors": 1},
        ),
        (
            [{"id": "tool-missing-requestor", "role": "tool", "error": True}],
            {"unattributed_tool_errors": 1, "malformed_tool_results": 1},
        ),
        (
            [
                {
                    "id": "tool-missing-error",
                    "role": "tool",
                    "requestor": "assistant",
                }
            ],
            {"tool_errors_total": 0, "malformed_tool_results": 1},
        ),
        (
            [{"role": "tool", "tool_messages": "not-a-list"}],
            {
                "tool_results_total": 0,
                "tool_errors_total": 0,
                "malformed_tool_results": 1,
            },
        ),
        (
            [
                {
                    "id": "tool-ok",
                    "role": "tool",
                    "requestor": "assistant",
                    "error": False,
                }
            ],
            {"tool_errors_total": 0, "malformed_tool_results": 0},
        ),
    ],
)
def test_too_many_errors_untrusted_provenance_fires_group_counter(
    monkeypatch,
    messages,
    expected_counts,
):
    response = _run_response(
        reward=0.6,
        termination_reason="too_many_errors",
    )
    response.result["messages"] = messages

    result, _ = _generate(monkeypatch, response)

    reason = "termination_too_many_errors_untrusted_provenance"
    assert result.metrics is None
    assert result.audit["evaluator_reward"] == 0.6
    assert result.audit["training_reward"] is None
    assert result.audit["submitted"] is False
    assert result.audit["finished"] is False
    evidence = result.audit["tool_error_evidence"]
    for key, expected in expected_counts.items():
        assert evidence[key] == expected
    stamp_rollout_metadata(result, "actor", 3, 0, 8)
    counters = {}
    assert apply_group_boundary_policy([result], counters) == reason
    assert counters == {reason: 1}


@pytest.mark.parametrize("evaluator_reward", [float("nan"), float("inf")])
def test_non_finite_evaluator_reward_is_fatal(monkeypatch, evaluator_reward):
    response = _run_response(reward=evaluator_reward)

    with pytest.raises(Tau2TerminationContractError, match="invalid evaluator reward"):
        _generate(monkeypatch, response)


def test_schema_validation_error_for_present_missing_termination_is_fatal(monkeypatch):
    payload = _judge_boundary_exception().boundary.model_dump()
    del payload["result"]["termination_reason"]
    with pytest.raises(ValidationError) as exc_info:
        Tau2BoundaryResponse.model_validate(payload)

    with pytest.raises(Tau2TerminationContractError, match="missing termination_reason"):
        _generate(monkeypatch, exc_info.value)


@pytest.mark.parametrize("termination_reason", [None, "new_unknown_reason"])
def test_missing_or_unknown_termination_reason_is_fatal(
    monkeypatch,
    termination_reason,
):
    response = _run_response(termination_reason=termination_reason)
    response.result["messages"] = []

    with pytest.raises(Tau2TerminationContractError, match="missing or unknown"):
        _generate(monkeypatch, response)


def test_judge_invalid_boundary_is_rewardless_and_retains_capture(monkeypatch):
    result, _ = _generate(monkeypatch, _judge_boundary_exception())

    assert result.metrics is None
    assert result.training_texts == []
    assert result.model_version == 8
    assert result.audit["reward"] is None
    assert result.audit["evaluator_reward"] is None
    assert result.audit["training_reward"] is None
    assert result.audit["reward_available"] is False
    assert result.audit["termination_reason"] == "agent_stop"
    assert result.audit["submitted"] is False
    assert result.audit["finished"] is False
    assert result.audit["disposition"] == "drop_group"
    assert result.audit["labelled_policy_tokens"] == 1
    assert result.audit["model_calls"][0]["token_start"] == 1
    assert len(result.audit["auxiliary_model_calls"]) == 2
    assert result.audit["judge_sampling"]["enable_thinking"] is True
    assert result.audit["actions"] == [
        {
            "turn_idx": 2,
            "tool_calls": [{"name": "lookup", "arguments": {"id": "123"}}],
        }
    ]
    failure = result.audit["boundary_failure"]
    assert failure == {
        "reason": "judge_reward_invalid",
        "detail": "judge output failed strict assertion validation",
        "producer_stage": "judge",
        "detail_code": "invalid_json",
        "attempt_count": 2,
        "diagnostic": "judge output failed strict assertion validation",
    }
    stamp_rollout_metadata(result, "actor", 4, 0, 8)
    counters = {}
    assert apply_group_boundary_policy([result], counters) == "judge_reward_invalid"
    assert counters == {"judge_reward_invalid": 1}
