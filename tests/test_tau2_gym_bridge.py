import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
import requests
from omegaconf import OmegaConf
from pydantic import ValidationError

import pipelinerl.domains.tau2.dataset as tau2_dataset
from pipelinerl.domains.tau2.client import (
    NEMO_GYM_TITO_PATCH_SHA,
    Tau2BoundaryFailure,
    Tau2GymClient,
    Tau2GymContractError,
    Tau2GymSettings,
    validate_executed_gym_config,
    validate_tau2_gym_sync,
)
from pipelinerl.domains.tau2.dataset import load_tau2_problems
from pipelinerl.entrypoints.run_tau2_gym import build_gym_config, prepare_tau2_data


POLICY_URLS = ["http://actor-0:8000/v1", "http://actor-1:8000/v1"]


def _settings(**overrides) -> Tau2GymSettings:
    values = {
        "head_url": "http://gym-head:11000",
        "policy_model_name": "qwen3.5-9b-policy",
        "user_model_url": "https://frozen-aux.example/v1",
        "user_model_name": "qwen3.5-27b-user",
        "judge_model_url": "https://frozen-aux.example/v1",
        "judge_model_name": "qwen3.5-27b-judge",
        "policy_thinking_enabled": True,
        "user_thinking_enabled": True,
        "judge_thinking_enabled": True,
        "judge_temperature": 0.6,
        "judge_top_p": 0.95,
        "judge_top_k": 20,
        "judge_seed": 17,
        "judge_initial_max_tokens": 1024,
        "judge_retry_max_tokens": 2048,
        "auxiliary_model_timeout_s": 300.0,
        "validation_interval_s": 0,
    }
    values.update(overrides)
    return Tau2GymSettings(**values)


def _config(**overrides) -> dict:
    values = {
        "policy_urls": POLICY_URLS,
        "policy_model_name": "qwen3.5-9b-policy",
        "user_model_url": "https://frozen-aux.example/v1",
        "user_model_name": "qwen3.5-27b-user",
        "judge_model_url": "https://frozen-aux.example/v1",
        "judge_model_name": "qwen3.5-27b-judge",
        "policy_thinking_enabled": True,
        "user_thinking_enabled": True,
        "judge_thinking_enabled": True,
        "judge_temperature": 0.6,
        "judge_top_p": 0.95,
        "judge_top_k": 20,
        "judge_seed": 17,
        "judge_initial_max_tokens": 1024,
        "judge_retry_max_tokens": 2048,
        "auxiliary_model_timeout_s": 300.0,
        "request_timeout_s": 3600.0,
        "host": "gym-host",
        "head_port": 11000,
        "service_port_start": 12000,
        "max_steps": 200,
        "policy_api_key_env": "PIPELINERL_LLM_TOKEN",
        "user_api_key_env": "TAU2_USER_API_KEY",
        "uses_reasoning_parser": True,
    }
    values.update(overrides)
    return build_gym_config(**values)


def _problem() -> dict:
    return {
        "config": {"domain": "retail"},
        "task": {"id": "task-1"},
        "seed": 7,
        "evaluation_type": "all",
        "save_dir": None,
        "user_voice_settings": None,
        "user_persona_config": None,
        "verbose_logs": False,
        "audio_debug": False,
        "audio_taps": False,
        "auto_review": False,
        "review_mode": "full",
        "hallucination_feedback": None,
        "responses_create_params": {"input": [], "tools": []},
    }


def _auxiliary_calls() -> list[dict]:
    return [
        {
            "role": "user_simulator",
            "call_index": 1,
            "attempt_index": 1,
            "requested_model_alias": "qwen3.5-27b-user",
            "response_model_alias": "qwen3.5-27b-user",
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "latency_s": 1.5,
            "finish_reason": "stop",
            "completion_budget": None,
            "output_character_count": 80,
            "output_sha256": None,
            "failure_code": None,
        },
        {
            "role": "judge",
            "call_index": 1,
            "attempt_index": 1,
            "requested_model_alias": "qwen3.5-27b-judge",
            "response_model_alias": "qwen3.5-27b-user",
            "prompt_tokens": 200,
            "completion_tokens": 30,
            "latency_s": 2.5,
            "finish_reason": "stop",
            "completion_budget": 1024,
            "output_character_count": 120,
            "output_sha256": "a" * 64,
            "failure_code": None,
        },
    ]


def _judge_sampling() -> dict:
    return {
        "requested_model_alias": "qwen3.5-27b-judge",
        "enable_thinking": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "seed": 17,
        "initial_max_tokens": 1024,
        "retry_max_tokens": 2048,
        "auxiliary_model_timeout_s": 300.0,
    }


def _completed_payload() -> dict:
    return {
        **_problem(),
        "schema_version": 1,
        "outcome": "completed",
        "reward": 1.0,
        "response": {"output": []},
        "result": {
            "messages": [],
            "termination_reason": "agent_stop",
            "reward_info": {"reward": 1.0, "info": {"nl": None}},
        },
        "auxiliary_model_calls": _auxiliary_calls(),
        "judge_sampling": _judge_sampling(),
        "duration": 3.0,
        "num_steps": 4,
        "num_agent_calls": 2,
        "min_prompt_tokens": 10,
        "min_completion_tokens": 5,
        "mean_prompt_tokens": 10,
        "mean_completion_tokens": 5,
        "max_prompt_tokens": 10,
        "max_completion_tokens": 5,
    }


def _boundary_payload() -> dict:
    marker = {
        "reason": "judge_reward_invalid",
        "producer_stage": "judge",
        "detail_code": "result_count_mismatch",
        "attempt_count": 2,
        "diagnostic": "judge output failed strict assertion validation",
    }
    auxiliary_calls = _auxiliary_calls()
    auxiliary_calls[-1]["failure_code"] = marker["detail_code"]
    auxiliary_calls.append(
        {
            **auxiliary_calls[-1],
            "attempt_index": 2,
            "completion_budget": 2048,
        }
    )
    return {
        **_problem(),
        "schema_version": 1,
        "outcome": "boundary_failure",
        **marker,
        "response": {"output": []},
        "result": {
            "messages": [],
            "termination_reason": "agent_stop",
            "reward_info": {
                "info": {"nl": {"pipelinerl_judge_reward_invalid": marker}},
            }
        },
        "auxiliary_model_calls": auxiliary_calls,
        "judge_sampling": _judge_sampling(),
        "duration": 3.0,
        "num_steps": 4,
        "num_agent_calls": 2,
    }


def test_generated_config_has_one_agent_per_policy_endpoint():
    config = _config()
    bindings = validate_executed_gym_config(config, POLICY_URLS, _settings())

    assert config["pipelinerl_gym_patch_sha"] == NEMO_GYM_TITO_PATCH_SHA
    assert set(bindings) == set(POLICY_URLS)
    assert bindings[POLICY_URLS[0]].agent_url == "http://gym-host:12004"
    assert bindings[POLICY_URLS[1]].agent_url == "http://gym-host:12005"
    policy_proxy = config["pipelinerl_policy_0"]["responses_api_models"]["vllm_model"]
    assert policy_proxy["base_url"] == POLICY_URLS[0]
    assert policy_proxy["uses_reasoning_parser"] is True
    assert policy_proxy["chat_template_kwargs"] == {"enable_thinking": True}
    assert config["pipelinerl_policy_thinking_enabled"] is True


def test_mapping_validation_rejects_policy_endpoint_drift():
    config = _config()
    config["pipelinerl_policy_1"]["responses_api_models"]["vllm_model"]["base_url"] = (
        "http://actor-wrong:8000/v1"
    )

    with pytest.raises(ValueError, match="is not bound"):
        validate_executed_gym_config(config, POLICY_URLS, _settings())


def test_generated_config_rejects_auxiliary_policy_endpoint_collision():
    with pytest.raises(ValueError, match="auxiliary-model endpoint must differ"):
        _config(user_model_url=POLICY_URLS[0], judge_model_url=POLICY_URLS[0])


@pytest.mark.parametrize(
    "updates",
    [
        {"judge_temperature": float("inf")},
        {"auxiliary_model_timeout_s": float("nan")},
        {"request_timeout_s": float("inf")},
    ],
)
def test_generated_config_rejects_nonfinite_auxiliary_limits(updates):
    with pytest.raises(ValueError):
        _config(**updates)


def test_load_tau2_problems_stamps_pipeline_metadata(
    tmp_path: Path,
    monkeypatch,
):
    path = tmp_path / "telecom.jsonl"
    rows = [
        {
            "config": {"agent": "tau2"},
            "task": {"id": "task-1"},
            "seed": 7,
            "evaluation_type": "env",
            "responses_create_params": {"input": [], "tools": []},
        },
        {
            "config": {"agent": "tau2"},
            "task": {"id": "task-2"},
            "seed": 8,
            "evaluation_type": "env",
            "responses_create_params": {"input": [], "tools": []},
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    monkeypatch.setattr(
        tau2_dataset,
        "validate_tau2_prepared_data",
        lambda *_args, **_kwargs: SimpleNamespace(
            rows_by_dataset={"telecom": rows}
        ),
    )
    problems = load_tau2_problems(
        ["telecom"],
        data_files={"telecom": str(path)},
        prepared_data_manifest=str(tmp_path / "manifest.json"),
        seed=3,
    )

    assert {problem["task_id"] for problem in problems} == {"task-1", "task-2"}
    assert all(problem["dataset"] == "telecom" for problem in problems)
    assert all(problem["domain"] == "tau2" for problem in problems)


class _FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    def raise_for_status(self):
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                request_info=SimpleNamespace(
                    real_url="https://frozen-aux.example/health"
                ),
                history=(),
                status=self.status,
                message="test readiness error",
            )
        return None

    async def json(self):
        return self.payload


class _FakeSession:
    def __init__(
        self,
        config: dict,
        *,
        user_health_results=None,
        user_models_payload=None,
        run_payload=None,
        run_status=200,
    ):
        self.config = config
        self.get_urls = []
        self.post_urls = []
        self.user_health_results = list(user_health_results or [])
        self.user_models_payload = user_models_payload
        self.run_payload = run_payload
        self.run_status = run_status

    def get(self, url, *, timeout=None):
        self.get_urls.append(url)
        if url.endswith("/global_config_dict_yaml"):
            return _FakeResponse(OmegaConf.to_yaml(OmegaConf.create(self.config)))
        if url.endswith("/v1/models"):
            return _FakeResponse(
                self.user_models_payload
                or {
                    "data": [
                        {"id": "qwen3.5-27b-user"},
                        {"id": "qwen3.5-27b-judge"},
                    ]
                }
            )
        if url.endswith("/health") and self.user_health_results:
            result = self.user_health_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return _FakeResponse({"status": "ok"}, status=result)
        return _FakeResponse({"status": "ok"})

    def post(self, url, *, json, timeout):
        self.post_urls.append(url)
        return _FakeResponse(
            copy.deepcopy(self.run_payload or _completed_payload()),
            status=self.run_status,
        )


def test_client_routes_to_bound_agent_and_periodically_revalidates():
    async def run_test():
        session = _FakeSession(_config())
        client = Tau2GymClient(_settings(), POLICY_URLS)
        problem = _problem()

        first = await client.run(POLICY_URLS[1], problem, session)
        second = await client.run(POLICY_URLS[1], problem, session)

        assert first.reward == second.reward == 1.0
        assert session.post_urls == ["http://gym-host:12005/run"] * 2
        assert session.get_urls.count("http://gym-head:11000/global_config_dict_yaml") == 2

        assert session.get_urls.count(
            "https://frozen-aux.example/health"
        ) == 2
        assert session.get_urls.count(
            "https://frozen-aux.example/v1/models"
        ) == 2
    asyncio.run(run_test())


def test_client_raises_typed_rewardless_judge_boundary():
    async def run_test():
        session = _FakeSession(
            _config(),
            run_payload=_boundary_payload(),
            run_status=409,
        )
        client = Tau2GymClient(_settings(), POLICY_URLS)

        with pytest.raises(Tau2BoundaryFailure) as exc_info:
            await client.run(POLICY_URLS[0], _problem(), session)

        boundary = exc_info.value.boundary
        assert boundary.reason == "judge_reward_invalid"
        assert boundary.producer_stage == "judge"
        assert boundary.attempt_count == 2
        assert not hasattr(boundary, "reward")

    asyncio.run(run_test())


@pytest.mark.parametrize(
    "case",
    [
        "marker-leak",
        "marker-mismatch",
        "numeric-boundary",
        "nested-reward",
        "judge-breakdown",
    ],
)
def test_client_fails_closed_on_judge_boundary_contract_drift(case):
    async def run_test():
        status = 200
        if case == "marker-leak":
            payload = _completed_payload()
            payload["result"] = _boundary_payload()["result"]
        else:
            status = 409
            payload = _boundary_payload()
            if case == "marker-mismatch":
                payload["detail_code"] = "different"
            elif case == "numeric-boundary":
                payload["reward"] = 0.0
            elif case == "nested-reward":
                payload["result"]["reward_info"]["reward"] = 0.0
            else:
                payload["result"]["reward_info"][
                    "reward_breakdown"
                ] = {"NL_ASSERTION": 0.0}
        session = _FakeSession(
            _config(),
            run_payload=payload,
            run_status=status,
        )
        client = Tau2GymClient(_settings(), POLICY_URLS)

        with pytest.raises(Tau2GymContractError):
            await client.run(POLICY_URLS[0], _problem(), session)

    asyncio.run(run_test())




@pytest.mark.parametrize(
    "case",
    [
        "missing-reason",
        "extra-field",
        "unknown-reason",
        "unknown-stage",
        "auxiliary-order",
        "sampling",
        "result",
        "response-capture",
        "input-capture",
        "marker-type",
        "call-audit",
    ],
)
def test_client_rejects_every_malformed_judge_boundary_surface(case):
    async def run_test():
        payload = _boundary_payload()
        if case == "missing-reason":
            payload.pop("reason")
        elif case == "extra-field":
            payload["unexpected"] = True
        elif case == "unknown-reason":
            payload["reason"] = "other"
        elif case == "unknown-stage":
            payload["producer_stage"] = "other"
        elif case == "auxiliary-order":
            payload["auxiliary_model_calls"][-1]["attempt_index"] = 3
        elif case == "sampling":
            payload["judge_sampling"].pop("seed")
        elif case == "result":
            payload["result"].pop("messages")
        elif case == "response-capture":
            payload["response"].pop("output")
        elif case == "input-capture":
            payload["responses_create_params"].pop("input")
        elif case == "marker-type":
            payload["result"]["reward_info"]["info"]["nl"][
                "pipelinerl_judge_reward_invalid"
            ] = "invalid"
        else:
            payload["auxiliary_model_calls"][-1][
                "failure_code"
            ] = "invalid_json"

        session = _FakeSession(
            _config(),
            run_payload=payload,
            run_status=409,
        )
        client = Tau2GymClient(_settings(), POLICY_URLS)
        with pytest.raises(Tau2GymContractError):
            await client.run(POLICY_URLS[0], _problem(), session)

    asyncio.run(run_test())


def test_malformed_completed_response_remains_pydantic_validation_error():
    async def run_test():
        payload = _completed_payload()
        payload["response"].pop("output")
        session = _FakeSession(_config(), run_payload=payload)
        client = Tau2GymClient(_settings(), POLICY_URLS)
        with pytest.raises(ValidationError):
            await client.run(POLICY_URLS[0], _problem(), session)

    asyncio.run(run_test())


@pytest.mark.parametrize(
    ("updates", "pattern"),
    [
        ({"judge_model_url": "https://other.example/v1"}, "same shared service"),
        ({"judge_model_name": "qwen3.5-27b-user"}, "aliases must be distinct"),
        ({"policy_thinking_enabled": False}, "requires policy, user, and judge thinking"),
        ({"judge_thinking_enabled": False}, "requires policy, user, and judge thinking"),
        ({"judge_temperature": float("inf")}, "finite"),
        ({"judge_retry_max_tokens": 2047}, "at least twice"),
        ({"auxiliary_model_timeout_s": 3600.0}, "below request_timeout_s"),
    ],
)
def test_settings_reject_invalid_auxiliary_contract(updates, pattern):
    with pytest.raises(ValueError, match=pattern):
        _settings(**updates)


@pytest.mark.parametrize(
    ("mutation", "pattern"),
    (
        ("top_level_thinking", "does not match"),
        ("proxy_thinking", "wrong thinking"),
        ("missing_proxy_thinking", "wrong thinking"),
        ("reasoning_parser", "must use the reasoning parser"),
    ),
)
def test_executed_config_rejects_policy_runtime_drift(mutation, pattern):
    config = _config()
    proxy = config["pipelinerl_policy_0"]["responses_api_models"]["vllm_model"]
    if mutation == "top_level_thinking":
        config["pipelinerl_policy_thinking_enabled"] = False
    elif mutation == "proxy_thinking":
        proxy["chat_template_kwargs"] = {"enable_thinking": False}
    elif mutation == "missing_proxy_thinking":
        proxy.pop("chat_template_kwargs")
    else:
        proxy["uses_reasoning_parser"] = False
    with pytest.raises(ValueError, match=pattern):
        validate_executed_gym_config(config, POLICY_URLS, _settings())


def test_build_config_rejects_disabled_policy_runtime():
    with pytest.raises(ValueError, match="requires policy, user, and judge thinking"):
        _config(policy_thinking_enabled=False)
    with pytest.raises(ValueError, match="requires the policy reasoning parser"):
        _config(uses_reasoning_parser=False)


def test_executed_config_rejects_proxy_and_sampling_drift():
    config = _config()
    config["pipelinerl_tau2_judge"]["responses_api_models"]["openai_model"][
        "extra_body"
    ]["top_k"] = 19
    with pytest.raises(ValueError, match="thinking or top-k"):
        validate_executed_gym_config(config, POLICY_URLS, _settings())

    config = _config()
    config["pipelinerl_judge_seed"] = 18
    with pytest.raises(ValueError, match="does not match"):
        validate_executed_gym_config(config, POLICY_URLS, _settings())


@pytest.mark.parametrize(
    "transient",
    [
        pytest.param(503, id="http-503"),
        pytest.param(
            aiohttp.ClientConnectionError("simulator restarting"),
            id="connection-error",
        ),
    ],
)
def test_user_simulator_readiness_retries_transient_failures(transient):
    async def run_test():
        session = _FakeSession(
            _config(),
            user_health_results=[transient, 200],
        )
        client = Tau2GymClient(_settings(), POLICY_URLS)
        sleep = AsyncMock()

        with patch(
            "pipelinerl.domains.tau2.client.asyncio.sleep",
            new=sleep,
        ):
            await client.ensure_valid(session)

        sleep.assert_awaited_once()
        assert session.get_urls.count(
            "https://frozen-aux.example/health"
        ) == 2

    asyncio.run(run_test())


def test_user_simulator_readiness_exhaustion_is_actor_retryable():
    async def run_test():
        session = _FakeSession(
            _config(),
            user_health_results=[503],
        )
        settings = _settings().model_copy(
            update={"startup_timeout_s": 0}
        )
        client = Tau2GymClient(settings, POLICY_URLS)
        sleep = AsyncMock()

        with patch(
            "pipelinerl.domains.tau2.client.asyncio.sleep",
            new=sleep,
        ):
            with pytest.raises(TimeoutError, match="did not recover"):
                await client.ensure_valid(session)

        sleep.assert_not_awaited()

    asyncio.run(run_test())


@pytest.mark.parametrize(
    ("payload", "pattern"),
    [
        (
            {"data": [{"id": "wrong-model"}]},
            "expected exactly",
        ),
        (
            {
                "data": [
                    {"id": "qwen3.5-27b-judge"},
                    {"id": "qwen3.5-27b-user"},
                ]
            },
            "expected exactly",
        ),
        ({"data": "not-a-list"}, "malformed"),
    ],
)
def test_user_simulator_identity_failure_does_not_retry(
    payload,
    pattern,
):
    async def run_test():
        session = _FakeSession(
            _config(),
            user_models_payload=payload,
        )
        client = Tau2GymClient(_settings(), POLICY_URLS)
        sleep = AsyncMock()

        with patch(
            "pipelinerl.domains.tau2.client.asyncio.sleep",
            new=sleep,
        ):
            with pytest.raises(ValueError, match=pattern):
                await client.ensure_valid(session)

        sleep.assert_not_awaited()

    asyncio.run(run_test())


def test_tau2_data_is_initialized_once_before_parallel_agents(tmp_path: Path):
    gym_root = tmp_path / "nemo-gym"
    env = {"NEMO_GYM_TAU2_BENCH_REF": "runtime-sha"}

    with patch("subprocess.run") as run:
        prepare_tau2_data(gym_root, env)

    run.assert_called_once()
    args, kwargs = run.call_args
    assert args[0][0] == str(gym_root / ".venv" / "bin" / "python")
    assert "ensure_tau2_data_dir" in args[0][2]
    assert kwargs == {"cwd": gym_root, "env": env, "check": True}


class _SyncResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_launch_validation_retries_only_service_readiness_errors():
    config = _config()
    responses = [
        requests.ConnectionError("user simulator booting"),
        _SyncResponse({"status": "ok"}),
        _SyncResponse({"data": [{"id": "qwen3.5-27b-user"}, {"id": "qwen3.5-27b-judge"}]}),
        _SyncResponse(config),
        _SyncResponse({"status": "ok"}),
        _SyncResponse({"status": "ok"}),
    ]

    with (
        patch("pipelinerl.domains.tau2.client.requests.get", side_effect=responses) as get,
        patch("pipelinerl.domains.tau2.client.time.sleep") as sleep,
    ):
        bindings = validate_tau2_gym_sync(_settings(), POLICY_URLS)

    assert set(bindings) == set(POLICY_URLS)
    assert get.call_count == 6
    sleep.assert_called_once_with(1.0)


@pytest.mark.parametrize(
    ("payload", "pattern"),
    [
        ({}, "malformed"),
        ({"data": [{"id": "wrong-model"}]}, "expected exactly"),
    ],
)
def test_launch_validation_rejects_bad_user_model_identity_without_retry(
    payload,
    pattern,
):
    responses = [
        _SyncResponse({"status": "ok"}),
        _SyncResponse(payload),
    ]
    with (
        patch(
            "pipelinerl.domains.tau2.client.requests.get",
            side_effect=responses,
        ) as get,
        patch("pipelinerl.domains.tau2.client.time.sleep") as sleep,
    ):
        with pytest.raises(ValueError, match=pattern):
            validate_tau2_gym_sync(_settings(), POLICY_URLS)

    assert get.call_count == 2
    sleep.assert_not_called()


def test_launch_validation_does_not_retry_bad_executed_config():
    config = _config()
    config["pipelinerl_policy_1"]["responses_api_models"]["vllm_model"]["base_url"] = (
        "http://actor-wrong:8000/v1"
    )

    with (
        patch(
            "pipelinerl.domains.tau2.client.requests.get",
            side_effect=[
                _SyncResponse({"status": "ok"}),
                _SyncResponse({"data": [{"id": "qwen3.5-27b-user"}, {"id": "qwen3.5-27b-judge"}]}),
                _SyncResponse(config),
            ],
        ) as get,
        patch("pipelinerl.domains.tau2.client.time.sleep") as sleep,
    ):
        with pytest.raises(ValueError, match="is not bound"):
            validate_tau2_gym_sync(_settings(), POLICY_URLS)

    assert get.call_count == 3
    sleep.assert_not_called()


def test_launch_validation_stops_at_startup_deadline():
    settings = _settings().model_copy(update={"startup_timeout_s": 1.0})

    with (
        patch(
            "pipelinerl.domains.tau2.client.requests.get",
            side_effect=requests.ConnectionError("still booting"),
        ) as get,
        patch(
            "pipelinerl.domains.tau2.client.time.monotonic",
            side_effect=[0.0, 0.1, 1.1],
        ),
        patch("pipelinerl.domains.tau2.client.time.sleep") as sleep,
    ):
        with pytest.raises(requests.ConnectionError, match="still booting"):
            validate_tau2_gym_sync(settings, POLICY_URLS)

    assert get.call_count == 2
    sleep.assert_called_once_with(0.9)
