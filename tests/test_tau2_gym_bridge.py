import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from pipelinerl.domains.tau2.client import (
    NEMO_GYM_TITO_PATCH_SHA,
    Tau2GymClient,
    Tau2GymSettings,
    validate_executed_gym_config,
)
from pipelinerl.domains.tau2.dataset import load_tau2_problems
from pipelinerl.entrypoints.run_tau2_gym import build_gym_config, prepare_tau2_data


POLICY_URLS = ["http://actor-0:8000/v1", "http://actor-1:8000/v1"]


def _settings() -> Tau2GymSettings:
    return Tau2GymSettings(
        head_url="http://gym-head:11000",
        policy_model_name="google/gemma-4-26b-a4b",
        user_model_url="https://frozen-user.example/v1",
        user_model_name="gpt-user-sim",
        validation_interval_s=0,
    )


def _config() -> dict:
    return build_gym_config(
        policy_urls=POLICY_URLS,
        policy_model_name="google/gemma-4-26b-a4b",
        user_model_url="https://frozen-user.example/v1",
        user_model_name="gpt-user-sim",
        host="gym-host",
        head_port=11000,
        service_port_start=12000,
        max_steps=200,
        policy_api_key_env="PIPELINERL_LLM_TOKEN",
        user_api_key_env="TAU2_USER_API_KEY",
        uses_reasoning_parser=True,
    )


def test_generated_config_has_one_agent_per_policy_endpoint():
    config = _config()
    bindings = validate_executed_gym_config(config, POLICY_URLS, _settings())

    assert config["pipelinerl_gym_patch_sha"] == NEMO_GYM_TITO_PATCH_SHA
    assert set(bindings) == set(POLICY_URLS)
    assert bindings[POLICY_URLS[0]].agent_url == "http://gym-host:12003"
    assert bindings[POLICY_URLS[1]].agent_url == "http://gym-host:12004"
    assert config["pipelinerl_policy_0"]["responses_api_models"]["vllm_model"]["base_url"] == POLICY_URLS[0]


def test_mapping_validation_rejects_policy_endpoint_drift():
    config = _config()
    config["pipelinerl_policy_1"]["responses_api_models"]["vllm_model"]["base_url"] = (
        "http://actor-wrong:8000/v1"
    )

    with pytest.raises(ValueError, match="is not bound"):
        validate_executed_gym_config(config, POLICY_URLS, _settings())


def test_generated_config_rejects_user_policy_alias():
    with pytest.raises(ValueError, match="user-model endpoint must differ"):
        build_gym_config(
            policy_urls=POLICY_URLS,
            policy_model_name="google/gemma-4-26b-a4b",
            user_model_url=POLICY_URLS[0],
            user_model_name="gpt-user-sim",
            host="gym-host",
            head_port=11000,
            service_port_start=12000,
            max_steps=200,
            policy_api_key_env="PIPELINERL_LLM_TOKEN",
            user_api_key_env="TAU2_USER_API_KEY",
            uses_reasoning_parser=True,
        )


def test_load_tau2_problems_stamps_pipeline_metadata(tmp_path: Path):
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

    problems = load_tau2_problems(["telecom"], data_files={"telecom": str(path)}, seed=3)

    assert {problem["task_id"] for problem in problems} == {"task-1", "task-2"}
    assert all(problem["dataset"] == "telecom" for problem in problems)
    assert all(problem["domain"] == "tau2" for problem in problems)


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    def raise_for_status(self):
        return None

    async def json(self):
        return self.payload


class _FakeSession:
    def __init__(self, config: dict):
        self.config = config
        self.get_urls = []
        self.post_urls = []

    def get(self, url):
        self.get_urls.append(url)
        if url.endswith("/global_config_dict_yaml"):
            return _FakeResponse(OmegaConf.to_yaml(OmegaConf.create(self.config)))
        return _FakeResponse({"status": "ok"})

    def post(self, url, *, json, timeout):
        self.post_urls.append(url)
        return _FakeResponse(
            {
                "reward": 1.0,
                "responses_create_params": json["responses_create_params"],
                "response": {"output": []},
                "result": {"passed": True},
            }
        )


def test_client_routes_to_bound_agent_and_periodically_revalidates():
    async def run_test():
        session = _FakeSession(_config())
        client = Tau2GymClient(_settings(), POLICY_URLS)
        problem = {"responses_create_params": {"input": [], "tools": []}}

        first = await client.run(POLICY_URLS[1], problem, session)
        second = await client.run(POLICY_URLS[1], problem, session)

        assert first.reward == second.reward == 1.0
        assert session.post_urls == ["http://gym-host:12004/run"] * 2
        assert session.get_urls.count("http://gym-head:11000/global_config_dict_yaml") == 2

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
