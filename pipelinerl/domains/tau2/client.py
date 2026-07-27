import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import aiohttp
import requests
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict


NEMO_GYM_SHA = "5f92a73217258074b74b7be26526c69f0ce3075d"
NEMO_GYM_TITO_PATCH_SHA = "0f712c906579803045d8f62576a7db01adbfb6e26a0cc52390b64f6b1be3a1b2"
TAU2_RUNTIME_SHA = "befd120003fb55f48b498f6549556dcaf74582d5"
TAU2_DATA_SHA = "ce4013b0afe03c873488878b72851414f92f458b"

logger = logging.getLogger(__name__)

_GYM_SHA_KEY = "pipelinerl_nemo_gym_sha"
_GYM_PATCH_SHA_KEY = "pipelinerl_gym_patch_sha"
_TAU2_RUNTIME_SHA_KEY = "pipelinerl_tau2_runtime_sha"
_TAU2_DATA_SHA_KEY = "pipelinerl_tau2_data_sha"
_POLICY_MODEL_NAME_KEY = "pipelinerl_policy_model_name"
_USER_MODEL_URL_KEY = "pipelinerl_user_model_url"
_USER_MODEL_NAME_KEY = "pipelinerl_user_model_name"


class Tau2GymSettings(BaseModel):
    head_url: str
    policy_model_name: str
    user_model_url: str
    user_model_name: str
    validation_interval_s: float = 30.0
    request_timeout_s: float = 3600.0
    startup_timeout_s: float = 300.0


class Tau2RunResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    reward: float
    responses_create_params: dict[str, Any]
    response: dict[str, Any]
    result: dict[str, Any]
    duration: float | None = None
    num_steps: int | None = None
    num_agent_calls: int | None = None


@dataclass(frozen=True)
class Tau2AgentBinding:
    policy_base_url: str
    agent_name: str
    agent_url: str


def normalize_openai_base_url(url: str) -> str:
    normalized = str(url).rstrip("/")
    return normalized if normalized.endswith("/v1") else f"{normalized}/v1"


def _server_config(
    config: Mapping[str, Any],
    server_name: str,
    server_type: str,
    implementation: str,
) -> Mapping[str, Any]:
    try:
        server = config[server_name][server_type][implementation]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Executed Gym config is missing {server_name}.{server_type}.{implementation}"
        ) from exc
    if not isinstance(server, Mapping):
        raise ValueError(f"Executed Gym server config {server_name!r} is not a mapping")
    return server


def parse_executed_gym_config(payload: Any) -> dict[str, Any]:
    if isinstance(payload, str):
        config = OmegaConf.create(payload)
    elif isinstance(payload, Mapping):
        config = OmegaConf.create(dict(payload))
    else:
        raise ValueError(f"Gym head returned unsupported config payload type {type(payload).__name__}")
    plain = OmegaConf.to_container(config, resolve=False)
    if not isinstance(plain, dict):
        raise ValueError("Gym head returned a non-mapping executed config")
    return plain


def validate_executed_gym_config(
    config: Mapping[str, Any],
    actor_llm_urls: Sequence[str],
    settings: Tau2GymSettings,
) -> dict[str, Tau2AgentBinding]:
    expected_pins = {
        _GYM_SHA_KEY: NEMO_GYM_SHA,
        _GYM_PATCH_SHA_KEY: NEMO_GYM_TITO_PATCH_SHA,
        _TAU2_RUNTIME_SHA_KEY: TAU2_RUNTIME_SHA,
        _TAU2_DATA_SHA_KEY: TAU2_DATA_SHA,
    }
    for key, expected in expected_pins.items():
        if config.get(key) != expected:
            raise ValueError(f"Executed Gym config {key}={config.get(key)!r}, expected {expected!r}")

    if config.get(_POLICY_MODEL_NAME_KEY) != settings.policy_model_name:
        raise ValueError("Executed Gym policy model does not match PipelineRL")
    if normalize_openai_base_url(str(config.get(_USER_MODEL_URL_KEY, ""))) != normalize_openai_base_url(
        settings.user_model_url
    ):
        raise ValueError("Executed Gym user-model endpoint does not match PipelineRL configuration")
    if config.get(_USER_MODEL_NAME_KEY) != settings.user_model_name:
        raise ValueError("Executed Gym user-model name does not match PipelineRL configuration")

    policy_urls = [normalize_openai_base_url(url) for url in actor_llm_urls]
    if len(policy_urls) != len(set(policy_urls)):
        raise ValueError("PipelineRL actor LLM URLs must be unique")
    if normalize_openai_base_url(settings.user_model_url) in policy_urls:
        raise ValueError("Tau2 user-model endpoint must differ from every policy endpoint")
    if settings.user_model_name == settings.policy_model_name:
        raise ValueError("Tau2 user-model name must differ from the policy model name")

    user_model = _server_config(config, "pipelinerl_tau2_user", "responses_api_models", "openai_model")
    if normalize_openai_base_url(str(user_model.get("openai_base_url", ""))) != normalize_openai_base_url(
        settings.user_model_url
    ):
        raise ValueError("Executed Gym user proxy points to the wrong endpoint")
    if user_model.get("openai_model") != settings.user_model_name:
        raise ValueError("Executed Gym user proxy points to the wrong model")

    bindings: dict[str, Tau2AgentBinding] = {}
    for index, policy_url in enumerate(policy_urls):
        proxy_name = f"pipelinerl_policy_{index}"
        agent_name = f"pipelinerl_tau2_agent_{index}"
        policy_proxy = _server_config(config, proxy_name, "responses_api_models", "vllm_model")
        agent = _server_config(config, agent_name, "responses_api_agents", "tau2")

        if normalize_openai_base_url(str(policy_proxy.get("base_url", ""))) != policy_url:
            raise ValueError(f"Gym policy proxy {proxy_name} is not bound to {policy_url}")
        if policy_proxy.get("model") != settings.policy_model_name:
            raise ValueError(f"Gym policy proxy {proxy_name} serves the wrong model")
        if agent.get("model_server") != {"type": "responses_api_models", "name": proxy_name}:
            raise ValueError(f"Gym Tau2 agent {agent_name} is not bound to {proxy_name}")
        if agent.get("user_model_server") != {
            "type": "responses_api_models",
            "name": "pipelinerl_tau2_user",
        }:
            raise ValueError(f"Gym Tau2 agent {agent_name} is not bound to the frozen user model")

        host = agent.get("host")
        port = agent.get("port")
        if not host or not isinstance(port, int):
            raise ValueError(f"Gym Tau2 agent {agent_name} has no executed host/port")
        bindings[policy_url] = Tau2AgentBinding(
            policy_base_url=policy_url,
            agent_name=agent_name,
            agent_url=f"http://{host}:{port}",
        )

    return bindings


def _head_config_url(head_url: str) -> str:
    return f"{head_url.rstrip('/')}/global_config_dict_yaml"


def validate_tau2_gym_sync(
    settings: Tau2GymSettings | Mapping[str, Any],
    actor_llm_urls: Sequence[str],
) -> dict[str, Tau2AgentBinding]:
    parsed_settings = Tau2GymSettings.model_validate(settings)
    deadline = time.monotonic() + parsed_settings.startup_timeout_s
    retry_delay_s = 1.0
    while True:
        try:
            response = requests.get(_head_config_url(parsed_settings.head_url), timeout=10.0)
            response.raise_for_status()
            config = parse_executed_gym_config(response.json())
            bindings = validate_executed_gym_config(config, actor_llm_urls, parsed_settings)
            for binding in bindings.values():
                health = requests.get(f"{binding.agent_url}/", timeout=10.0)
                health.raise_for_status()
            return bindings
        except requests.RequestException as exc:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                raise
            sleep_s = min(retry_delay_s, remaining_s)
            logger.warning("Tau2 Gym is not ready; retrying in %.1fs: %s", sleep_s, exc)
            time.sleep(sleep_s)
            retry_delay_s = min(retry_delay_s * 2, 10.0)


class Tau2GymClient:
    def __init__(
        self,
        settings: Tau2GymSettings | Mapping[str, Any],
        actor_llm_urls: Sequence[str],
    ) -> None:
        self.settings = Tau2GymSettings.model_validate(settings)
        self.actor_llm_urls = tuple(actor_llm_urls)
        self._bindings: dict[str, Tau2AgentBinding] = {}
        self._validated_at = 0.0
        self._validation_lock = asyncio.Lock()

    async def _fetch_executed_config(self, session: aiohttp.ClientSession) -> dict[str, Any]:
        async with session.get(_head_config_url(self.settings.head_url)) as response:
            response.raise_for_status()
            return parse_executed_gym_config(await response.json())

    async def _check_agent(self, session: aiohttp.ClientSession, binding: Tau2AgentBinding) -> None:
        async with session.get(f"{binding.agent_url}/") as response:
            response.raise_for_status()

    async def ensure_valid(self, session: aiohttp.ClientSession) -> None:
        now = time.monotonic()
        if self._bindings and now - self._validated_at < self.settings.validation_interval_s:
            return
        async with self._validation_lock:
            now = time.monotonic()
            if self._bindings and now - self._validated_at < self.settings.validation_interval_s:
                return
            config = await self._fetch_executed_config(session)
            bindings = validate_executed_gym_config(config, self.actor_llm_urls, self.settings)
            await asyncio.gather(*(self._check_agent(session, binding) for binding in bindings.values()))
            self._bindings = bindings
            self._validated_at = time.monotonic()

    async def run(
        self,
        policy_base_url: str,
        problem: Mapping[str, Any],
        session: aiohttp.ClientSession,
    ) -> Tau2RunResponse:
        await self.ensure_valid(session)
        policy_url = normalize_openai_base_url(policy_base_url)
        try:
            binding = self._bindings[policy_url]
        except KeyError as exc:
            raise ValueError(f"No Tau2 Gym agent is bound to policy endpoint {policy_url}") from exc

        timeout = aiohttp.ClientTimeout(total=self.settings.request_timeout_s)
        async with session.post(f"{binding.agent_url}/run", json=dict(problem), timeout=timeout) as response:
            response.raise_for_status()
            return Tau2RunResponse.model_validate(await response.json())
