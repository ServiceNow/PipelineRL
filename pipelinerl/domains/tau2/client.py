import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import aiohttp
import requests
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


NEMO_GYM_SHA = "5f92a73217258074b74b7be26526c69f0ce3075d"
NEMO_GYM_TITO_PATCH_SHA = "2ef95bf3cd7f045a134554fbee18e4ba224c9a191fc736c613278b519ed5d6f6"
TAU2_RUNTIME_SHA = "befd120003fb55f48b498f6549556dcaf74582d5"
TAU2_DATA_SHA = "ce4013b0afe03c873488878b72851414f92f458b"

logger = logging.getLogger(__name__)

_GYM_SHA_KEY = "pipelinerl_nemo_gym_sha"
_GYM_PATCH_SHA_KEY = "pipelinerl_gym_patch_sha"
_TAU2_RUNTIME_SHA_KEY = "pipelinerl_tau2_runtime_sha"
_TAU2_DATA_SHA_KEY = "pipelinerl_tau2_data_sha"
_POLICY_MODEL_NAME_KEY = "pipelinerl_policy_model_name"
_POLICY_THINKING_KEY = "pipelinerl_policy_thinking_enabled"
_USER_MODEL_URL_KEY = "pipelinerl_user_model_url"
_USER_MODEL_NAME_KEY = "pipelinerl_user_model_name"
_JUDGE_MODEL_URL_KEY = "pipelinerl_judge_model_url"
_JUDGE_MODEL_NAME_KEY = "pipelinerl_judge_model_name"
_USER_THINKING_KEY = "pipelinerl_user_thinking_enabled"
_JUDGE_THINKING_KEY = "pipelinerl_judge_thinking_enabled"
_JUDGE_TEMPERATURE_KEY = "pipelinerl_judge_temperature"
_JUDGE_TOP_P_KEY = "pipelinerl_judge_top_p"
_JUDGE_TOP_K_KEY = "pipelinerl_judge_top_k"
_JUDGE_SEED_KEY = "pipelinerl_judge_seed"
_JUDGE_INITIAL_MAX_TOKENS_KEY = "pipelinerl_judge_initial_max_tokens"
_JUDGE_RETRY_MAX_TOKENS_KEY = "pipelinerl_judge_retry_max_tokens"
_AUXILIARY_MODEL_TIMEOUT_KEY = "pipelinerl_auxiliary_model_timeout_s"
_REQUEST_TIMEOUT_KEY = "pipelinerl_request_timeout_s"
_JUDGE_MARKER_KEY = "pipelinerl_judge_reward_invalid"
_JUDGE_FAILURE_CODES = {
    "missing_content",
    "invalid_json",
    "invalid_object",
    "missing_results",
    "invalid_results",
    "result_count_mismatch",
    "invalid_result",
    "expected_outcome_mismatch",
    "invalid_met_expectation",
    "missing_reasoning",
    "finish_reason_length",
}


class Tau2GymSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    head_url: str
    policy_model_name: str
    user_model_url: str
    user_model_name: str
    judge_model_url: str
    judge_model_name: str
    policy_thinking_enabled: bool
    user_thinking_enabled: bool
    judge_thinking_enabled: bool
    judge_temperature: float = Field(ge=0)
    judge_top_p: float = Field(gt=0, le=1)
    judge_top_k: int = Field(gt=0)
    judge_seed: int
    judge_initial_max_tokens: int = Field(gt=0)
    judge_retry_max_tokens: int = Field(gt=0)
    auxiliary_model_timeout_s: float = Field(gt=0)
    validation_interval_s: float = 30.0
    request_timeout_s: float = 3600.0
    startup_timeout_s: float = 300.0

    @model_validator(mode="after")
    def validate_auxiliary_contract(self):
        if normalize_openai_base_url(self.user_model_url) != normalize_openai_base_url(
            self.judge_model_url
        ):
            raise ValueError("Tau2 user and judge endpoints must be the same shared service")
        if len({self.policy_model_name, self.user_model_name, self.judge_model_name}) != 3:
            raise ValueError("Tau2 policy, user, and judge model aliases must be distinct")
        if not (
            self.policy_thinking_enabled
            and self.user_thinking_enabled
            and self.judge_thinking_enabled
        ):
            raise ValueError("Tau2 run 1 requires policy, user, and judge thinking")
        if self.judge_retry_max_tokens < 2 * self.judge_initial_max_tokens:
            raise ValueError("Tau2 judge retry budget must be at least twice the initial budget")
        if self.auxiliary_model_timeout_s >= self.request_timeout_s:
            raise ValueError("Tau2 auxiliary timeout must be below request_timeout_s")
        return self


class Tau2AuxiliaryCall(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    role: Literal["user_simulator", "judge"]
    call_index: int = Field(gt=0)
    attempt_index: int = Field(gt=0)
    requested_model_alias: str = Field(min_length=1)
    response_model_alias: str = Field(min_length=1)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    latency_s: float = Field(ge=0)
    finish_reason: str = Field(min_length=1)
    completion_budget: int | None = Field(default=None, gt=0)
    output_character_count: int = Field(ge=0)
    output_sha256: str | None
    failure_code: str | None

    @model_validator(mode="after")
    def validate_role_contract(self):
        if (
            not self.requested_model_alias
            or not self.response_model_alias
            or not self.finish_reason
        ):
            raise ValueError(
                "Tau2 auxiliary call has an empty identity or finish reason"
            )
        if self.role == "user_simulator":
            if (
                self.output_sha256 is not None
                or self.failure_code not in (None, "empty_message")
            ):
                raise ValueError("Tau2 user call has invalid output metadata")
        else:
            if (
                self.completion_budget is None
                or self.failure_code not in ({None} | _JUDGE_FAILURE_CODES)
            ):
                raise ValueError("Tau2 judge call has invalid output metadata")
            if self.output_sha256 is not None and (
                len(self.output_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in self.output_sha256
                )
            ):
                raise ValueError("Tau2 judge call has invalid output SHA256")
        return self


class Tau2JudgeSampling(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    requested_model_alias: str = Field(min_length=1)
    enable_thinking: Literal[True]
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    top_k: int = Field(gt=0)
    seed: int
    initial_max_tokens: int = Field(gt=0)
    retry_max_tokens: int = Field(gt=0)
    auxiliary_model_timeout_s: float = Field(gt=0)


class _Tau2RunPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: int
    outcome: str
    config: dict[str, Any]
    task: dict[str, Any]
    seed: int
    evaluation_type: str
    save_dir: None
    user_voice_settings: None
    user_persona_config: None
    verbose_logs: bool
    audio_debug: bool
    audio_taps: bool
    auto_review: bool
    review_mode: str
    hallucination_feedback: None
    responses_create_params: dict[str, Any]
    response: dict[str, Any]
    result: dict[str, Any]
    auxiliary_model_calls: list[Tau2AuxiliaryCall]
    judge_sampling: Tau2JudgeSampling
    duration: float = Field(ge=0)
    num_steps: int = Field(ge=0)
    num_agent_calls: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_result_and_capture(self):
        if not isinstance(self.responses_create_params.get("input"), list):
            raise ValueError("Tau2 response has no captured input list")
        if not isinstance(self.response.get("output"), list):
            raise ValueError("Tau2 response has no captured output list")
        if not isinstance(self.result.get("messages"), list):
            raise ValueError("Tau2 response has no result message list")
        if (
            not isinstance(self.result.get("termination_reason"), str)
            or not self.result["termination_reason"]
        ):
            raise ValueError("Tau2 response has no termination reason")
        if not isinstance(self.result.get("reward_info"), Mapping):
            raise ValueError("Tau2 response has no reward info")
        return self


class _Tau2CompletedResponse(_Tau2RunPayload):
    schema_version: Literal[1]
    outcome: Literal["completed"]
    evaluation_type: Literal["all"]
    verbose_logs: Literal[False]
    audio_debug: Literal[False]
    audio_taps: Literal[False]
    auto_review: Literal[False]
    review_mode: Literal["full"]
    reward: float
    min_prompt_tokens: float | None = None
    min_completion_tokens: float | None = None
    mean_prompt_tokens: float | None = None
    mean_completion_tokens: float | None = None
    max_prompt_tokens: float | None = None
    max_completion_tokens: float | None = None


class Tau2RunResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    reward: float
    responses_create_params: dict[str, Any]
    response: dict[str, Any]
    result: dict[str, Any]
    duration: float | None = None
    num_steps: int | None = None
    num_agent_calls: int | None = None


class Tau2BoundaryResponse(_Tau2RunPayload):
    schema_version: Literal[1]
    outcome: Literal["boundary_failure"]
    evaluation_type: Literal["all"]
    verbose_logs: Literal[False]
    audio_debug: Literal[False]
    audio_taps: Literal[False]
    auto_review: Literal[False]
    review_mode: Literal["full"]
    reason: Literal["judge_reward_invalid"]
    producer_stage: Literal["judge"]
    detail_code: str = Field(min_length=1)
    attempt_count: int = Field(ge=1, le=2)
    diagnostic: str = Field(min_length=1)


class Tau2GymContractError(RuntimeError):
    pass


class Tau2BoundaryFailure(RuntimeError):
    def __init__(self, boundary: Tau2BoundaryResponse):
        super().__init__(
            f"Tau2 boundary failure {boundary.reason}: {boundary.detail_code}"
        )
        self.boundary = boundary


def _judge_marker_from_result(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    reward_info = result.get("reward_info")
    if not isinstance(reward_info, Mapping):
        return None
    info = reward_info.get("info")
    if not isinstance(info, Mapping):
        return None
    nl_info = info.get("nl")
    if not isinstance(nl_info, Mapping):
        return None
    if _JUDGE_MARKER_KEY not in nl_info:
        return None
    marker = nl_info[_JUDGE_MARKER_KEY]
    if not isinstance(marker, Mapping):
        raise Tau2GymContractError("Tau2 judge-invalid marker is malformed")
    return marker


def _validate_boundary_marker(boundary: Tau2BoundaryResponse) -> None:
    reward_info = boundary.result["reward_info"]
    if "reward" in reward_info:
        raise Tau2GymContractError(
            "Tau2 boundary leaked its internal reward"
        )
    reward_breakdown = reward_info.get("reward_breakdown")
    if (
        isinstance(reward_breakdown, Mapping)
        and "NL_ASSERTION" in reward_breakdown
    ):
        raise Tau2GymContractError(
            "Tau2 boundary leaked its invalid judge score"
        )
    marker = _judge_marker_from_result(boundary.result)
    expected = {
        "reason": boundary.reason,
        "producer_stage": boundary.producer_stage,
        "detail_code": boundary.detail_code,
        "attempt_count": boundary.attempt_count,
        "diagnostic": boundary.diagnostic,
    }
    if marker != expected:
        raise Tau2GymContractError(
            "Tau2 boundary response does not match its judge-invalid marker"
        )


def _validate_response_contract(
    payload: _Tau2RunPayload,
    settings: Tau2GymSettings,
) -> None:
    expected_sampling = {
        "requested_model_alias": settings.judge_model_name,
        "enable_thinking": True,
        "temperature": settings.judge_temperature,
        "top_p": settings.judge_top_p,
        "top_k": settings.judge_top_k,
        "seed": settings.judge_seed,
        "initial_max_tokens": settings.judge_initial_max_tokens,
        "retry_max_tokens": settings.judge_retry_max_tokens,
        "auxiliary_model_timeout_s": settings.auxiliary_model_timeout_s,
    }
    if payload.judge_sampling.model_dump() != expected_sampling:
        raise Tau2GymContractError("Tau2 response has wrong executed judge sampling")
    approved_aliases = {
        settings.user_model_name,
        settings.judge_model_name,
    }
    for call in payload.auxiliary_model_calls:
        expected_requested = (
            settings.user_model_name
            if call.role == "user_simulator"
            else settings.judge_model_name
        )
        if call.requested_model_alias != expected_requested:
            raise Tau2GymContractError("Tau2 auxiliary call has wrong requested alias")
        if call.response_model_alias not in approved_aliases:
            raise Tau2GymContractError("Tau2 auxiliary call has unapproved response alias")

    for role in ("user_simulator", "judge"):
        role_calls = [
            call
            for call in payload.auxiliary_model_calls
            if call.role == role
        ]
        previous = None
        for call in role_calls:
            expected_position = (1, 1)
            if previous is not None:
                expected_position = (
                    (
                        previous.call_index,
                        previous.attempt_index + 1,
                    )
                    if previous.failure_code is not None
                    else (previous.call_index + 1, 1)
                )
            if (call.call_index, call.attempt_index) != expected_position:
                raise Tau2GymContractError(
                    "Tau2 auxiliary call ordering is invalid"
                )
            previous = call

    judge_calls = [
        call
        for call in payload.auxiliary_model_calls
        if call.role == "judge"
    ]
    if isinstance(payload, Tau2BoundaryResponse):
        if not judge_calls or (
            judge_calls[-1].attempt_index != payload.attempt_count
            or judge_calls[-1].failure_code != payload.detail_code
        ):
            raise Tau2GymContractError(
                "Tau2 judge boundary does not match its call audit"
            )
    elif judge_calls and judge_calls[-1].failure_code is not None:
        raise Tau2GymContractError(
            "Tau2 completed response ends with a failed judge call"
        )


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
    if normalize_openai_base_url(str(config.get(_JUDGE_MODEL_URL_KEY, ""))) != (
        normalize_openai_base_url(settings.judge_model_url)
    ):
        raise ValueError("Executed Gym judge-model endpoint does not match PipelineRL configuration")
    if config.get(_JUDGE_MODEL_NAME_KEY) != settings.judge_model_name:
        raise ValueError("Executed Gym judge-model name does not match PipelineRL configuration")

    executed_auxiliary = {
        _POLICY_THINKING_KEY: settings.policy_thinking_enabled,
        _USER_THINKING_KEY: settings.user_thinking_enabled,
        _JUDGE_THINKING_KEY: settings.judge_thinking_enabled,
        _JUDGE_TEMPERATURE_KEY: settings.judge_temperature,
        _JUDGE_TOP_P_KEY: settings.judge_top_p,
        _JUDGE_TOP_K_KEY: settings.judge_top_k,
        _JUDGE_SEED_KEY: settings.judge_seed,
        _JUDGE_INITIAL_MAX_TOKENS_KEY: settings.judge_initial_max_tokens,
        _JUDGE_RETRY_MAX_TOKENS_KEY: settings.judge_retry_max_tokens,
        _AUXILIARY_MODEL_TIMEOUT_KEY: settings.auxiliary_model_timeout_s,
        _REQUEST_TIMEOUT_KEY: settings.request_timeout_s,
    }
    for key, expected in executed_auxiliary.items():
        if config.get(key) != expected:
            raise ValueError(f"Executed Gym config {key} does not match PipelineRL")

    policy_urls = [normalize_openai_base_url(url) for url in actor_llm_urls]
    if len(policy_urls) != len(set(policy_urls)):
        raise ValueError("PipelineRL actor LLM URLs must be unique")
    shared_auxiliary_url = normalize_openai_base_url(settings.user_model_url)
    if shared_auxiliary_url in policy_urls:
        raise ValueError("Tau2 auxiliary-model endpoint must differ from every policy endpoint")

    user_model = _server_config(config, "pipelinerl_tau2_user", "responses_api_models", "openai_model")
    if normalize_openai_base_url(str(user_model.get("openai_base_url", ""))) != normalize_openai_base_url(
        settings.user_model_url
    ):
        raise ValueError("Executed Gym user proxy points to the wrong endpoint")
    if user_model.get("openai_model") != settings.user_model_name:
        raise ValueError("Executed Gym user proxy points to the wrong model")
    if user_model.get("extra_body") != {
        "chat_template_kwargs": {"enable_thinking": True}
    }:
        raise ValueError("Executed Gym user proxy has wrong thinking configuration")

    judge_model = _server_config(
        config,
        "pipelinerl_tau2_judge",
        "responses_api_models",
        "openai_model",
    )
    if normalize_openai_base_url(str(judge_model.get("openai_base_url", ""))) != (
        normalize_openai_base_url(settings.judge_model_url)
    ):
        raise ValueError("Executed Gym judge proxy points to the wrong endpoint")
    if judge_model.get("openai_model") != settings.judge_model_name:
        raise ValueError("Executed Gym judge proxy points to the wrong model")
    if judge_model.get("extra_body") != {
        "chat_template_kwargs": {"enable_thinking": True},
        "top_k": settings.judge_top_k,
    }:
        raise ValueError("Executed Gym judge proxy has wrong thinking or top-k configuration")

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
        if policy_proxy.get("chat_template_kwargs") != {"enable_thinking": True}:
            raise ValueError(f"Gym policy proxy {proxy_name} has wrong thinking configuration")
        if policy_proxy.get("uses_reasoning_parser") is not True:
            raise ValueError(f"Gym policy proxy {proxy_name} must use the reasoning parser")
        if agent.get("model_server") != {"type": "responses_api_models", "name": proxy_name}:
            raise ValueError(f"Gym Tau2 agent {agent_name} is not bound to {proxy_name}")
        expected_agent_fields = {
            "user_model_server": {
                "type": "responses_api_models",
                "name": "pipelinerl_tau2_user",
            },
            "judge_model_server": {
                "type": "responses_api_models",
                "name": "pipelinerl_tau2_judge",
            },
            "user_model_name": settings.user_model_name,
            "judge_model_name": settings.judge_model_name,
            "user_thinking_enabled": True,
            "judge_thinking_enabled": True,
            "judge_temperature": settings.judge_temperature,
            "judge_top_p": settings.judge_top_p,
            "judge_top_k": settings.judge_top_k,
            "judge_seed": settings.judge_seed,
            "judge_initial_max_tokens": settings.judge_initial_max_tokens,
            "judge_retry_max_tokens": settings.judge_retry_max_tokens,
            "auxiliary_model_timeout_s": settings.auxiliary_model_timeout_s,
        }
        for key, expected in expected_agent_fields.items():
            if agent.get(key) != expected:
                raise ValueError(f"Gym Tau2 agent {agent_name} has wrong {key}")

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


def _user_health_url(user_model_url: str) -> str:
    return (
        normalize_openai_base_url(user_model_url).removesuffix("/v1")
        + "/health"
    )


def _user_models_url(user_model_url: str) -> str:
    return f"{normalize_openai_base_url(user_model_url)}/models"


def _validate_auxiliary_models_payload(
    payload: Any,
    expected_models: Sequence[str],
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("Tau2 auxiliary service returned malformed /v1/models")
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("Tau2 auxiliary service returned malformed /v1/models")
    model_ids = []
    for item in data:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            raise ValueError("Tau2 auxiliary service returned malformed /v1/models")
        model_ids.append(item["id"])
    if model_ids != list(expected_models):
        raise ValueError(
            "Tau2 auxiliary service model aliases "
            f"{model_ids!r}, expected exactly {list(expected_models)!r}"
        )


def validate_tau2_gym_sync(
    settings: Tau2GymSettings | Mapping[str, Any],
    actor_llm_urls: Sequence[str],
) -> dict[str, Tau2AgentBinding]:
    parsed_settings = Tau2GymSettings.model_validate(settings)
    deadline = time.monotonic() + parsed_settings.startup_timeout_s
    retry_delay_s = 1.0
    while True:
        try:
            user_health = requests.get(
                _user_health_url(parsed_settings.user_model_url),
                timeout=10.0,
            )
            user_health.raise_for_status()
            user_models = requests.get(
                _user_models_url(parsed_settings.user_model_url),
                timeout=10.0,
            )
            user_models.raise_for_status()
            _validate_auxiliary_models_payload(
                user_models.json(),
                (
                    parsed_settings.user_model_name,
                    parsed_settings.judge_model_name,
                ),
            )
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

    async def _check_user_simulator(
        self,
        session: aiohttp.ClientSession,
    ) -> None:
        deadline = time.monotonic() + self.settings.startup_timeout_s
        retry_delay_s = 1.0
        while True:
            try:
                timeout = aiohttp.ClientTimeout(total=10.0)
                async with session.get(
                    _user_health_url(self.settings.user_model_url),
                    timeout=timeout,
                ) as response:
                    response.raise_for_status()
                async with session.get(
                    _user_models_url(self.settings.user_model_url),
                    timeout=timeout,
                ) as response:
                    response.raise_for_status()
                    _validate_auxiliary_models_payload(
                        await response.json(),
                        (
                            self.settings.user_model_name,
                            self.settings.judge_model_name,
                        ),
                    )
                return
            except aiohttp.ClientResponseError as exc:
                if exc.status < 500:
                    raise
                transient_error = exc
            except (
                aiohttp.ClientConnectionError,
                asyncio.TimeoutError,
            ) as exc:
                transient_error = exc

            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                raise TimeoutError(
                    "Tau2 user simulator did not recover before the "
                    "readiness deadline"
                ) from transient_error
            sleep_s = min(retry_delay_s, remaining_s)
            logger.warning(
                "Tau2 user simulator is restarting; retrying in %.1fs: %s",
                sleep_s,
                transient_error,
            )
            await asyncio.sleep(sleep_s)
            retry_delay_s = min(retry_delay_s * 2, 10.0)

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
            await self._check_user_simulator(session)
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
            if response.status == 409:
                try:
                    boundary = Tau2BoundaryResponse.model_validate(await response.json())
                    _validate_boundary_marker(boundary)
                    _validate_response_contract(boundary, self.settings)
                except (ValidationError, TypeError, ValueError) as exc:
                    raise Tau2GymContractError(
                        "Tau2 Gym returned malformed boundary response"
                    ) from exc
                raise Tau2BoundaryFailure(boundary)
            response.raise_for_status()
            completed = _Tau2CompletedResponse.model_validate(await response.json())
            _validate_response_contract(completed, self.settings)
            if _judge_marker_from_result(completed.result) is not None:
                raise Tau2GymContractError(
                    "Tau2 Gym leaked judge-invalid marker in completed response"
                )
            return Tau2RunResponse.model_validate(completed.model_dump())
