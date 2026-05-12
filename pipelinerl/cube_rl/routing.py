from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass
from typing import Any

import ray

from cube_harness.llm import LLMRouteLease

logger = logging.getLogger(__name__)


def _api_base(base_url: str) -> str:
    return base_url if base_url.endswith("/v1") else f"{base_url}/v1"


def _route_model_name(llm: dict[str, Any]) -> str:
    model_name = llm.get("served_model_name") or llm["model_name"]
    return model_name if str(model_name).startswith("openai/") else f"openai/{model_name}"


def estimate_prompt_tokens(prompt: Any) -> int:
    """Cheap admission-control estimate used before the real LLM call."""

    chars = 0
    for message in getattr(prompt, "messages", []) or []:
        if isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = getattr(message, "content", "")
        chars += len(str(content))
    tools = getattr(prompt, "tools", None)
    if tools:
        chars += len(str(tools))
    return max(1, chars // 4)


@dataclass(frozen=True)
class RoutedVLLM:
    server_id: int
    api_base: str
    model_name: str
    api_key: str = "EMPTY"


class RayVLLMRouter:
    """Synchronous cube-harness router backed by a non-blocking Ray actor."""

    def __init__(self, router_actor: Any, *, poll_interval_s: float = 0.01):
        self._router_actor = router_actor
        self._poll_interval_s = max(0.001, float(poll_interval_s))

    def acquire(self, config: Any, prompt: Any) -> LLMRouteLease:
        estimated_tokens = estimate_prompt_tokens(prompt) + int(getattr(config, "max_completion_tokens", 0) or 0)
        wait_started_at = time.perf_counter()
        while True:
            lease = ray.get(self._router_actor.try_acquire.remote(estimated_tokens=estimated_tokens))
            if lease is not None:
                return LLMRouteLease(
                    route_id=lease["route_id"],
                    api_base=lease["api_base"],
                    api_key=lease["api_key"],
                    model_name=lease["model_name"],
                    metadata={
                        "route_id": lease["route_id"],
                        "vllm_server_id": lease["server_id"],
                        "vllm_api_base": lease["api_base"],
                        "vllm_model_name": lease["model_name"],
                        "vllm_estimated_tokens": estimated_tokens,
                        "vllm_lease_wait_s": time.perf_counter() - wait_started_at,
                    },
                )
            time.sleep(self._poll_interval_s)

    def release(
        self,
        lease: LLMRouteLease,
        response: Any | None = None,
        error: BaseException | None = None,
    ) -> None:
        prompt_tokens = 0
        completion_tokens = 0
        if response is not None and getattr(response, "usage", None) is not None:
            prompt_tokens = int(getattr(response.usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(response.usage, "completion_tokens", 0) or 0)
        ray.get(
            self._router_actor.release.remote(
                route_id=lease.route_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                errored=error is not None,
            )
        )


@ray.remote(max_restarts=0, max_task_retries=0)
class VLLMRouterActor:
    """Tracks per-vLLM active generation load and leases one request at a time."""

    def __init__(self, routes: list[dict[str, Any]], max_inflight_per_server: int):
        if not routes:
            raise ValueError("VLLMRouterActor requires at least one route")
        if max_inflight_per_server < 1:
            raise ValueError("max_inflight_per_server must be >= 1")

        self._routes = [
            RoutedVLLM(
                server_id=i,
                api_base=_api_base(str(route["base_url"])),
                model_name=_route_model_name(route),
            )
            for i, route in enumerate(routes)
        ]
        self._max_inflight = int(max_inflight_per_server)
        self._inflight = [0 for _ in self._routes]
        self._active_tokens = [0 for _ in self._routes]
        self._latency_ema = [0.0 for _ in self._routes]
        self._errors = [0 for _ in self._routes]
        self._requests = [0 for _ in self._routes]
        self._suppressed_until = [0.0 for _ in self._routes]
        self._leases: dict[str, tuple[int, float, int]] = {}
        self._lease_ids = itertools.count()
        self._error_suppression_threshold = 3
        self._suppression_cooldown_s = 30.0

    def try_acquire(self, estimated_tokens: int = 0) -> dict[str, Any] | None:
        now = time.time()
        eligible = [
            route.server_id
            for route in self._routes
            if self._inflight[route.server_id] < self._max_inflight
            and self._suppressed_until[route.server_id] <= now
        ]
        if not eligible:
            # If every server is suppressed, prefer slow progress over a hard stall.
            eligible = [
                route.server_id
                for route in self._routes
                if self._inflight[route.server_id] < self._max_inflight
            ]
        if not eligible:
            return None

        server_id = min(
            eligible,
            key=lambda i: (
                self._inflight[i],
                self._active_tokens[i],
                self._latency_ema[i],
                i,
            ),
        )
        route = self._routes[server_id]
        route_id = f"{server_id}:{next(self._lease_ids)}"
        token_load = max(1, int(estimated_tokens or 1))
        self._inflight[server_id] += 1
        self._active_tokens[server_id] += token_load
        self._requests[server_id] += 1
        self._leases[route_id] = (server_id, time.perf_counter(), token_load)
        return {
            "route_id": route_id,
            "server_id": server_id,
            "api_base": route.api_base,
            "api_key": route.api_key,
            "model_name": route.model_name,
        }

    def release(
        self,
        *,
        route_id: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        errored: bool = False,
    ) -> None:
        lease = self._leases.pop(route_id, None)
        if lease is None:
            logger.warning("Ignoring unknown vLLM route lease release: %s", route_id)
            return

        server_id, started_at, estimated_tokens = lease
        self._inflight[server_id] = max(0, self._inflight[server_id] - 1)
        self._active_tokens[server_id] = max(0, self._active_tokens[server_id] - estimated_tokens)
        latency = time.perf_counter() - started_at
        previous = self._latency_ema[server_id]
        self._latency_ema[server_id] = latency if previous <= 0 else previous * 0.9 + latency * 0.1
        if errored:
            self._errors[server_id] += 1
            if self._errors[server_id] >= self._error_suppression_threshold:
                self._suppressed_until[server_id] = time.time() + self._suppression_cooldown_s
        else:
            self._errors[server_id] = 0
            self._suppressed_until[server_id] = 0.0

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        return {
            "max_inflight_per_server": self._max_inflight,
            "servers": [
                {
                    "server_id": route.server_id,
                    "api_base": route.api_base,
                    "inflight": self._inflight[route.server_id],
                    "active_tokens": self._active_tokens[route.server_id],
                    "latency_ema": self._latency_ema[route.server_id],
                    "errors": self._errors[route.server_id],
                    "requests": self._requests[route.server_id],
                    "suppressed": self._suppressed_until[route.server_id] > now,
                    "suppressed_remaining_s": max(0.0, self._suppressed_until[route.server_id] - now),
                }
                for route in self._routes
            ],
        }
