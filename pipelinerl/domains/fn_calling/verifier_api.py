"""Verifier utilities for the fn_calling domain."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from typing import Any, Callable, Iterable, Literal

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)

AnswerStatus = Literal["correct", "wrong", "no_answer", "unparsable"]
_VALID_STATUSES: set[str] = {"correct", "wrong", "no_answer", "unparsable"}
_DEFAULT_REWARD_FN_ENV = "FN_CALLING_REWARD_FN"
_TOOL_BLOCK = re.compile(r"<tool_calls>(.*?)</tool_calls>", re.DOTALL | re.IGNORECASE)


def _json_loads(value: str) -> Any:
    return json.loads(value)


def _normalize_args(value: Any) -> Any:
    if value is None or value == "":
        return {}
    if isinstance(value, (dict, list, bool, int, float)):
        return value
    if isinstance(value, str):
        return _json_loads(value)
    raise ValueError(f"Unsupported argument payload: {type(value)}")


def _canonicalize(entries: Iterable[Any]) -> list[tuple[str, str]]:
    canon: list[tuple[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Tool call entry must be a mapping")
        fn_block = entry.get("function") if isinstance(entry.get("function"), dict) else None
        name = entry.get("name") or (fn_block or {}).get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool call missing name")
        args = entry.get("arguments")
        if args is None and fn_block:
            args = fn_block.get("arguments")
        canon_args = json.dumps(_normalize_args(args), sort_keys=True, ensure_ascii=False)
        canon.append((name.strip(), canon_args))
    return canon


def _parse_prediction(generation: str | None) -> list[tuple[str, str]]:
    if not generation:
        return []
    block = _TOOL_BLOCK.findall(generation)
    if not block:
        return []
    payload = block[-1].strip()
    if not payload:
        return []
    data = _json_loads(payload)
    if not isinstance(data, list):
        raise ValueError("tool_calls block must be a list")
    return _canonicalize(data)


def _parse_label(tool_calls: Iterable[Any] | None) -> list[tuple[str, str]]:
    if not tool_calls:
        return []
    return _canonicalize(tool_calls)


def _has_irrelevant_flag(reward_context: dict[str, Any], extra_info: dict[str, Any]) -> bool:
    for source in (reward_context, extra_info):
        flag = source.get("irrelevant_tool_call")
        if isinstance(flag, bool):
            return flag
        if isinstance(flag, str) and flag.lower() in {"1", "true", "yes"}:
            return True
    return "irrelevant_tool_call" in json.dumps(reward_context, ensure_ascii=False)


def evaluate_fn_calling_answer(
    *,
    generation: str,
    reward_context: dict[str, Any] | None,
    extra_info: dict[str, Any] | None = None,
) -> AnswerStatus:
    reward_context = reward_context or {}
    extra_info = extra_info or {}

    try:
        expected = _parse_label(reward_context.get("tool_calls"))
        predicted = _parse_prediction(generation)
    except ValueError as exc:
        LOGGER.debug("Failed to parse fn_calling sample: %s", exc)
        return "unparsable"

    if _has_irrelevant_flag(reward_context, extra_info):
        return "wrong" if predicted else "correct"

    expected_counter = Counter(expected)
    predicted_counter = Counter(predicted)

    if not predicted_counter and not expected_counter:
        return "correct"
    if not predicted_counter and expected_counter:
        return "no_answer"
    if predicted_counter and not expected_counter:
        return "wrong"
    return "correct" if predicted_counter == expected_counter else "wrong"


def _ensure_mapping(data: dict[str, Any] | str | None) -> dict[str, Any]:
    if data is None:
        return {}
    if isinstance(data, dict):
        return data
    if isinstance(data, str) and data.strip():
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            LOGGER.debug("Failed to decode fn_calling payload: %s", data[:128])
            return {}
    return {}


class FnCallingVerificationRequest(BaseModel):
    """Payload accepted by the fn_calling verifier service."""

    generation: str
    reward_context: dict[str, Any] = Field(default_factory=dict)
    extra_info: dict[str, Any] = Field(default_factory=dict)


@lru_cache(maxsize=None)
def _import_callable(path: str) -> Callable[..., Any]:
    module_name, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, attr_name)
    if not callable(fn):  # pragma: no cover - defensive guard
        raise TypeError(f"Object at '{path}' is not callable")
    return fn


def _invoke_reward_fn(
    reward_fn_path: str,
    generation: str,
    reward_context: dict[str, Any],
    extra_info: dict[str, Any],
) -> AnswerStatus:
    fn = _import_callable(reward_fn_path)
    try:
        result = fn(generation=generation, reward_context=reward_context, extra_info=extra_info)
    except TypeError:
        result = fn(generation, reward_context)
    status = str(result).strip().lower()
    if status not in _VALID_STATUSES:
        raise ValueError(f"Reward function returned invalid status '{result}'")
    return status  # type: ignore[return-value]


def _execute_reward_job(
    reward_fn_path: str | None,
    generation: str,
    reward_context: dict[str, Any],
    extra_info: dict[str, Any],
) -> AnswerStatus:
    if reward_fn_path:
        return _invoke_reward_fn(reward_fn_path, generation, reward_context, extra_info)
    return evaluate_fn_calling_answer(
        generation=generation,
        reward_context=reward_context,
        extra_info=extra_info,
    )


async def verify_fn_calling_answer_rpc(
    *,
    session: aiohttp.ClientSession,
    host: str,
    port: int,
    generation: str,
    reward_context: dict[str, Any] | str | None,
    extra_info: dict[str, Any] | str | None = None,
) -> AnswerStatus:
    payload = {
        "generation": generation,
        "reward_context": _ensure_mapping(reward_context),
        "extra_info": _ensure_mapping(extra_info),
    }
    async with session.post(f"http://{host}:{port}/verify_answer", json=payload) as response:
        body = await response.text()
        if response.status != 200:
            LOGGER.error("fn_calling verifier returned %s: %s", response.status, body[:512])
            raise ValueError("fn_calling verifier request failed")
        data = json.loads(body)
        status = str(data.get("answer_status", "")).strip().lower()
        if status not in _VALID_STATUSES:
            raise ValueError(f"fn_calling verifier produced invalid status '{status}'")
        return status  # type: ignore[return-value]


class AgenticToolsEnvironment:
    """FastAPI wrapper that exposes a deterministic fn_calling verifier."""

    def __init__(
        self,
        *,
        reward_fn_path: str | None = None,
        max_workers: int = 4,
        keepalive_timeout_s: int = 60,
    ) -> None:
        self._reward_fn_path = reward_fn_path or os.environ.get(_DEFAULT_REWARD_FN_ENV)
        self._max_workers = max_workers
        self._keepalive_timeout_s = keepalive_timeout_s

    def launch(self, port: int) -> None:
        app = FastAPI()

        with ProcessPoolExecutor(max_workers=self._max_workers) as process_pool:
            @app.post("/verify_answer")
            async def verify(request: FnCallingVerificationRequest):
                loop = asyncio.get_running_loop()
                try:
                    answer_status = await loop.run_in_executor(
                        process_pool,
                        _execute_reward_job,
                        self._reward_fn_path,
                        request.generation,
                        dict(request.reward_context),
                        dict(request.extra_info),
                    )
                except Exception as exc:  # pragma: no cover - server-side diagnostics
                    LOGGER.exception("fn_calling reward function failed")
                    raise HTTPException(status_code=500, detail=str(exc))
                return JSONResponse(content={"answer_status": answer_status})

            @app.get("/health")
            async def health():
                return {"status": "ok"}

            uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=self._keepalive_timeout_s)