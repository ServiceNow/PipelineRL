"""Sandbox-backed verification utilities for the coding domain."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import requests
import aiohttp
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*([\s\S]*?)```", re.IGNORECASE)
_DEFAULT_SANDBOX_URL = os.environ.get("CODING_SANDBOX_URL", "http://sandbox:8080/run_code")


class CodingVerificationRequest(BaseModel):
    """Payload accepted by the coding verifier environment."""

    prediction: str | None = None
    reward_context: dict[str, Any] | str | None = Field(default_factory=dict)
    extra_info: dict[str, Any] | str | None = None


@dataclass
class CodingTestResult:
    index: int
    kind: str
    status: str
    input: str | None = None
    expected: str | None = None
    assertion: str | None = None
    stdout: str = ""
    stderr: str = ""
    elapsed: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "status": self.status,
            "input": self.input,
            "expected": self.expected,
            "assertion": self.assertion,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "elapsed": self.elapsed,
        }


@dataclass
class CodingVerificationSummary:
    passed: int = 0
    total: int = 0
    compile_error: bool = False
    runtime_error: bool = False
    timeout_error: bool = False
    empty_response: bool = False
    error: str | None = None
    call_type: str | None = None
    fn_name: str | None = None
    tests: list[CodingTestResult] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "total": self.total,
            "compile_error": self.compile_error,
            "runtime_error": self.runtime_error,
            "timeout_error": self.timeout_error,
            "empty_response": self.empty_response,
            "error": self.error,
            "call_type": self.call_type,
            "fn_name": self.fn_name,
            "tests": [test.to_dict() for test in self.tests],
        }


def _extract_code(prediction: str | None) -> str:
    if not prediction:
        return ""
    match = _CODE_FENCE_RE.findall(prediction)
    if match:
        return match[-1].strip()
    return prediction.strip()


def _ensure_dict(data: dict[str, Any] | str | None) -> dict[str, Any]:
    if data is None:
        return {}
    if isinstance(data, dict):
        return data
    try:
        return json.loads(data)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to parse reward context: %s", exc)
        return {}


def _normalize_output(text: str | None) -> str:
    if text is None:
        return ""
    return text.strip()


def _compose_script(user_code: str, extra_snippet: str | None) -> str:
    if not extra_snippet:
        return user_code
    return f"{user_code.rstrip()}\n\n{extra_snippet.strip()}\n"


def _convert_bytes_to_mb(value: int) -> int:
    if value <= 0:
        return 0
    return max(16, int(math.ceil(value / (1024 * 1024))))


def _has_compile_error(response: dict[str, Any]) -> bool:
    compile_result = response.get("compile_result") or {}
    if not compile_result:
        return False
    stderr = str(compile_result.get("stderr", ""))
    return_code = compile_result.get("return_code")
    if return_code is not None and return_code != 0:
        return True
    return bool(stderr.strip())


def _is_timeout(response: dict[str, Any]) -> bool:
    run_result = response.get("run_result") or {}
    status = (run_result.get("status") or response.get("status") or "").lower()
    return status == "timeout"


def _has_runtime_error(response: dict[str, Any]) -> bool:
    if _has_compile_error(response) or _is_timeout(response):
        return False
    run_result = response.get("run_result") or {}
    return_code = run_result.get("return_code")
    if return_code is None:
        return False
    if return_code != 0:
        return True
    stderr = str(run_result.get("stderr", ""))
    return "Traceback" in stderr


def _get_run_field(response: dict[str, Any], field: str) -> str:
    run_result = response.get("run_result") or {}
    value = run_result.get(field)
    if value is None:
        return ""
    return str(value)


def _get_status_text(response: dict[str, Any]) -> str:
    run_result = response.get("run_result") or {}
    status = run_result.get("status")
    if status:
        return str(status)
    overall = response.get("status")
    return str(overall or "")


def _post_to_sandbox(
    *,
    code: str,
    stdin: str,
    sandbox_url: str,
    compile_timeout_s: float,
    run_timeout_s: float,
    request_timeout_s: float,
    memory_limit_mb: int,
    language: str,
) -> tuple[dict[str, Any], float]:
    payload = {
        "compile_timeout": compile_timeout_s,
        "run_timeout": run_timeout_s,
        "code": code,
        "stdin": stdin,
        "memory_limit_MB": memory_limit_mb,
        "language": language,
        "files": {},
        "fetch_files": [],
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    start = time.perf_counter()
    try:
        response = requests.post(
            sandbox_url,
            headers=headers,
            data=json.dumps(payload),
            timeout=request_timeout_s,
        )
        response.raise_for_status()
        data = response.json()
    except requests.Timeout as exc:
        data = {
            "status": "Timeout",
            "message": str(exc),
            "compile_result": None,
            "run_result": {"status": "Timeout", "stdout": "", "stderr": str(exc)},
        }
    except requests.RequestException as exc:  # pragma: no cover - network failures
        data = {
            "status": "NetworkError",
            "message": str(exc),
            "compile_result": None,
            "run_result": {"status": "Error", "stdout": "", "stderr": str(exc)},
        }
    elapsed = time.perf_counter() - start
    return data, elapsed


def evaluate_coding_prediction(
    prediction: str | None,
    reward_context: dict[str, Any] | str | None,
    *,
    extra_info: dict[str, Any] | str | None = None,
    sandbox_url: str | None = None,
    compile_timeout_s: float = 5.0,
    run_timeout_s: float = 5.0,
    request_timeout_s: float = 15.0,
    memory_limit_mb: int = 512,
    language: str = "python",
) -> CodingVerificationSummary:
    """Run generated code inside the sandbox and collect pass/fail statistics."""

    sandbox_target = sandbox_url or _DEFAULT_SANDBOX_URL
    context = _ensure_dict(reward_context)
    summary = CodingVerificationSummary(
        call_type=context.get("call_type"),
        fn_name=context.get("fn_name"),
    )

    candidate_code = _extract_code(prediction)
    if not candidate_code:
        summary.empty_response = True
        summary.error = "empty_prediction"
        return summary

    call_type = (context.get("call_type") or "assert").lower().strip()
    tests: list[dict[str, Any]] = []
    if call_type == "assert":
        for idx, assertion in enumerate(context.get("assert_case", []) or []):
            if not assertion:
                continue
            tests.append({"kind": "assert", "assertion": assertion, "index": idx})
    elif call_type == "std":
        inputs = context.get("inputs") or []
        outputs = context.get("outputs") or []
        total = min(len(inputs), len(outputs))
        for idx in range(total):
            tests.append(
                {
                    "kind": "std",
                    "input": inputs[idx],
                    "expected": outputs[idx],
                    "index": idx,
                }
            )
    else:
        summary.error = f"unsupported_call_type:{call_type}"
        return summary

    if not tests:
        summary.error = "no_tests"
        return summary

    for raw_test in tests:
        idx = summary.total
        summary.total += 1
        if raw_test["kind"] == "assert":
            script = _compose_script(candidate_code, raw_test["assertion"])
            stdin = ""
        else:
            script = candidate_code
            stdin = raw_test.get("input", "") or ""

        response, elapsed = _post_to_sandbox(
            code=script,
            stdin=stdin,
            sandbox_url=sandbox_target,
            compile_timeout_s=compile_timeout_s,
            run_timeout_s=run_timeout_s,
            request_timeout_s=request_timeout_s,
            memory_limit_mb=memory_limit_mb,
            language=language,
        )

        if _has_compile_error(response):
            status = "compile_error"
            summary.compile_error = True
            summary.error = "compile_error"
        elif _is_timeout(response):
            status = "timeout"
            summary.timeout_error = True
            summary.error = "timeout"
        elif _has_runtime_error(response):
            status = "runtime_error"
            summary.runtime_error = True
            summary.error = "runtime_error"
        else:
            run_status_text = _get_status_text(response).lower()
            if raw_test["kind"] == "std":
                produced = _normalize_output(_get_run_field(response, "stdout"))
                expected = _normalize_output(raw_test.get("expected"))
                status = "passed" if produced == expected else "failed"
            else:
                succeeded = run_status_text in ("finished", "success", "")
                status = "passed" if succeeded else "failed"
                if not succeeded:
                    summary.runtime_error = True
                    summary.error = summary.error or f"run_status:{run_status_text or 'unknown'}"

        stdout = _get_run_field(response, "stdout")
        stderr = _get_run_field(response, "stderr")
        test_result = CodingTestResult(
            index=idx,
            kind=raw_test["kind"],
            status=status,
            input=raw_test.get("input"),
            expected=raw_test.get("expected"),
            assertion=raw_test.get("assertion"),
            stdout=stdout,
            stderr=stderr,
            elapsed=elapsed,
        )
        summary.tests.append(test_result)
        if status == "passed":
            summary.passed += 1
        if status == "compile_error":
            break

    return summary


def _rpc_failure_summary(reason: str, *, status: int | None = None, body: str | None = None) -> dict[str, Any]:
    details = reason
    if status is not None:
        details = f"{reason}:{status}"
    if body:
        details = f"{details}:{body[:256]}"
    return {
        "passed": 0,
        "total": 0,
        "compile_error": False,
        "runtime_error": True,
        "timeout_error": False,
        "empty_response": False,
        "error": f"verifier_rpc_error:{details}",
        "call_type": None,
        "fn_name": None,
        "tests": [],
    }


async def verify_coding_solution_rpc(
    session: aiohttp.ClientSession,
    host: str,
    port: int,
    *,
    prediction: str | None,
    reward_context: dict[str, Any] | str | None,
    extra_info: dict[str, Any] | str | None,
) -> dict[str, Any]:
    """Call a remote coding verifier via HTTP RPC."""

    payload = {
        "prediction": prediction,
        "reward_context": reward_context,
        "extra_info": extra_info,
    }
    url = f"http://{host}:{port}/verify_solution"
    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                text = await response.text()
                logger.warning(
                    "Coding verifier RPC failed with %s: %s", response.status, text[:256]
                )
                return _rpc_failure_summary("http_status", status=response.status, body=text)
            return await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        logger.warning("Coding verifier RPC request error: %s", exc)
        return _rpc_failure_summary("client_error", body=str(exc))


class CodingSandboxEnvironment:
    """Environment server that proxies requests to the shared sandbox executor."""

    def __init__(
        self,
        *,
        sandbox_url: str | None = None,
        compile_timeout_s: float = 5.0,
        run_timeout_s: float = 5.0,
        request_timeout_s: float = 15.0,
        memory_limit_bytes: int = 512 * 1024 * 1024,
        language: str = "python",
        max_workers: int = 4,
    ) -> None:
        self.sandbox_url = sandbox_url or _DEFAULT_SANDBOX_URL
        self.compile_timeout_s = compile_timeout_s
        self.run_timeout_s = run_timeout_s
        self.request_timeout_s = request_timeout_s
        self.memory_limit_mb = _convert_bytes_to_mb(memory_limit_bytes)
        self.language = language
        self.max_workers = max_workers

    def launch(self, port: int) -> None:
        app = FastAPI()
        executor = ThreadPoolExecutor(max_workers=self.max_workers)

        @app.post("/verify_solution")
        async def verify_endpoint(request: CodingVerificationRequest):
            loop = asyncio.get_running_loop()

            def _evaluate() -> dict[str, Any]:
                summary = evaluate_coding_prediction(
                    prediction=request.prediction,
                    reward_context=request.reward_context,
                    extra_info=request.extra_info,
                    sandbox_url=self.sandbox_url,
                    compile_timeout_s=self.compile_timeout_s,
                    run_timeout_s=self.run_timeout_s,
                    request_timeout_s=self.request_timeout_s,
                    memory_limit_mb=self.memory_limit_mb,
                    language=self.language,
                )
                return summary.to_payload()

            result = await loop.run_in_executor(executor, _evaluate)
            return JSONResponse(content=result)

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=60)