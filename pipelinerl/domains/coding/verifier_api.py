from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from sandbox_fusion import (
    RunCodeRequest,
    set_sandbox_endpoint,
    run_code_async,
)

logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\n([\s\S]*?)```", re.IGNORECASE)

_ENDPOINT_CONFIGURED = False


@dataclass
class CodingTestResult:
    index: int
    kind: str
    status: str
    input: str | None = None
    expected: str | None = None
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

    # Truncate at end
    for marker in [
        "[END FINAL RESPONSE]",
        "<|end|>",
        "</s>",
        "<|im_end|>",
        "<|endoftext|>",
        "\n---\n",
    ]:
        if marker in prediction:
            prediction = prediction.split(marker)[0]

    matches = _CODE_FENCE_RE.findall(prediction)
    if not matches:
        # prevent reasoning traces from being treated as code
        return ""

    code_blocks = [m.strip() for m in matches if ">>>" not in m]

    if not code_blocks:
        code_blocks = [m.strip() for m in matches]

    # return the longest block (code should be longer than examples)
    return max(code_blocks, key=len)


def _ensure_dict(data: dict[str, Any] | str | None) -> dict[str, Any]:
    if data is None:
        return {}
    if isinstance(data, dict):
        return data
    try:
        return json.loads(data)
    except Exception:
        return {}


def _normalize_output(text: str | None) -> str:
    if text is None:
        return ""
    return text.strip()


def _build_stdin_test_script(user_code: str, stdin_input: str) -> str:
    escaped_input = json.dumps(stdin_input)
    return f'''
import sys
import os
from io import StringIO, BytesIO

__name__ = "__main__"

class _GracefulExit(Exception):
    pass

def _safe_exit(code=0):
    raise _GracefulExit(code)

sys.exit = _safe_exit
os._exit = _safe_exit
import builtins
builtins.quit = _safe_exit
builtins.exit = _safe_exit

class _StdinWrapper:
    def __init__(self, text):
        self._text_io = StringIO(text)
        self._byte_io = BytesIO(text.encode('utf-8'))
        self.buffer = self._byte_io

    def read(self, *args):
        return self._text_io.read(*args)

    def readline(self, *args):
        return self._text_io.readline(*args)

    def readlines(self, *args):
        return self._text_io.readlines(*args)

    def __iter__(self):
        return iter(self._text_io)

_original_stdin = sys.stdin
sys.stdin = _StdinWrapper({escaped_input})

try:
{_indent_code(user_code, 4)}
except (_GracefulExit, SystemExit, KeyboardInterrupt):
    pass
finally:
    sys.stdin = _original_stdin
'''


def _build_fn_test_script(user_code: str, fn_name: str, args: list) -> str:
    # Use repr() instead of json.dumps() to get valid Python literals (don't break verification)
    args_repr = repr(args)
    return f'''
import sys
import os

class _GracefulExit(Exception):
    pass

def _safe_exit(code=0):
    raise _GracefulExit(code)

sys.exit = _safe_exit
os._exit = _safe_exit
import builtins
builtins.quit = _safe_exit
builtins.exit = _safe_exit

try:
{_indent_code(user_code, 4)}

    _test_args = {args_repr}
    _result = {fn_name}(*_test_args)
    print(_result)
except (_GracefulExit, SystemExit, KeyboardInterrupt):
    pass
'''


def _indent_code(code: str, spaces: int) -> str:
    indent = " " * spaces
    lines = code.split("\n")
    return "\n".join(indent + line if line.strip() else line for line in lines)


@dataclass
class _SandboxResult:
    status: str
    stdout: str
    stderr: str
    elapsed: float
    timeout: bool


async def _run_code_sandboxfusion(
    code: str,
    *,
    timeout: float,
    sandbox_endpoint: str,
) -> _SandboxResult:
    """Run code using SandboxFusion."""
    global _ENDPOINT_CONFIGURED

    if not _ENDPOINT_CONFIGURED:
        set_sandbox_endpoint(sandbox_endpoint)
        _ENDPOINT_CONFIGURED = True
        logger.info("Configured SandboxFusion endpoint: %s", sandbox_endpoint)

    start = time.perf_counter()
    try:
        request = RunCodeRequest(
            code=code,
            language="python",
            run_timeout=timeout,
        )
        response = await run_code_async(request)
        elapsed = time.perf_counter() - start

        # Extract results from response
        status = response.status.value if hasattr(response.status, 'value') else str(response.status)
        stdout = ""
        stderr = ""

        if response.run_result:
            stdout = response.run_result.stdout or ""
            stderr = response.run_result.stderr or ""

        is_timeout = "timeout" in status.lower() or "timeout" in (response.message or "").lower()

        logger.debug(
            "SandboxFusion executed in %.3fs, status=%s, stdout=%d bytes, stderr=%d bytes",
            elapsed, status, len(stdout), len(stderr),
        )

        return _SandboxResult(
            status=status,
            stdout=stdout,
            stderr=stderr,
            elapsed=elapsed,
            timeout=is_timeout,
        )
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - start
        logger.debug("SandboxFusion request timed out after %.1fs", elapsed)
        return _SandboxResult(
            status="timeout",
            stdout="",
            stderr="Request timed out",
            elapsed=elapsed,
            timeout=True,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.warning("SandboxFusion error: %s", exc)
        return _SandboxResult(
            status="error",
            stdout="",
            stderr=str(exc),
            elapsed=elapsed,
            timeout=False,
        )


async def evaluate_coding_prediction_async(
    prediction: str | None,
    reward_context: dict[str, Any] | str | None,
    *,
    extra_info: dict[str, Any] | str | None = None,
    timeout_per_test: float = 5.0,
    max_tests: int = 15,
    sandbox_endpoint: str = "http://127.0.0.1:8080",
) -> CodingVerificationSummary:
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

    call_type = (context.get("call_type") or "std").lower().strip()
    inputs = context.get("inputs", [])
    outputs = context.get("outputs", [])
    fn_name = context.get("fn_name")

    if not inputs or not outputs:
        summary.error = "no_tests"
        return summary

    num_tests = min(len(inputs), len(outputs), max_tests)

    async def _run_test(idx: int) -> tuple[int, str, str, str, _SandboxResult]:
        test_input = inputs[idx]
        expected_output = outputs[idx]
        if call_type == "std":
            script = _build_stdin_test_script(candidate_code, str(test_input))
            expected_str = _normalize_output(str(expected_output))
        elif call_type == "fn" and fn_name:
            args = test_input if isinstance(test_input, list) else [test_input]
            script = _build_fn_test_script(candidate_code, fn_name, args)
            # For fn tests, expected_output is often stored as [return_value] to match input format.
            # Unwrap single-element lists to get the actual expected return value.
            if isinstance(expected_output, list) and len(expected_output) == 1:
                expected_val = expected_output[0]
            else:
                expected_val = expected_output
            expected_str = _normalize_output(str(expected_val))
        else:
            script = _build_stdin_test_script(candidate_code, str(test_input))
            expected_str = _normalize_output(str(expected_output))

        result = await _run_code_sandboxfusion(
            script,
            timeout=timeout_per_test,
            sandbox_endpoint=sandbox_endpoint,
        )
        return idx, str(test_input), expected_str, call_type, result

    tasks = [
        asyncio.create_task(_run_test(idx))
        for idx in range(num_tests)
    ]
    logger.debug("Running %d tests concurrently via SandboxFusion", num_tests)
    results = await asyncio.gather(*tasks)

    for idx, test_input, expected_str, kind, result in results:
        summary.total += 1

        if result.timeout:
            status = "timeout"
            summary.timeout_error = True
            summary.error = summary.error or "timeout"
        elif result.status.lower() in {"failed", "error", "sandboxerror"} or result.stderr:
            if "SyntaxError" in result.stderr or "IndentationError" in result.stderr:
                status = "compile_error"
                summary.compile_error = True
                summary.error = summary.error or "compile_error"
            else:
                status = "runtime_error"
                summary.runtime_error = True
                summary.error = summary.error or "runtime_error"
        else:
            actual_output = _normalize_output(result.stdout)
            if actual_output == expected_str:
                status = "passed"
                summary.passed += 1
            else:
                status = "failed"

        summary.tests.append(
            CodingTestResult(
                index=idx,
                kind=kind,
                status=status,
                input=str(test_input)[:500],
                expected=expected_str[:500],
                stdout=result.stdout[:500],
                stderr=result.stderr[:500],
                elapsed=result.elapsed,
            )
        )

    logger.debug(
        "Verification complete: %d/%d passed (compile_error=%s, runtime_error=%s, timeout=%s)",
        summary.passed, summary.total, summary.compile_error, summary.runtime_error, summary.timeout_error,
    )
    return summary


def evaluate_coding_prediction(
    prediction: str | None,
    reward_context: dict[str, Any] | str | None,
    *,
    extra_info: dict[str, Any] | str | None = None,
    timeout_per_test: float = 5.0,
    max_tests: int = 15,
    sandbox_endpoint: str = "http://127.0.0.1:8080",
) -> CodingVerificationSummary:
    return asyncio.run(
        evaluate_coding_prediction_async(
            prediction,
            reward_context,
            extra_info=extra_info,
            timeout_per_test=timeout_per_test,
            max_tests=max_tests,
            sandbox_endpoint=sandbox_endpoint,
        )
    )
