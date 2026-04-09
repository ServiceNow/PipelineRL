from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from sandbox_fusion import (
    RunCodeRequest,
    set_sandbox_endpoint,
    run_code_async,
)

logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\n([\s\S]*?)```", re.IGNORECASE)
_IMPORT_PREAMBLE = """from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
sys.setrecursionlimit(50000)
"""

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

    return code_blocks[-1] if code_blocks else ""


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


def _get_stripped_lines(text: str | None) -> list[str]:
    normalized = _normalize_output(text)
    if not normalized:
        return []
    return [line.strip() for line in normalized.split("\n")]


def _convert_line_to_decimals(line: str) -> list[Decimal] | None:
    try:
        return [Decimal(token) for token in line.split()]
    except Exception:
        return None


def _outputs_match(actual: str | None, expected: str | None) -> bool:
    actual_lines = _get_stripped_lines(actual)
    expected_lines = _get_stripped_lines(expected)
    if actual_lines == expected_lines:
        return True
    if len(actual_lines) != len(expected_lines):
        return False

    for actual_line, expected_line in zip(actual_lines, expected_lines):
        if actual_line == expected_line:
            continue
        actual_decimals = _convert_line_to_decimals(actual_line)
        expected_decimals = _convert_line_to_decimals(expected_line)
        if actual_decimals is None or expected_decimals is None:
            return False
        if actual_decimals != expected_decimals:
            return False
    return True


def _parse_function_args(test_input: Any) -> list[Any]:
    if isinstance(test_input, list):
        return test_input
    if not isinstance(test_input, str):
        return [test_input]
    try:
        return [json.loads(line) for line in test_input.split("\n")]
    except Exception:
        return [test_input]


def _parse_function_expected(expected_output: Any) -> Any:
    if isinstance(expected_output, list) and len(expected_output) == 1:
        expected_output = expected_output[0]
    if not isinstance(expected_output, str):
        return expected_output
    try:
        return json.loads(expected_output)
    except Exception:
        return expected_output


def _stderr_indicates_failure(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(
        marker in lowered
        for marker in (
            "traceback",
            "syntaxerror",
            "indentationerror",
            "exception",
        )
    )


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
{_indent_code(_IMPORT_PREAMBLE, 4)}
{_indent_code(user_code, 4)}
except (_GracefulExit, SystemExit, KeyboardInterrupt):
    pass
finally:
    sys.stdin = _original_stdin
'''


def _build_fn_test_script(user_code: str, fn_name: str, args: list, expected: Any) -> str:
    args_repr = repr(args)
    expected_repr = repr(expected)
    fn_name_repr = repr(fn_name)
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

def _normalize_result(value):
    if isinstance(value, tuple):
        return list(value)
    return value

try:
{_indent_code(_IMPORT_PREAMBLE, 4)}
{_indent_code(user_code, 4)}

    _test_args = {args_repr}
    _expected = {expected_repr}
    if "Solution" in globals():
        _callable = getattr(Solution(), {fn_name_repr})
    else:
        _callable = globals()[{fn_name_repr}]
    _result = _normalize_result(_callable(*_test_args))
    print("__LCB_PASS__" if _result == _expected else "__LCB_FAIL__")
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
    max_tests: int = 0,
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

    total_available_tests = min(len(inputs), len(outputs))
    num_tests = min(total_available_tests, max_tests) if max_tests > 0 else total_available_tests

    async def _run_test(idx: int) -> tuple[int, str, str, str, _SandboxResult]:
        test_input = inputs[idx]
        expected_output = outputs[idx]
        if call_type == "std":
            script = _build_stdin_test_script(candidate_code, str(test_input))
            expected_str = _normalize_output(str(expected_output))
        elif call_type == "fn" and fn_name:
            args = _parse_function_args(test_input)
            expected_val = _parse_function_expected(expected_output)
            script = _build_fn_test_script(candidate_code, fn_name, args, expected_val)
            expected_str = "__LCB_PASS__"
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
        elif result.status.lower() in {"failed", "error", "sandboxerror"} or _stderr_indicates_failure(result.stderr):
            if "SyntaxError" in result.stderr or "IndentationError" in result.stderr:
                status = "compile_error"
                summary.compile_error = True
                summary.error = summary.error or "compile_error"
            else:
                status = "runtime_error"
                summary.runtime_error = True
                summary.error = summary.error or "runtime_error"
        else:
            actual_output = result.stdout
            if _outputs_match(actual_output, expected_str):
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
    max_tests: int = 0,
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
