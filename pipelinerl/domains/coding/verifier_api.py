"""Python code verification using mcp-run-python sandbox."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\n([\s\S]*?)```", re.IGNORECASE)

# lazy init Sandbox instance
_SANDBOX_INSTANCE = None
_SANDBOX_LOCK = asyncio.Lock()


class CodingVerificationRequest(BaseModel):
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
    except Exception:
        return {}


def _normalize_output(text: str | None) -> str:
    if text is None:
        return ""
    return text.strip()


def _build_stdin_test_script(user_code: str, stdin_input: str) -> str:
    """script that simulates stdin input for stdout-style problems."""
    # Escape the input for embedding in the script
    escaped_input = json.dumps(stdin_input)
    return f'''
import sys
from io import StringIO, BytesIO

# Ensure __name__ == "__main__" so guarded code blocks execute
__name__ = "__main__"

# Create a stdin wrapper that supports both .read() and .buffer.read()
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

# Simulate stdin
_original_stdin = sys.stdin
sys.stdin = _StdinWrapper({escaped_input})

try:
{_indent_code(user_code, 4)}
finally:
    sys.stdin = _original_stdin
'''


def _build_fn_test_script(user_code: str, fn_name: str, args: list) -> str:
    """Build script that calls a function with given arguments."""
    args_repr = json.dumps(args)
    return f'''
{user_code}

# Test the function
_test_args = {args_repr}
_result = {fn_name}(*_test_args)
print(_result)
'''


def _indent_code(code: str, spaces: int) -> str:
    """Indent code block by specified number of spaces."""
    indent = " " * spaces
    lines = code.split("\n")
    return "\n".join(indent + line if line.strip() else line for line in lines)


async def _get_sandbox():
    """Get or create the sandbox instance."""
    global _SANDBOX_INSTANCE

    async with _SANDBOX_LOCK:
        if _SANDBOX_INSTANCE is not None:
            return _SANDBOX_INSTANCE

        try:
            from mcp_run_python import code_sandbox

            # Create sandbox context manager and enter it
            sandbox_cm = code_sandbox()
            sandbox = await sandbox_cm.__aenter__()
            _SANDBOX_INSTANCE = (sandbox, sandbox_cm)

            # Warmup execution
            logger.info("Warming up mcp-run-python sandbox...")
            await sandbox.eval("print('sandbox ready')")
            logger.info("Sandbox ready")

            return _SANDBOX_INSTANCE
        except ImportError:
            logger.error("mcp-run-python not installed. Install with: pip install mcp-run-python")
            raise
        except Exception as e:
            logger.error("Failed to initialize sandbox: %s", e)
            raise


async def _reset_sandbox():
    """Reset the sandbox instance after a failure."""
    global _SANDBOX_INSTANCE

    async with _SANDBOX_LOCK:
        if _SANDBOX_INSTANCE is not None:
            sandbox, sandbox_cm = _SANDBOX_INSTANCE
            try:
                await sandbox_cm.__aexit__(None, None, None)
            except Exception as e:
                logger.warning("Error cleaning up sandbox during reset: %s", e)
            _SANDBOX_INSTANCE = None


async def _run_in_sandbox(code: str, timeout: float = 10.0, retry_on_error: bool = True) -> dict[str, Any]:
    """Execute code in the mcp-run-python sandbox."""
    try:
        sandbox_tuple = await asyncio.wait_for(_get_sandbox(), timeout=30.0)
        sandbox = sandbox_tuple[0]

        start = time.perf_counter()
        result = await asyncio.wait_for(sandbox.eval(code), timeout=timeout)
        elapsed = time.perf_counter() - start

        # mcp-run-python returns:
        # - {"status": "success", "output": [...], "return_value": ...} on success
        # - {"status": "run-error", "output": [...], "error": "..."} on error
        status = result.get("status", "unknown")
        # Normalize status: mcp-run-python uses "run-error", we use "error"
        if status == "run-error":
            status = "error"

        # Capture error message from mcp-run-python's "error" field
        stderr = result.get("error", "") or ""

        return {
            "status": status,
            "stdout": "\n".join(result.get("output", [])),
            "stderr": stderr,
            "return_value": result.get("return_value"),
            "elapsed": elapsed,
            "timeout": False,
        }
    except asyncio.TimeoutError:
        return {
            "status": "timeout",
            "stdout": "",
            "stderr": "Execution timed out",
            "return_value": None,
            "elapsed": timeout,
            "timeout": True,
        }
    except Exception as e:
        error_msg = str(e) or f"Sandbox error: {type(e).__name__}"
        logger.warning("Sandbox execution error: %s (%s)", error_msg, type(e).__name__)

        # Try to reset and retry once if this looks like a sandbox crash
        if retry_on_error:
            logger.info("Attempting sandbox reset and retry...")
            try:
                await _reset_sandbox()
                return await _run_in_sandbox(code, timeout=timeout, retry_on_error=False)
            except Exception as retry_e:
                error_msg = f"Retry failed: {retry_e}" if str(retry_e) else f"Retry failed: {type(retry_e).__name__}"

        return {
            "status": "error",
            "stdout": "",
            "stderr": error_msg,
            "return_value": None,
            "elapsed": 0,
            "timeout": False,
        }


async def evaluate_coding_prediction_async(
    prediction: str | None,
    reward_context: dict[str, Any] | str | None,
    *,
    extra_info: dict[str, Any] | str | None = None,
    timeout_per_test: float = 5.0,
    max_tests: int = 15,
) -> CodingVerificationSummary:
    """Run generated code in sandbox and collect pass/fail statistics."""

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

    # Limit number of tests
    num_tests = min(len(inputs), len(outputs), max_tests)

    for idx in range(num_tests):
        test_input = inputs[idx]
        expected_output = outputs[idx]
        summary.total += 1

        # Build test script based on call type
        if call_type == "std":
            # stdin/stdout style
            script = _build_stdin_test_script(candidate_code, str(test_input))
            expected_str = _normalize_output(str(expected_output))
        elif call_type == "fn" and fn_name:
            # Function call style
            args = test_input if isinstance(test_input, list) else [test_input]
            script = _build_fn_test_script(candidate_code, fn_name, args)
            expected_str = _normalize_output(str(expected_output))
        else:
            # Fallback to stdin style
            script = _build_stdin_test_script(candidate_code, str(test_input))
            expected_str = _normalize_output(str(expected_output))

        # Run in sandbox
        result = await _run_in_sandbox(script, timeout=timeout_per_test)

        # Determine test status
        if result["timeout"]:
            status = "timeout"
            summary.timeout_error = True
            summary.error = summary.error or "timeout"
        elif result["status"] == "error" or "Error" in result.get("stderr", ""):
            # Check for compile vs runtime error
            stderr = result.get("stderr", "")
            if "SyntaxError" in stderr or "IndentationError" in stderr:
                status = "compile_error"
                summary.compile_error = True
                summary.error = summary.error or "compile_error"
            else:
                status = "runtime_error"
                summary.runtime_error = True
                summary.error = summary.error or "runtime_error"
        else:
            # Compare output
            actual_output = _normalize_output(result.get("stdout", ""))
            if actual_output == expected_str:
                status = "passed"
                summary.passed += 1
            else:
                status = "failed"

        test_result = CodingTestResult(
            index=idx,
            kind=call_type,
            status=status,
            input=str(test_input)[:500],  # Truncate for storage
            expected=expected_str[:500],
            stdout=result.get("stdout", "")[:500],
            stderr=result.get("stderr", "")[:500],
            elapsed=result.get("elapsed"),
        )
        summary.tests.append(test_result)

        # Stop on compile or runtime error (no point continuing)
        if status in ("compile_error", "runtime_error"):
            break

    return summary


def evaluate_coding_prediction(
    prediction: str | None,
    reward_context: dict[str, Any] | str | None,
    *,
    extra_info: dict[str, Any] | str | None = None,
    timeout_per_test: float = 5.0,
    max_tests: int = 15,
) -> CodingVerificationSummary:
    """Synchronous wrapper for evaluate_coding_prediction_async."""
    return asyncio.run(
        evaluate_coding_prediction_async(
            prediction,
            reward_context,
            extra_info=extra_info,
            timeout_per_test=timeout_per_test,
            max_tests=max_tests,
        )
    )


def _rpc_failure_summary(reason: str, *, status: int | None = None, body: str | None = None) -> dict[str, Any]:
    """Create a failure summary for RPC errors."""
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
    """Environment server that uses mcp-run-python for code execution."""

    def __init__(
        self,
        *,
        timeout_per_test: float = 5.0,
        max_tests: int = 15,
        max_workers: int = 4,
    ) -> None:
        self.timeout_per_test = timeout_per_test
        self.max_tests = max_tests
        self.max_workers = max_workers

    def launch(self, port: int) -> None:
        app = FastAPI()

        @app.post("/verify_solution")
        async def verify_endpoint(request: CodingVerificationRequest):
            summary = await evaluate_coding_prediction_async(
                prediction=request.prediction,
                reward_context=request.reward_context,
                extra_info=request.extra_info,
                timeout_per_test=self.timeout_per_test,
                max_tests=self.max_tests,
            )
            return JSONResponse(content=summary.to_payload())

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=60)


async def cleanup_sandbox():
    """Clean up the sandbox instance."""
    global _SANDBOX_INSTANCE

    async with _SANDBOX_LOCK:
        if _SANDBOX_INSTANCE is not None:
            sandbox, sandbox_cm = _SANDBOX_INSTANCE
            try:
                await sandbox_cm.__aexit__(None, None, None)
            except Exception as e:
                logger.warning("Error cleaning up sandbox: %s", e)
            _SANDBOX_INSTANCE = None
