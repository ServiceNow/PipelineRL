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

# Sandbox pool (lazily initialized)
_SANDBOX_POOL: "SandboxPool | None" = None
_SANDBOX_POOL_LOCK = asyncio.Lock()


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
import os
from io import StringIO, BytesIO

# Ensure __name__ == "__main__" so guarded code blocks execute
__name__ = "__main__"

# Override exit functions to prevent them from killing the sandbox
class _GracefulExit(Exception):
    pass

def _safe_exit(code=0):
    raise _GracefulExit(code)

sys.exit = _safe_exit
os._exit = _safe_exit
# Override builtins quit/exit which also raise SystemExit
import builtins
builtins.quit = _safe_exit
builtins.exit = _safe_exit

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
except (_GracefulExit, SystemExit, KeyboardInterrupt):
    pass  # Gracefully handle exit calls and interrupts
finally:
    sys.stdin = _original_stdin
'''


def _build_fn_test_script(user_code: str, fn_name: str, args: list) -> str:
    """script that calls a function with given arguments."""
    args_repr = json.dumps(args)
    return f'''
import sys
import os

# Override exit functions to prevent them from killing the sandbox
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

    # Test the function
    _test_args = {args_repr}
    _result = {fn_name}(*_test_args)
    print(_result)
except (_GracefulExit, SystemExit, KeyboardInterrupt):
    pass  # Gracefully handle exit calls and interrupts
'''


def _indent_code(code: str, spaces: int) -> str:
    indent = " " * spaces
    lines = code.split("\n")
    return "\n".join(indent + line if line.strip() else line for line in lines)


@dataclass
class _SandboxInstance:
    sandbox: Any
    sandbox_cm: Any
    healthy: bool = True
    use_count: int = 0
    last_error: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SandboxPool:
    def __init__(self, size: int = 4):
        self.size = size
        self.instances: list[_SandboxInstance | None] = [None] * size
        self.init_locks: list[asyncio.Lock] = [asyncio.Lock() for _ in range(size)]
        self._next_idx = 0
        self._idx_lock = asyncio.Lock()
        self._respawn_tasks: set[asyncio.Task] = set()

    async def _create_sandbox(self) -> tuple[Any, Any]:
        """Create a new sandbox instance."""
        from mcp_run_python import code_sandbox
        sandbox_cm = code_sandbox()
        sandbox = await sandbox_cm.__aenter__()
        await sandbox.eval("print('sandbox ready')")
        return sandbox, sandbox_cm

    async def _init_instance(self, idx: int) -> _SandboxInstance:
        """Initialize or reinitialize a sandbox instance."""
        async with self.init_locks[idx]:
            # Check if already initialized
            if self.instances[idx] is not None and self.instances[idx].healthy:
                return self.instances[idx]

            # Cleanup old
            if self.instances[idx] is not None:
                try:
                    await self.instances[idx].sandbox_cm.__aexit__(None, None, None)
                except Exception as e:
                    logger.debug("Cleanup error for sandbox %d: %s", idx, e)

            try:
                sandbox, sandbox_cm = await self._create_sandbox()
                self.instances[idx] = _SandboxInstance(
                    sandbox=sandbox,
                    sandbox_cm=sandbox_cm,
                    healthy=True,
                    use_count=0,
                )
                logger.info("Sandbox %d initialized", idx)
                return self.instances[idx]
            except Exception as e:
                logger.error("Failed to initialize sandbox %d: %s", idx, e)
                raise

    async def _get_next_idx(self) -> int:
        """Get next sandbox index (round-robin)."""
        async with self._idx_lock:
            idx = self._next_idx
            self._next_idx = (self._next_idx + 1) % self.size
            return idx

    async def _respawn_in_background(self, idx: int):
        """Respawn a failed sandbox in the background."""
        try:
            await self._init_instance(idx)
        except Exception as e:
            logger.warning("Background respawn of sandbox %d failed: %s", idx, e)

    def _schedule_respawn(self, idx: int):
        """Schedule a background respawn task."""
        task = asyncio.create_task(self._respawn_in_background(idx))
        self._respawn_tasks.add(task)
        task.add_done_callback(self._respawn_tasks.discard)

    async def execute(self, code: str, timeout: float = 10.0) -> dict[str, Any]:
        """Execute code in an available sandbox."""
        # sandbox in round-robin order with fallback
        start_idx = await self._get_next_idx()

        for attempt in range(self.size):
            idx = (start_idx + attempt) % self.size

            if self.instances[idx] is None or not self.instances[idx].healthy:
                try:
                    await asyncio.wait_for(self._init_instance(idx), timeout=30.0)
                except Exception:
                    continue  # next sandbox

            instance = self.instances[idx]
            if instance is None or not instance.healthy:
                continue

            # Try to acquire this sandbox
            try:
                async with asyncio.timeout(0.1):
                    await instance.lock.acquire()
            except asyncio.TimeoutError:
                continue  # Sandbox busy, try next

            try:
                start = time.perf_counter()
                result = await asyncio.wait_for(
                    instance.sandbox.eval(code),
                    timeout=timeout
                )
                elapsed = time.perf_counter() - start
                instance.use_count += 1

                status = result.get("status", "unknown")
                if status == "run-error":
                    status = "error"

                return {
                    "status": status,
                    "stdout": "\n".join(result.get("output", [])),
                    "stderr": result.get("error", "") or "",
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
                logger.warning("Sandbox %d error: %s", idx, error_msg)

                # mark as unhealthy and respawn
                instance.healthy = False
                instance.last_error = error_msg
                self._schedule_respawn(idx)

                # don't fail immediately
                continue

            finally:
                instance.lock.release()

        # All sandboxes failed
        return {
            "status": "error",
            "stdout": "",
            "stderr": "All sandboxes unavailable",
            "return_value": None,
            "elapsed": 0,
            "timeout": False,
        }

    async def shutdown(self):
        # Cancel pending respawn tasks
        for task in self._respawn_tasks:
            task.cancel()

        for idx, instance in enumerate(self.instances):
            if instance is not None:
                try:
                    await instance.sandbox_cm.__aexit__(None, None, None)
                except Exception as e:
                    logger.debug("Shutdown cleanup error for sandbox %d: %s", idx, e)
                self.instances[idx] = None


async def _get_sandbox_pool(pool_size: int = 4) -> SandboxPool:
    global _SANDBOX_POOL

    async with _SANDBOX_POOL_LOCK:
        if _SANDBOX_POOL is not None:
            return _SANDBOX_POOL

        _SANDBOX_POOL = SandboxPool(size=pool_size)
        logger.info("Created sandbox pool with %d instances", pool_size)
        return _SANDBOX_POOL


async def _run_in_sandbox(code: str, timeout: float = 10.0, pool_size: int = 4) -> dict[str, Any]:
    try:
        pool = await asyncio.wait_for(_get_sandbox_pool(pool_size), timeout=5.0)
        return await pool.execute(code, timeout=timeout)
    except Exception as e:
        error_msg = str(e) or f"Pool error: {type(e).__name__}"
        logger.error("Sandbox pool error: %s", error_msg)
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
    pool_size: int = 4,
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

    for idx in range(num_tests):
        test_input = inputs[idx]
        expected_output = outputs[idx]
        summary.total += 1

        if call_type == "std":
            # stdin/stdout style
            script = _build_stdin_test_script(candidate_code, str(test_input))
            expected_str = _normalize_output(str(expected_output))
        elif call_type == "fn" and fn_name:
            args = test_input if isinstance(test_input, list) else [test_input]
            script = _build_fn_test_script(candidate_code, fn_name, args)
            expected_str = _normalize_output(str(expected_output))
        else:
            script = _build_stdin_test_script(candidate_code, str(test_input))
            expected_str = _normalize_output(str(expected_output))

        result = await _run_in_sandbox(script, timeout=timeout_per_test, pool_size=pool_size)

        if result["timeout"]:
            status = "timeout"
            summary.timeout_error = True
            summary.error = summary.error or "timeout"
        elif result["status"] == "error" or "Error" in result.get("stderr", ""):
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
    pool_size: int = 4,
) -> CodingVerificationSummary:
    return asyncio.run(
        evaluate_coding_prediction_async(
            prediction,
            reward_context,
            extra_info=extra_info,
            timeout_per_test=timeout_per_test,
            max_tests=max_tests,
            pool_size=pool_size,
        )
    )


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
    global _SANDBOX_POOL

    async with _SANDBOX_POOL_LOCK:
        if _SANDBOX_POOL is not None:
            try:
                await _SANDBOX_POOL.shutdown()
            except Exception as e:
                logger.warning("Error cleaning up sandbox pool: %s", e)
            _SANDBOX_POOL = None
