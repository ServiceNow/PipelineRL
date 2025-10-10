import asyncio
import atexit
import inspect
import json
import logging
import os
import re
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, AsyncIterator, List

import multiprocessing

from fastapi import HTTPException
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from tapeagents.core import Action, Observation
from tapeagents.environment import Environment
from tapeagents.mcp import MCPClient, MCPEnvironment, NoTool
from tapeagents.remote_environment import EnvironmentServer
from tapeagents.tool_calling import FunctionSpec, ToolCallAction, ToolResult, ToolSpec
from mcp.types import CallToolResult, TextContent

from pipelinerl.domains.math.verifier_api import verify_answer
from pipelinerl.domains.mcp.steps import MathAnswer

logger = logging.getLogger(__name__)


_CONNECTION_ERROR_PATTERNS = (
    "closedresourceerror",
    "brokenresourceerror",
    "broken pipe",
    "connectionreseterror",
    "timed out while waiting for response",
)


_MCP_WORKER_STATE: dict[str, Any] | None = None


def _shutdown_mcp_worker() -> None:
    global _MCP_WORKER_STATE
    if not _MCP_WORKER_STATE:
        return
    loop: asyncio.AbstractEventLoop = _MCP_WORKER_STATE["loop"]
    client: MCPClient = _MCP_WORKER_STATE["client"]
    try:
        loop.run_until_complete(client.close())
    except Exception:
        logger.warning("Failed to close MCP client in worker", exc_info=True)
    finally:
        loop.close()
        _MCP_WORKER_STATE = None


def _initialize_mcp_worker(
    config_path: str,
    tools_whitelist: list[str] | tuple[str, ...] | None,
    use_cache: bool,
    read_timeout_seconds: int,
) -> None:
    """Initializer for the ProcessPool workers that own MCP runtimes."""
    global _MCP_WORKER_STATE
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    client = MCPClient(
        config_path=config_path,
        use_cache=use_cache,
        read_timeout_seconds=read_timeout_seconds,
    )
    loop.run_until_complete(client.start_servers())
    _MCP_WORKER_STATE = {
        "loop": loop,
        "client": client,
        "tools_whitelist": list(tools_whitelist or []),
    }
    atexit.register(_shutdown_mcp_worker)


def _call_tool_in_worker(tool_name: str, tool_arguments: Any) -> dict[str, Any]:
    """Execute an MCP tool call inside a worker process."""
    if not _MCP_WORKER_STATE:
        raise RuntimeError("MCP worker not initialized")
    loop: asyncio.AbstractEventLoop = _MCP_WORKER_STATE["loop"]
    client: MCPClient = _MCP_WORKER_STATE["client"]
    whitelist: list[str] = _MCP_WORKER_STATE.get("tools_whitelist", [])
    if whitelist and tool_name not in whitelist:
        raise NoTool(f"Tool {tool_name} not allowed by whitelist")
    result = loop.run_until_complete(client.call_tool(tool_name, tool_arguments))
    return result.model_dump(exclude_none=True)


class _RemoteCallError(RuntimeError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


def _invoke_environment_method(
    environment: Environment,
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    loop: asyncio.AbstractEventLoop,
) -> Any:
    attr = getattr(environment, method_name)
    if inspect.iscoroutinefunction(attr):
        return loop.run_until_complete(attr(*args, **kwargs))
    result = attr(*args, **kwargs)
    if inspect.isawaitable(result):
        return loop.run_until_complete(result)
    return result


def _environment_process_main(env_cfg_container: dict[str, Any], conn) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        env_cfg = OmegaConf.create(env_cfg_container)
        environment: Environment = instantiate(env_cfg)
    except Exception:
        conn.send(
            (
                "exception",
                {
                    "type": "EnvironmentBootstrapError",
                    "message": "Failed to instantiate environment",
                    "traceback": traceback.format_exc(),
                },
            )
        )
        conn.close()
        loop.close()
        return

    async_methods = {
        name
        for name in ("ainitialize", "areset", "aclose", "astep", "areact")
        if hasattr(environment, name) and inspect.iscoroutinefunction(getattr(environment, name))
    }
    sync_methods = {
        name
        for name in (
            "initialize",
            "reset",
            "close",
            "start_task",
            "actions",
            "tools_description",
            "mark_healthy",
            "is_healthy",
            "step",
            "react",
        )
        if callable(getattr(environment, name, None))
    }

    conn.send(("capabilities", {"sync": list(sync_methods), "async": list(async_methods)}))

    running = True
    while running:
        try:
            message = conn.recv()
        except EOFError:
            break
        if not isinstance(message, tuple) or len(message) != 3:
            continue
        command, args, kwargs = message
        if command == "__shutdown__":
            running = False
            conn.send(("ok", None))
            break
        try:
            result = _invoke_environment_method(environment, command, args, kwargs, loop)
            conn.send(("ok", result))
        except Exception as exc:
            conn.send(
                (
                    "exception",
                    {
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            )

    try:
        if "aclose" in async_methods:
            loop.run_until_complete(environment.aclose())
        elif "close" in sync_methods:
            environment.close()
    except Exception:
        logger.debug("Failed to close environment during shutdown", exc_info=True)
    finally:
        conn.close()
        loop.close()


class _ProcessEnvironmentProxy:
    def __init__(self, env_cfg: DictConfig):
        self._ctx = multiprocessing.get_context("spawn")
        self._parent_conn, child_conn = self._ctx.Pipe()
        cfg_container = OmegaConf.to_container(env_cfg, resolve=True)
        self._process = self._ctx.Process(
            target=_environment_process_main,
            args=(cfg_container, child_conn),
        )
        self._process.daemon = False
        self._process.start()
        self._lock = threading.Lock()
        self._closed = False
        try:
            status, payload = self._parent_conn.recv()
        except EOFError as error:
            raise _RemoteCallError("Environment process terminated prematurely") from error
        if status == "exception":
            raise _RemoteCallError(payload.get("message", "Environment bootstrap failed"), payload)
        if status != "capabilities":
            raise _RemoteCallError("Unexpected handshake from environment process")
        self._sync_methods = set(payload.get("sync", []))
        self._async_methods = set(payload.get("async", []))

    def supports_async(self, name: str) -> bool:
        return name in self._async_methods

    def supports_sync(self, name: str) -> bool:
        return name in self._sync_methods

    def _ensure_alive(self) -> None:
        if self._closed:
            raise _RemoteCallError("Environment proxy is closed")
        if not self._process.is_alive():
            raise _RemoteCallError("Environment process died unexpectedly")

    def _call_remote(self, method: str, *args: Any, **kwargs: Any) -> Any:
        self._ensure_alive()
        with self._lock:
            try:
                self._parent_conn.send((method, args, kwargs))
                status, payload = self._parent_conn.recv()
            except EOFError as error:
                raise _RemoteCallError("Lost connection to environment process") from error
        if status == "ok":
            return payload
        if status == "exception":
            raise _RemoteCallError(payload.get("message", "Remote call failed"), payload)
        raise _RemoteCallError(f"Unexpected response type: {status}")

    def start_task(self, task: dict) -> dict:
        return self._call_remote("start_task", task)

    def actions(self) -> tuple[type[Action], ...]:
        return tuple(self._call_remote("actions"))

    def tools_description(self) -> str:
        return self._call_remote("tools_description")

    def initialize(self):
        if self.supports_sync("initialize"):
            return self._call_remote("initialize")
        if self.supports_async("ainitialize"):
            return self._call_remote("ainitialize")
        return None

    async def ainitialize(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.initialize)

    def reset(self) -> None:
        if self.supports_sync("reset"):
            self._call_remote("reset")
        elif self.supports_async("areset"):
            self._call_remote("areset")

    async def areset(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.reset)

    def step(self, action: Action) -> Observation:
        if self.supports_sync("step"):
            return self._call_remote("step", action)
        if self.supports_async("astep"):
            return self._call_remote("astep", action)
        raise _RemoteCallError("Remote environment does not support step or astep")

    async def astep(self, action: Action) -> Observation:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.step, action)

    def react(self, tape) -> Any:
        if self.supports_sync("react"):
            return self._call_remote("react", tape)
        if self.supports_async("areact"):
            return self._call_remote("areact", tape)
        raise _RemoteCallError("Remote environment does not support react or areact")

    async def areact(self, tape) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.react, tape)

    def mark_healthy(self) -> None:
        if self.supports_sync("mark_healthy"):
            self._call_remote("mark_healthy")

    def is_healthy(self) -> bool:
        if self.supports_sync("is_healthy"):
            return bool(self._call_remote("is_healthy"))
        return True

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self.supports_sync("close"):
                self._call_remote("close")
            elif self.supports_async("aclose"):
                self._call_remote("aclose")
        except _RemoteCallError:
            logger.debug("Remote close failed", exc_info=True)
        finally:
            self._shutdown()

    async def aclose(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.close)

    def _shutdown(self) -> None:
        if self._closed:
            return
        try:
            with self._lock:
                if self._process.is_alive():
                    self._parent_conn.send(("__shutdown__", (), {}))
                    try:
                        self._parent_conn.recv()
                    except EOFError:
                        pass
        except Exception:
            logger.debug("Failed to send shutdown to environment process", exc_info=True)
        finally:
            self._parent_conn.close()
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
            self._closed = True

    def __del__(self) -> None:
        try:
            self._shutdown()
        except Exception:
            pass
class EnvironmentServerWithVerifier(EnvironmentServer):
    """Environment server that includes the verify_answer endpoint."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.process_pool = ProcessPoolExecutor(max_workers=4)
    
    def create_app(self):
        app = super().create_app()
        
        class VerifyAnswerRequest(BaseModel):
            prediction: str
            gold: str
            strict: bool = True
            max_prediction_length: int = 1000
        
        @app.post("/verify_answer")
        async def verify_answer_endpoint(request: VerifyAnswerRequest):
            try:
                # Run verification in the process pool to avoid blocking the main thread
                loop = asyncio.get_event_loop()
                answer_status = await loop.run_in_executor(
                    self.process_pool, 
                    partial(
                        verify_answer, 
                        request.prediction, 
                        request.gold, 
                        request.strict, 
                        request.max_prediction_length
                    )
                )
                return {"answer_status": answer_status}
            except Exception as e:
                logger.exception(f"Error in verify_answer: {e}")
                raise HTTPException(status_code=500, detail=f"Error verifying answer: {str(e)}")
        
        return app
    
    def shutdown(self):
        super().shutdown()
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=True)


class MCPEnvironmentServer:

    def __init__(self,
        n_envs: int,
        host: str,
        mcp_target: str,
        mcp_config_path: str,
        mcp_tools_whitelist: List[str],
        exp_path: str,
        env_call_timeout: int = 60,
        mcp_read_timeout_seconds: int = 10,
    ):
        # Remote environment server configuration
        self.n_envs = n_envs
        self.host = host
        self.env_call_timeout = env_call_timeout
        # Individual web environment configuration
        self.mcp_target = mcp_target
        self.mcp_config_path = mcp_config_path
        self.mcp_tools_whitelist = mcp_tools_whitelist
        self.exp_path = exp_path
        self.mcp_read_timeout_seconds = mcp_read_timeout_seconds


    def launch(self, port: int):
        """
        Serve the environment in TapeAgent with verify_answer endpoint.
        """
        env_server = EnvironmentServerWithVerifier(
            n_envs=self.n_envs, 
            host=self.host, 
            port=port, 
            env_call_timeout=self.env_call_timeout
        )
        env_server.launch(OmegaConf.create({
            "_target_": self.mcp_target,
            "config_path": self.mcp_config_path,
            "tools_whitelist": self.mcp_tools_whitelist,
            "read_timeout_seconds": self.mcp_read_timeout_seconds,
        }))


class EmbeddedMCPEnvironment(MCPEnvironment):
    def __init__(
        self,
        *args,
        math_answer_description: str = "Submit the final answer in LaTeX \\boxed{} format.",
        **kwargs,
    ) -> None:
        config_path = kwargs.get("config_path", "")
        use_cache = kwargs.get("use_cache", False)
        read_timeout_seconds = kwargs.get("read_timeout_seconds", 10)
        runtime_pool_workers = kwargs.pop("runtime_pool_workers", 0)
        offload_tools = tuple(kwargs.pop("offload_tools", ()))

        super().__init__(*args, **kwargs)
        self._broken = False
        self._last_failure_reason: str | None = None
        self._runtime_guard_installed: bool = False
        self._runtime_pool: ProcessPoolExecutor | None = None
        self._runtime_pool_lock = threading.Lock()
        self._runtime_pool_workers = runtime_pool_workers
        self._offload_tools = set(offload_tools)
        self._config_path = getattr(self.client, "config_path", config_path)
        self._use_cache = getattr(self.client, "use_cache", use_cache)
        self._read_timeout_seconds = getattr(self.client, "read_timeout_seconds", read_timeout_seconds)

        # try to catch time wasting patterns before execution
        self._python_blocklist = (
            (re.compile(r"\bsys\s*\.\s*exit\s*\(", re.IGNORECASE), "sys.exit"),
            (re.compile(r"\bos\s*\.\s*_exit\s*\(", re.IGNORECASE), "os._exit"),
            (re.compile(r"\bexit\s*\(", re.IGNORECASE), "exit"),
            (re.compile(r"\bquit\s*\(", re.IGNORECASE), "quit"),
            (re.compile(r"raise\s+systemexit", re.IGNORECASE), "raise SystemExit"),
            (re.compile(r"from\s+sys\s+import\s+exit", re.IGNORECASE), "from sys import exit"),
            (
                re.compile(r"__import__\s*\(\s*['\"]os['\"]\s*\)\s*\.\s*_exit", re.IGNORECASE),
                "__import__('os')._exit",
            ),
            (
                re.compile(r"__import__\s*\(\s*['\"]sys['\"]\s*\)\s*\.\s*exit", re.IGNORECASE),
                "__import__('sys').exit",
            ),
        )
        self._math_answer_spec = ToolSpec(
            function=FunctionSpec(
                name="MathAnswer",
                description=math_answer_description,
                parameters={
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "Final answer expressed in LaTeX \\boxed{} format.",
                        }
                    },
                    "required": ["answer"],
                },
            )
        )

    def initialize(self):
        super().initialize()
        self._reset_health()
        self._ensure_math_answer_tool()

    async def ainitialize(self) -> None:
        self.loop = asyncio.get_running_loop()
        await super().ainitialize()
        self._reset_health()
        self._ensure_math_answer_tool()
        await self._install_runtime_guard()

    def actions(self):
        base_actions = super().actions()
        if not any(
            getattr(action, "function", None) and action.function.name == "MathAnswer"
            for action in base_actions
        ):
            base_actions = base_actions + (self._math_answer_spec,)
        return base_actions

    def _should_offload(self, tool_name: str) -> bool:
        return bool(self._runtime_pool_workers) and tool_name in self._offload_tools

    def _ensure_runtime_pool(self) -> ProcessPoolExecutor:
        if self._runtime_pool is not None:
            return self._runtime_pool
        with self._runtime_pool_lock:
            if self._runtime_pool is not None:
                return self._runtime_pool
            cpu_count = os.cpu_count() or 1
            default_workers = max(1, cpu_count // 2)
            max_workers = self._runtime_pool_workers or default_workers
            whitelist = tuple(self.tools_whitelist) if getattr(self, "tools_whitelist", None) else tuple()
            self._runtime_pool = ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_initialize_mcp_worker,
                initargs=(
                    self._config_path,
                    whitelist,
                    bool(self._use_cache),
                    int(self._read_timeout_seconds),
                ),
            )
            return self._runtime_pool

    @staticmethod
    def _make_error_call_result(tool_name: str, message: str) -> CallToolResult:
        return CallToolResult(
            content=[TextContent(type="text", text=message)],
            isError=True,
        )

    def _resolve_pool_future_sync(self, future, tool_name: str) -> CallToolResult:
        try:
            payload = future.result()
            return CallToolResult.model_validate(payload)
        except NoTool:
            logger.exception(f"Tool {tool_name} not found in MCP client")
            return self._make_error_call_result(tool_name, f"Tool {tool_name} not found")
        except KeyError as error:
            logger.exception(f"KeyError when executing MCP tool call: {error}")
            return self._make_error_call_result(
                tool_name, f"Error executing tool {tool_name}: KeyError {error}"
            )
        except Exception as error:
            logger.exception(f"Error executing MCP tool call: {error}")
            return self._make_error_call_result(
                tool_name, f"Error executing tool {tool_name}: {error}"
            )

    async def _resolve_pool_future_async(self, future, tool_name: str) -> CallToolResult:
        try:
            payload = await asyncio.wrap_future(future)
            return CallToolResult.model_validate(payload)
        except NoTool:
            logger.exception(f"Tool {tool_name} not found in MCP client")
            return self._make_error_call_result(tool_name, f"Tool {tool_name} not found")
        except KeyError as error:
            logger.exception(f"KeyError when executing MCP tool call: {error}")
            return self._make_error_call_result(
                tool_name, f"Error executing tool {tool_name}: KeyError {error}"
            )
        except Exception as error:
            logger.exception(f"Error executing MCP tool call: {error}")
            return self._make_error_call_result(
                tool_name, f"Error executing tool {tool_name}: {error}"
            )

    def _shutdown_runtime_pool(self) -> None:
        if self._runtime_pool is not None:
            self._runtime_pool.shutdown(wait=True)
            self._runtime_pool = None

    def _execute_tool_via_pool_sync(self, action: ToolCallAction) -> ToolResult:
        start = time.perf_counter()
        future = self._ensure_runtime_pool().submit(
            _call_tool_in_worker,
            action.function.name,
            action.function.arguments,
        )
        call_result = self._resolve_pool_future_sync(future, action.function.name)
        observation = ToolResult(tool_call_id=getattr(action, "id", ""), content=call_result)
        observation.metadata.other["action_execution_time"] = time.perf_counter() - start
        observation.metadata.other["action_kind"] = action.kind
        return observation

    async def _execute_tool_via_pool_async(self, action: ToolCallAction) -> ToolResult:
        start = time.perf_counter()
        future = self._ensure_runtime_pool().submit(
            _call_tool_in_worker,
            action.function.name,
            action.function.arguments,
        )
        call_result = await self._resolve_pool_future_async(future, action.function.name)
        observation = ToolResult(tool_call_id=getattr(action, "id", ""), content=call_result)
        observation.metadata.other["action_execution_time"] = time.perf_counter() - start
        observation.metadata.other["action_kind"] = action.kind
        return observation

    def step(self, action: Action) -> Observation:
        if not isinstance(action, ToolCallAction):
            return super().step(action)

        outcome, message = self._precheck_tool_action(action)
        if outcome == "math_answer":
            return self._create_math_answer(action)
        if outcome == "error":
            return self._make_error_tool_result(action, message or "")

        try:
            observation = self._execute_tool_call_sync(action)
        except BaseException:
            self._broken = True
            raise

        return self._postprocess_after_tool(action, observation)

    async def astep(self, action: Action) -> Observation:
        if not isinstance(action, ToolCallAction):
            return await super().astep(action)

        outcome, message = self._precheck_tool_action(action)
        if outcome == "math_answer":
            return self._create_math_answer(action)
        if outcome == "error":
            return self._make_error_tool_result(action, message or "")

        try:
            observation = await self._execute_tool_call_async(action)
        except BaseException:
            self._broken = True
            raise

        return self._postprocess_after_tool(action, observation)

    def _precheck_tool_action(self, action: ToolCallAction) -> tuple[str, str | None]:
        if action.function.name == "MathAnswer":
            return "math_answer", None
        if self._broken:
            return "error", self._backend_unavailable_message()
        if action.function.name == "run_python_code":
            block_message = self._check_python_safety(action.function.arguments)
            if block_message is not None:
                return "error", block_message
        return "ok", None

    def _execute_tool_call_sync(self, action: ToolCallAction) -> Observation:
        if self._should_offload(action.function.name):
            return self._execute_tool_via_pool_sync(action)
        return super().step(action)

    async def _execute_tool_call_async(self, action: ToolCallAction) -> Observation:
        if self._should_offload(action.function.name):
            return await self._execute_tool_via_pool_async(action)
        return await super().astep(action)

    def _postprocess_after_tool(
        self,
        action: ToolCallAction,
        observation: Observation,
    ) -> Observation:
        if action.function.name != "MathAnswer":
            return self._postprocess_tool_observation(action, observation)
        return observation

    def _ensure_math_answer_tool(self) -> None:
        if not any(
            getattr(tool, "function", None) and tool.function.name == "MathAnswer"
            for tool in self.tools
        ):
            self.tools.append(self._math_answer_spec)

    def _reset_health(self) -> None:
        self._broken = False
        self._last_failure_reason = None
        self._runtime_guard_installed = False

    def _create_math_answer(self, action: ToolCallAction) -> MathAnswer:
        answer_value = self._extract_answer(action.function.arguments)
        math_answer = MathAnswer(answer=answer_value)
        math_answer.metadata.other.update({
            "action_kind": "MathAnswer",
            "tool_call_id": getattr(action, "id", ""),
            "action_execution_time": 0.0,
        })
        return math_answer

    def mark_healthy(self) -> None:
        self._reset_health()

    def is_healthy(self) -> bool:
        return not self._broken

    def close(self) -> None:
        self._shutdown_runtime_pool()
        super().close()

    async def aclose(self) -> None:
        self._shutdown_runtime_pool()
        await super().aclose()

    @staticmethod
    def _guard_snippet() -> str:
        """generate Python code that installs safety guards"""
        return (
            "import builtins, sys, os, time, atexit\n"
            "try:\n"
            "    _PIPELINERL_TIME_LIMIT = float(os.environ.get('PIPELINERL_PY_TIMEOUT', '30'))\n"
            "except ValueError:\n"
            "    _PIPELINERL_TIME_LIMIT = 30.0\n"
            "_PIPELINERL_START = time.perf_counter()\n"
            "class _ExitBlocked(RuntimeError):\n"
            "    pass\n"
            "def _blocked_exit(*_args, **_kwargs):\n"
            "    raise _ExitBlocked('exit() and os._exit() are disabled in this environment.')\n"
            "for _target in (builtins, sys):\n"
            "    for _name in ('exit', 'quit'):\n"
            "        if hasattr(_target, _name):\n"
            "            setattr(_target, _name, _blocked_exit)\n"
            "if hasattr(os, '_exit'):\n"
            "    os._exit = _blocked_exit\n"
            "def _pipelinerl_trace(frame, event, arg):\n"
            "    if event == 'line' and (time.perf_counter() - _PIPELINERL_START) > _PIPELINERL_TIME_LIMIT:\n"
            "        sys.settrace(None)\n"
            "        raise RuntimeError(f'Python execution timed out after {_PIPELINERL_TIME_LIMIT} seconds.')\n"
            "    return _pipelinerl_trace\n"
            "sys.settrace(_pipelinerl_trace)\n"
            "atexit.register(lambda: sys.settrace(None))\n"
        )

    async def _install_runtime_guard(self) -> None:
        """Install runtime safety guard in the Python environment."""
        if self._runtime_guard_installed or not getattr(self, "client", None):
            return
        try:
            snippet = self._guard_snippet()
            if self._should_offload("run_python_code"):
                future = self._ensure_runtime_pool().submit(
                    _call_tool_in_worker,
                    "run_python_code",
                    {"python_code": snippet},
                )
                await self._resolve_pool_future_async(future, "run_python_code")
            else:
                await self.client.call_tool(
                    "run_python_code",
                    {"python_code": snippet},
                )
            self._runtime_guard_installed = True
            logger.debug("Runtime guard installed successfully")
        except Exception:
            logger.warning("Failed to install runtime guard in MCP environment", exc_info=True)

    def _postprocess_tool_observation(
        self,
        action: ToolCallAction,
        observation: Observation,
    ) -> Observation:
        if not isinstance(observation, ToolResult):
            return observation
        call_result = observation.content
        if not isinstance(call_result, CallToolResult):
            return observation
        if not getattr(call_result, "isError", False):
            return observation
        error_text = self._extract_call_result_text(call_result)
        if not self._is_connection_error_message(error_text):
            return observation
        logger.warning(
            "MCP backend failure detected for tool %s: %s",
            action.function.name,
            error_text,
        )
        return self._handle_connection_failure(action, observation, error_text)

    @staticmethod
    def _extract_call_result_text(call_result: CallToolResult) -> str:
        if not isinstance(call_result.content, list):
            return ""
        parts: list[str] = []
        for block in call_result.content:
            if isinstance(block, TextContent) and isinstance(block.text, str):
                parts.append(block.text)
        return "\n".join(parts).strip()

    @staticmethod
    def _is_connection_error_message(message: str) -> bool:
        lowered = message.lower()
        return any(pattern in lowered for pattern in _CONNECTION_ERROR_PATTERNS)

    def _handle_connection_failure(
        self,
        action: ToolCallAction,
        observation: ToolResult,
        error_text: str,
    ) -> ToolResult:
        """Mark environment as broken and update observation."""
        self._broken = True
        failure_message = (
            "Python tool backend became unavailable (connection lost). "
            "Environment will restart after this attempt; stop issuing additional tool calls."
        )
        if error_text:
            failure_message = f"{failure_message}\nOriginal error: {error_text}"

        observation.content = CallToolResult(
            content=[TextContent(type="text", text=failure_message)],
            isError=True,
        )
        observation.metadata.other.setdefault("action_execution_time", observation.metadata.other.get("action_execution_time", 0.0))
        observation.metadata.other["connection_failure"] = True
        observation.metadata.other["original_error"] = error_text
        self._last_failure_reason = failure_message
        return observation

    def _backend_unavailable_message(self) -> str:
        """Get message for unavailable backend."""
        return self._last_failure_reason or (
            "Python tool backend is restarting after a connection failure. "
            "Abort this attempt and wait for a fresh environment."
        )

    @staticmethod
    def _extract_answer(arguments: dict | str | None) -> str:
        """Extract answer string from arguments."""
        if arguments is None:
            return ""
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                return str(parsed.get("answer", "")) if isinstance(parsed, dict) else str(parsed)
            except json.JSONDecodeError:
                return arguments
        if isinstance(arguments, dict):
            return str(arguments.get("answer", ""))
        return str(arguments)

    def _check_python_safety(self, arguments: dict | str | None) -> str | None:
        """check for Python code problems"""
        code = self._extract_python_code(arguments)
        if not code:
            return None
        for pattern, label in self._python_blocklist:
            if pattern.search(code):
                return (
                    f"Python execution rejected: forbidden call detected ({label}). "
                    "Use pure computation without exiting the runtime."
                )
        return None

    @staticmethod
    def _extract_python_code(arguments: dict | str | None) -> str:
        if arguments is None:
            return ""
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                if isinstance(parsed, dict):
                    return str(parsed.get("python_code", parsed.get("code", "")))
                return str(parsed)
            except json.JSONDecodeError:
                return arguments
        if isinstance(arguments, dict):
            return str(arguments.get("python_code", arguments.get("code", "")))
        return str(arguments)

    def _make_error_tool_result(self, action: ToolCallAction, message: str) -> ToolResult:
        result = CallToolResult(
            content=[TextContent(type="text", text=message)],
            isError=True,
        )
        tool_result = ToolResult(
            tool_call_id=getattr(action, "id", ""),
            content=result,
        )
        tool_result.metadata.other["action_execution_time"] = 0.0
        tool_result.metadata.other["action_kind"] = action.kind
        return tool_result


class EmbeddedEnvironmentWorker:
    def __init__(self, env_cfg: DictConfig, concurrency: int = 1):
        # make repeated instantiations stable even if the caller changes its copy
        self._env_cfg = OmegaConf.create(env_cfg)
        self._cfg_signature = self._make_cfg_signature(self._env_cfg)
        self._concurrency = max(1, concurrency)
        self._init_lock = asyncio.Lock()
        self._available: asyncio.Queue[_ProcessEnvironmentProxy] | None = None
        self._all_envs: set[_ProcessEnvironmentProxy] = set()

    @staticmethod
    def _make_cfg_signature(cfg: DictConfig) -> str:
        try:
            container = OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            container = OmegaConf.to_container(cfg, resolve=False)
        return json.dumps(container, sort_keys=True, default=str)

    @property
    def concurrency(self) -> int:
        return self._concurrency

    def matches(self, env_cfg: DictConfig) -> bool:
        return self._cfg_signature == self._make_cfg_signature(env_cfg)

    def set_concurrency(self, concurrency: int) -> None:
        self._concurrency = max(1, concurrency)

    async def _ensure_pool(self) -> None:
        if self._available is None:
            self._available = asyncio.Queue()
        if len(self._all_envs) >= self._concurrency:
            return
        async with self._init_lock:
            if len(self._all_envs) >= self._concurrency:
                return
            missing = self._concurrency - len(self._all_envs)
            for _ in range(missing):
                environment = _ProcessEnvironmentProxy(self._env_cfg)
                try:
                    await self._init_and_reset(environment)
                except Exception:
                    logger.exception("Failed to initialize embedded environment instance")
                    await self._close(environment)
                    raise
                self._all_envs.add(environment)
                await self._available.put(environment)

    @asynccontextmanager
    async def alifecycle(self) -> AsyncIterator[Environment]:
        """Context manager for environment lifecycle with automatic health checking."""
        await self._ensure_pool()
        assert self._available is not None
        
        environment = await self._available.get()
        try:
            await self._reset(environment)
            yield environment
        finally:
            try:
                unhealthy = (
                    hasattr(environment, "is_healthy")
                    and not environment.is_healthy()  # type: ignore
                )
            except Exception:
                logger.warning("Failed to query embedded environment health; replacing", exc_info=True)
                unhealthy = True
            is_healthy = not unhealthy
            
            if is_healthy:
                # try to reset and recycle healthy environment
                try:
                    await self._reset(environment)
                    if hasattr(environment, "mark_healthy"):
                        environment.mark_healthy()  # type: ignore
                    await self._available.put(environment)
                except Exception:
                    logger.exception("Failed to recycle embedded environment; replacing")
                    await self._replace(environment)
            else:
                # environment is unhealthy, replace it
                logger.warning("Embedded environment is unhealthy, replacing")
                await self._replace(environment)

    async def _replace(self, environment: Environment) -> None:
        """Replace a broken environment with a new one."""
        if environment in self._all_envs:
            self._all_envs.remove(environment)
        try:
            await self._close(environment)
        except Exception:
            logger.exception("Failed to close environment during replacement")
        # Refill the pool
        await self._ensure_pool()

    async def _init_and_reset(self, env: Environment) -> None:
        # init
        if hasattr(env, "ainitialize") and inspect.iscoroutinefunction(env.ainitialize):
            await env.ainitialize()  # type: ignore
        else:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, env.initialize)
        
        # reset
        await self._reset(env)

    async def _reset(self, env: Environment) -> None:
        if hasattr(env, "areset") and inspect.iscoroutinefunction(env.areset):
            await env.areset()  # type: ignore
        else:
            reset_fn = getattr(env, "reset", None)
            if callable(reset_fn):
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, reset_fn)

    async def _close(self, env: Environment) -> None:
        loop = asyncio.get_running_loop()
        
        # try async close first
        if hasattr(env, "aclose") and inspect.iscoroutinefunction(env.aclose):
            try:
                await env.aclose()  # type: ignore
                return
            except Exception as e:
                logger.debug(f"Async close failed: {e}, trying sync close")
        
        # fallback to sync close
        try:
            await loop.run_in_executor(None, env.close)
        except Exception as e:
            logger.debug(f"Sync close failed: {e}")
