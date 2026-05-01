import asyncio
import logging
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from cube.container import ContainerBackend
from pipelinerl.domains.terminalbench.environment import ContainerEnvironment

logger = logging.getLogger(__name__)


class ContainerEnvironmentServer:
    """
    Manages on-demand container environments exposed via HTTP.

    Each incoming /start_task request launches a fresh container using the
    task's ContainerConfig. Concurrency is capped at n_parallel_envs; requests
    beyond that receive a 503 immediately.
    """

    def __init__(
        self,
        cube_backend: ContainerBackend,
        n_parallel_envs: int = 8,
        host: str = "0.0.0.0",
        env_call_timeout: int = 120,
        start_task_timeout: int = 600,
    ) -> None:
        self.cube_backend = cube_backend
        self.n_parallel_envs = n_parallel_envs
        self.host = host
        self.env_call_timeout = env_call_timeout
        self.start_task_timeout = start_task_timeout

        self.sessions: dict[str, ContainerEnvironment] = {}
        self._active = 0
        self._lock = asyncio.Lock()

    async def _try_acquire(self) -> bool:
        async with self._lock:
            if self._active >= self.n_parallel_envs:
                return False
            self._active += 1
            return True

    async def _release_slot(self) -> None:
        async with self._lock:
            self._active = max(0, self._active - 1)

    def create_app(self) -> FastAPI:
        app = FastAPI(title="Container Environment Server")

        class SessionRequest(BaseModel):
            session_id: str

        class ExecRequest(BaseModel):
            session_id: str
            command: str
            timeout: int | None = None

        class StartTaskRequest(BaseModel):
            task_data: dict = {}

        class EvaluateRequest(BaseModel):
            session_id: str
            archive_b64: str
            test_timeout_sec: int = 900

        def _get_env(session_id: str) -> ContainerEnvironment:
            if session_id not in self.sessions:
                raise HTTPException(status_code=400, detail=f"Invalid or expired session: {session_id}")
            return self.sessions[session_id]

        @app.post("/start_task")
        async def start_task(request: StartTaskRequest):
            if not await self._try_acquire():
                raise HTTPException(status_code=503, detail="Server at capacity")

            task_id = request.task_data.get("id", "unknown")
            session_id = str(uuid.uuid4())
            env = ContainerEnvironment(self.cube_backend)
            self.sessions[session_id] = env
            logger.info(
                f"[{task_id}] Session {session_id[:8]} acquired "
                f"(active={self._active}/{self.n_parallel_envs})"
            )
            loop = asyncio.get_running_loop()
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, env.start_task, request.task_data),
                    timeout=self.start_task_timeout,
                )
            except Exception as e:
                self.sessions.pop(session_id, None)
                await self._release_slot()
                if isinstance(e, asyncio.TimeoutError):
                    logger.error(f"[{task_id}] start_task timed out for session {session_id[:8]}")
                    raise HTTPException(status_code=503, detail="start_task timed out")
                logger.exception(f"[{task_id}] start_task failed for session {session_id[:8]}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            return {"session_id": session_id, **result}

        @app.post("/release")
        async def release(request: SessionRequest):
            env = _get_env(request.session_id)
            task_id = env._task_id
            self.sessions.pop(request.session_id, None)
            loop = asyncio.get_running_loop()
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, env.release),
                    timeout=self.env_call_timeout,
                )
            except Exception as e:
                logger.warning(f"[{task_id}] Error releasing session {request.session_id[:8]}: {e}")
            finally:
                await self._release_slot()
            logger.info(
                f"[{task_id}] Session {request.session_id[:8]} released "
                f"(active={self._active}/{self.n_parallel_envs})"
            )
            return {"status": "ok"}

        @app.post("/exec")
        async def exec_command(request: ExecRequest):
            env = _get_env(request.session_id)
            loop = asyncio.get_running_loop()
            call_timeout = (request.timeout or 0) + 5 or self.env_call_timeout
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, env.exec, request.command, request.timeout),
                    timeout=call_timeout,
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=503, detail=f"exec timed out: {request.command!r}")
            return result

        @app.post("/evaluate")
        async def evaluate(request: EvaluateRequest):
            env = _get_env(request.session_id)
            loop = asyncio.get_running_loop()
            call_timeout = request.test_timeout_sec + 30
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, env.evaluate, request.archive_b64, request.test_timeout_sec
                    ),
                    timeout=call_timeout,
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=503, detail="evaluate timed out")
            return result

        @app.get("/health")
        async def health():
            return {
                "status": "ok",
                "active": self._active,
                "n_parallel_envs": self.n_parallel_envs,
                "free": self.n_parallel_envs - self._active,
            }

        return app

    def launch(self, port: int) -> None:
        """Start the HTTP server. Blocking."""
        app = self.create_app()
        logger.info(
            f"Starting Container Environment Server at http://{self.host}:{port} "
            f"(n_parallel_envs={self.n_parallel_envs})"
        )
        uvicorn.run(app, host=self.host, port=port, timeout_keep_alive=3600, log_level="info")
