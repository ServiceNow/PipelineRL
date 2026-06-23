"""Remote server hosting per-task proot terminal sandboxes over HTTP.

Plays the same role as ``miniwob``'s ``WebEnvironmentServer`` but with no
TapeAgents dependency: a small ``aiohttp`` app (aiohttp is already a core
PipelineRL dependency) manages a pool of ``TerminalSession`` objects. Blocking
proot operations run in a thread pool so the event loop stays responsive.

Launched on each ``kind="environment"`` job by
``pipelinerl/entrypoints/run_environment.py`` via ``launch(port)``.

Endpoints:
    GET  /health                       -> {status, active, capacity}
    POST /start_task {task_data}       -> {session_id, build_ok, started, init_ok}
    POST /step       {session_id, command} -> {output, success, exit_code}
    POST /finish     {session_id}      -> {passed, output}
    POST /close      {session_id}      -> {status}
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict

from aiohttp import web

from .environment import TerminalSession

logger = logging.getLogger(__name__)


class TerminalEnvironmentServer:
    def __init__(
        self,
        bases_dir: str,
        n_envs: int,
        host: str = "0.0.0.0",
        proot_bin: str = "proot",
        nameserver: str = "10.150.0.10",
        command_timeout: float = 60.0,
        verifier_timeout: float = 180.0,
        max_observation_chars: int = 4000,
        check_initial_state: bool = True,
        cache_dir: str | None = None,
    ):
        self.bases_dir = Path(bases_dir)
        self.n_envs = n_envs
        self.host = host
        self.proot_bin = proot_bin
        self.nameserver = nameserver
        self.command_timeout = command_timeout
        self.verifier_timeout = verifier_timeout
        self.max_observation_chars = max_observation_chars
        self.check_initial_state = check_initial_state
        self.cache_dir = cache_dir

        self._sessions: Dict[str, TerminalSession] = {}
        self._lock = asyncio.Lock()
        # Each concurrent session may have one blocking proot call in flight.
        self._executor = ThreadPoolExecutor(max_workers=n_envs + 4)

    async def _run(self, fn, *args):
        return await asyncio.get_event_loop().run_in_executor(self._executor, partial(fn, *args))

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response(
            {"status": "ok", "active": len(self._sessions), "capacity": self.n_envs}
        )

    async def start_task(self, request: web.Request) -> web.Response:
        body = await request.json()
        task = body["task_data"]

        async with self._lock:
            if len(self._sessions) >= self.n_envs:
                return web.json_response({"error": "capacity reached"}, status=503)
            session_id = str(uuid.uuid4())
            self._sessions[session_id] = None  # reserve the slot

        session = TerminalSession(
            bases_dir=self.bases_dir,
            proot_bin=self.proot_bin,
            nameserver=self.nameserver,
            command_timeout=self.command_timeout,
            verifier_timeout=self.verifier_timeout,
            max_observation_chars=self.max_observation_chars,
            check_initial_state=self.check_initial_state,
            cache_dir=self.cache_dir,
        )
        try:
            flags = await self._run(session.start, task)
        except Exception as e:
            logger.exception("start_task failed: %s", e)
            await self._run(session.close)
            async with self._lock:
                self._sessions.pop(session_id, None)
            return web.json_response({"error": str(e)}, status=500)

        if not flags.get("started"):
            # Unbuildable / unrunnable task: free the slot, no session handle.
            await self._run(session.close)
            async with self._lock:
                self._sessions.pop(session_id, None)
            return web.json_response({"session_id": None, **flags})

        self._sessions[session_id] = session
        return web.json_response({"session_id": session_id, **flags})

    def _get(self, session_id: str) -> TerminalSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise web.HTTPNotFound(text=f"unknown session {session_id}")
        return session

    async def step(self, request: web.Request) -> web.Response:
        body = await request.json()
        session = self._get(body["session_id"])
        result = await self._run(session.exec, body["command"])
        return web.json_response(result)

    async def finish(self, request: web.Request) -> web.Response:
        body = await request.json()
        session = self._get(body["session_id"])
        result = await self._run(session.finish)
        return web.json_response(result)

    async def close(self, request: web.Request) -> web.Response:
        body = await request.json()
        session_id = body["session_id"]
        session = self._sessions.get(session_id)
        if session is not None:
            await self._run(session.close)
        async with self._lock:
            self._sessions.pop(session_id, None)
        return web.json_response({"status": "ok"})

    def launch(self, port: int):
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.add_routes(
            [
                web.get("/health", self.health),
                web.post("/start_task", self.start_task),
                web.post("/step", self.step),
                web.post("/finish", self.finish),
                web.post("/close", self.close),
            ]
        )
        logger.info(
            "TerminalEnvironmentServer on %s:%d (n_envs=%d, bases=%s)",
            self.host, port, self.n_envs, self.bases_dir,
        )
        web.run_app(app, host=self.host, port=port, access_log=None)
