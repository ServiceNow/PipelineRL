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
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict

from aiohttp import web

from .environment import TerminalSession

logger = logging.getLogger(__name__)


def _start_local_disk_logger() -> None:
    """Diagnostic: periodically log node-local ephemeral usage and its breakdown.

    The 16 GiB EAI ephemeral cap is per node, but 8 env servers run per inference
    node, so elect a single logger via an atomic node-local marker to avoid 8x
    duplicate output. Set ``PL_TERMINAL_DISK_LOG_INTERVAL=0`` to disable.
    """
    interval = float(os.environ.get("PL_TERMINAL_DISK_LOG_INTERVAL", "60"))
    if interval <= 0:
        return
    marker = Path(tempfile.gettempdir()) / "pl_terminal_disk_logger.lock"

    def _claim() -> bool:
        try:
            fd = os.open(str(marker), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            return False

    if not _claim():
        # Marker exists: take over only if the holder is dead (stale from a prior
        # run on a reused node); otherwise another live env server owns logging.
        try:
            holder = int(marker.read_text().strip() or "-1")
            os.kill(holder, 0)
            return
        except (ValueError, ProcessLookupError, PermissionError, FileNotFoundError):
            marker.unlink(missing_ok=True)
            if not _claim():
                return

    def _loop() -> None:
        while True:
            try:
                du = shutil.disk_usage("/")
                used, total = du.used, du.total
                logger.info(
                    "[disk] ephemeral '/' used=%.2f GiB / %.2f GiB (%.0f%%)",
                    used / 2**30, total / 2**30, 100 * used / max(total, 1),
                )
                # Top local dirs. -x stays on the overlay fs (won't descend NFS
                # mounts or /proc,/sys,/dev). /tmp logged separately in case it is
                # a distinct mount.
                for cmd in (["du", "-xhd1", "/"], ["du", "-shx", "/tmp"]):
                    try:
                        out = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
                        lines = [l for l in out.stdout.splitlines() if l.strip()]
                        logger.info("[disk] %s:\n%s", " ".join(cmd), "\n".join(lines[-20:]))
                    except Exception as e:
                        logger.warning("[disk] %s failed: %s", " ".join(cmd), e)
            except Exception:
                logger.warning("[disk] logger iteration failed", exc_info=True)
            time.sleep(interval)

    threading.Thread(target=_loop, daemon=True, name="disk-logger").start()
    logger.info("[disk] local ephemeral logger started (interval=%.0fs)", interval)


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
        _start_local_disk_logger()
        web.run_app(app, host=self.host, port=port, access_log=None)
