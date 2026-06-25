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

    # k8s ephemeral-storage = container writable layer + emptyDir + pod logs. An
    # emptyDir is its own mount, which `du -x /` skips, so enumerate every
    # non-NFS/non-virtual mount and du each one. That surfaces a scratch volume
    # the overlay-only view misses.
    _SKIP_FSTYPES = {
        "nfs", "nfs4", "proc", "sysfs", "cgroup", "cgroup2", "devpts", "mqueue",
        "devtmpfs", "tmpfs", "fusectl", "debugfs", "tracefs", "securityfs",
        "pstore", "bpf", "configfs", "autofs", "binfmt_misc", "hugetlbfs", "rpc_pipefs",
    }

    def _local_mounts() -> list[str]:
        points: list[str] = []
        try:
            for line in Path("/proc/mounts").read_text().splitlines():
                parts = line.split()
                if len(parts) < 3:
                    continue
                mnt, fstype = parts[1], parts[2]
                if fstype in _SKIP_FSTYPES:
                    continue
                points.append(mnt)
        except Exception:
            points = ["/"]
        # de-dup, keep "/" first
        seen, ordered = set(), []
        for p in (["/"] + points):
            if p not in seen:
                seen.add(p)
                ordered.append(p)
        return ordered

    def _loop() -> None:
        first = True
        while True:
            try:
                du = shutil.disk_usage("/")
                logger.info(
                    "[disk] backing fs '/' used=%.2f GiB / %.2f GiB (%.0f%%)",
                    du.used / 2**30, du.total / 2**30, 100 * du.used / max(du.total, 1),
                )
                if first:
                    try:
                        out = subprocess.run(["df", "-h"], capture_output=True, text=True, timeout=20)
                        logger.info("[disk] df -h:\n%s", out.stdout)
                    except Exception as e:
                        logger.warning("[disk] df failed: %s", e)
                    first = False
                # Per-mount usage for local (non-NFS) mounts: the ephemeral-counted
                # bytes live here. -x so each du stays within its own mount.
                for mnt in _local_mounts():
                    try:
                        out = subprocess.run(["du", "-shx", mnt], capture_output=True, text=True, timeout=45)
                        if out.stdout.strip():
                            logger.info("[disk] du -shx %s -> %s", mnt, out.stdout.split()[0])
                    except Exception as e:
                        logger.warning("[disk] du %s failed: %s", mnt, e)
                # /tmp is on the overlay and is the observed consumer; drill into it
                # to name the producer (terminal_env_* sessions vs vLLM/torch JIT
                # scratch like torchinductor_*/triton/tmp*).
                try:
                    out = subprocess.run(
                        "du -xhd1 /tmp 2>/dev/null | sort -rh | head -15",
                        shell=True, capture_output=True, text=True, timeout=45,
                    )
                    if out.stdout.strip():
                        logger.info("[disk] du -xhd1 /tmp (top):\n%s", out.stdout)
                except Exception as e:
                    logger.warning("[disk] /tmp drill failed: %s", e)
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
        max_session_disk_bytes: int = 1536 * 2**20,
        max_session_rss_bytes: int = 16 * 2**30,
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
        self.max_session_disk_bytes = max_session_disk_bytes
        self.max_session_rss_bytes = max_session_rss_bytes

        self._sessions: Dict[str, TerminalSession] = {}
        self._lock = asyncio.Lock()
        # Each concurrent session may have one blocking proot call in flight.
        self._executor = ThreadPoolExecutor(max_workers=n_envs + 4)
        # Background cleanup tasks (kept referenced so they aren't GC'd mid-flight).
        self._bg_tasks: set = set()

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
            max_session_disk_bytes=self.max_session_disk_bytes,
            max_session_rss_bytes=self.max_session_rss_bytes,
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
        # Free the slot immediately, then run cleanup off the response path. Session
        # cleanup includes the shared-rootfs eviction rmtree (NFS), which can exceed
        # the client's /close timeout and was holding the slot; the refcount+lock in
        # cleanup keep eviction race-safe even with the slot already reused.
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            task = asyncio.create_task(self._run(session.close))
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)
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
