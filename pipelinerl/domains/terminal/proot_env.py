"""proot-backed terminal environment for TMax-style tasks.

A drop-in replacement for the Apptainer-instance runner in tmax
``rl_data/generator/env.py``, built for environments where privileged or
user-namespace container runtimes are blocked (e.g. an eai/Toolkit job whose
mount namespace is locked by the parent). proot translates filesystem and
root-id syscalls in userspace via ptrace, so it needs no mount namespace, no
setuid helper, and no daemon. See ``TMAX_ENV_RECIPE.md`` for the analysis.

One task maps to one persistent proot bash session on a PTY. Commands are sent
with a unique completion marker so we can recover stdout and the exit code. The
session is long-lived on purpose: tasks that start a background daemon (the
def's ``/.singularity.d/env/*.sh`` startup hooks, sourced at session start) keep
that daemon alive across agent commands, which is what the verifier checks.
"""
from __future__ import annotations

import errno
import fcntl
import logging
import os
import pty
import queue
import re
import shlex
import shutil
import subprocess
import tempfile
import termios
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

#: Bind these host paths into every proot session.
_BIND_PATHS = ("/proc", "/dev", "/sys", "/etc/resolv.conf")
#: Minimal clean environment inside the container.
_CONTAINER_ENV = [
    "HOME=/home/user",
    "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "TERM=xterm",
    "DEBIAN_FRONTEND=noninteractive",
    "LC_ALL=C.UTF-8",
    "LANG=C.UTF-8",
]


def _proot_argv(proot_bin: str, rootfs: Path, cwd: str) -> List[str]:
    argv = [proot_bin, "-0", "-r", str(rootfs)]
    for b in _BIND_PATHS:
        if Path(b).exists():
            argv += ["-b", b]
    argv += ["-w", cwd, "--kill-on-exit", "/usr/bin/env", "-i", *_CONTAINER_ENV]
    return argv


def _post_body(def_text: str) -> str:
    """Extract the ``%post`` body from an Apptainer ``.def``."""
    body, in_post = [], False
    for line in def_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("%"):
            in_post = stripped.lower().startswith("%post")
            continue
        if in_post:
            body.append(line)
    return "\n".join(body)


class ProotTerminalEnvironment:
    """Persistent proot bash session for a single terminal task."""

    def __init__(
        self,
        base_rootfs: str | Path,
        proot_bin: str = "proot",
        nameserver: str = "10.150.0.10",
        read_timeout: float = 30.0,
        shell_init_timeout: float = 60.0,
        build_timeout: float = 1800.0,
        verifier_timeout: float = 180.0,
        work_dir: Optional[str | Path] = None,
    ):
        self.base_rootfs = Path(base_rootfs).resolve()
        self.proot_bin = proot_bin
        self.nameserver = nameserver
        self.read_timeout = read_timeout
        self.shell_init_timeout = shell_init_timeout
        self.build_timeout = build_timeout
        self.verifier_timeout = verifier_timeout

        self._owns_work_dir = work_dir is None
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="terminal_env_"))
        self.rootfs = self.work_dir / "rootfs"

        self.shell_process: Optional[subprocess.Popen] = None
        self.master_fd: Optional[int] = None
        self.slave_fd: Optional[int] = None
        self.output_queue: "queue.Queue[str]" = queue.Queue()
        self.reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._marker = f"__CMD_DONE__{uuid.uuid4().hex}__"

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------
    def build(self, container_def: str) -> Tuple[bool, str]:
        """Clone the base rootfs and apply the task's ``%post``."""
        t0 = time.perf_counter()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if self.rootfs.exists():
            shutil.rmtree(self.rootfs, ignore_errors=True)
        subprocess.run(["cp", "-a", str(self.base_rootfs), str(self.rootfs)], check=True)
        resolv = self.rootfs / "etc/resolv.conf"
        resolv.unlink(missing_ok=True)
        resolv.write_text(f"nameserver {self.nameserver}\noptions ndots:0\n")

        post = _post_body(container_def)
        proc = self._proot_run(post, cwd="/root", timeout=self.build_timeout)
        logger.info("Built task rootfs in %.1fs (rc=%d)", time.perf_counter() - t0, proc.returncode)
        if proc.returncode != 0:
            return False, ((proc.stderr or proc.stdout) or "")[-1000:]
        return True, ""

    def _proot_run(self, script: str, cwd: str = "/home/user", timeout: float = 600.0) -> subprocess.CompletedProcess:
        """Run a script in a throwaway (non-persistent) proot invocation."""
        argv = _proot_argv(self.proot_bin, self.rootfs, cwd) + ["/bin/bash", "-c", script]
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)

    # ------------------------------------------------------------------
    # persistent shell (PTY) lifecycle — ported from tmax env.py
    # ------------------------------------------------------------------
    def _reader_loop(self) -> None:
        fd = self.master_fd
        if fd is None:
            return
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        while not self._stop_event.is_set() and self.shell_process and self.shell_process.poll() is None:
            try:
                data = os.read(fd, 16384)
                if data:
                    self.output_queue.put_nowait(data.decode("utf-8", errors="replace"))
                    continue
            except BlockingIOError:
                pass
            except OSError as e:
                if getattr(e, "errno", None) in (errno.EBADF, errno.EIO):
                    break
                raise
            time.sleep(0.005)

    def _drain(self) -> str:
        chunks: List[str] = []
        while True:
            try:
                chunks.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return "".join(chunks)

    def _read_until_marker(self, timeout: Optional[float]) -> Tuple[str, Optional[int]]:
        deadline = time.time() + (timeout if timeout is not None else self.read_timeout)
        buf: List[str] = []
        while time.time() < deadline:
            chunk = self._drain()
            if chunk:
                buf.append(chunk)
                joined = "".join(buf)
                for line in joined.splitlines():
                    if self._marker in line and ":" in line:
                        head, _, tail = line.rpartition(":")
                        if head.endswith(self._marker):
                            try:
                                code = int(tail.strip())
                            except ValueError:
                                code = None
                            return joined[: joined.find(self._marker)], code
            time.sleep(0.002)
        return "".join(buf), None

    def start(self, source_startup_hooks: bool = True) -> bool:
        """Start the persistent proot bash session on a PTY."""
        self.master_fd, self.slave_fd = pty.openpty()
        try:
            attrs = termios.tcgetattr(self.slave_fd)
            attrs[3] = attrs[3] & ~termios.ECHO
            termios.tcsetattr(self.slave_fd, termios.TCSANOW, attrs)
        except Exception:
            pass

        argv = _proot_argv(self.proot_bin, self.rootfs, "/home/user") + ["/bin/bash"]
        self.shell_process = subprocess.Popen(
            argv, stdin=self.slave_fd, stdout=self.slave_fd, stderr=self.slave_fd,
            close_fds=True, start_new_session=True,
        )
        self._stop_event.clear()
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()

        time.sleep(0.2)
        if self.shell_process.poll() is not None:
            logger.error("proot shell exited early (rc=%s)", self.shell_process.returncode)
            return False

        init = (
            "set -o pipefail 2>/dev/null; export PS1=''; export HOME=/home/user; "
            "cd /home/user 2>/dev/null || true; "
            f"printf '{self._marker}:0\\n'"
        )
        os.write(self.master_fd, (init + "\n").encode())
        _, code = self._read_until_marker(self.shell_init_timeout)
        if code is None:
            logger.error("proot shell init timed out after %.0fs", self.shell_init_timeout)
            return False

        if source_startup_hooks:
            # Apptainer auto-sources these on exec; proot does not. Sourcing them
            # here starts the task's background services inside this long-lived
            # session so verifier port checks pass.
            self.exec(
                'for f in /.singularity.d/env/*.sh; do [ -f "$f" ] && source "$f" 2>/dev/null; done; true'
            )
        return True

    def exec(self, command: str, timeout: Optional[float] = None) -> Tuple[bool, str]:
        """Run a command in the persistent shell, returning (success, output)."""
        if not self.shell_process or self.shell_process.poll() is not None:
            return False, "shell is not running"
        if not self.reader_thread or not self.reader_thread.is_alive():
            return False, "reader thread is not alive"

        self._drain()
        command = command.strip()
        stripped = command.rstrip()
        is_background = stripped.endswith("&") and not stripped.endswith("&&")
        if "<<" in command or is_background:
            wrapped = f"{command}\ncode=$?; printf '{self._marker}:%s\\n' \"$code\""
        else:
            wrapped = f"{{ {command}; }}; code=$?; printf '{self._marker}:%s\\n' \"$code\""

        try:
            os.write(self.master_fd, (wrapped + "\n").encode())
        except OSError as e:
            return False, f"command write failed: {e}"

        raw, code = self._read_until_marker(timeout)
        if code is None:
            return False, f"command timed out. Partial output:\n{raw[:1000]}"
        cleaned = ANSI_RE.sub("", raw).replace("\r", "")
        return code == 0, cleaned

    # ------------------------------------------------------------------
    # verifiers
    # ------------------------------------------------------------------
    def _run_pytest(self, test_text: str, name: str) -> Tuple[bool, str]:
        host_path = self.work_dir / name
        host_path.write_text(test_text, encoding="utf-8")
        container_path = f"/home/user/{name}"
        shutil.copy(host_path, self.rootfs / "home/user" / name)
        q = shlex.quote(container_path)
        return self.exec(
            f"cd /home/user && python3 -m pytest -q --no-header {q}",
            timeout=self.verifier_timeout,
        )

    def run_initial_tests(self, test_text: str) -> bool:
        ok, out = self._run_pytest(test_text, "test_initial_state.py")
        if not ok:
            logger.info("initial-state tests failed:\n%s", out[-800:])
        return ok

    def run_final_tests(self, test_text: str) -> Tuple[bool, str]:
        ok, out = self._run_pytest(test_text, "test_final_state.py")
        return ok, out

    # ------------------------------------------------------------------
    def cleanup(self) -> None:
        self._stop_event.set()
        if self.reader_thread:
            self.reader_thread.join(timeout=1.0)
            self.reader_thread = None
        if self.shell_process and self.shell_process.poll() is None:
            try:
                os.write(self.master_fd, b"exit\n")
                self.shell_process.wait(timeout=2)
            except Exception:
                self.shell_process.kill()
        self.shell_process = None
        for fd in (self.master_fd, self.slave_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self.master_fd = self.slave_fd = None
        if self._owns_work_dir and self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)

    def __enter__(self) -> "ProotTerminalEnvironment":
        return self

    def __exit__(self, *_exc) -> None:
        self.cleanup()
