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

Fast-reset (mirrors tmax ``_materialize_writable_home_user``): the built task
rootfs (base + ``%post``) is created once per ``(base, container_def)`` and kept
read-only, shared across all env-server processes on the node via a file lock.
proot has no copy-on-write layer, so per session we bind a fresh small writable
``/home/user`` (and ``/tmp``) over the read-only base instead of copying the
whole 620MB tree. This keeps local ephemeral usage tiny under high concurrency
and avoids rebuilding the same task once per rollout (group size 32).
"""
from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import logging
import os
import pty
import queue
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
import termios
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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

#: Default root for the shared, read-only per-task rootfs cache when no
#: ``cache_dir`` is given: node-local disk. A mounted ``cache_dir`` (see
#: ``ProotTerminalEnvironment``) moves the big trees off the local ephemeral
#: quota. Override the local default with PL_TERMINAL_CACHE_DIR.
_DEFAULT_ROOTFS_ROOT = Path(os.environ.get("PL_TERMINAL_CACHE_DIR", Path(tempfile.gettempdir()) / "pl_terminal_cache"))
#: Locks and refcounts always live on node-local disk: the lock only needs to
#: coordinate the env-server processes on one node, and keeping it off the shared
#: mount avoids relying on NFS file locking.
_META_ROOT = Path(tempfile.gettempdir()) / "pl_terminal_meta"


def _proot_argv(proot_bin: str, rootfs: Path, cwd: str, binds: Sequence[str] = ()) -> List[str]:
    argv = [proot_bin, "-0", "-r", str(rootfs)]
    for b in _BIND_PATHS:
        if Path(b).exists():
            argv += ["-b", b]
    for spec in binds:
        argv += ["-b", spec]
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


def _task_key(base_name: str, container_def: str) -> str:
    h = hashlib.sha256()
    h.update(base_name.encode("utf-8"))
    h.update(b"\0")
    h.update(container_def.encode("utf-8"))
    return h.hexdigest()[:16]


@contextlib.contextmanager
def _locked(lock_path: Path):
    """Exclusive cross-process lock. The lock file lives outside the cached
    rootfs dir so eviction can remove the rootfs without dropping the lock."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _force_rmtree(path: Path) -> None:
    """Remove a tree that may have had its write bits stripped (read-only base)."""
    if path.exists():
        subprocess.run(["chmod", "-R", "u+w", str(path)], check=False)
        shutil.rmtree(path, ignore_errors=True)


def _dir_size_bytes(path: Path) -> int:
    """Apparent local-disk usage of a tree, or -1 if it can't be measured."""
    try:
        out = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True, timeout=30)
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.split()[0])
    except Exception:
        pass
    return -1


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
        cache_dir: Optional[str | Path] = None,
        max_session_disk_bytes: int = 1536 * 2**20,
        disk_check_interval: float = 3.0,
    ):
        self.base_rootfs = Path(base_rootfs).resolve()
        self.proot_bin = proot_bin
        self.nameserver = nameserver
        self.read_timeout = read_timeout
        self.shell_init_timeout = shell_init_timeout
        self.build_timeout = build_timeout
        self.verifier_timeout = verifier_timeout
        # Hard cap on this session's node-local writable scratch (/home/user +
        # /tmp binds). A single runaway agent command (e.g. ``cat /dev/zero >f``)
        # can otherwise fill the node's 16GiB ephemeral quota and evict the whole
        # replica. proot can't mount a quota'd fs, so a monitor thread enforces it
        # by aborting and freeing the session. 0 disables.
        self.max_session_disk_bytes = max_session_disk_bytes
        self.disk_check_interval = disk_check_interval

        self._owns_work_dir = work_dir is None
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="terminal_env_"))

        # Where the big read-only task rootfs trees live. A mounted ``cache_dir``
        # keeps them off the 16GiB local ephemeral quota; scope by hostname so
        # nodes sharing the mount don't collide or evict each other's trees.
        if cache_dir:
            self._rootfs_root = Path(cache_dir) / socket.gethostname()
        else:
            self._rootfs_root = _DEFAULT_ROOTFS_ROOT

        self._sid = uuid.uuid4().hex
        # Set by build(): the shared read-only task rootfs (under _rootfs_root),
        # its node-local meta dir (refcounts), and this session's small writable
        # overlay dirs bound over the read-only base.
        self.rootfs: Optional[Path] = None
        self._task_cache_dir: Optional[Path] = None
        self._task_meta_dir: Optional[Path] = None
        self._ref: Optional[Path] = None
        self.session_home = self.work_dir / "home"
        self.session_tmp = self.work_dir / "tmp"

        self.shell_process: Optional[subprocess.Popen] = None
        self.master_fd: Optional[int] = None
        self.slave_fd: Optional[int] = None
        self.output_queue: "queue.Queue[str]" = queue.Queue()
        self.reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._disk_exceeded = threading.Event()
        self._disk_monitor_thread: Optional[threading.Thread] = None
        self._marker = f"__CMD_DONE__{uuid.uuid4().hex}__"

    # ------------------------------------------------------------------
    # build (shared, read-only) + per-session writable overlay
    # ------------------------------------------------------------------
    def build(self, container_def: str) -> Tuple[bool, str]:
        """Ensure the shared task rootfs exists, then materialize this session's
        writable ``/home/user``. The rootfs (base + ``%post``) is built once per
        ``(base, container_def)`` and shared read-only across processes."""
        t0 = time.perf_counter()
        key = _task_key(self.base_rootfs.name, container_def)
        self._task_cache_dir = self._rootfs_root / key
        self._task_meta_dir = _META_ROOT / key
        task_rootfs = self._task_cache_dir / "rootfs"
        ready = self._task_cache_dir / ".ready"
        lock = _META_ROOT / "_locks" / f"{key}.lock"

        with _locked(lock):
            if not ready.exists():
                ok, err = self._build_shared_rootfs(task_rootfs, container_def, t0)
                if not ok:
                    return False, err
                self._task_cache_dir.mkdir(parents=True, exist_ok=True)
                ready.touch()
            # Register this session before releasing the lock so a concurrent
            # eviction cannot remove the rootfs out from under us.
            self._ref = self._task_meta_dir / "refs" / self._sid
            self._ref.parent.mkdir(parents=True, exist_ok=True)
            self._ref.write_text("")

        self.rootfs = task_rootfs
        # Materialize a small writable /home/user for this session from the
        # read-only base, plus a writable /tmp. cp -a preserves the stripped
        # write bits, so restore them on the copy.
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if self.session_home.exists():
            _force_rmtree(self.session_home)
        src_home = task_rootfs / "home/user"
        if src_home.exists():
            subprocess.run(["cp", "-a", str(src_home), str(self.session_home)], check=True)
            subprocess.run(["chmod", "-R", "u+w", str(self.session_home)], check=False)
        else:
            self.session_home.mkdir(parents=True, exist_ok=True)
        self.session_tmp.mkdir(parents=True, exist_ok=True)
        return True, ""

    def _build_shared_rootfs(self, task_rootfs: Path, container_def: str, t0: float) -> Tuple[bool, str]:
        """Clone the base rootfs and apply ``%post`` once, then make it read-only.
        Caller holds the per-task build lock."""
        _force_rmtree(task_rootfs)
        self._task_cache_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", "-a", str(self.base_rootfs), str(task_rootfs)], check=True)
        resolv = task_rootfs / "etc/resolv.conf"
        resolv.unlink(missing_ok=True)
        resolv.write_text(f"nameserver {self.nameserver}\noptions ndots:0\n")

        post = _post_body(container_def)
        argv = _proot_argv(self.proot_bin, task_rootfs, cwd="/root") + ["/bin/bash", "-c", post]
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=self.build_timeout)
        logger.info("Built task rootfs in %.1fs (rc=%d)", time.perf_counter() - t0, proc.returncode)
        if proc.returncode != 0:
            _force_rmtree(task_rootfs)
            return False, ((proc.stderr or proc.stdout) or "")[-1000:]
        # Strip write bits so concurrent sessions sharing this base cannot
        # corrupt it; each session writes only to its bound /home/user and /tmp.
        subprocess.run(["chmod", "-R", "a-w", str(task_rootfs)], check=False)
        return True, ""

    def _session_binds(self) -> List[str]:
        return [
            f"{self.session_home}:/home/user",
            f"{self.session_tmp}:/tmp",
        ]

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

    def _disk_monitor_loop(self) -> None:
        """Abort the session if its writable scratch exceeds the cap.

        Frees the scratch immediately on trip so the node's ephemeral quota is
        released without waiting for ``/close``. Marks the session via
        ``_disk_exceeded`` so ``exec``/verifier return a clean failure."""
        cap = self.max_session_disk_bytes
        while not self._stop_event.wait(self.disk_check_interval):
            if not self.shell_process or self.shell_process.poll() is not None:
                return
            size = _dir_size_bytes(self.work_dir)
            if size < 0 or size <= cap:
                continue
            logger.warning(
                "session %s scratch %.1f MiB exceeded cap %.1f MiB; aborting",
                self._sid, size / 2**20, cap / 2**20,
            )
            self._disk_exceeded.set()
            proc = self.shell_process
            if proc and proc.poll() is None:
                proc.kill()
            # Reclaim the node-local bytes now, not at /close.
            _force_rmtree(self.session_home)
            _force_rmtree(self.session_tmp)
            return

    def start(self, source_startup_hooks: bool = True) -> bool:
        """Start the persistent proot bash session on a PTY."""
        if self.rootfs is None:
            logger.error("start() called before build()")
            return False
        self.master_fd, self.slave_fd = pty.openpty()
        try:
            attrs = termios.tcgetattr(self.slave_fd)
            attrs[3] = attrs[3] & ~termios.ECHO
            termios.tcsetattr(self.slave_fd, termios.TCSANOW, attrs)
        except Exception:
            pass

        argv = _proot_argv(self.proot_bin, self.rootfs, "/home/user", binds=self._session_binds()) + ["/bin/bash"]
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

        if self.max_session_disk_bytes > 0:
            self._disk_monitor_thread = threading.Thread(target=self._disk_monitor_loop, daemon=True)
            self._disk_monitor_thread.start()
        return True

    def exec(self, command: str, timeout: Optional[float] = None) -> Tuple[bool, str]:
        """Run a command in the persistent shell, returning (success, output)."""
        if self._disk_exceeded.is_set():
            return False, "session aborted: local scratch disk limit exceeded"
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
        if self._disk_exceeded.is_set():
            return False, "session aborted: local scratch disk limit exceeded"
        # /home/user is bound to the writable session home, so write the test
        # file straight into it instead of touching the read-only base.
        (self.session_home / name).write_text(test_text, encoding="utf-8")
        q = shlex.quote(f"/home/user/{name}")
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
        if self._disk_monitor_thread:
            self._disk_monitor_thread.join(timeout=1.0)
            self._disk_monitor_thread = None
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

        # Release this session's hold on the shared rootfs and evict it once no
        # session is using it, so the node-local cache stays bounded.
        self._release_shared_rootfs()

        if self._owns_work_dir and self.work_dir.exists():
            # Report the node-local writable scratch this session held (per-session
            # /home/user + /tmp). Aggregated across concurrent sessions this is the
            # quantity that overflows the 16GiB ephemeral cap, so log it to size n_envs.
            size = _dir_size_bytes(self.work_dir)
            if size >= 0:
                logger.info("session %s local scratch at close: %.1f MiB", self._sid, size / (1024 * 1024))
            shutil.rmtree(self.work_dir, ignore_errors=True)

    def _release_shared_rootfs(self) -> None:
        if self._task_meta_dir is None:
            return
        key = self._task_meta_dir.name
        lock = _META_ROOT / "_locks" / f"{key}.lock"
        try:
            with _locked(lock):
                if self._ref is not None:
                    try:
                        self._ref.unlink()
                    except FileNotFoundError:
                        pass
                refs_dir = self._task_meta_dir / "refs"
                remaining = list(refs_dir.glob("*")) if refs_dir.exists() else []
                if not remaining:
                    # No session is using this task: drop both the read-only
                    # rootfs (on the mount) and the node-local meta dir.
                    if self._task_cache_dir is not None:
                        _force_rmtree(self._task_cache_dir)
                    _force_rmtree(self._task_meta_dir)
        except Exception:
            logger.warning("shared rootfs cleanup failed for %s", self._task_cache_dir, exc_info=True)
        finally:
            self._task_cache_dir = None
            self._task_meta_dir = None
            self._ref = None

    def __enter__(self) -> "ProotTerminalEnvironment":
        return self

    def __exit__(self, *_exc) -> None:
        self.cleanup()
