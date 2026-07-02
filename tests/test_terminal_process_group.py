import os
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from pipelinerl.domains.terminal import proot_env
from pipelinerl.domains.terminal.proot_env import ProotTerminalEnvironment, _terminate_process_group


def _pid_running(pid: int) -> bool:
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=5)
    stat = out.stdout.strip()
    return bool(stat) and "Z" not in stat


def test_terminate_process_group_kills_child_process():
    child_code = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    parent_code = (
        "import subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', sys.argv[1]]); "
        "print(child.pid, flush=True); "
        "time.sleep(60)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", parent_code, child_code],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert proc.stdout is not None
        child_pid = int(proc.stdout.readline().strip())
        assert _pid_running(child_pid)

        _terminate_process_group(proc, grace_seconds=0.1)

        deadline = time.time() + 5.0
        while time.time() < deadline and _pid_running(child_pid):
            time.sleep(0.05)

        assert proc.poll() is not None
        assert not _pid_running(child_pid)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=1)
        except Exception:
            pass


def test_read_until_marker_waits_for_split_exit_code():
    env = ProotTerminalEnvironment.__new__(ProotTerminalEnvironment)
    env._marker = "__CMD_DONE__test__"
    env.read_timeout = 0.05
    chunks = iter([
        "visible output\necho __CMD_DONE__test__:not-an-exit\n__CMD_DONE__test__:",
        "0\n",
    ])
    env._drain = lambda: next(chunks, "")

    raw, code = env._read_until_marker(timeout=0.2)

    assert code == 0
    assert raw == "visible output\necho __CMD_DONE__test__:not-an-exit\n"


def test_read_until_marker_waits_for_complete_exit_code_line():
    env = ProotTerminalEnvironment.__new__(ProotTerminalEnvironment)
    env._marker = "__CMD_DONE__test__"
    env.read_timeout = 0.05
    chunks = iter([
        "visible output\n__CMD_DONE__test__:1",
        "2\n",
    ])
    env._drain = lambda: next(chunks, "")

    raw, code = env._read_until_marker(timeout=0.2)

    assert code == 12
    assert raw == "visible output\n"
    assert env._drain() == ""


def test_start_returns_false_when_startup_hook_aborts(monkeypatch, tmp_path):
    class DummyProcess:
        returncode = None

        def poll(self):
            return None

    class DummyThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    env = ProotTerminalEnvironment(
        base_rootfs=tmp_path,
        work_dir=tmp_path / "work",
        max_session_disk_bytes=0,
        max_session_rss_bytes=0,
    )
    env.rootfs = tmp_path
    env._session_binds = lambda: []
    env._read_until_marker = lambda timeout: ("", 0)

    def aborting_exec(command):
        env._abort_reason = "timeout"
        return False, "session aborted"

    env.exec = aborting_exec
    monkeypatch.setattr(proot_env.pty, "openpty", lambda: (10, 11))
    monkeypatch.setattr(proot_env.termios, "tcgetattr", lambda fd: [0, 0, 0, 0])
    monkeypatch.setattr(proot_env.termios, "tcsetattr", lambda *args: None)
    monkeypatch.setattr(proot_env.subprocess, "Popen", lambda *args, **kwargs: DummyProcess())
    monkeypatch.setattr(proot_env.threading, "Thread", DummyThread)
    monkeypatch.setattr(proot_env.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(proot_env.os, "write", lambda fd, data: len(data))
    monkeypatch.setattr(proot_env, "_proot_argv", lambda *args, **kwargs: ["proot"])

    assert not env.start()
    assert env._abort_reason == "timeout"


def test_build_shared_rootfs_stages_base_and_uses_reflink(monkeypatch, tmp_path):
    meta_root = tmp_path / "meta"
    monkeypatch.setattr(proot_env, "_META_ROOT", meta_root)
    base = tmp_path / "bases" / "base_software_engineering"
    (base / "home/user").mkdir(parents=True)
    (base / "home/user/.bashrc").write_text("export READY=1\n")

    env = ProotTerminalEnvironment(
        base_rootfs=base,
        work_dir=tmp_path / "work",
        cache_dir=tmp_path / "cache",
        max_session_disk_bytes=0,
        max_session_rss_bytes=0,
    )
    env._task_cache_dir = tmp_path / "task_cache"
    task_rootfs = env._task_cache_dir / "rootfs"
    commands = []

    def fake_run(argv, *args, **kwargs):
        commands.append(list(argv))
        if argv[:2] == ["cp", "-a"]:
            src = argv[-2]
            dst = argv[-1]
            shutil.copytree(src, dst)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(proot_env.subprocess, "run", fake_run)

    ok, err = env._build_shared_rootfs(task_rootfs, "%post\necho ok", time.perf_counter())

    assert ok
    assert err == ""
    base_stage = meta_root / "_base_stages" / base.name
    cp_commands = [cmd for cmd in commands if cmd[:2] == ["cp", "-a"]]
    assert cp_commands[0][2] == str(base)
    assert cp_commands[1][:3] == ["cp", "-a", "--reflink=auto"]
    assert cp_commands[1][3] == str(base_stage)
    assert (task_rootfs / "home/user/.bashrc").exists()


def test_release_shared_rootfs_retains_current_and_evicts_old(monkeypatch, tmp_path):
    meta_root = tmp_path / "meta"
    monkeypatch.setattr(proot_env, "_META_ROOT", meta_root)
    base = tmp_path / "base"
    base.mkdir()
    env = ProotTerminalEnvironment(
        base_rootfs=base,
        work_dir=tmp_path / "work",
        cache_dir=tmp_path / "cache",
        rootfs_retention_seconds=60.0,
        max_session_disk_bytes=0,
        max_session_rss_bytes=0,
    )
    env._rootfs_root = tmp_path / "rootfs_cache"

    fresh_cache = env._rootfs_root / "fresh"
    fresh_meta = meta_root / "fresh"
    fresh_ref = fresh_meta / "refs" / env._sid
    fresh_cache.mkdir(parents=True)
    fresh_ref.parent.mkdir(parents=True)
    fresh_ref.write_text("")
    env._task_cache_dir = fresh_cache
    env._task_meta_dir = fresh_meta
    env._ref = fresh_ref

    old_cache = env._rootfs_root / "old"
    old_meta = meta_root / "old"
    old_cache.mkdir(parents=True)
    (old_meta / "refs").mkdir(parents=True)
    (old_meta / "last_used").write_text(str(time.time() - 120.0))

    env._release_shared_rootfs()

    assert fresh_cache.exists()
    assert fresh_meta.exists()
    assert not fresh_ref.exists()
    assert (fresh_meta / "last_used").exists()
    assert not old_cache.exists()
    assert not old_meta.exists()


def test_release_shared_rootfs_evicts_immediately_without_retention(monkeypatch, tmp_path):
    meta_root = tmp_path / "meta"
    monkeypatch.setattr(proot_env, "_META_ROOT", meta_root)
    base = tmp_path / "base"
    base.mkdir()
    env = ProotTerminalEnvironment(
        base_rootfs=base,
        work_dir=tmp_path / "work",
        cache_dir=tmp_path / "cache",
        max_session_disk_bytes=0,
        max_session_rss_bytes=0,
    )
    env._rootfs_root = tmp_path / "rootfs_cache"
    cache_dir = env._rootfs_root / "task"
    meta_dir = meta_root / "task"
    ref = meta_dir / "refs" / env._sid
    cache_dir.mkdir(parents=True)
    ref.parent.mkdir(parents=True)
    ref.write_text("")
    env._task_cache_dir = cache_dir
    env._task_meta_dir = meta_dir
    env._ref = ref

    env._release_shared_rootfs()

    assert not cache_dir.exists()
    assert not meta_dir.exists()
