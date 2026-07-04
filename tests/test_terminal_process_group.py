import json
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


def _started_env_for_exec():
    env = ProotTerminalEnvironment.__new__(ProotTerminalEnvironment)
    env._disk_exceeded = SimpleNamespace(is_set=lambda: False)
    env._abort_reason = None
    env.shell_process = SimpleNamespace(poll=lambda: None)
    env.reader_thread = SimpleNamespace(is_alive=lambda: True)
    env.master_fd = 11
    env._marker = "__CMD_DONE__test__"
    env.read_timeout = 0.05
    env._drain = lambda: ""
    env._read_until_marker = lambda timeout: ("ok\n", 0)
    return env


def test_exec_bash_precheck_rejects_unterminated_quote_without_touching_session(monkeypatch):
    env = _started_env_for_exec()
    writes = []
    monkeypatch.setattr(proot_env.os, "write", lambda fd, data: writes.append(data.decode()) or len(data))

    ok, out, abort_kind = env.exec('echo "unterminated', timeout=1.0)

    assert not ok
    assert abort_kind is None
    assert "bash syntax error (exit 2):" in out
    assert "unexpected EOF" in out
    assert writes == []

    ok, out, abort_kind = env.exec("echo ok", timeout=1.0)

    assert ok
    assert abort_kind is None
    assert "echo ok" in writes[-1]


def test_exec_bash_precheck_rejects_unterminated_heredoc_without_touching_session(monkeypatch):
    env = _started_env_for_exec()
    writes = []
    monkeypatch.setattr(proot_env.os, "write", lambda fd, data: writes.append(data.decode()) or len(data))

    ok, out, abort_kind = env.exec("cat <<EOF\nhello", timeout=1.0)

    assert not ok
    assert abort_kind is None
    assert "bash syntax error (exit 2):" in out
    assert "here-document" in out
    assert writes == []

    ok, out, abort_kind = env.exec("cat <<EOF\nhello\nEOF", timeout=1.0)

    assert ok
    assert abort_kind is None
    assert "cat <<EOF" in writes[-1]


def test_exec_bash_precheck_preserves_background_commands(monkeypatch):
    env = _started_env_for_exec()
    writes = []
    monkeypatch.setattr(proot_env.os, "write", lambda fd, data: writes.append(data.decode()) or len(data))

    ok, out, abort_kind = env.exec("sleep 1 &", timeout=1.0)

    assert ok
    assert abort_kind is None
    assert "sleep 1 &" in writes[-1]


def test_exec_bash_precheck_timeout_falls_through_to_normal_exec(monkeypatch):
    env = _started_env_for_exec()
    writes = []
    monkeypatch.setattr(proot_env.os, "write", lambda fd, data: writes.append(data.decode()) or len(data))

    captured_env = {}

    def timeout_run(*args, **kwargs):
        captured_env["value"] = kwargs.get("env")
        raise subprocess.TimeoutExpired(args[0], kwargs.get("timeout", 5.0))

    monkeypatch.setattr(proot_env.subprocess, "run", timeout_run)

    ok, out, abort_kind = env.exec('echo "unterminated', timeout=1.0)

    assert ok
    assert abort_kind is None
    assert captured_env["value"] is not os.environ
    assert captured_env["value"]["LC_ALL"] == "C"
    assert writes
    assert 'echo "unterminated' in writes[-1]


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
        return False, "session aborted", env._abort_reason

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


def test_session_delta_manifest_tracks_touched_top_level_dirs(tmp_path):
    root = tmp_path / "rootfs"
    (root / "app").mkdir(parents=True)
    (root / "opt").mkdir(parents=True)
    (root / "home/user").mkdir(parents=True)
    (root / "tmp").mkdir(parents=True)
    before = proot_env._tree_metadata(root)

    (root / "app/new.txt").write_text("abc")
    (root / "opt/config.txt").write_text("hello")
    (root / "home/user/ignore.txt").write_text("private")
    (root / "tmp/ignore.txt").write_text("private")
    (root / "outside").symlink_to("/etc/passwd")

    manifest = proot_env._write_session_dirs_manifest(root, before)

    assert manifest["version"] == 1
    assert isinstance(manifest["build_completed_at"], float)
    assert manifest["dirs"] == [
        {"name": "app", "bytes": 3},
        {"name": "opt", "bytes": 5},
    ]
    assert manifest["total_bytes"] == 8
    assert json.loads((root / ".session_dirs.json").read_text()) == manifest


def test_shared_rootfs_contamination_detector_reports_modified_files(tmp_path):
    root = tmp_path / "rootfs"
    (root / "etc").mkdir(parents=True)
    (root / "app").mkdir()
    clean = root / "etc/clean.txt"
    touched = root / "etc/touched.txt"
    delta_touched = root / "app/touched.txt"
    manifest = {
        "version": 1,
        "build_completed_at": 1000.0,
        "dirs": [{"name": "app", "bytes": 7}],
        "total_bytes": 7,
    }
    manifest_path = root / ".session_dirs.json"
    manifest_path.write_text(json.dumps(manifest))
    clean.write_text("clean")
    touched.write_text("dirty")
    delta_touched.write_text("private")
    for path, mtime in (
        (manifest_path, 1000.0),
        (clean, 1001.0),
        (touched, 1003.5),
        (delta_touched, 1003.5),
    ):
        os.utime(path, (mtime, mtime))

    env = ProotTerminalEnvironment(
        base_rootfs=tmp_path,
        work_dir=tmp_path / "work",
        max_session_disk_bytes=0,
        max_session_rss_bytes=0,
    )
    env.rootfs = root

    assert env._detect_shared_rootfs_contamination() == 1


def test_shared_rootfs_contamination_detector_counts_delta_dirs_when_isolation_off(tmp_path):
    root = tmp_path / "rootfs"
    (root / "etc").mkdir(parents=True)
    (root / "app").mkdir()
    touched = root / "etc/touched.txt"
    delta_touched = root / "app/touched.txt"
    manifest = {
        "version": 1,
        "build_completed_at": 1000.0,
        "dirs": [{"name": "app", "bytes": 7}],
        "total_bytes": 7,
    }
    manifest_path = root / ".session_dirs.json"
    manifest_path.write_text(json.dumps(manifest))
    touched.write_text("dirty")
    delta_touched.write_text("private")
    for path in (manifest_path,):
        os.utime(path, (1000.0, 1000.0))
    for path in (touched, delta_touched):
        os.utime(path, (1003.5, 1003.5))

    env = ProotTerminalEnvironment(
        base_rootfs=tmp_path,
        work_dir=tmp_path / "work",
        max_session_disk_bytes=0,
        max_session_rss_bytes=0,
        session_delta_isolation=False,
    )
    env.rootfs = root

    assert env._detect_shared_rootfs_contamination() == 2


def test_shared_rootfs_contamination_detector_ignores_untouched_files(tmp_path):
    root = tmp_path / "rootfs"
    (root / "etc").mkdir(parents=True)
    clean = root / "etc/clean.txt"
    manifest = {
        "version": 1,
        "build_completed_at": 1000.0,
        "dirs": [],
        "total_bytes": 0,
    }
    manifest_path = root / ".session_dirs.json"
    manifest_path.write_text(json.dumps(manifest))
    clean.write_text("clean")
    os.utime(manifest_path, (1000.0, 1000.0))
    os.utime(clean, (1001.0, 1001.0))
    env = ProotTerminalEnvironment(
        base_rootfs=tmp_path,
        work_dir=tmp_path / "work",
        max_session_disk_bytes=0,
        max_session_rss_bytes=0,
    )
    env.rootfs = root

    assert env._detect_shared_rootfs_contamination() == 0


def test_session_delta_binds_include_reflinked_copies(monkeypatch, tmp_path):
    root = tmp_path / "rootfs"
    (root / "app").mkdir(parents=True)
    (root / "app/file.txt").write_text("abc")
    (root / ".session_dirs.json").write_text(json.dumps({
        "version": 1,
        "build_completed_at": 1000.0,
        "dirs": [{"name": "app", "bytes": 3}],
        "total_bytes": 3,
    }))
    env = ProotTerminalEnvironment(
        base_rootfs=tmp_path,
        work_dir=tmp_path / "work",
        max_session_disk_bytes=0,
        max_session_rss_bytes=0,
    )
    commands = []

    real_run = proot_env.subprocess.run

    def recording_run(argv, *args, **kwargs):
        commands.append(list(argv))
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(proot_env.subprocess, "run", recording_run)

    ok, err = env._materialize_session_delta_dirs(root)

    assert ok
    assert err == ""
    copied = env.session_deltas / "app" / "file.txt"
    assert copied.read_text() == "abc"
    assert [cmd[:3] for cmd in commands if cmd[:2] == ["cp", "-a"]] == [["cp", "-a", "--reflink=auto"]]
    binds = env._session_binds()
    assert binds[0] == f"{env.session_deltas / 'app'}:/app"
    assert binds[-2:] == [f"{env.session_home}:/home/user", f"{env.session_tmp}:/tmp"]


def test_session_delta_over_cap_is_not_runnable(monkeypatch, tmp_path):
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
        session_delta_max_bytes=2,
    )
    key = proot_env._task_key(base.name, "def")
    task_cache = env._rootfs_root / key
    root = task_cache / "rootfs"
    (root / "app").mkdir(parents=True)
    (root / "app/file.txt").write_text("abc")
    (root / ".session_dirs.json").write_text(json.dumps({
        "version": 1,
        "build_completed_at": 1000.0,
        "dirs": [{"name": "app", "bytes": 3}],
        "total_bytes": 3,
    }))
    (task_cache / ".ready").touch()

    ok, err = env.build("def")

    assert not ok
    assert "over cap" in err
    assert env._session_delta_binds == []
    env.cleanup()


def test_session_delta_isolation_disabled_bypasses_stale_over_cap_manifest(monkeypatch, tmp_path):
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
        session_delta_max_bytes=2,
        session_delta_isolation=False,
        contamination_check=False,
    )
    key = proot_env._task_key(base.name, "def")
    task_cache = env._rootfs_root / key
    root = task_cache / "rootfs"
    (root / "app").mkdir(parents=True)
    (root / "app/file.txt").write_text("abc")
    (root / ".session_dirs.json").write_text(json.dumps({
        "version": 999,
        "build_completed_at": 1000.0,
        "dirs": [{"name": "app", "bytes": 3}],
        "total_bytes": 3,
    }))
    (task_cache / ".ready").touch()

    ok, err = env.build("def")

    assert ok
    assert err == ""
    assert env._session_delta_binds == []
    assert env._session_binds() == [
        f"{env.session_home}:/home/user",
        f"{env.session_tmp}:/tmp",
    ]
    env.cleanup()


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
