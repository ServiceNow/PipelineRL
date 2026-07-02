import os
import signal
import subprocess
import sys
import time

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
