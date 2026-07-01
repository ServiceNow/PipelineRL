import os
import signal
import subprocess
import sys
import time

from pipelinerl.domains.terminal.proot_env import _terminate_process_group


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
