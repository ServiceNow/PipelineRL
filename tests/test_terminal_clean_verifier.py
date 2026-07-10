import hashlib
import subprocess
from pathlib import Path
import shutil
from types import SimpleNamespace
import threading

from pipelinerl.domains.terminal.environment import TerminalSession
from pipelinerl.domains.terminal import proot_env
from pipelinerl.domains.terminal.proot_env import ProotTerminalEnvironment


_TEST_SOURCE = "def test_ok():\n    assert True\n"


def _env(tmp_path):
    env = ProotTerminalEnvironment.__new__(ProotTerminalEnvironment)
    env.work_dir = tmp_path
    env.proot_bin = "proot"
    env.rootfs = tmp_path / "rootfs"
    env.rootfs.mkdir()
    env.verifier_timeout = 5.0
    env.clean_verifier = True
    env._session_binds = lambda: [f"{tmp_path}/home:/home/user", f"{tmp_path}/tmp:/tmp"]
    env._verifier_integrity = None
    env._verifier_source_sha256 = None
    env._abort_reason = None
    env._disk_exceeded = threading.Event()
    env._verifier_integrity_override = None
    return env


def _bind_source(argv):
    return next(
        Path(spec.split(":", 1)[0])
        for spec in argv
        if ":/tmp/.pl_verifier_" in spec
    )


def test_clean_verifier_uses_private_bind_clean_env_and_hash(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(proot_env, "_force_rmtree", lambda path: shutil.rmtree(path, ignore_errors=True))

    class Process:
        returncode = 0

        def communicate(self, timeout):
            seen["timeout"] = timeout
            seen["source"] = _bind_source(seen["argv"])
            assert seen["source"].stat().st_mode & 0o077 == 0
            assert seen["source"].joinpath("test_final_state.py").read_text() == _TEST_SOURCE
            return "1 passed in 0.01s\n", None

    def fake_popen(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return Process()

    monkeypatch.setattr(proot_env.subprocess, "Popen", fake_popen)
    env = _env(tmp_path)

    ok, output, abort_kind = env._run_clean_pytest(_TEST_SOURCE, "test_final_state.py")

    assert ok
    assert output.startswith("1 passed")
    assert abort_kind is None
    assert env.verifier_metadata() == {
        "verifier_integrity": "ok",
        "verifier_source_sha256": hashlib.sha256(_TEST_SOURCE.encode()).hexdigest(),
    }
    command_line = " ".join(seen["argv"])
    assert "/usr/bin/python3" in command_line
    assert "--noconftest" in command_line
    assert "/usr/bin/python3 -I -m pytest" in command_line
    assert "PYTHONNOUSERSITE=1" in command_line
    assert "-w" in seen["argv"]
    assert "/home/user" in seen["argv"]
    assert seen["kwargs"]["start_new_session"] is True
    assert seen["kwargs"]["stderr"] == subprocess.STDOUT
    assert not seen["source"].exists()


def test_clean_verifier_source_change_is_explicit_fail_override(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(proot_env, "_force_rmtree", lambda path: shutil.rmtree(path, ignore_errors=True))

    class Process:
        returncode = 0

        def communicate(self, timeout):
            seen["source"] = _bind_source(seen["argv"])
            seen["source"].joinpath("test_final_state.py").write_text("def test_fake(): pass")
            return "1 passed\n", None

    def fake_popen(argv, **kwargs):
        seen["argv"] = argv
        return Process()

    monkeypatch.setattr(proot_env.subprocess, "Popen", fake_popen)
    env = _env(tmp_path)

    ok, output, abort_kind = env._run_clean_pytest(_TEST_SOURCE, "test_final_state.py")

    assert not ok
    assert "source changed" in output
    assert abort_kind is None
    assert env.verifier_metadata()["verifier_integrity"] == "tampered"
    assert env.verifier_metadata()["verifier_integrity_override"] == "fail"


def test_clean_verifier_timeout_is_infrastructure_drop(tmp_path, monkeypatch):
    monkeypatch.setattr(proot_env, "_force_rmtree", lambda path: shutil.rmtree(path, ignore_errors=True))
    class Process:
        returncode = None

        def communicate(self, timeout):
            raise subprocess.TimeoutExpired("proot", timeout, output="partial")

    monkeypatch.setattr(proot_env.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(proot_env, "_terminate_process_group", lambda proc: None)
    env = _env(tmp_path)

    ok, output, abort_kind = env._run_clean_pytest(_TEST_SOURCE, "test_final_state.py")

    assert not ok
    assert "verifier timed out" in output
    assert abort_kind == "verifier_integrity"
    assert env.verifier_metadata()["verifier_integrity"] == "error"
    assert env.verifier_metadata()["verifier_integrity_override"] == "drop"


def test_finish_includes_integrity_metadata_only_when_server_provides_it(tmp_path):
    session = TerminalSession.__new__(TerminalSession)
    session._env = SimpleNamespace(
        run_final_tests=lambda text: (True, "ok", 1, 1, None),
        verifier_metadata=lambda: {"verifier_integrity": "ok", "verifier_source_sha256": "abc"},
    )
    session._final_test = _TEST_SOURCE
    session.max_observation_chars = 100

    result = session.finish()

    assert result["passed"] is True
    assert result["verifier_integrity"] == "ok"
    assert result["verifier_source_sha256"] == "abc"


def test_finish_default_mode_has_no_integrity_fields():
    session = TerminalSession.__new__(TerminalSession)
    session._env = SimpleNamespace(
        run_final_tests=lambda text: (True, "ok", 1, 1, None),
        verifier_metadata=lambda: {},
    )
    session._final_test = _TEST_SOURCE
    session.max_observation_chars = 100

    result = session.finish()

    assert "verifier_integrity" not in result
    assert "verifier_source_sha256" not in result


def test_clean_verifier_respects_existing_resource_abort(tmp_path, monkeypatch):
    env = _env(tmp_path)
    env._disk_exceeded.set()
    monkeypatch.setattr(proot_env.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not launch")))

    ok, output, abort_kind = env._run_clean_pytest(_TEST_SOURCE, "test_final_state.py")

    assert not ok
    assert "session aborted" in output
    assert abort_kind == "verifier_integrity"
    assert env.verifier_metadata()["verifier_integrity_override"] == "drop"
