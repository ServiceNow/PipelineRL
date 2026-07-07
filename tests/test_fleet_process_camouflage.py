import logging
from pathlib import Path

from pipelinerl.entrypoints import run_environment_fleet


def test_spawned_serve_child_sets_process_comm(monkeypatch, tmp_path):
    comm_path = tmp_path / "comm"

    def fake_instantiate(_cfg):
        class FakeServer:
            def launch(self, port):
                comm_path.write_text(Path("/proc/self/comm").read_text().strip())

        return FakeServer()

    monkeypatch.setattr(run_environment_fleet.hydra.utils, "instantiate", fake_instantiate)

    proc = run_environment_fleet._spawn({"_target_": "unused"}, 7777)
    proc.join(timeout=5)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)

    assert proc.exitcode == 0
    assert comm_path.read_text().strip() == "plenv-serve"


def test_set_process_comm_failure_logs_and_continues(monkeypatch, caplog):
    def fail_cdll(*args, **kwargs):
        raise RuntimeError("ctypes unavailable")

    monkeypatch.setattr(run_environment_fleet.ctypes, "CDLL", fail_cdll)

    with caplog.at_level(logging.WARNING):
        run_environment_fleet._set_process_comm("plenv-fleet")

    assert "failed to set process comm to plenv-fleet" in caplog.text
    assert "ctypes unavailable" in caplog.text
