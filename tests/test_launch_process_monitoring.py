from pathlib import Path

from pipelinerl import launch


class FakeTrainerState:
    def __init__(self, exp_path: Path, use_fast_llm: bool, weight_broadcast: bool):
        self.training_done = True
        self.started = False

    def start_listening(self):
        self.started = True

    def wait_for_training_done(self, timeout: float | None = None):
        raise AssertionError("training_done was already set")


class FakeProcessHandle:
    def __init__(self, pid: int, kind: str, poll_results: list[int | None]):
        self.pid = pid
        self.args = [kind]
        self._poll_results = poll_results
        self.terminated = False
        self.waited = False

    def poll(self):
        if self.terminated:
            return -15
        if self._poll_results:
            return self._poll_results.pop(0)
        return None

    def wait(self):
        self.waited = True
        return -15 if self.terminated else 0


def test_watch_processes_stops_remaining_helpers_after_training_completion(monkeypatch, tmp_path):
    handles = {
        100: FakeProcessHandle(100, "finetune", [0]),
        101: FakeProcessHandle(101, "redis", [None]),
        102: FakeProcessHandle(102, "actor", [None]),
    }
    terminated_pids = []

    def terminate_with_children(pid: int):
        terminated_pids.append(pid)
        handles[pid].terminated = True

    monkeypatch.setattr(launch, "TrainerState", FakeTrainerState)
    monkeypatch.setattr(launch, "terminate_with_children", terminate_with_children)
    monkeypatch.setattr(launch.time, "sleep", lambda seconds: None)

    processes = [
        launch.LaunchedProcess(kind="finetune", handle=handles[100]),
        launch.LaunchedProcess(kind="redis", handle=handles[101]),
        launch.LaunchedProcess(kind="actor", handle=handles[102]),
    ]

    launch.watch_processes_running(tmp_path, processes, use_fast_llm=True)

    assert terminated_pids == [101, 102]
    assert not handles[100].terminated
    assert handles[101].waited
    assert handles[102].waited
