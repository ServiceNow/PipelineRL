import logging

import pandas as pd

from pipelinerl.domains.terminal.load_tasks import load_problems


def _row(task_id: str) -> dict:
    return {
        "domain": "file_operations",
        "task_id": task_id,
        "description": f"task {task_id}",
        "container_def": "Bootstrap: docker",
        "test_initial_state": "def test_initial(): pass",
        "test_final_state": "def test_final(): pass",
        "task_complexity": "moderate task",
        "command_complexity": "simple",
    }


def test_load_problems_filters_task_blocklist(monkeypatch, caplog) -> None:
    rows = [_row("keep-1"), _row("drop-1"), _row("keep-2")]

    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **_: "tasks.parquet")
    monkeypatch.setattr("pandas.read_parquet", lambda path: pd.DataFrame(rows))

    with caplog.at_level(logging.INFO, logger="pipelinerl.domains.terminal.load_tasks"):
        problems = load_problems(
            ["terminal_train"],
            task_blocklist=["drop-1"],
            train_ratio=1.0,
            seed=0,
        )

    assert {problem["task_id"] for problem in problems} == {"keep-1", "keep-2"}
    assert all(problem["id"] == i for i, problem in enumerate(problems))
    assert "task blocklist: dropped 1/3 tasks" in caplog.text
