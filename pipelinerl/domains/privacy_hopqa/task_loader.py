"""Minimal task loader used by the privacy_hopqa domain."""


import json
from pathlib import Path
from typing import Dict, Union

from .paths import DEFAULT_TASK_DATA_ROOT


def resolve_task_path(path_like: str | Path, data_dir: str | Path = DEFAULT_TASK_DATA_ROOT) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    if path.parts[:3] == ("drbench", "data", "tasks"):
        path = Path(*path.parts[3:]) if len(path.parts) > 3 else Path("")
    elif path.parts[:2] == ("drbench", "data"):
        path = Path(*path.parts[2:]) if len(path.parts) > 2 else Path("")
    elif path.parts[:2] == ("data", "tasks"):
        path = Path(*path.parts[2:]) if len(path.parts) > 2 else Path("")
    return Path(data_dir).expanduser() / path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Task:
    """Thin wrapper that loads a DRBench task's task.json config."""

    def __init__(self, task_path: Union[str, Path], data_dir: str | Path = DEFAULT_TASK_DATA_ROOT):
        self.data_dir = Path(data_dir).expanduser()
        self.task_path = resolve_task_path(task_path, data_dir=self.data_dir)
        task_file = self.task_path / "task.json"
        if not task_file.exists():
            raise FileNotFoundError(f"Task configuration file not found at {task_file}")
        self.task_config = _read_json(task_file)

    def get_task_config(self) -> Dict:
        return self.task_config
