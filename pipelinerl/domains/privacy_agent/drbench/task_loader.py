"""Minimal task loader used by the privacy_agent domain."""


import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from .paths import DEFAULT_TASK_DATA_ROOT

LOCAL_CORPUS_SKIP_FILENAMES = {"qa_dict.json", "file_dict.json"}


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


def _task_relative_key(task_root: Path, source_path: Path) -> str:
    return source_path.relative_to(task_root).as_posix()


class Task:
    """Thin task wrapper for loading DRBench task configs and private files."""

    def __init__(
        self,
        task_path: Union[str, Path],
        ignore_config: bool = False,
        data_dir: str | Path = DEFAULT_TASK_DATA_ROOT,
    ):
        self.data_dir = Path(data_dir).expanduser()
        self.task_path = resolve_task_path(task_path, data_dir=self.data_dir)

        if ignore_config:
            self.task_config = None
            self.eval_config = None
            self.env_config = None
            return

        config_dir = self.task_path
        task_file = config_dir / "task.json"
        env_file = config_dir / "env.json"
        eval_file = config_dir / "eval.json"

        if not task_file.exists():
            raise FileNotFoundError(f"Task configuration file not found at {task_file}")

        self.task_config = _read_json(task_file)
        self.env_config = _read_json(env_file) if env_file.exists() else {"env_files": []}
        self.eval_config = _read_json(eval_file) if eval_file.exists() else None

    def get_task_and_eval(self) -> Tuple[Dict, Optional[Dict]]:
        return self.task_config, self.eval_config

    def get_task_config(self) -> Dict:
        return self.task_config

    def get_eval_config(self) -> Optional[Dict]:
        return self.eval_config

    def get_path(self) -> str:
        return str(self.task_path)

    def get_id(self) -> str:
        return self.task_config["task_id"]

    def get_env_files(self) -> Dict[str, Path]:
        env_files = {}
        task_root = self.task_path.parent
        for env_file in self.env_config.get("env_files", []):
            source = env_file.get("source")
            if not source:
                continue
            source_path = resolve_task_path(source, data_dir=self.data_dir)
            env_files[_task_relative_key(task_root, source_path)] = source_path
        return env_files

    def get_local_files_list(self) -> list[str]:
        file_paths = [str(path) for path in self.get_env_files().values()]
        for file_path in file_paths:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Task file does not exist: {file_path}")
        return file_paths

    def get_all_task_files_list(self, prefer_md: bool = True) -> list[str]:
        files_dir = self.task_path.parent / "files"
        if not files_dir.exists():
            raise FileNotFoundError(f"Task files directory does not exist: {files_dir}")

        file_paths: list[str] = []
        for path in sorted(files_dir.rglob("*")):
            if not path.is_file() or path.name in LOCAL_CORPUS_SKIP_FILENAMES:
                continue
            if prefer_md and path.suffix.lower() == ".pdf" and path.with_suffix(".md").exists():
                continue
            file_paths.append(str(path))
        return file_paths
