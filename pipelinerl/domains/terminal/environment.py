"""Per-task terminal sandbox session, backed by a proot environment.

A ``TerminalSession`` owns one ``ProotTerminalEnvironment`` for a single task.
``TerminalEnvironmentServer`` manages many of these concurrently and exposes them
over HTTP. No TapeAgents dependency: actions and observations are plain dicts on
the wire.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .proot_env import ProotTerminalEnvironment

logger = logging.getLogger(__name__)


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    return f"{text[:head]}\n... [output truncated] ...\n{text[-(max_chars - head):]}"


class TerminalSession:
    """One proot sandbox bound to a single task."""

    def __init__(
        self,
        bases_dir: Path,
        proot_bin: str,
        nameserver: str,
        command_timeout: float,
        verifier_timeout: float,
        max_observation_chars: int,
        check_initial_state: bool,
        cache_dir: Optional[str | Path] = None,
    ):
        self.bases_dir = bases_dir
        self.proot_bin = proot_bin
        self.nameserver = nameserver
        self.command_timeout = command_timeout
        self.verifier_timeout = verifier_timeout
        self.max_observation_chars = max_observation_chars
        self.check_initial_state = check_initial_state
        self.cache_dir = cache_dir

        self._env: Optional[ProotTerminalEnvironment] = None
        self._final_test: str = ""

    def _resolve_base_rootfs(self, tmax_domain: str) -> Optional[Path]:
        for name in (f"base_{tmax_domain}", "base_software_engineering", "base_intricate"):
            candidate = self.bases_dir / name
            if candidate.exists():
                return candidate
        return None

    def start(self, task: dict) -> dict:
        """Build the task rootfs and start the persistent session.

        Returns flags (``build_ok`` / ``started`` / ``init_ok``) so the caller can
        record a normal negative outcome instead of treating an unbuildable task
        as a server error.
        """
        base = self._resolve_base_rootfs(task["tmax_domain"])
        if base is None:
            logger.error("no base rootfs for domain %s under %s", task["tmax_domain"], self.bases_dir)
            return {"build_ok": False, "started": False, "init_ok": False}

        self._final_test = task["test_final_state"]
        self._env = ProotTerminalEnvironment(
            base_rootfs=base,
            proot_bin=self.proot_bin,
            nameserver=self.nameserver,
            verifier_timeout=self.verifier_timeout,
            cache_dir=self.cache_dir,
        )
        build_ok, build_err = self._env.build(task["container_def"])
        if not build_ok:
            logger.warning("build failed for %s: %s", task.get("task_id"), build_err[:200])
            return {"build_ok": False, "started": False, "init_ok": False}

        if not self._env.start():
            return {"build_ok": True, "started": False, "init_ok": False}

        init_ok = True
        if self.check_initial_state:
            init_ok = self._env.run_initial_tests(task["test_initial_state"])
        return {"build_ok": True, "started": True, "init_ok": init_ok}

    def exec(self, command: str) -> dict:
        if self._env is None:
            raise RuntimeError("session not started")
        success, output = self._env.exec(command, timeout=self.command_timeout)
        return {
            "output": truncate(output, self.max_observation_chars) or "(no output)",
            "success": success,
            "exit_code": 0 if success else 1,
        }

    def finish(self) -> dict:
        if self._env is None:
            raise RuntimeError("session not started")
        passed, output = self._env.run_final_tests(self._final_test)
        return {"passed": passed, "output": truncate(output, self.max_observation_chars)}

    def close(self) -> None:
        if self._env is not None:
            self._env.cleanup()
            self._env = None
