import base64
import io
import logging
import shlex
import tarfile
import time
from pathlib import Path

from cube.container import Container, ContainerBackend, ContainerConfig

logger = logging.getLogger(__name__)


class ContainerEnvironment:
    """
    Wraps a single container as a terminal environment for one task.

    A fresh container is launched in start_task() using the task's ContainerConfig
    and stopped in release(). No container is held between tasks.
    """

    def __init__(self, backend: ContainerBackend) -> None:
        self.backend = backend
        self.container: Container | None = None
        self._task_id: str = ""

    def start_task(self, task_data: dict) -> dict:
        """Launch a fresh container and prepare it for the task."""
        self._task_id = task_data.get("id", "unknown")
        container_config = ContainerConfig(**task_data.get("container_config", {}))
        logger.info(f"[{self._task_id}] Launching container (image={container_config.image})")

        t0 = time.monotonic()
        self.container = self.backend.launch(container_config)
        logger.info(
            f"[{self._task_id}] Container {self.container.id} running "
            f"(launch took {time.monotonic() - t0:.1f}s)"
        )

        self.container.exec("rm -rf /app && mkdir -p /app", timeout=30)

        if archive_b64 := task_data.get("archive_b64"):
            logger.debug(f"[{self._task_id}] Uploading task archive to /app")
            self._upload_archive_subset(
                archive_b64,
                target_dir="/app",
                skip_prefixes=("tests/", "solution/", "task.toml", "instruction.md"),
            )

        for cmd in task_data.get("setup_commands", []):
            logger.debug(f"[{self._task_id}] Setup: {cmd}")
            result = self.container.exec(cmd)
            if result.exit_code != 0:
                logger.warning(
                    f"[{self._task_id}] Setup command failed (exit {result.exit_code}): {cmd}\n{result.stderr}"
                )

        logger.info(f"[{self._task_id}] Task ready in container {self.container.id}")
        return {
            "task_id": self._task_id,
            "description": task_data.get("description", ""),
        }

    def exec(self, command: str, timeout: int | None = None) -> dict:
        """Run a shell command in the container and return stdout/stderr/exit_code."""
        assert self.container is not None, "No active container"
        logger.debug(f"[{self._task_id}] exec: {command!r}")
        result = self.container.exec(command, timeout=timeout)
        logger.debug(
            f"[{self._task_id}] exec done in {result.duration_seconds:.1f}s "
            f"(exit={result.exit_code}): {command!r}"
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "duration_seconds": result.duration_seconds,
        }

    def evaluate(self, archive_b64: str, test_timeout_sec: int = 900) -> dict:
        """Upload tests from the archive, run test.sh, and return the reward."""
        assert self.container is not None, "No active container"
        logger.info(f"[{self._task_id}] Running evaluation (timeout={test_timeout_sec}s)")

        self.container.exec("mkdir -p /tests /logs/verifier")
        logger.debug(f"[{self._task_id}] Uploading test archive")
        self._upload_archive_subset(archive_b64, target_dir="/", include_prefix="tests/")
        self.container.exec("chmod +x /tests/test.sh")

        t0 = time.monotonic()
        self.container.exec("cd /app && bash /tests/test.sh", timeout=test_timeout_sec)
        logger.debug(f"[{self._task_id}] test.sh finished in {time.monotonic() - t0:.1f}s")

        reward_result = self.container.exec("cat /logs/verifier/reward.txt 2>/dev/null || echo 0")
        try:
            reward = float(reward_result.stdout.strip().split()[0])
        except (ValueError, IndexError):
            reward = 0.0

        logger.info(f"[{self._task_id}] Evaluation complete: reward={reward}")
        return {"reward": reward}

    def release(self) -> None:
        """Stop the container. Called when the task session ends."""
        if self.container is not None:
            logger.info(f"[{self._task_id}] Stopping container {self.container.id}")
            try:
                self.container.stop()
                logger.info(f"[{self._task_id}] Container {self.container.id} stopped")
            except Exception as e:
                logger.warning(f"[{self._task_id}] Error stopping container {self.container.id}: {e}")
            self.container = None

    # ── Private helpers ────────────────────────────────────────────

    def _upload_archive_subset(
        self,
        archive_b64: str,
        target_dir: str,
        skip_prefixes: tuple[str, ...] = (),
        include_prefix: str | None = None,
    ) -> None:
        archive_bytes = base64.b64decode(archive_b64)
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue

                if include_prefix is not None:
                    if not member.name.startswith(include_prefix):
                        continue
                    rel = member.name[len(include_prefix):]
                else:
                    if any(member.name.startswith(p) for p in skip_prefixes):
                        continue
                    rel = member.name

                f = tar.extractfile(member)
                if f is None:
                    continue
                content = f.read()

                remote_path = f"{target_dir.rstrip('/')}/{rel}"
                parent = shlex.quote(str(Path(remote_path).parent))
                remote = shlex.quote(remote_path)
                b64 = base64.b64encode(content).decode("ascii")
                logger.debug(f"[{self._task_id}] Uploading {remote_path}")
                self.container.exec(f"mkdir -p {parent}")  # type: ignore[union-attr]
                self.container.exec(  # type: ignore[union-attr]
                    f'printf "%s" {shlex.quote(b64)} | base64 -d > {remote}'
                )
