"""Simple file-based synchronization for distributed test processes."""

import time
from pathlib import Path


class SyncPoint:
    """File-based synchronization point for coordinating subprocesses."""

    def __init__(self, sync_dir: Path, name: str):
        """Create a sync point.

        Args:
            sync_dir: Directory for sync files
            name: Name of this sync point (e.g., "baseline_done")
        """
        self.sync_file = sync_dir / f"{name}.sync"
        self.sync_dir = sync_dir

    def signal(self):
        """Signal that this point is reached."""
        self.sync_file.touch()
        # Force filesystem sync to ensure file is visible immediately
        import os
        fd = os.open(str(self.sync_file.parent), os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
        print(f"[Sync] Signaled: {self.sync_file.name}")

    def wait(self, timeout: float = 60):
        """Wait for this point to be signaled.

        Args:
            timeout: Maximum time to wait in seconds

        Raises:
            TimeoutError: If sync point not reached within timeout
        """
        start = time.time()
        while not self.sync_file.exists():
            if time.time() - start > timeout:
                raise TimeoutError(
                    f"Timeout waiting for sync point: {self.sync_file.name}"
                )
            time.sleep(0.1)
        print(f"[Sync] Reached: {self.sync_file.name}")

    def clear(self):
        """Clear this sync point."""
        if self.sync_file.exists():
            self.sync_file.unlink()


def create_sync_dir(base_dir: Path) -> Path:
    """Create a directory for sync files.

    Args:
        base_dir: Base temporary directory

    Returns:
        Path to sync directory
    """
    sync_dir = base_dir / "sync"
    sync_dir.mkdir(exist_ok=True)
    return sync_dir


def write_weight_update_request(sync_dir: Path, request):
    """Write WeightUpdateRequest to JSON file.

    Args:
        sync_dir: Sync directory
        request: WeightUpdateRequest object
    """
    import json

    request_file = sync_dir / "weight_update_request.json"
    with open(request_file, "w") as f:
        json.dump(request.model_dump(), f)
    print(f"[Sync] Wrote weight update request to {request_file.name}")


def read_weight_update_request(sync_dir: Path):
    """Read WeightUpdateRequest from JSON file.

    Args:
        sync_dir: Sync directory

    Returns:
        WeightUpdateRequest object
    """
    import json
    from pipelinerl.finetune_loop import WeightUpdateRequest

    request_file = sync_dir / "weight_update_request.json"

    # Wait for file to exist
    import time
    timeout = 60
    start = time.time()
    while not request_file.exists():
        if time.time() - start > timeout:
            raise TimeoutError(f"Timeout waiting for {request_file.name}")
        time.sleep(0.1)

    with open(request_file, "r") as f:
        data = json.load(f)

    print(f"[Sync] Read weight update request from {request_file.name}")
    return WeightUpdateRequest(**data)
