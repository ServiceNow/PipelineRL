"""Per-worker shared resources for privacy_hopqa rollouts."""

import threading
from pathlib import Path

from .local_index import StaticLocalIndex
from .paths import DEFAULT_LOCAL_INDEX_ROOT

_LOCAL_INDEX_CACHE: dict[tuple, StaticLocalIndex] = {}
_CACHE_LOCK = threading.Lock()


def get_local_index_path(task_id: str, index_root: str | Path | None = None) -> Path:
    root = Path(index_root or DEFAULT_LOCAL_INDEX_ROOT).expanduser()
    return root / task_id


def get_static_local_index(
    task_id: str,
    index_root: str | Path | None = None,
    embedding_model: str | None = None,
    embedding_provider: str | None = None,
    embedding_base_url: str | None = None,
    embedding_api_key: str | None = None,
    embedding_device: str | None = None,
    semantic_threshold: float = 0.7,
    mmap_mode: str | None = "r",
) -> StaticLocalIndex:
    index_path = get_local_index_path(task_id=task_id, index_root=index_root)
    key = (
        str(index_path.resolve()),
        embedding_model,
        embedding_provider,
        embedding_base_url,
        embedding_device,
        semantic_threshold,
        mmap_mode,
    )
    with _CACHE_LOCK:
        index = _LOCAL_INDEX_CACHE.get(key)
        if index is None:
            if not index_path.exists():
                raise FileNotFoundError(
                    f"Missing prebuilt local index for task {task_id} at {index_path}. "
                    "Use the MosaicProject helper to build local indices, or set use_remote_local_search=true."
                )
            index = StaticLocalIndex.load(
                storage_dir=index_path,
                embedding_model=embedding_model,
                embedding_provider=embedding_provider,
                embedding_base_url=embedding_base_url,
                embedding_api_key=embedding_api_key,
                embedding_device=embedding_device,
                semantic_threshold=semantic_threshold,
                mmap_mode=mmap_mode,
            )
            _LOCAL_INDEX_CACHE[key] = index
        return index
