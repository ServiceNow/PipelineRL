"""Per-worker shared resources for privacy_hopqa rollouts."""

import threading
from pathlib import Path

from .local_index import StaticLocalIndex
from .paths import DEFAULT_LOCAL_INDEX_ROOT

_BROWSECOMP_CACHE: dict[tuple, object] = {}
_LOCAL_INDEX_CACHE: dict[tuple, StaticLocalIndex] = {}
_CACHE_LOCK = threading.Lock()


def _browsecomp_key(config, device: str) -> tuple:
    return (
        config.browsecomp_index_glob,
        config.browsecomp_model_name,
        config.browsecomp_corpus,
        int(config.browsecomp_top_k),
        int(config.browsecomp_max_chars),
        device,
    )


def get_shared_browsecomp(config, device: str = "cpu"):
    key = _browsecomp_key(config, device)
    with _CACHE_LOCK:
        tool = _BROWSECOMP_CACHE.get(key)
        if tool is None:
            # Local BrowseComp fallback is rarely used for hopqa, so keep the import lazy.
            from pipelinerl.domains.privacy_agent.drbench.agent_tools.browsecomp_search_tool import BrowseCompSearchTool

            tool = BrowseCompSearchTool(config=config, vector_store=None, device=device)
            _BROWSECOMP_CACHE[key] = tool
        return tool


def get_local_index_path(task_id: str, index_root: str | Path | None = None) -> Path:
    root = Path(index_root or DEFAULT_LOCAL_INDEX_ROOT).expanduser()
    return root / task_id


def get_static_local_index(
    task_id: str,
    index_root: str | Path | None = None,
    embedding_model: str | None = None,
    embedding_provider: str | None = None,
    semantic_threshold: float = 0.7,
    mmap_mode: str | None = "r",
) -> StaticLocalIndex:
    index_path = get_local_index_path(task_id=task_id, index_root=index_root)
    key = (
        str(index_path.resolve()),
        embedding_model,
        embedding_provider,
        semantic_threshold,
        mmap_mode,
    )
    with _CACHE_LOCK:
        index = _LOCAL_INDEX_CACHE.get(key)
        if index is None:
            if not index_path.exists():
                raise FileNotFoundError(
                    f"Missing prebuilt local index for task {task_id} at {index_path}. "
                    "Run pipelinerl.domains.privacy_agent.build_local_indices first."
                )
            index = StaticLocalIndex.load(
                storage_dir=index_path,
                embedding_model=embedding_model,
                embedding_provider=embedding_provider,
                semantic_threshold=semantic_threshold,
                mmap_mode=mmap_mode,
            )
            _LOCAL_INDEX_CACHE[key] = index
        return index
