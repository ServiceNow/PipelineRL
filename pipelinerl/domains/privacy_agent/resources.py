"""Per-worker shared resources for privacy_agent rollouts."""


import threading
from pathlib import Path

from .drbench.agent_tools.browsecomp_search_tool import BrowseCompSearchTool
from .drbench.paths import DEFAULT_LOCAL_INDEX_ROOT
from .helper_client import PrivacyHelperClient
from .local_index import StaticLocalIndex

_BROWSECOMP_CACHE: dict[tuple, object] = {}
_HELPER_CLIENT_CACHE: dict[tuple[str, float], PrivacyHelperClient] = {}
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
            # BrowseComp indices are expensive to load, so each rollout worker keeps
            # one shared instance per configuration tuple.
            tool = BrowseCompSearchTool(config=config, vector_store=None, device=device)
            _BROWSECOMP_CACHE[key] = tool
        return tool


def get_shared_helper_client(base_url: str, timeout_s: float = 30.0) -> PrivacyHelperClient:
    key = (base_url.rstrip("/"), float(timeout_s))
    with _CACHE_LOCK:
        client = _HELPER_CLIENT_CACHE.get(key)
        if client is None:
            client = PrivacyHelperClient(base_url=base_url, timeout_s=timeout_s)
            client.health()
            _HELPER_CLIENT_CACHE[key] = client
        return client


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
            # Each worker opens a task index at most once and then reuses the
            # same read-only embedding matrix across many rollouts.
            index = StaticLocalIndex.load(
                storage_dir=index_path,
                embedding_model=embedding_model,
                embedding_provider=embedding_provider,
                semantic_threshold=semantic_threshold,
                mmap_mode=mmap_mode,
            )
            _LOCAL_INDEX_CACHE[key] = index
        return index
