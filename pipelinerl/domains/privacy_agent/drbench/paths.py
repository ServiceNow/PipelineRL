import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PRIVACY_AGENT_ROOT = REPO_ROOT / ".privacy_agent"
DEFAULT_INPUTS_ROOT = DEFAULT_PRIVACY_AGENT_ROOT / "inputs"
DEFAULT_BROWSECOMP_ROOT = DEFAULT_PRIVACY_AGENT_ROOT / "browsecomp"


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser() if value else default


DEFAULT_ANNOTATIONS_PATH = _env_path(
    "PRIVACY_AGENT_ANNOTATIONS_PATH",
    DEFAULT_INPUTS_ROOT / "annotations_2026-04-06.json",
)
DEFAULT_CURATED_CHAINS_PATH = _env_path(
    "PRIVACY_AGENT_CURATED_PATH",
    DEFAULT_INPUTS_ROOT / "curated_chains.jsonl",
)
DEFAULT_TASK_DATA_ROOT = _env_path(
    "PRIVACY_AGENT_TASK_DATA_ROOT",
    DEFAULT_INPUTS_ROOT / "tasks",
)
DEFAULT_LOCAL_INDEX_ROOT = _env_path(
    "PRIVACY_AGENT_LOCAL_INDEX_ROOT",
    DEFAULT_PRIVACY_AGENT_ROOT / "local_indices",
)
DEFAULT_BROWSECOMP_INDEX_GLOB = os.getenv(
    "PRIVACY_AGENT_BROWSECOMP_INDEX_GLOB",
    str(DEFAULT_BROWSECOMP_ROOT / "indexes" / "qwen3-embedding-4b" / "corpus.shard*_of_*.pkl"),
)
DEFAULT_BROWSECOMP_CORPUS = "Tevatron/browsecomp-plus-corpus"
