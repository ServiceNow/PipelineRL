"""Runtime configuration for the privacy_hopqa domain."""

import os
from dataclasses import dataclass
from pathlib import Path

from .paths import (
    DEFAULT_BROWSECOMP_CORPUS,
    DEFAULT_BROWSECOMP_INDEX_GLOB,
)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


def _load_env_file() -> None:
    if load_dotenv is None:
        return

    env_path_value = os.getenv("PRIVACY_AGENT_ENV_FILE")
    if not env_path_value:
        return

    env_path = Path(env_path_value).expanduser()
    if env_path.is_file():
        load_dotenv(env_path, override=True)


_load_env_file()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
VLLM_API_KEY = os.getenv("VLLM_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")

VLLM_API_URL = os.getenv("VLLM_API_URL")
VLLM_EMBEDDING_URL = os.getenv("VLLM_EMBEDDING_URL")
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
DRBENCH_DATA_DIR = os.getenv("DRBENCH_DATA_DIR")
DRBENCH_DOCKER_IMAGE = os.getenv("DRBENCH_DOCKER_IMAGE", "drbench-services")
DRBENCH_DOCKER_TAG = os.getenv("DRBENCH_DOCKER_TAG", "latest")

DRBENCH_LLM_PROVIDER = os.getenv("DRBENCH_LLM_PROVIDER", "openai")
DRBENCH_EMBEDDING_PROVIDER = os.getenv("DRBENCH_EMBEDDING_PROVIDER", "huggingface")
DRBENCH_EMBEDDING_MODEL = os.getenv("DRBENCH_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
DRBENCH_EMBEDDING_DEVICE = os.getenv("DRBENCH_EMBEDDING_DEVICE", "cpu")

os.environ.setdefault("DRBENCH_EMBEDDING_PROVIDER", DRBENCH_EMBEDDING_PROVIDER)
os.environ.setdefault("DRBENCH_EMBEDDING_MODEL", DRBENCH_EMBEDDING_MODEL)
os.environ.setdefault("DRBENCH_EMBEDDING_DEVICE", DRBENCH_EMBEDDING_DEVICE)


@dataclass(slots=True)
class RunConfig:
    model: str | None = None
    max_iterations: int = 10
    concurrent_actions: int = 3
    semantic_threshold: float = 0.7
    run_dir: Path | None = None
    log_searches: bool = True
    log_prompts: bool = True
    log_generations: bool = True
    verbose: bool = False
    no_web: bool = False
    llm_provider: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    report_style: str = "research_report"
    browsecomp_enabled: bool = False
    browsecomp_index_glob: str = DEFAULT_BROWSECOMP_INDEX_GLOB
    browsecomp_model_name: str = "Qwen/Qwen3-Embedding-4B"
    browsecomp_corpus: str = DEFAULT_BROWSECOMP_CORPUS
    browsecomp_top_k: int = 5
    browsecomp_max_chars: int = 8000
    local_document_search_mode: str = "retrieve_only"

    @property
    def local_document_search_synthesis(self) -> bool:
        return self.local_document_search_mode == "synthesize"

    def get_llm_provider(self) -> str:
        return self.llm_provider or DRBENCH_LLM_PROVIDER

    def get_embedding_provider(self) -> str:
        return self.embedding_provider or DRBENCH_EMBEDDING_PROVIDER

    def get_embedding_model(self) -> str | None:
        return self.embedding_model or DRBENCH_EMBEDDING_MODEL

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "max_iterations": self.max_iterations,
            "concurrent_actions": self.concurrent_actions,
            "semantic_threshold": self.semantic_threshold,
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "log_searches": self.log_searches,
            "log_prompts": self.log_prompts,
            "log_generations": self.log_generations,
            "verbose": self.verbose,
            "no_web": self.no_web,
            "llm_provider": self.get_llm_provider(),
            "embedding_provider": self.get_embedding_provider(),
            "embedding_model": self.get_embedding_model(),
            "report_style": self.report_style,
            "browsecomp_enabled": self.browsecomp_enabled,
            "browsecomp_index_glob": self.browsecomp_index_glob,
            "browsecomp_model_name": self.browsecomp_model_name,
            "browsecomp_corpus": self.browsecomp_corpus,
            "browsecomp_top_k": self.browsecomp_top_k,
            "browsecomp_max_chars": self.browsecomp_max_chars,
            "local_document_search_mode": self.local_document_search_mode,
        }
