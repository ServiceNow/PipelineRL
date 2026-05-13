"""Configuration helpers for the privacy_hopqa domain."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from .paths import (
    DEFAULT_ANNOTATIONS_PATH,
    DEFAULT_BROWSECOMP_CORPUS,
    DEFAULT_BROWSECOMP_INDEX_GLOB,
    DEFAULT_CURATED_CHAINS_PATH,
    DEFAULT_LOCAL_INDEX_ROOT,
    DEFAULT_TASK_DATA_ROOT,
)


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, DictConfig):
        return dict(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _extract_settings_block(cfg: DictConfig | Mapping[str, Any] | None) -> dict[str, Any]:
    root = _to_plain_dict(cfg)
    if not root:
        return {}

    direct = _to_plain_dict(root.get("privacy_hopqa"))
    if direct:
        return direct

    dataset_loader_params = _to_plain_dict(root.get("dataset_loader_params"))
    per_domain = _to_plain_dict(dataset_loader_params.get("per_domain_params"))
    domain_block = _to_plain_dict(per_domain.get("privacy_hopqa"))
    if domain_block:
        return domain_block

    return dataset_loader_params


@dataclass(slots=True)
class PrivacyHopQASettings:
    annotations_path: Path = DEFAULT_ANNOTATIONS_PATH
    curated_path: Path = DEFAULT_CURATED_CHAINS_PATH
    sample_size: int = 20
    max_examples: int | None = None
    task_data_root: Path = DEFAULT_TASK_DATA_ROOT
    local_index_root: Path = DEFAULT_LOCAL_INDEX_ROOT
    local_index_mmap_mode: str = "r"
    helper_service_url: str | None = None
    helper_timeout_s: float = 30.0
    generation_context_limit_tokens: int = 16384
    generation_context_margin_tokens: int = 128
    hop_plan_max_tokens: int = 4096
    doc_choose_max_tokens: int = 4096
    doc_read_max_tokens: int = 4096
    hop_resolve_max_tokens: int = 4096
    reader_min_answer_confidence: float = 0.75
    use_remote_local_search: bool = True
    use_remote_browsecomp: bool = True
    capture_mode: str = "all_calls"
    answer_match_f1_threshold: float = 0.75
    answer_match_mode: str = "exact_or_f1"
    reward_mode: str = "all_hops_correct"
    max_iterations: int = 21
    iteration_budget_per_hop: int = 3
    hop_plan_attempts: int = 1
    enable_generic_retrieval_fallback: bool = False
    enable_local_search_bootstrap: bool = False
    max_parallel_retrieval_actions: int = 3
    retrieval_top_k: int = 5
    choose_top_k: int = 3
    max_candidate_cards: int = 20
    chooser_excerpt_chars: int = 500
    reader_excerpt_chars: int = 2500
    retrieval_context_mode: str = "windowed_evidence"
    retrieval_source_chars: int = 0
    retrieval_chunk_chars: int = 4000
    retrieval_chunk_overlap_chars: int = 800
    reader_window_chars: int = 16000
    reader_window_overlap_chars: int = 1000
    max_windows_per_parent: int = 0
    large_parent_window_policy: str = "top_with_neighbors"
    read_all_windows_max_count: int = 10
    large_parent_top_windows: int = 4
    large_parent_neighbor_windows: int = 1
    large_parent_max_read_windows: int = 0
    max_history_entries: int = 10
    local_search_threshold: float = 0.0
    semantic_threshold: float = 0.7
    browsecomp_enabled: bool = True
    browsecomp_device: str = "cpu"
    browsecomp_index_glob: str = DEFAULT_BROWSECOMP_INDEX_GLOB
    browsecomp_model_name: str = "Qwen/Qwen3-Embedding-4B"
    browsecomp_corpus: str = DEFAULT_BROWSECOMP_CORPUS
    browsecomp_top_k: int = 5
    browsecomp_max_chars: int = 8000
    llm_provider: str = "vllm"
    embedding_provider: str | None = "huggingface"
    embedding_model: str | None = "Qwen/Qwen3-Embedding-4B"
    log_prompts: bool = False
    verbose: bool = False
    no_web: bool = False
    parallel_searches: bool = True
    max_parallel_doc_reads: int = 6

    @classmethod
    def from_cfg(cls, cfg: DictConfig | Mapping[str, Any] | None) -> "PrivacyHopQASettings":
        data = _extract_settings_block(cfg)
        return cls(
            annotations_path=Path(data.get("annotations_path", DEFAULT_ANNOTATIONS_PATH)).expanduser(),
            curated_path=Path(data.get("curated_path", DEFAULT_CURATED_CHAINS_PATH)).expanduser(),
            sample_size=int(data.get("sample_size", 20)),
            max_examples=(int(data["max_examples"]) if data.get("max_examples") is not None else None),
            task_data_root=Path(data.get("task_data_root", DEFAULT_TASK_DATA_ROOT)).expanduser(),
            local_index_root=Path(data.get("local_index_root", DEFAULT_LOCAL_INDEX_ROOT)).expanduser(),
            local_index_mmap_mode=str(data.get("local_index_mmap_mode", "r")),
            helper_service_url=(str(data["helper_service_url"]) if data.get("helper_service_url") is not None else None),
            helper_timeout_s=float(data.get("helper_timeout_s", 30.0)),
            generation_context_limit_tokens=int(data.get("generation_context_limit_tokens", 16384)),
            generation_context_margin_tokens=int(data.get("generation_context_margin_tokens", 128)),
            hop_plan_max_tokens=int(data.get("hop_plan_max_tokens", 4096)),
            doc_choose_max_tokens=int(data.get("doc_choose_max_tokens", 4096)),
            doc_read_max_tokens=int(data.get("doc_read_max_tokens", 4096)),
            hop_resolve_max_tokens=int(data.get("hop_resolve_max_tokens", 4096)),
            reader_min_answer_confidence=float(data.get("reader_min_answer_confidence", 0.75)),
            use_remote_local_search=bool(data.get("use_remote_local_search", True)),
            use_remote_browsecomp=bool(data.get("use_remote_browsecomp", True)),
            capture_mode=str(data.get("capture_mode", "all_calls")),
            answer_match_f1_threshold=float(data.get("answer_match_f1_threshold", 0.75)),
            answer_match_mode=str(data.get("answer_match_mode", "exact_or_f1")),
            reward_mode=str(data.get("reward_mode", "all_hops_correct")),
            max_iterations=int(data.get("max_iterations", 21)),
            iteration_budget_per_hop=int(data.get("iteration_budget_per_hop", 3)),
            hop_plan_attempts=int(data.get("hop_plan_attempts", 1)),
            enable_generic_retrieval_fallback=bool(data.get("enable_generic_retrieval_fallback", False)),
            enable_local_search_bootstrap=bool(data.get("enable_local_search_bootstrap", False)),
            max_parallel_retrieval_actions=int(data.get("max_parallel_retrieval_actions", 3)),
            retrieval_top_k=int(data.get("retrieval_top_k", 5)),
            choose_top_k=int(data.get("choose_top_k", 3)),
            max_candidate_cards=int(data.get("max_candidate_cards", 20)),
            chooser_excerpt_chars=int(data.get("chooser_excerpt_chars", 500)),
            reader_excerpt_chars=int(data.get("reader_excerpt_chars", 2500)),
            retrieval_context_mode=str(data.get("retrieval_context_mode", "windowed_evidence")),
            retrieval_source_chars=int(data.get("retrieval_source_chars", 0)),
            retrieval_chunk_chars=int(data.get("retrieval_chunk_chars", 4000)),
            retrieval_chunk_overlap_chars=int(data.get("retrieval_chunk_overlap_chars", 800)),
            reader_window_chars=int(data.get("reader_window_chars", 16000)),
            reader_window_overlap_chars=int(data.get("reader_window_overlap_chars", 1000)),
            max_windows_per_parent=int(data.get("max_windows_per_parent", 0)),
            large_parent_window_policy=str(data.get("large_parent_window_policy", "top_with_neighbors")),
            read_all_windows_max_count=int(data.get("read_all_windows_max_count", 10)),
            large_parent_top_windows=int(data.get("large_parent_top_windows", 4)),
            large_parent_neighbor_windows=int(data.get("large_parent_neighbor_windows", 1)),
            large_parent_max_read_windows=int(data.get("large_parent_max_read_windows", 0)),
            max_history_entries=int(data.get("max_history_entries", 10)),
            local_search_threshold=float(data.get("local_search_threshold", 0.0)),
            semantic_threshold=float(data.get("semantic_threshold", 0.7)),
            browsecomp_enabled=bool(data.get("browsecomp_enabled", True)),
            browsecomp_device=str(data.get("browsecomp_device", "cpu")),
            browsecomp_index_glob=str(data.get("browsecomp_index_glob", DEFAULT_BROWSECOMP_INDEX_GLOB)),
            browsecomp_model_name=str(data.get("browsecomp_model_name", "Qwen/Qwen3-Embedding-4B")),
            browsecomp_corpus=str(data.get("browsecomp_corpus", DEFAULT_BROWSECOMP_CORPUS)),
            browsecomp_top_k=int(data.get("browsecomp_top_k", data.get("retrieval_top_k", 5))),
            browsecomp_max_chars=int(data.get("browsecomp_max_chars", 8000)),
            llm_provider=str(data.get("llm_provider", "vllm")),
            embedding_provider=(str(data["embedding_provider"]) if data.get("embedding_provider") is not None else None),
            embedding_model=(str(data["embedding_model"]) if data.get("embedding_model") is not None else None),
            log_prompts=bool(data.get("log_prompts", False)),
            verbose=bool(data.get("verbose", False)),
            no_web=bool(data.get("no_web", False)),
            parallel_searches=bool(data.get("parallel_searches", True)),
            max_parallel_doc_reads=int(data.get("max_parallel_doc_reads", 6)),
        )

    def dataset_loader_kwargs(self) -> dict[str, Any]:
        return {
            "annotations_path": str(self.annotations_path),
            "curated_path": str(self.curated_path),
            "sample_size": self.sample_size,
            "max_examples": self.max_examples,
        }
