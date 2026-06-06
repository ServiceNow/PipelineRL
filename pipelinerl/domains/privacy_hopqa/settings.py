"""Configuration helpers for the privacy_hopqa domain."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from .paths import DEFAULT_LOCAL_INDEX_ROOT, DEFAULT_TASK_DATA_ROOT


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


def _optional_path(value: Any, default: Path | None = None) -> Path | None:
    if value is None:
        return default
    if str(value).strip().lower() in {"", "none", "null"}:
        return None
    return Path(value).expanduser()


@dataclass(slots=True)
class PrivacyHopQASettings:
    task_data_root: Path = DEFAULT_TASK_DATA_ROOT
    local_index_root: Path | None = DEFAULT_LOCAL_INDEX_ROOT
    local_index_mmap_mode: str = "r"
    helper_service_url: str | None = None
    privacy_reward_service_url: str | None = None
    helper_timeout_s: float = 180.0
    privacy_reward_timeout_s: float = 30.0
    generation_context_limit_tokens: int = 16384
    generation_context_margin_tokens: int = 128
    hop_plan_max_tokens: int = 4096
    doc_choose_max_tokens: int = 4096
    doc_read_max_tokens: int = 4096
    hop_resolve_max_tokens: int = 4096
    reader_min_answer_confidence: float = 0.75
    use_remote_local_search: bool = True
    use_remote_browsecomp: bool = True
    capture_mode: str = "planning_only"
    answer_match_f1_threshold: float = 0.75
    answer_match_mode: str = "exact_or_f1"
    reward_mode: str = "hop_accuracy"
    training_reward_mode: str = "outcome"
    hop_step_source_bonus: float = 0.25
    hop_step_doc_bonus: float = 0.25
    privacy_reward_weight: float = 0.0
    privacy_reward_threshold: float = 0.5
    privacy_reward_context_hops: int = 2
    constrained_resolver: bool = False
    stop_after_incorrect_hop: bool = False
    # In stepwise training modes, individual planning traces whose
    # specific (stage, hop_number, iteration) appears in the rollout's
    # error_records (JSON parse failures, context overflows, resolver
    # self-wipes, exceptions) get an error-local reward. This is strictly
    # per-trace: we never blanket-zero all traces in a hop just because the hop
    # failed to produce an answer. Each rollout's padding_reward is set to the
    # max prefix reward achieved, so error-terminated rollouts pad to their
    # best progress rather than to a zero'd error step.
    zero_error_step_reward: bool = True
    max_iterations: int = 14
    iteration_budget_per_hop: int = 2
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
    reader_window_chars: int = 10000
    reader_window_overlap_chars: int = 1000
    max_windows_per_parent: int = 0
    large_parent_window_policy: str = "top_with_neighbors"
    read_all_windows_max_count: int = 5
    large_parent_top_windows: int = 4
    large_parent_neighbor_windows: int = 1
    large_parent_max_read_windows: int = 0
    max_history_entries: int = 10
    local_search_threshold: float = 0.0
    semantic_threshold: float = 0.7
    browsecomp_enabled: bool = True
    browsecomp_top_k: int = 5
    browsecomp_max_chars: int = 8000
    llm_provider: str = "vllm"
    embedding_provider: str | None = "huggingface"
    embedding_model: str | None = "Qwen/Qwen3-Embedding-4B"
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_device: str | None = None
    log_prompts: bool = False
    verbose: bool = False
    no_web: bool = False
    parallel_searches: bool = True
    max_parallel_doc_reads: int = 5
    planner_privacy_prompt: bool = False

    @classmethod
    def from_cfg(cls, cfg: DictConfig | Mapping[str, Any] | None) -> "PrivacyHopQASettings":
        data = _extract_settings_block(cfg)
        return cls(
            task_data_root=Path(data.get("task_data_root", DEFAULT_TASK_DATA_ROOT)).expanduser(),
            local_index_root=_optional_path(data.get("local_index_root"), DEFAULT_LOCAL_INDEX_ROOT),
            local_index_mmap_mode=str(data.get("local_index_mmap_mode", "r")),
            helper_service_url=(str(data["helper_service_url"]) if data.get("helper_service_url") is not None else None),
            privacy_reward_service_url=(
                str(data["privacy_reward_service_url"]) if data.get("privacy_reward_service_url") is not None else None
            ),
            helper_timeout_s=float(data.get("helper_timeout_s", 180.0)),
            privacy_reward_timeout_s=float(data.get("privacy_reward_timeout_s", 30.0)),
            generation_context_limit_tokens=int(data.get("generation_context_limit_tokens", 16384)),
            generation_context_margin_tokens=int(data.get("generation_context_margin_tokens", 128)),
            hop_plan_max_tokens=int(data.get("hop_plan_max_tokens", 4096)),
            doc_choose_max_tokens=int(data.get("doc_choose_max_tokens", 4096)),
            doc_read_max_tokens=int(data.get("doc_read_max_tokens", 4096)),
            hop_resolve_max_tokens=int(data.get("hop_resolve_max_tokens", 4096)),
            reader_min_answer_confidence=float(data.get("reader_min_answer_confidence", 0.75)),
            use_remote_local_search=bool(data.get("use_remote_local_search", True)),
            use_remote_browsecomp=bool(data.get("use_remote_browsecomp", True)),
            capture_mode=str(data.get("capture_mode", "planning_only")),
            answer_match_f1_threshold=float(data.get("answer_match_f1_threshold", 0.75)),
            answer_match_mode=str(data.get("answer_match_mode", "exact_or_f1")),
            reward_mode=str(data.get("reward_mode", "hop_accuracy")),
            training_reward_mode=str(data.get("training_reward_mode", "outcome")),
            hop_step_source_bonus=float(data.get("hop_step_source_bonus", 0.25)),
            hop_step_doc_bonus=float(data.get("hop_step_doc_bonus", data.get("hop_step_source_bonus", 0.25))),
            privacy_reward_weight=float(data.get("privacy_reward_weight", 0.0)),
            privacy_reward_threshold=float(data.get("privacy_reward_threshold", 0.5)),
            privacy_reward_context_hops=int(data.get("privacy_reward_context_hops", 2)),
            constrained_resolver=bool(data.get("constrained_resolver", False)),
            stop_after_incorrect_hop=bool(data.get("stop_after_incorrect_hop", False)),
            zero_error_step_reward=bool(data.get("zero_error_step_reward", True)),
            max_iterations=int(data.get("max_iterations", 14)),
            iteration_budget_per_hop=int(data.get("iteration_budget_per_hop", 2)),
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
            reader_window_chars=int(data.get("reader_window_chars", 10000)),
            reader_window_overlap_chars=int(data.get("reader_window_overlap_chars", 1000)),
            max_windows_per_parent=int(data.get("max_windows_per_parent", 0)),
            large_parent_window_policy=str(data.get("large_parent_window_policy", "top_with_neighbors")),
            read_all_windows_max_count=int(data.get("read_all_windows_max_count", 5)),
            large_parent_top_windows=int(data.get("large_parent_top_windows", 4)),
            large_parent_neighbor_windows=int(data.get("large_parent_neighbor_windows", 1)),
            large_parent_max_read_windows=int(data.get("large_parent_max_read_windows", 0)),
            max_history_entries=int(data.get("max_history_entries", 10)),
            local_search_threshold=float(data.get("local_search_threshold", 0.0)),
            semantic_threshold=float(data.get("semantic_threshold", 0.7)),
            browsecomp_enabled=bool(data.get("browsecomp_enabled", True)),
            browsecomp_top_k=int(data.get("browsecomp_top_k", data.get("retrieval_top_k", 5))),
            browsecomp_max_chars=int(data.get("browsecomp_max_chars", 8000)),
            llm_provider=str(data.get("llm_provider", "vllm")),
            embedding_provider=(str(data["embedding_provider"]) if data.get("embedding_provider") is not None else None),
            embedding_model=(str(data["embedding_model"]) if data.get("embedding_model") is not None else None),
            embedding_base_url=(str(data["embedding_base_url"]) if data.get("embedding_base_url") is not None else None),
            embedding_api_key=(str(data["embedding_api_key"]) if data.get("embedding_api_key") is not None else None),
            embedding_device=(str(data["embedding_device"]) if data.get("embedding_device") is not None else None),
            log_prompts=bool(data.get("log_prompts", False)),
            verbose=bool(data.get("verbose", False)),
            no_web=bool(data.get("no_web", False)),
            parallel_searches=bool(data.get("parallel_searches", True)),
            max_parallel_doc_reads=int(data.get("max_parallel_doc_reads", 5)),
            planner_privacy_prompt=bool(data.get("planner_privacy_prompt", False)),
        )
