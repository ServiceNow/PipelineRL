"""Rollout entrypoint for the privacy_agent domain."""


import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import aiohttp
from omegaconf import DictConfig

from pipelinerl.llm import TrainableLLM
from pipelinerl.rollouts import BaseMetrics, RolloutResult

from .composite_store import CompositeVectorStore
from .drbench.config import RunConfig
from .drbench.drbench_agent import DrBenchAgent
from .drbench.llm_adapter import PrivacyAgentLLMAdapter
from .drbench.session_cache import SessionCache
from .drbench.task_loader import Task
from .drbench.vector_store import VectorStore
from .resources import get_shared_browsecomp, get_shared_helper_client, get_static_local_index
from .reward import score_chain_answers
from .settings import PrivacyAgentSettings

logger = logging.getLogger(__name__)


class PrivacyAgentMetrics(BaseMetrics):
    correct_hops: int = 0
    total_hops: int = 0
    hop_accuracy: float = 0.0
    final_correct: bool = False
    chain_complete: bool = False
    llm_calls_total: int = 0
    llm_calls_captured: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    captured_prompt_tokens: int = 0
    captured_output_tokens: int = 0
    captured_total_tokens: int = 0
    iterations: int = 0
    max_iterations: int = 0
    total_actions: int = 0
    completed_actions: int = 0
    failed_actions: int = 0
    pending_actions: int = 0
    in_progress_actions: int = 0
    vector_store_total_documents: int = 0
    vector_store_static_documents: int = 0
    vector_store_overlay_documents: int = 0
    search_queries: int = 0
    duplicate_preventions: int = 0
    intended_web_docs: int = 0
    retrieved_expected_web_docs: int = 0
    intended_local_docs: int = 0
    retrieved_expected_local_docs: int = 0
    error: str | None = None


def _build_run_paths(cfg: DictConfig, problem: dict) -> tuple[Path, Path]:
    rollout_id = uuid.uuid4().hex[:8]
    root = (
        Path(cfg.output_dir)
        / "privacy_agent_runs"
        / f"{problem['task_id']}_{problem['chain_id']}_{rollout_id}"
    )
    workspace_dir = root / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return root, workspace_dir


def _get_action_plan_stats(report_metadata: dict[str, Any]) -> dict[str, Any]:
    action_plan_stats = report_metadata.get("action_plan_stats")
    if isinstance(action_plan_stats, dict):
        return action_plan_stats
    return {}


async def generate_privacy_agent_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    del session
    start = time.time()
    result = await asyncio.to_thread(_run_privacy_agent_rollout_sync, cfg, llm, problem)
    result.latency = time.time() - start
    return result


def _run_privacy_agent_rollout_sync(cfg: DictConfig, llm: TrainableLLM, problem: dict) -> RolloutResult:
    settings = PrivacyAgentSettings.from_cfg(cfg)
    run_root, workspace_dir = _build_run_paths(cfg, problem)
    run_cfg = RunConfig(
        model=llm.model_name,
        max_iterations=settings.max_iterations,
        concurrent_actions=settings.concurrent_actions,
        semantic_threshold=settings.semantic_threshold,
        run_dir=run_root,
        log_searches=settings.log_searches,
        log_prompts=settings.log_prompts,
        log_generations=settings.log_generations,
        verbose=settings.verbose,
        no_web=settings.no_web,
        llm_provider=settings.llm_provider,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        report_style=settings.report_style,
        browsecomp_enabled=settings.browsecomp_enabled,
        browsecomp_index_glob=settings.browsecomp_index_glob,
        browsecomp_model_name=settings.browsecomp_model_name,
        browsecomp_corpus=settings.browsecomp_corpus,
        browsecomp_top_k=settings.browsecomp_top_k,
        browsecomp_max_chars=settings.browsecomp_max_chars,
    )

    task = Task(Path(problem["task_id"]) / "config", data_dir=settings.task_data_root)
    helper_client = None
    if (
        settings.use_remote_embeddings
        or settings.use_remote_local_search
        or settings.use_remote_browsecomp
    ):
        if not settings.helper_service_url:
            raise ValueError(
                "privacy_agent remote helper mode is enabled, but helper_service_url is not configured"
            )
        helper_client = get_shared_helper_client(
            base_url=settings.helper_service_url,
            timeout_s=settings.helper_timeout_s,
        )
    static_index = get_static_local_index(
        task_id=problem["task_id"],
        index_root=settings.local_index_root,
        embedding_model=run_cfg.get_embedding_model(),
        embedding_provider=run_cfg.get_embedding_provider(),
        semantic_threshold=run_cfg.semantic_threshold,
        mmap_mode=settings.local_index_mmap_mode,
    )
    shared_browsecomp = None
    if run_cfg.browsecomp_enabled and not settings.use_remote_browsecomp:
        shared_browsecomp = get_shared_browsecomp(run_cfg, device=settings.browsecomp_device)

    adapter = PrivacyAgentLLMAdapter(
        llm=llm,
        capture_mode=settings.capture_mode,
    )

    answers: dict[str, str] = {}
    error: str | None = None
    final_report = ""
    diagnostics: dict[str, Any] = {}
    report_metadata: dict[str, Any] = {}
    vector_store_stats: dict[str, Any] = {}
    session_cache_stats: dict[str, Any] = {}
    try:
        session_cache = SessionCache(session_id=str(uuid.uuid4())[:8])
        session_dir = run_root / "session"
        overlay_dir = session_dir / "overlay_store"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_store = VectorStore(
            run_config=run_cfg,
            storage_dir=str(overlay_dir),
            embedding_model=run_cfg.get_embedding_model(),
            embedding_provider=run_cfg.get_embedding_provider(),
            session_cache=session_cache,
            helper_client=helper_client if settings.use_remote_embeddings else None,
        )
        vector_store = CompositeVectorStore(
            static_local_index=static_index,
            overlay_store=overlay_store,
            storage_dir=session_dir,
            task_id=problem["task_id"],
            embedding_model=run_cfg.get_embedding_model(),
            embedding_provider=run_cfg.get_embedding_provider(),
            semantic_threshold=run_cfg.semantic_threshold,
            session_cache=session_cache,
            helper_client=helper_client,
            use_remote_local_search=settings.use_remote_local_search,
        )
        agent = DrBenchAgent(
            model=run_cfg.model,
            run_config=run_cfg,
            workspace_dir=str(workspace_dir),
            max_iterations=run_cfg.max_iterations,
            vector_store_base_dir=str(run_root / "vector_stores"),
            embedding_model=run_cfg.get_embedding_model(),
            embedding_provider=run_cfg.get_embedding_provider(),
            concurrent_actions=run_cfg.concurrent_actions,
            verbose=run_cfg.verbose,
            shared_browsecomp=shared_browsecomp,
            helper_client=helper_client,
            use_remote_browsecomp=settings.use_remote_browsecomp,
            llm_adapter=adapter,
            vector_store=vector_store,
            session_cache=session_cache,
            preindexed_local_documents=True,
        )
        final_report, _ = agent.generate_report(
            query=problem["numbered_questions"],
            task_id=problem["task_id"],
            local_files=task.get_local_files_list(),
            extract_insights=False,
            as_dict=False,
        )
        answers = dict(getattr(agent.report_assembler, "_parsed_answers", {}) or {})
        report_metadata = dict(agent.get_report_metadata() or {})
        vector_store_stats = dict(agent.vector_store.get_stats() or {})
        session_cache_stats = dict(agent.session_cache.get_stats() or {})
        diagnostics = agent.vector_store.get_retrieval_diagnostics(problem.get("expected_doc_ids", []))
    except Exception as exc:
        logger.exception("privacy_agent rollout failed for chain %s", problem.get("chain_id"))
        error = str(exc)

    score = score_chain_answers(
        problem,
        answers,
        f1_threshold=settings.answer_match_f1_threshold,
    )
    action_plan_stats = _get_action_plan_stats(report_metadata)
    group_id = str(problem.get("chain_id") or problem.get("problem_id"))
    base_metadata = {
        "chain_id": problem.get("chain_id"),
        "task_id": problem.get("task_id"),
        "company": problem.get("company"),
        "problem_id": problem.get("problem_id"),
        "parsed_answers": answers,
        "final_report_length": len(final_report),
        "final_report_preview": final_report[:1000],
        "report_metadata": report_metadata,
        "retrieval_diagnostics": diagnostics,
    }
    training_texts = adapter.make_training_texts(group_id=group_id, base_metadata=base_metadata)
    for trace in training_texts:
        trace.reward = score["reward"]

    expected_doc_ids = problem.get("expected_doc_ids", [])
    intended_web_docs = len([doc_id for doc_id in expected_doc_ids if str(doc_id).startswith("web/")])
    intended_local_docs = len(
        [doc_id for doc_id in expected_doc_ids if str(doc_id).startswith("local/")]
    )
    metrics = PrivacyAgentMetrics(
        reward=score["reward"],
        success=(score["correct_hops"] == score["total_hops"] and error is None),
        no_error=error is None,
        no_answer=not bool(answers),
        correct_hops=score["correct_hops"],
        total_hops=score["total_hops"],
        hop_accuracy=score["hop_accuracy"],
        final_correct=bool(score["final_correct"]),
        chain_complete=bool(score["chain_complete"]),
        llm_calls_total=adapter.total_calls,
        llm_calls_captured=len(training_texts),
        prompt_tokens=adapter.total_prompt_tokens,
        output_tokens=adapter.total_output_tokens,
        total_tokens=adapter.total_prompt_tokens + adapter.total_output_tokens,
        captured_prompt_tokens=adapter.captured_prompt_tokens,
        captured_output_tokens=adapter.captured_output_tokens,
        captured_total_tokens=adapter.captured_prompt_tokens + adapter.captured_output_tokens,
        iterations=int(action_plan_stats.get("current_iteration") or 0),
        max_iterations=int(run_cfg.max_iterations),
        total_actions=int(action_plan_stats.get("total_actions") or 0),
        completed_actions=int(action_plan_stats.get("completed") or 0),
        failed_actions=int(action_plan_stats.get("failed") or 0),
        pending_actions=int(action_plan_stats.get("pending") or 0),
        in_progress_actions=int(action_plan_stats.get("in_progress") or 0),
        vector_store_total_documents=int(vector_store_stats.get("total_documents") or 0),
        vector_store_static_documents=int(vector_store_stats.get("static_documents") or 0),
        vector_store_overlay_documents=int(vector_store_stats.get("overlay_documents") or 0),
        search_queries=len(diagnostics.get("search_log", [])),
        duplicate_preventions=int(session_cache_stats.get("duplicate_preventions") or 0),
        intended_web_docs=intended_web_docs,
        retrieved_expected_web_docs=len(diagnostics.get("retrieved_expected_web_docids", [])),
        intended_local_docs=intended_local_docs,
        retrieved_expected_local_docs=len(diagnostics.get("retrieved_expected_local_suffixes", [])),
        error=error,
    )
    return RolloutResult(
        training_texts=training_texts,
        metrics=metrics,
        latency=0.0,
        dataset_name=problem.get("dataset"),
        group_id=group_id,
        domain="privacy_agent",
    )
