"""Rollout entrypoint for the privacy_hopqa domain."""

from __future__ import annotations

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

from .helper_client import AsyncPrivacyHopQAHelperClient
from .task_loader import Task
from .resources import (
    get_shared_browsecomp,
    get_static_local_index,
)

from .agent import PrivacyHopQAAgent, is_helper_infrastructure_error
from .llm_adapter import PrivacyHopQALLMAdapter, is_llm_infrastructure_error
from .reward import answers_match, score_chain_answers
from .settings import PrivacyHopQASettings

logger = logging.getLogger(__name__)


class PrivacyHopQAMetrics(BaseMetrics):
    correct_hops: int = 0
    total_hops: int = 0
    hop_accuracy: float = 0.0
    final_correct: bool = False
    chain_complete: bool = False
    n_question_hops: int = 0
    n_unique_docs: int = 0
    n_cross_doc_transitions: int = 0
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
    searches_total: int = 0
    docs_read: int = 0
    candidate_windows_before_parent_trim: int = 0
    candidate_windows_after_parent_trim: int = 0
    candidate_parent_docs_before_parent_trim: int = 0
    candidate_parent_docs_after_parent_trim: int = 0
    selected_parent_docs: int = 0
    selected_parent_windows_available: int = 0
    selected_windows_read: int = 0
    selected_window_read_ratio: float = 0.0
    large_parent_docs_selected: int = 0
    large_parent_docs_limited: int = 0
    large_parent_windows_available: int = 0
    large_parent_windows_read: int = 0
    large_parent_window_read_ratio: float = 0.0
    max_selected_parent_windows_available: int = 0
    max_selected_parent_windows_read: int = 0
    gold_parent_hops: int = 0
    gold_parent_retrieved_before_trim_hops: int = 0
    gold_parent_retrieved_hops: int = 0
    gold_parent_read_hops: int = 0
    gold_parent_retrieved_before_trim_rate: float = 0.0
    gold_parent_retrieved_rate: float = 0.0
    gold_parent_read_rate: float = 0.0
    gold_parent_missed_after_trim_hops: int = 0
    gold_parent_retrieved_not_read_hops: int = 0
    gold_parent_read_and_correct_hops: int = 0
    gold_parent_read_but_incorrect_hops: int = 0
    gold_parent_read_but_no_answer_hops: int = 0
    gold_parent_read_answer_accuracy: float = 0.0
    gold_reader_correct_hops: int = 0
    gold_reader_correct_but_final_incorrect_hops: int = 0
    any_reader_correct_hops: int = 0
    any_reader_correct_but_final_incorrect_hops: int = 0
    hops_resolved: int = 0
    total_errors: int = 0
    parse_errors_total: int = 0
    context_overflow_errors: int = 0
    hop_plan_parse_errors: int = 0
    doc_choose_parse_errors: int = 0
    doc_read_parse_errors: int = 0
    hop_resolve_parse_errors: int = 0
    error: str | None = None


def _build_run_paths(cfg: DictConfig, problem: dict) -> tuple[Path, Path]:
    rollout_id = uuid.uuid4().hex[:8]
    root = (
        Path(cfg.output_dir)
        / "privacy_hopqa_runs"
        / f"{problem['task_id']}_{problem['chain_id']}_{rollout_id}"
    )
    workspace_dir = root / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return root, workspace_dir


async def generate_privacy_hopqa_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    start = time.time()
    result = await _run_privacy_hopqa_rollout_async(cfg, llm, problem, session)
    result.latency = time.time() - start
    return result


async def _run_privacy_hopqa_rollout_async(
    cfg: DictConfig, llm: TrainableLLM, problem: dict, session: aiohttp.ClientSession
) -> RolloutResult:
    settings = PrivacyHopQASettings.from_cfg(cfg)
    run_root, workspace_dir = _build_run_paths(cfg, problem)

    task = Task(Path(problem["task_id"]) / "config", data_dir=settings.task_data_root)
    task_config = task.get_task_config() or {}
    company_info = dict(task_config.get("company_info") or {})
    company_name = str(company_info.get("name") or problem.get("company") or "").strip() or None
    company_description = str(company_info.get("description") or "").strip() or None

    helper_client = None
    if settings.use_remote_local_search or settings.use_remote_browsecomp:
        if not settings.helper_service_url:
            raise ValueError("privacy_hopqa remote helper mode is enabled, but helper_service_url is not configured")
        helper_client = AsyncPrivacyHopQAHelperClient(
            base_url=settings.helper_service_url,
            timeout_s=settings.helper_timeout_s,
            session=session,
        )

    static_index = get_static_local_index(
        task_id=problem["task_id"],
        index_root=settings.local_index_root,
        embedding_model=settings.embedding_model,
        embedding_provider=settings.embedding_provider,
        semantic_threshold=settings.semantic_threshold,
        mmap_mode=settings.local_index_mmap_mode,
    )
    shared_browsecomp = None
    if settings.browsecomp_enabled and not settings.use_remote_browsecomp:
        shared_browsecomp = get_shared_browsecomp(settings, device=settings.browsecomp_device)

    adapter = PrivacyHopQALLMAdapter(
        llm=llm,
        session=session,
        capture_mode=settings.capture_mode,
        rollout_id=run_root.name,
        task_id=str(problem.get("task_id") or ""),
        chain_id=str(problem.get("chain_id") or ""),
    )

    answers: dict[str, str] = {}
    error: str | None = None
    report_metadata: dict[str, Any] = {}
    try:
        agent = PrivacyHopQAAgent(
            settings=settings,
            llm_adapter=adapter,
            run_root=run_root,
            workspace_dir=workspace_dir,
            task_id=str(problem["task_id"]),
            numbered_questions=str(problem.get("numbered_questions") or problem.get("task") or ""),
            company_name=company_name,
            company_description=company_description,
            hops=list(problem.get("hops") or []),
            helper_client=helper_client,
            static_local_index=static_index,
            browsecomp_tool=shared_browsecomp,
        )
        agent_result = await agent.run()
        final_report = agent_result.final_report
        answers = dict(agent_result.parsed_answers)
        report_metadata = dict(agent_result.report_metadata)
        retrieval_summary = dict(report_metadata.get("retrieval_summary") or {})
        error_summary = dict(report_metadata.get("error_summary") or {})
        hops_resolved = sum(1 for hop in agent_result.hop_states if hop.get("answer"))
        searches_total = len(agent_result.action_records)
        docs_read = sum(len(hop.get("selected_doc_ids", [])) for hop in agent_result.hop_states)
        iterations_used = int(report_metadata.get("iterations_used") or 0)
    except Exception as exc:
        if is_llm_infrastructure_error(exc) or is_helper_infrastructure_error(exc):
            raise
        logger.exception("privacy_hopqa rollout failed for chain %s", problem.get("chain_id"))
        error = str(exc)
        final_report = ""
        hops_resolved = 0
        searches_total = 0
        docs_read = 0
        iterations_used = 0
        error_summary = {}
        retrieval_summary = {}

    score = score_chain_answers(
        problem,
        answers,
        f1_threshold=settings.answer_match_f1_threshold,
        reward_mode=settings.reward_mode,
        answer_match_mode=settings.answer_match_mode,
    )
    per_hop_scores = {str(item.get("hop")): item for item in score.get("per_hop", [])}
    hop_states_for_metrics = []
    if "agent_result" in locals():
        hop_states_for_metrics = list(agent_result.hop_states)
    gold_parent_read_and_correct_hops = 0
    gold_parent_read_but_incorrect_hops = 0
    gold_parent_read_but_no_answer_hops = 0
    gold_reader_correct_hops = 0
    gold_reader_correct_but_final_incorrect_hops = 0
    any_reader_correct_hops = 0
    any_reader_correct_but_final_incorrect_hops = 0
    for hop_state in hop_states_for_metrics:
        hop_number = str(hop_state.get("hop_number"))
        hop_score = per_hop_scores.get(hop_number) or {}
        final_correct_for_hop = bool(hop_score.get("correct"))
        accepted_answers = hop_score.get("accepted_answers") or []
        if not hop_state.get("gold_parent_read"):
            continue
        if final_correct_for_hop:
            gold_parent_read_and_correct_hops += 1
        else:
            gold_parent_read_but_incorrect_hops += 1
            if not hop_state.get("answer"):
                gold_parent_read_but_no_answer_hops += 1
        gold_doc_id = str(hop_state.get("gold_doc_id") or "")
        gold_reader_correct = False
        any_reader_correct = False
        for reader_result in hop_state.get("reader_results") or []:
            if not reader_result.get("can_answer") or not str(reader_result.get("proposed_answer") or "").strip():
                continue
            reader_correct, _reader_score = answers_match(
                reader_result.get("proposed_answer"),
                accepted_answers,
                f1_threshold=settings.answer_match_f1_threshold,
                match_mode=settings.answer_match_mode,
            )
            if not reader_correct:
                continue
            any_reader_correct = True
            if gold_doc_id and (
                str(reader_result.get("parent_doc_id") or "") == gold_doc_id
                or str(reader_result.get("doc_id") or "") == gold_doc_id
            ):
                gold_reader_correct = True
        if gold_reader_correct:
            gold_reader_correct_hops += 1
            if not final_correct_for_hop:
                gold_reader_correct_but_final_incorrect_hops += 1
        if any_reader_correct:
            any_reader_correct_hops += 1
            if not final_correct_for_hop:
                any_reader_correct_but_final_incorrect_hops += 1
    gold_parent_read_answer_accuracy = (
        gold_parent_read_and_correct_hops / max(1, gold_parent_read_and_correct_hops + gold_parent_read_but_incorrect_hops)
    )
    n_question_hops = int(problem.get("n_question_hops") or problem.get("n_hops") or len(problem.get("hops") or []))
    n_unique_docs = int(problem.get("n_unique_docs") or 0)
    n_cross_doc_transitions = int(problem.get("n_cross_doc_transitions") or max(0, n_unique_docs - 1))
    group_id = str(problem.get("chain_id") or problem.get("problem_id"))
    base_metadata = {
        "chain_id": problem.get("chain_id"),
        "task_id": problem.get("task_id"),
        "company": problem.get("company"),
        "problem_id": problem.get("problem_id"),
        "parsed_answers": answers,
        "iterations_used": iterations_used,
        "searches_total": searches_total,
        "docs_read": docs_read,
        "hops_resolved": hops_resolved,
        "n_question_hops": n_question_hops,
        "n_unique_docs": n_unique_docs,
        "n_cross_doc_transitions": n_cross_doc_transitions,
        "report_metadata": report_metadata,
        "error_summary": error_summary,
        "retrieval_summary": retrieval_summary,
    }
    training_texts = adapter.make_training_texts(group_id=group_id, base_metadata=base_metadata)
    for trace in training_texts:
        trace.reward = score["reward"]

    metrics = PrivacyHopQAMetrics(
        reward=score["reward"],
        success=(score["correct_hops"] == score["total_hops"] and error is None),
        no_error=error is None,
        no_answer=not bool(answers),
        correct_hops=score["correct_hops"],
        total_hops=score["total_hops"],
        hop_accuracy=score["hop_accuracy"],
        final_correct=bool(score["final_correct"]),
        chain_complete=bool(score["chain_complete"]),
        n_question_hops=n_question_hops,
        n_unique_docs=n_unique_docs,
        n_cross_doc_transitions=n_cross_doc_transitions,
        llm_calls_total=adapter.total_calls,
        llm_calls_captured=len(training_texts),
        prompt_tokens=adapter.total_prompt_tokens,
        output_tokens=adapter.total_output_tokens,
        total_tokens=adapter.total_prompt_tokens + adapter.total_output_tokens,
        captured_prompt_tokens=adapter.captured_prompt_tokens,
        captured_output_tokens=adapter.captured_output_tokens,
        captured_total_tokens=adapter.captured_prompt_tokens + adapter.captured_output_tokens,
        iterations=iterations_used,
        max_iterations=settings.max_iterations,
        searches_total=searches_total,
        docs_read=docs_read,
        candidate_windows_before_parent_trim=int(retrieval_summary.get("candidate_windows_before_parent_trim") or 0),
        candidate_windows_after_parent_trim=int(retrieval_summary.get("candidate_windows_after_parent_trim") or 0),
        candidate_parent_docs_before_parent_trim=int(retrieval_summary.get("candidate_parent_docs_before_parent_trim") or 0),
        candidate_parent_docs_after_parent_trim=int(retrieval_summary.get("candidate_parent_docs_after_parent_trim") or 0),
        selected_parent_docs=int(retrieval_summary.get("selected_parent_docs") or 0),
        selected_parent_windows_available=int(retrieval_summary.get("selected_parent_windows_available") or 0),
        selected_windows_read=int(retrieval_summary.get("selected_windows_read") or 0),
        selected_window_read_ratio=float(retrieval_summary.get("selected_window_read_ratio") or 0.0),
        large_parent_docs_selected=int(retrieval_summary.get("large_parent_docs_selected") or 0),
        large_parent_docs_limited=int(retrieval_summary.get("large_parent_docs_limited") or 0),
        large_parent_windows_available=int(retrieval_summary.get("large_parent_windows_available") or 0),
        large_parent_windows_read=int(retrieval_summary.get("large_parent_windows_read") or 0),
        large_parent_window_read_ratio=float(retrieval_summary.get("large_parent_window_read_ratio") or 0.0),
        max_selected_parent_windows_available=int(retrieval_summary.get("max_selected_parent_windows_available") or 0),
        max_selected_parent_windows_read=int(retrieval_summary.get("max_selected_parent_windows_read") or 0),
        gold_parent_hops=int(retrieval_summary.get("gold_parent_hops") or 0),
        gold_parent_retrieved_before_trim_hops=int(retrieval_summary.get("gold_parent_retrieved_before_trim_hops") or 0),
        gold_parent_retrieved_hops=int(retrieval_summary.get("gold_parent_retrieved_hops") or 0),
        gold_parent_read_hops=int(retrieval_summary.get("gold_parent_read_hops") or 0),
        gold_parent_retrieved_before_trim_rate=float(retrieval_summary.get("gold_parent_retrieved_before_trim_rate") or 0.0),
        gold_parent_retrieved_rate=float(retrieval_summary.get("gold_parent_retrieved_rate") or 0.0),
        gold_parent_read_rate=float(retrieval_summary.get("gold_parent_read_rate") or 0.0),
        gold_parent_missed_after_trim_hops=int(retrieval_summary.get("gold_parent_missed_after_trim_hops") or 0),
        gold_parent_retrieved_not_read_hops=int(retrieval_summary.get("gold_parent_retrieved_not_read_hops") or 0),
        gold_parent_read_and_correct_hops=gold_parent_read_and_correct_hops,
        gold_parent_read_but_incorrect_hops=gold_parent_read_but_incorrect_hops,
        gold_parent_read_but_no_answer_hops=gold_parent_read_but_no_answer_hops,
        gold_parent_read_answer_accuracy=gold_parent_read_answer_accuracy,
        gold_reader_correct_hops=gold_reader_correct_hops,
        gold_reader_correct_but_final_incorrect_hops=gold_reader_correct_but_final_incorrect_hops,
        any_reader_correct_hops=any_reader_correct_hops,
        any_reader_correct_but_final_incorrect_hops=any_reader_correct_but_final_incorrect_hops,
        hops_resolved=hops_resolved,
        total_errors=int(error_summary.get("total_errors") or 0),
        parse_errors_total=int(error_summary.get("parse_errors_total") or 0),
        context_overflow_errors=int(error_summary.get("context_overflow_errors") or 0),
        hop_plan_parse_errors=int((error_summary.get("parse_errors_by_stage") or {}).get("hop_plan") or 0),
        doc_choose_parse_errors=int((error_summary.get("parse_errors_by_stage") or {}).get("doc_choose") or 0),
        doc_read_parse_errors=int((error_summary.get("parse_errors_by_stage") or {}).get("doc_read") or 0),
        hop_resolve_parse_errors=int((error_summary.get("parse_errors_by_stage") or {}).get("hop_resolve") or 0),
        error=error,
    )
    return RolloutResult(
        training_texts=training_texts,
        metrics=metrics,
        latency=0.0,
        dataset_name=problem.get("dataset"),
        group_id=group_id,
        domain="privacy_hopqa",
    )
