"""Core hop-wise QA agent for the privacy_hopqa domain."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .llm_adapter import PrivacyHopQALLMAdapter, is_llm_infrastructure_error
from .prompts import (
    build_doc_choose_prompt,
    build_doc_reader_prompt,
    build_hop_plan_prompt,
    build_hop_resolve_prompt,
)
from .reporting import build_deterministic_report, parse_answer_lines
from .settings import PrivacyHopQASettings
from .timeline import write_timeline_artifacts
from .utils import extract_json

logger = logging.getLogger(__name__)


def _truncate(text: str, limit: int) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


def _normalize_query(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _clamp_confidence(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, numeric))


def _normalize_answer(value: str | None) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _current_timestamp() -> float:
    return time.perf_counter()


@dataclass
class RetrievalActionRecord:
    id: str
    type: str
    description: str
    parameters: dict[str, Any]
    priority: float = 0.5
    expected_output: str = ""
    status: str = "pending"
    actual_output: dict[str, Any] | None = None
    error: str | None = None
    execution_time: float | None = None
    iteration: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateDoc:
    doc_id: str
    source: str
    title: str
    locator: str
    excerpt: str
    score: float | None
    rank: int
    queries: list[str] = field(default_factory=list)
    search_action_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def merge(self, other: "CandidateDoc") -> None:
        if other.rank < self.rank:
            self.rank = other.rank
        if other.score is not None and (self.score is None or other.score > self.score):
            self.score = other.score
        for query in other.queries:
            if query not in self.queries:
                self.queries.append(query)
        for action_id in other.search_action_ids:
            if action_id not in self.search_action_ids:
                self.search_action_ids.append(action_id)
        if len(other.excerpt) > len(self.excerpt):
            self.excerpt = other.excerpt
        if not self.locator and other.locator:
            self.locator = other.locator
        if not self.title and other.title:
            self.title = other.title

    def as_card(self, chooser_chars: int) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "title": self.title,
            "best_rank": self.rank,
            "hit_count": len(self.queries),
            "top_queries": self.queries[:2],
            "snippet": _truncate(self.excerpt, chooser_chars),
        }

    def as_reader_input(self, excerpt_chars: int) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "title": self.title,
            "locator": self.locator,
            "queries": self.queries,
            "excerpt": _truncate(self.excerpt, excerpt_chars),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReaderResult:
    doc_id: str
    source: str
    title: str
    locator: str
    can_answer: bool
    proposed_answer: str
    justification: str
    confidence: float
    missing_information: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HopState:
    hop_number: int
    question: str
    status: str = "pending"
    attempts: int = 0
    answer: str | None = None
    justification: str | None = None
    confidence: float = 0.0
    resolution_reason: str | None = None
    search_history: list[dict[str, Any]] = field(default_factory=list)
    selected_doc_ids: list[str] = field(default_factory=list)
    reader_results: list[dict[str, Any]] = field(default_factory=list)
    candidate_doc_ids: list[str] = field(default_factory=list)
    candidate_cards: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PrivacyHopQAAgentResult:
    final_report: str
    parsed_answers: dict[str, str]
    hop_states: list[dict[str, Any]]
    action_records: list[dict[str, Any]]
    timeline_events: list[dict[str, Any]]
    error_records: list[dict[str, Any]]
    report_metadata: dict[str, Any]


class PrivacyHopQAAgent:
    def __init__(
        self,
        *,
        settings: PrivacyHopQASettings,
        llm_adapter: PrivacyHopQALLMAdapter,
        run_root: Path,
        workspace_dir: Path,
        task_id: str,
        numbered_questions: str,
        company_name: str | None,
        company_description: str | None,
        hops: list[dict[str, Any]],
        helper_client: Any = None,
        static_local_index: Any = None,
        browsecomp_tool: Any = None,
    ):
        self.settings = settings
        self.llm_adapter = llm_adapter
        self.run_root = Path(run_root)
        self.workspace_dir = Path(workspace_dir)
        self.task_id = task_id
        self.numbered_questions = numbered_questions
        self.company_name = company_name
        self.company_description = company_description
        self.hop_states = [HopState(hop_number=int(hop["hop_number"]), question=str(hop.get("question") or "")) for hop in hops]
        self.helper_client = helper_client
        self.static_local_index = static_local_index
        self.browsecomp_tool = browsecomp_tool
        self.action_records: list[RetrievalActionRecord] = []
        self.timeline_events: list[dict[str, Any]] = []
        self.error_records: list[dict[str, Any]] = []
        self._errors_by_stage: Counter[str] = Counter()
        self._errors_by_kind: Counter[str] = Counter()
        self._errors_by_stage_kind: dict[str, Counter[str]] = defaultdict(Counter)
        self.prompt_dir = self.run_root / "prompts"
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        self._run_started_at = _current_timestamp()

    def _elapsed_s(self) -> float:
        return _current_timestamp() - self._run_started_at

    def _record_event(
        self,
        *,
        stage: str,
        lane: str,
        label: str,
        start_s: float,
        end_s: float,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.timeline_events.append(
            {
                "stage": stage,
                "lane": lane,
                "label": label,
                "start_s": round(start_s, 6),
                "end_s": round(end_s, 6),
                "meta": meta or {},
            }
        )

    def _save_prompt(self, name: str, prompt: str) -> None:
        if not self.settings.log_prompts:
            return
        path = self.prompt_dir / f"{name}.txt"
        path.write_text(prompt, encoding="utf-8")

    def _classify_error_kind(self, exc: Exception | str) -> str:
        text = str(exc).lower()
        if "no valid json found" in text or ("json" in text and "line 1 column 1" in text):
            return "parse_error"
        if "maximum context length" in text or "context length" in text:
            return "context_overflow"
        return "other"

    def _record_error(
        self,
        *,
        stage: str,
        hop_number: int | None,
        iteration: int | None,
        exc: Exception | str,
        lane: str | None = None,
        label: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        kind = self._classify_error_kind(exc)
        record = {
            "stage": stage,
            "kind": kind,
            "hop_number": hop_number,
            "iteration": iteration,
            "message": str(exc),
            "elapsed_s": round(self._elapsed_s(), 6),
        }
        if meta:
            record["meta"] = dict(meta)
        self.error_records.append(record)
        self._errors_by_stage[stage] += 1
        self._errors_by_kind[kind] += 1
        self._errors_by_stage_kind[stage][kind] += 1

        point_s = self._elapsed_s()
        self._record_event(
            stage="error",
            lane=lane or f"error:{stage}",
            label=label or f"{stage} {kind}",
            start_s=point_s,
            end_s=point_s + 0.001,
            meta={
                "stage": stage,
                "kind": kind,
                "hop_number": hop_number,
                "iteration": iteration,
                "message": str(exc),
                **(meta or {}),
            },
        )

    def _error_summary(self) -> dict[str, Any]:
        by_stage = {stage: int(count) for stage, count in sorted(self._errors_by_stage.items())}
        by_kind = {kind: int(count) for kind, count in sorted(self._errors_by_kind.items())}
        by_stage_kind = {
            stage: {kind: int(count) for kind, count in sorted(counter.items())}
            for stage, counter in sorted(self._errors_by_stage_kind.items())
        }
        parse_by_stage = {
            stage: int(counter.get("parse_error", 0))
            for stage, counter in sorted(self._errors_by_stage_kind.items())
        }
        return {
            "total_errors": int(sum(self._errors_by_stage.values())),
            "parse_errors_total": int(self._errors_by_kind.get("parse_error", 0)),
            "context_overflow_errors": int(self._errors_by_kind.get("context_overflow", 0)),
            "errors_by_stage": by_stage,
            "errors_by_kind": by_kind,
            "errors_by_stage_kind": by_stage_kind,
            "parse_errors_by_stage": parse_by_stage,
        }

    def _resolved_answers(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for hop in self.hop_states:
            if hop.answer:
                rows.append(
                    {
                        "hop_number": hop.hop_number,
                        "question": hop.question,
                        "answer": hop.answer,
                        "justification": hop.justification or "",
                        "confidence": round(hop.confidence, 3),
                    }
                )
        return rows

    def _materialize_question(self, question: str) -> str:
        resolved = {str(hop.hop_number): hop.answer for hop in self.hop_states if hop.answer}

        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            return resolved.get(key, match.group(0))

        return re.sub(r"\((\d+)\)", replace, question)

    def _current_hop(self) -> HopState | None:
        for hop in self.hop_states:
            if not hop.answer:
                return hop
        return None

    def _recent_search_history(self, hop: HopState) -> list[dict[str, Any]]:
        return hop.search_history[-self.settings.max_history_entries :]

    def _recent_reader_results(self, hop: HopState) -> list[dict[str, Any]]:
        return hop.reader_results[-self.settings.max_history_entries :]

    def _unread_candidates(self, hop: HopState, candidates: list[CandidateDoc]) -> list[CandidateDoc]:
        return [candidate for candidate in candidates if candidate.doc_id not in hop.selected_doc_ids]

    def _mark_selected_documents(self, hop: HopState, selected: list[CandidateDoc]) -> None:
        for doc in selected:
            if doc.doc_id not in hop.selected_doc_ids:
                hop.selected_doc_ids.append(doc.doc_id)

    def _next_iteration_cap(self) -> int:
        total_hops = max(len(self.hop_states), 1)
        return min(self.settings.max_iterations, (2 * total_hops) + 2)

    def _fallback_actions(self, hop: HopState, iteration: int) -> list[RetrievalActionRecord]:
        question = self._materialize_question(hop.question)
        actions: list[RetrievalActionRecord] = []
        if not self.settings.no_web:
            actions.append(
                RetrievalActionRecord(
                    id=f"iter{iteration}_web_fallback",
                    type="web_search",
                    description=f"Search the web for hop {hop.hop_number}",
                    parameters={"query": question},
                    priority=0.7,
                    expected_output="Candidate web evidence",
                )
            )
        actions.append(
            RetrievalActionRecord(
                id=f"iter{iteration}_local_fallback",
                type="local_document_search",
                description=f"Search local documents for hop {hop.hop_number}",
                parameters={"query": question},
                priority=0.7,
                expected_output="Candidate local evidence",
            )
        )
        return actions[: self.settings.max_parallel_retrieval_actions]

    async def _plan_retrieval_actions(self, hop: HopState, iteration: int) -> list[RetrievalActionRecord]:
        started = self._elapsed_s()
        planned_actions: list[RetrievalActionRecord] = []
        retry_guidance: str | None = None
        attempt_count = 0
        fallback_used = False
        for plan_attempt in range(max(1, self.settings.hop_plan_attempts)):
            attempt_count = plan_attempt + 1
            prompt = build_hop_plan_prompt(
                numbered_questions=self.numbered_questions,
                current_hop_number=hop.hop_number,
                current_hop_question=self._materialize_question(hop.question),
                resolved_answers=self._resolved_answers(),
                search_history=self._recent_search_history(hop),
                recent_reader_results=self._recent_reader_results(hop),
                company_name=self.company_name,
                company_description=self.company_description,
                max_parallel_retrieval_actions=self.settings.max_parallel_retrieval_actions,
                no_web=self.settings.no_web,
                retry_guidance=retry_guidance,
            )
            prompt_name = f"hop_plan_iter{iteration}_hop{hop.hop_number}"
            if plan_attempt > 0:
                prompt_name = f"{prompt_name}_attempt{plan_attempt + 1}"
            self._save_prompt(prompt_name, prompt)
            try:
                response = await self.llm_adapter.generate_text(prompt, log_name="hop_plan")
                payload = extract_json(response)
            except Exception as exc:
                if is_llm_infrastructure_error(exc):
                    raise
                logger.warning("hop planner failed for hop %s iteration %s attempt %s: %s", hop.hop_number, iteration, attempt_count, exc)
                self._record_error(stage="hop_plan", hop_number=hop.hop_number, iteration=iteration, exc=exc)
                payload = []
                retry_guidance = "The previous response was not valid JSON or could not be parsed. Return a valid JSON array of retrieval actions."
                continue

            raw_actions = payload if isinstance(payload, list) else []
            records: list[RetrievalActionRecord] = []
            seen_keys: set[tuple[str, str]] = set()
            seen_recent = {
                (entry.get("type", ""), _normalize_query(entry.get("query", "")))
                for entry in self._recent_search_history(hop)
            }
            invalid_count = 0
            duplicate_count = 0
            for idx, raw in enumerate(raw_actions[: self.settings.max_parallel_retrieval_actions]):
                if not isinstance(raw, dict):
                    invalid_count += 1
                    continue
                action_type = str(raw.get("type") or "").strip().lower()
                if action_type not in {"web_search", "local_document_search"}:
                    invalid_count += 1
                    continue
                if action_type == "web_search" and self.settings.no_web:
                    invalid_count += 1
                    continue
                parameters = raw.get("parameters") if isinstance(raw.get("parameters"), dict) else {}
                query = str(parameters.get("query") or raw.get("description") or "").strip()
                if not query:
                    invalid_count += 1
                    continue
                key = (action_type, _normalize_query(query))
                if key in seen_keys or key in seen_recent:
                    duplicate_count += 1
                    continue
                seen_keys.add(key)
                records.append(
                    RetrievalActionRecord(
                        id=f"iter{iteration}_{action_type}_{idx}",
                        type=action_type,
                        description=str(raw.get("description") or f"{action_type} for hop {hop.hop_number}"),
                        parameters={"query": query},
                        priority=float(raw.get("priority", 0.5)),
                        expected_output=str(raw.get("expected_output") or "Candidate evidence"),
                    )
                )
            if records:
                planned_actions = records
                break
            if raw_actions:
                retry_guidance = (
                    "The previous plan did not produce any usable new retrieval actions. "
                    f"Invalid actions: {invalid_count}. Duplicate or already-tried actions: {duplicate_count}. "
                    "Return up to the allowed number of concrete, non-duplicate retrieval actions for this hop, or [] if no useful retrieval remains."
                )
            else:
                retry_guidance = (
                    "The previous plan was empty. Return up to the allowed number of concrete retrieval actions for this hop, "
                    "or [] only if you truly believe no useful retrieval remains."
                )
        if not planned_actions and self.settings.enable_generic_retrieval_fallback:
            planned_actions = self._fallback_actions(hop, iteration)
            fallback_used = True
        ended = self._elapsed_s()
        self._record_event(
            stage="hop_plan",
            lane="plan",
            label=f"H{hop.hop_number} retrieval plan",
            start_s=started,
            end_s=ended,
            meta={
                "hop_number": hop.hop_number,
                "iteration": iteration,
                "plan_attempts": attempt_count,
                "fallback_used": fallback_used,
                "planned_actions": [
                    {
                        "action_id": action.id,
                        "type": action.type,
                        "query": str(action.parameters.get("query") or ""),
                    }
                    for action in planned_actions
                ],
            },
        )
        return planned_actions

    async def _web_hits(self, query: str) -> list[dict[str, Any]]:
        if self.settings.no_web:
            return []
        if self.helper_client is not None and self.settings.use_remote_browsecomp:
            return await self.helper_client.search_browsecomp(
                query=query,
                task_id=self.task_id,
                k=self.settings.retrieval_top_k,
                max_chars=self.settings.browsecomp_max_chars,
            )
        if self.browsecomp_tool is None:
            return []
        result = await asyncio.to_thread(self.browsecomp_tool.execute, query, SimpleNamespace(task_id=self.task_id))
        raw_hits = result.get("results")
        return raw_hits if isinstance(raw_hits, list) else []

    async def _local_hits(self, query: str) -> list[dict[str, Any]]:
        if self.helper_client is not None and self.settings.use_remote_local_search:
            return await self.helper_client.search_local(
                task_id=self.task_id,
                query=query,
                k=self.settings.retrieval_top_k,
                threshold=self.settings.local_search_threshold,
            )
        if self.static_local_index is None:
            return []
        return await asyncio.to_thread(
            self.static_local_index.semantic_search,
            query=query,
            top_k=self.settings.retrieval_top_k,
            threshold=self.settings.local_search_threshold,
        )

    def _candidate_from_web_hit(self, hit: dict[str, Any], *, query: str, action_id: str, rank: int) -> CandidateDoc:
        doc_id = str(hit.get("docid") or hit.get("doc_id") or "")
        url = str(hit.get("url") or "")
        title = url or doc_id or "web document"
        excerpt = _truncate(str(hit.get("text") or ""), self.settings.reader_excerpt_chars)
        score = hit.get("score")
        return CandidateDoc(
            doc_id=doc_id or f"web_missing_{action_id}_{rank}",
            source="web",
            title=title,
            locator=url,
            excerpt=excerpt,
            score=float(score) if score is not None else None,
            rank=rank,
            queries=[query],
            search_action_ids=[action_id],
            metadata={"source": hit.get("source"), "raw_docid": hit.get("raw_docid")},
        )

    def _candidate_from_local_hit(self, hit: dict[str, Any], *, query: str, action_id: str, rank: int) -> CandidateDoc:
        metadata = dict(hit.get("metadata") or {})
        file_path = str(metadata.get("file_path") or metadata.get("original_path") or "")
        title = Path(file_path).name if file_path else str(hit.get("doc_id") or "local document")
        score = hit.get("similarity_score", hit.get("relevance_score"))
        excerpt = _truncate(str(hit.get("content") or hit.get("preview") or ""), self.settings.reader_excerpt_chars)
        return CandidateDoc(
            doc_id=str(hit.get("doc_id") or f"local_missing_{action_id}_{rank}"),
            source="local",
            title=title,
            locator=file_path,
            excerpt=excerpt,
            score=float(score) if score is not None else None,
            rank=rank,
            queries=[query],
            search_action_ids=[action_id],
            metadata={"file_path": file_path, "folder_path": metadata.get("folder_path")},
        )

    async def _execute_retrieval_action(
        self, action: RetrievalActionRecord, hop: HopState, iteration: int
    ) -> tuple[RetrievalActionRecord, list[CandidateDoc], dict[str, Any]]:
        query = str(action.parameters.get("query") or "")
        start_s = self._elapsed_s()
        try:
            if action.type == "web_search":
                raw_hits = await self._web_hits(query)
                candidates = [
                    self._candidate_from_web_hit(hit, query=query, action_id=action.id, rank=rank)
                    for rank, hit in enumerate(raw_hits, start=1)
                ]
                output = {
                    "tool": "browsecomp_search",
                    "query": query,
                    "success": True,
                    "data_retrieved": bool(raw_hits),
                    "results_count": len(raw_hits),
                    "results": [candidate.as_card(self.settings.chooser_excerpt_chars) for candidate in candidates],
                }
            else:
                raw_hits = await self._local_hits(query)
                candidates = [
                    self._candidate_from_local_hit(hit, query=query, action_id=action.id, rank=rank)
                    for rank, hit in enumerate(raw_hits, start=1)
                ]
                output = {
                    "tool": "local_document_search",
                    "query": query,
                    "success": True,
                    "data_retrieved": bool(raw_hits),
                    "results_count": len(raw_hits),
                    "results": [candidate.as_card(self.settings.chooser_excerpt_chars) for candidate in candidates],
                }
            action.status = "completed"
            action.actual_output = output
            action.execution_time = self._elapsed_s() - start_s
            action.iteration = iteration
            history_entry = {
                "iteration": iteration,
                "hop_number": hop.hop_number,
                "action_id": action.id,
                "type": action.type,
                "query": query,
                "result_count": len(candidates),
                "top_doc_ids": [candidate.doc_id for candidate in candidates[:3]],
            }
        except Exception as exc:
            action.status = "failed"
            action.error = str(exc)
            action.execution_time = self._elapsed_s() - start_s
            action.iteration = iteration
            candidates = []
            history_entry = {
                "iteration": iteration,
                "hop_number": hop.hop_number,
                "action_id": action.id,
                "type": action.type,
                "query": query,
                "result_count": 0,
                "error": str(exc),
            }
            logger.warning("retrieval action %s failed: %s", action.id, exc)
            self._record_error(
                stage="search",
                hop_number=hop.hop_number,
                iteration=iteration,
                exc=exc,
                lane=f"error:search:{action.id}",
                label=f"{action.type} error",
                meta={"action_id": action.id, "query": query, "action_type": action.type},
            )
        end_s = self._elapsed_s()
        self._record_event(
            stage="search",
            lane=f"search:{action.id}",
            label=f"{action.type}: {_truncate(query, 48)}",
            start_s=start_s,
            end_s=end_s,
            meta={
                "hop_number": hop.hop_number,
                "iteration": iteration,
                "action_id": action.id,
                "action_type": action.type,
                "query": query,
                "result_count": history_entry.get("result_count", 0),
                "top_doc_ids": history_entry.get("top_doc_ids", []),
            },
        )
        return action, candidates, history_entry

    async def _run_retrieval_actions(
        self, actions: list[RetrievalActionRecord], hop: HopState, iteration: int
    ) -> list[CandidateDoc]:
        if not actions:
            return []
        self.action_records.extend(actions)
        results: list[tuple[RetrievalActionRecord, list[CandidateDoc], dict[str, Any]]] = []
        if self.settings.parallel_searches and len(actions) > 1:
            results = list(
                await asyncio.gather(*(self._execute_retrieval_action(action, hop, iteration) for action in actions))
            )
        else:
            for action in actions:
                results.append(await self._execute_retrieval_action(action, hop, iteration))

        candidates_by_doc: dict[str, CandidateDoc] = {}
        for action, candidates, history_entry in results:
            hop.search_history.append(history_entry)
            for candidate in candidates:
                if candidate.doc_id in hop.selected_doc_ids:
                    continue
                existing = candidates_by_doc.get(candidate.doc_id)
                if existing is None:
                    candidates_by_doc[candidate.doc_id] = candidate
                else:
                    existing.merge(candidate)

        ordered = sorted(
            candidates_by_doc.values(),
            key=lambda doc: (doc.rank, -(doc.score if doc.score is not None else -1.0), doc.doc_id),
        )
        trimmed = ordered[: self.settings.max_candidate_cards]
        hop.candidate_doc_ids = [candidate.doc_id for candidate in trimmed]
        hop.candidate_cards = [candidate.as_card(self.settings.chooser_excerpt_chars) for candidate in trimmed]
        return trimmed

    async def _choose_documents(self, hop: HopState, candidates: list[CandidateDoc], iteration: int) -> list[CandidateDoc]:
        if not candidates:
            return []
        if len(candidates) <= self.settings.choose_top_k:
            selected = list(candidates)
            now = self._elapsed_s()
            self._record_event(
                stage="doc_choose",
                lane="choose",
                label=f"H{hop.hop_number} choose docs (skipped)",
                start_s=now,
                end_s=now,
                meta={
                    "hop_number": hop.hop_number,
                    "iteration": iteration,
                    "candidate_count": len(candidates),
                    "selected_doc_ids": [doc.doc_id for doc in selected],
                    "skipped": True,
                },
            )
            self._mark_selected_documents(hop, selected)
            return selected
        prompt = build_doc_choose_prompt(
            current_hop_number=hop.hop_number,
            current_hop_question=self._materialize_question(hop.question),
            resolved_answers=self._resolved_answers(),
            candidate_cards=[candidate.as_card(self.settings.chooser_excerpt_chars) for candidate in candidates],
            choose_top_k=self.settings.choose_top_k,
        )
        self._save_prompt(f"doc_choose_iter{iteration}_hop{hop.hop_number}", prompt)
        started = self._elapsed_s()
        try:
            response = await self.llm_adapter.generate_text(prompt, log_name="doc_choose")
            payload = extract_json(response)
        except Exception as exc:
            if is_llm_infrastructure_error(exc):
                raise
            logger.warning("doc chooser failed for hop %s iteration %s: %s", hop.hop_number, iteration, exc)
            self._record_error(stage="doc_choose", hop_number=hop.hop_number, iteration=iteration, exc=exc)
            payload = {}
        selected_ids = payload.get("selected_doc_ids") if isinstance(payload, dict) else None
        valid_ids = {candidate.doc_id for candidate in candidates}
        normalized_ids: list[str] = []
        if isinstance(selected_ids, list):
            for raw_id in selected_ids:
                doc_id = str(raw_id)
                if doc_id in valid_ids and doc_id not in normalized_ids:
                    normalized_ids.append(doc_id)
        if not normalized_ids:
            normalized_ids = [candidate.doc_id for candidate in candidates[: self.settings.choose_top_k]]
        selected = [candidate for candidate in candidates if candidate.doc_id in normalized_ids][: self.settings.choose_top_k]
        ended = self._elapsed_s()
        self._record_event(
            stage="doc_choose",
            lane="choose",
            label=f"H{hop.hop_number} choose docs",
            start_s=started,
            end_s=ended,
            meta={
                "hop_number": hop.hop_number,
                "iteration": iteration,
                "candidate_count": len(candidates),
                "selected_doc_ids": [doc.doc_id for doc in selected],
            },
        )
        self._mark_selected_documents(hop, selected)
        return selected

    async def _read_document(self, hop: HopState, candidate: CandidateDoc, iteration: int) -> ReaderResult:
        prompt = build_doc_reader_prompt(
            current_hop_number=hop.hop_number,
            current_hop_question=self._materialize_question(hop.question),
            resolved_answers=self._resolved_answers(),
            document=candidate.as_reader_input(self.settings.reader_excerpt_chars),
        )
        self._save_prompt(f"doc_read_iter{iteration}_hop{hop.hop_number}_{candidate.doc_id.replace('/', '_')}", prompt)
        start_s = self._elapsed_s()
        try:
            response = await self.llm_adapter.generate_text(prompt, log_name="doc_read")
            payload = extract_json(response)
            can_answer = bool(payload.get("can_answer"))
            proposed_answer = str(payload.get("proposed_answer") or "").strip()
            justification = str(payload.get("justification") or "").strip()
            if can_answer and justification and f"[DOC:{candidate.doc_id}]" not in justification:
                justification = f"{justification} [DOC:{candidate.doc_id}]"
            result = ReaderResult(
                doc_id=candidate.doc_id,
                source=candidate.source,
                title=candidate.title,
                locator=candidate.locator,
                can_answer=can_answer and bool(proposed_answer),
                proposed_answer=proposed_answer,
                justification=justification,
                confidence=_clamp_confidence(payload.get("confidence"), default=0.0),
                missing_information=str(payload.get("missing_information") or "").strip(),
            )
        except Exception as exc:
            if is_llm_infrastructure_error(exc):
                raise
            logger.warning("doc read failed for %s: %s", candidate.doc_id, exc)
            self._record_error(
                stage="doc_read",
                hop_number=hop.hop_number,
                iteration=iteration,
                exc=exc,
                lane=f"error:read:{candidate.doc_id}",
                label=f"read error {candidate.doc_id}",
                meta={"doc_id": candidate.doc_id, "source": candidate.source},
            )
            result = ReaderResult(
                doc_id=candidate.doc_id,
                source=candidate.source,
                title=candidate.title,
                locator=candidate.locator,
                can_answer=False,
                proposed_answer="",
                justification="",
                confidence=0.0,
                missing_information=str(exc),
            )
        end_s = self._elapsed_s()
        self._record_event(
            stage="doc_read",
            lane=f"read:{candidate.doc_id}",
            label=_truncate(candidate.title or candidate.doc_id, 48),
            start_s=start_s,
            end_s=end_s,
            meta={
                "hop_number": hop.hop_number,
                "iteration": iteration,
                "doc_id": candidate.doc_id,
                "source": candidate.source,
                "title": candidate.title,
                "can_answer": result.can_answer,
                "confidence": round(result.confidence, 3),
            },
        )
        return result

    async def _read_documents(self, hop: HopState, candidates: list[CandidateDoc], iteration: int) -> list[ReaderResult]:
        results: list[ReaderResult] = []
        if self.settings.max_parallel_doc_reads > 1 and len(candidates) > 1:
            sem = asyncio.Semaphore(self.settings.max_parallel_doc_reads)

            async def _bounded_read(candidate: CandidateDoc) -> ReaderResult:
                async with sem:
                    return await self._read_document(hop, candidate, iteration)

            results = list(await asyncio.gather(*(_bounded_read(candidate) for candidate in candidates)))
        else:
            for candidate in candidates:
                results.append(await self._read_document(hop, candidate, iteration))

        for result in results:
            hop.reader_results.append(result.to_dict())
        return results

    def _fallback_resolution(self, reader_results: list[ReaderResult]) -> dict[str, Any]:
        positive = [result for result in reader_results if result.can_answer and result.proposed_answer]
        if not positive:
            return {
                "answered": False,
                "answer": "",
                "justification": "",
                "confidence": 0.0,
                "reason": "No selected document excerpt confidently answered the hop.",
                "next_step": "search_more",
                "selected_doc_ids": [],
            }
        grouped: dict[str, list[ReaderResult]] = {}
        for result in positive:
            grouped.setdefault(_normalize_answer(result.proposed_answer), []).append(result)
        best_key, best_group = max(grouped.items(), key=lambda item: (len(item[1]), max(res.confidence for res in item[1])))
        if len(best_group) == 1 and len(grouped) > 1:
            return {
                "answered": False,
                "answer": "",
                "justification": "",
                "confidence": 0.0,
                "reason": "Selected documents proposed conflicting answers.",
                "next_step": "search_more",
                "selected_doc_ids": [],
            }
        best = max(best_group, key=lambda res: res.confidence)
        return {
            "answered": True,
            "answer": best.proposed_answer,
            "justification": best.justification,
            "confidence": best.confidence,
            "reason": "Fallback resolver selected the strongest consistent document read.",
            "next_step": "done",
            "selected_doc_ids": [],
        }

    def _select_followup_documents(
        self,
        hop: HopState,
        unread_candidates: list[CandidateDoc],
        resolution: dict[str, Any],
    ) -> list[CandidateDoc]:
        if not unread_candidates:
            return []
        valid_ids = {candidate.doc_id for candidate in unread_candidates}
        raw_ids = resolution.get("selected_doc_ids")
        normalized_ids: list[str] = []
        if isinstance(raw_ids, list):
            for raw_id in raw_ids:
                doc_id = str(raw_id)
                if doc_id in valid_ids and doc_id not in normalized_ids:
                    normalized_ids.append(doc_id)
        if not normalized_ids:
            normalized_ids = [candidate.doc_id for candidate in unread_candidates[: self.settings.choose_top_k]]
        selected = [
            candidate for candidate in unread_candidates if candidate.doc_id in normalized_ids
        ][: self.settings.choose_top_k]
        self._mark_selected_documents(hop, selected)
        return selected

    async def _resolve_hop(
        self,
        hop: HopState,
        iteration: int,
        reader_results: list[ReaderResult],
        unread_candidates: list[CandidateDoc],
    ) -> dict[str, Any]:
        prompt = build_hop_resolve_prompt(
            current_hop_number=hop.hop_number,
            current_hop_question=self._materialize_question(hop.question),
            resolved_answers=self._resolved_answers(),
            search_history=self._recent_search_history(hop),
            reader_results=[result.to_dict() for result in reader_results],
            unread_candidate_cards=[candidate.as_card(self.settings.chooser_excerpt_chars) for candidate in unread_candidates],
            choose_top_k=self.settings.choose_top_k,
        )
        self._save_prompt(f"hop_resolve_iter{iteration}_hop{hop.hop_number}", prompt)
        started = self._elapsed_s()
        try:
            response = await self.llm_adapter.generate_text(prompt, log_name="hop_resolve")
            payload = extract_json(response)
            if not isinstance(payload, dict):
                payload = {}
            next_step = str(payload.get("next_step") or "search_more").strip().lower()
            if next_step not in {"done", "read_more", "search_more"}:
                next_step = "search_more"
            resolution = {
                "answered": bool(payload.get("answered")),
                "answer": str(payload.get("answer") or "").strip(),
                "justification": str(payload.get("justification") or "").strip(),
                "confidence": _clamp_confidence(payload.get("confidence"), default=0.0),
                "reason": str(payload.get("reason") or "").strip(),
                "next_step": next_step,
                "selected_doc_ids": payload.get("selected_doc_ids") if isinstance(payload.get("selected_doc_ids"), list) else [],
            }
        except Exception as exc:
            if is_llm_infrastructure_error(exc):
                raise
            logger.warning("hop resolver failed for hop %s iteration %s: %s", hop.hop_number, iteration, exc)
            self._record_error(stage="hop_resolve", hop_number=hop.hop_number, iteration=iteration, exc=exc)
            resolution = self._fallback_resolution(reader_results)
        ended = self._elapsed_s()
        self._record_event(
            stage="hop_resolve",
            lane="resolve",
            label=f"H{hop.hop_number} resolve",
            start_s=started,
            end_s=ended,
            meta={
                "hop_number": hop.hop_number,
                "iteration": iteration,
                "answered": bool(resolution.get("answered")),
                "confidence": round(float(resolution.get("confidence") or 0.0), 3),
                "next_step": resolution.get("next_step", "search_more"),
                "selected_doc_ids": list(resolution.get("selected_doc_ids") or []),
                "unread_candidate_count": len(unread_candidates),
            },
        )
        if resolution["answered"] and resolution["justification"]:
            cited_doc_ids = {reader.doc_id for reader in reader_results if reader.doc_id in resolution["justification"]}
            if not cited_doc_ids and reader_results:
                resolution["justification"] = f"{resolution['justification']} [DOC:{reader_results[0].doc_id}]".strip()
        if resolution["answered"]:
            resolution["next_step"] = "done"
            resolution["selected_doc_ids"] = []
        return resolution

    async def _work_candidate_pool(
        self,
        hop: HopState,
        iteration: int,
        candidates: list[CandidateDoc],
    ) -> dict[str, Any]:
        if not candidates:
            return {
                "answered": False,
                "answer": "",
                "justification": "",
                "confidence": 0.0,
                "reason": "No candidates were retrieved for this hop.",
                "next_step": "search_more",
                "selected_doc_ids": [],
            }

        unread_candidates = self._unread_candidates(hop, candidates)
        if not unread_candidates:
            return {
                "answered": False,
                "answer": "",
                "justification": "",
                "confidence": 0.0,
                "reason": "The current retrieval round only returned documents that were already read earlier in this hop.",
                "next_step": "search_more",
                "selected_doc_ids": [],
            }

        selected = await self._choose_documents(hop, unread_candidates, iteration)
        cumulative_reader_results: list[ReaderResult] = []
        resolution = {
            "answered": False,
            "answer": "",
            "justification": "",
            "confidence": 0.0,
            "reason": "",
            "next_step": "search_more",
            "selected_doc_ids": [],
        }

        while selected:
            batch_results = await self._read_documents(hop, selected, iteration)
            cumulative_reader_results.extend(batch_results)
            unread_candidates = self._unread_candidates(hop, candidates)
            resolution = await self._resolve_hop(
                hop,
                iteration,
                cumulative_reader_results,
                unread_candidates,
            )
            if resolution.get("answered"):
                return resolution
            if resolution.get("next_step") != "read_more" or not unread_candidates:
                return resolution
            selected = self._select_followup_documents(hop, unread_candidates, resolution)

        return resolution

    async def run(self) -> PrivacyHopQAAgentResult:
        max_iterations = self._next_iteration_cap()
        iterations_used = 0
        stalled = False
        for iteration in range(max_iterations):
            hop = self._current_hop()
            if hop is None:
                break
            iterations_used = iteration + 1
            hop.attempts += 1
            retrieval_actions = await self._plan_retrieval_actions(hop, iteration)
            if not retrieval_actions:
                hop.status = "pending"
                hop.resolution_reason = "No usable retrieval plan was produced for this hop in this iteration."
                stalled = True
                continue
            candidates = await self._run_retrieval_actions(retrieval_actions, hop, iteration)
            resolution = await self._work_candidate_pool(hop, iteration, candidates)
            hop.resolution_reason = resolution.get("reason") or ""
            if resolution.get("answered") and resolution.get("answer"):
                hop.answer = str(resolution.get("answer") or "").strip()
                hop.justification = str(resolution.get("justification") or "").strip()
                hop.confidence = _clamp_confidence(resolution.get("confidence"), default=0.0)
                hop.status = "answered"
                stalled = False
            else:
                hop.status = "pending"
                if not candidates:
                    stalled = True
                    continue

        report_started = self._elapsed_s()
        final_report = build_deterministic_report(
            numbered_questions=self.numbered_questions,
            hop_states=[hop.to_dict() for hop in self.hop_states],
        )
        report_path = self.workspace_dir / f"research_report_{int(time.time())}.md"
        report_path.write_text(final_report, encoding="utf-8")
        report_ended = self._elapsed_s()
        self._record_event(
            stage="report",
            lane="report",
            label="Final report",
            start_s=report_started,
            end_s=report_ended,
            meta={
                "resolved_hops": sum(1 for hop in self.hop_states if hop.answer),
                "report_path": str(report_path),
            },
        )

        error_summary = self._error_summary()
        summary = {
            "task_id": self.task_id,
            "chain_id": self.llm_adapter.chain_id,
            "iterations_used": iterations_used,
            "resolved_hops": sum(1 for hop in self.hop_states if hop.answer),
            "total_hops": len(self.hop_states),
            "searches_total": len(self.action_records),
            "docs_read": sum(len(hop.selected_doc_ids) for hop in self.hop_states),
            "duration_s": self._elapsed_s(),
            "stalled": stalled,
            "llm_calls_total": self.llm_adapter.total_calls,
            "prompt_tokens": self.llm_adapter.total_prompt_tokens,
            "output_tokens": self.llm_adapter.total_output_tokens,
            "total_tokens": self.llm_adapter.total_prompt_tokens + self.llm_adapter.total_output_tokens,
            **error_summary,
        }
        trace = {
            "summary": summary,
            "hop_states": [hop.to_dict() for hop in self.hop_states],
            "actions": [action.to_dict() for action in self.action_records],
            "error_records": list(self.error_records),
            "timeline_paths": {},
            "report_path": str(report_path),
        }
        timeline_paths = write_timeline_artifacts(
            self.run_root,
            summary=summary,
            events=self.timeline_events,
            trace=trace,
            report_text=final_report,
            report_path=str(report_path),
        )
        trace["timeline_paths"] = timeline_paths
        trace_path = self.run_root / "trace.json"
        trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")

        parsed_answers = parse_answer_lines(final_report)
        report_metadata = {
            **summary,
            "error_summary": error_summary,
            "trace_path": str(trace_path),
            "timeline_html": timeline_paths["html_path"],
            "report_path": str(report_path),
        }
        return PrivacyHopQAAgentResult(
            final_report=final_report,
            parsed_answers=parsed_answers,
            hop_states=[hop.to_dict() for hop in self.hop_states],
            action_records=[action.to_dict() for action in self.action_records],
            timeline_events=list(self.timeline_events),
            error_records=list(self.error_records),
            report_metadata=report_metadata,
        )
