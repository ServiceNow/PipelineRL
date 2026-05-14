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

import aiohttp

from .llm_adapter import PrivacyHopQALLMAdapter, is_llm_infrastructure_error
from .prompts import (
    build_doc_choose_prompt,
    build_doc_reader_prompt,
    build_hop_plan_prompt,
    build_hop_resolve_prompt,
)
from .reporting import build_deterministic_report, parse_answer_lines
from .reward import normalize_answer
from .settings import PrivacyHopQASettings
from .timeline import write_timeline_artifacts
from .utils import extract_json

logger = logging.getLogger(__name__)


class PrivacyHopQAHelperInfrastructureError(TimeoutError):
    """Retrieval-helper transport/server failure that should retry the rollout."""


def is_helper_infrastructure_error(exc: BaseException) -> bool:
    """Return true for helper transport/server failures, not invalid retrieval plans."""
    if isinstance(exc, PrivacyHopQAHelperInfrastructureError):
        return True
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status == 429 or exc.status >= 500
    if isinstance(
        exc,
        (
            aiohttp.ClientConnectionError,
            aiohttp.ClientPayloadError,
            aiohttp.ServerTimeoutError,
            asyncio.TimeoutError,
            TimeoutError,
        ),
    ):
        return True
    text = str(exc).lower()
    return "server disconnected" in text or "connection reset" in text


def _truncate(text: str, limit: int) -> str:
    value = str(text or "")
    if int(limit) <= 0:
        return ""
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


_LEGACY_RETRIEVAL_CONTEXT_MODES = {"legacy", "legacy_prefix", "document_prefix", "prefix"}
_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whose",
    "why",
    "with",
}


def _uses_legacy_retrieval_context(mode: str) -> bool:
    return str(mode or "").strip().lower() in _LEGACY_RETRIEVAL_CONTEXT_MODES


def _query_terms(query: str) -> set[str]:
    terms = set(re.findall(r"[a-z0-9][a-z0-9._-]*", query.lower()))
    return {term for term in terms if len(term) > 2 and term not in _QUERY_STOPWORDS}


def _window_spans(text: str, window_chars: int, overlap_chars: int) -> list[tuple[int, int]]:
    if not text:
        return [(0, 0)]
    window_chars = max(1, int(window_chars))
    overlap_chars = max(0, min(int(overlap_chars), window_chars - 1))
    step = max(1, window_chars - overlap_chars)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + window_chars)
        spans.append((start, end))
        if end >= len(text):
            break
        start += step
    return spans


def _query_match_score(query: str, text: str) -> int:
    terms = _query_terms(query)
    if not terms or not text:
        return 0
    lower = text.lower()
    return sum(1 for term in terms if term in lower)


def _query_snippet(text: str, query: str, limit: int) -> str:
    if not text:
        return ""
    terms = sorted(_query_terms(query), key=len, reverse=True)
    lower = text.lower()
    pos = -1
    for term in terms:
        pos = lower.find(term)
        if pos >= 0:
            break
    if pos < 0:
        return _truncate(text, limit)
    half = max(1, limit // 2)
    start = max(0, pos - half)
    end = min(len(text), start + limit)
    start = max(0, end - limit)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


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


def _dedupe_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        key = normalize_answer(text)
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _compact_reader_result_dict(raw: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "doc_id",
        "source",
        "title",
        "locator",
        "can_answer",
        "proposed_answer",
        "confidence",
        "parent_doc_id",
        "evidence_window_id",
        "window_index",
        "window_count",
    ):
        if key in raw:
            value = raw[key]
            if key in {"title", "locator"}:
                value = _truncate(str(value), 180)
            compact[key] = value
    justification = str(raw.get("justification") or "").strip()
    if justification:
        compact["justification"] = _truncate(justification, 600)
    missing = str(raw.get("missing_information") or "").strip()
    if missing:
        compact["missing_information"] = _truncate(missing, 240)
    metadata = raw.get("metadata")
    if isinstance(metadata, dict):
        useful_metadata = {
            key: metadata[key]
            for key in ("parent_doc_id", "window_index", "window_count", "best_rank", "hit_count", "top_queries")
            if key in metadata
        }
        if useful_metadata:
            compact["metadata"] = useful_metadata
    return compact


def _resolver_reader_result(result: "ReaderResult") -> dict[str, Any]:
    return _compact_reader_result_dict(result.to_dict())


def _resolver_candidate_card(candidate: "CandidateDoc", excerpt_chars: int) -> dict[str, Any]:
    excerpt_chars = max(0, int(excerpt_chars))
    card = candidate.as_card(max(1, excerpt_chars))
    compact: dict[str, Any] = {}
    for key in (
        "doc_id",
        "source",
        "title",
        "locator",
        "score",
        "rank",
        "parent_doc_id",
        "window_index",
        "window_count",
        "best_rank",
        "hit_count",
        "top_queries",
    ):
        if key in card:
            value = card[key]
            if key in {"title", "locator", "parent_doc_id"}:
                value = _truncate(str(value), 180)
            elif key == "top_queries" and isinstance(value, list):
                value = [_truncate(str(item), 100) for item in value[:5]]
            compact[key] = value
    excerpt = str(card.get("excerpt") or "").strip()
    if excerpt_chars > 0 and excerpt:
        compact["snippet"] = _truncate(excerpt, excerpt_chars)
    return compact


def _candidate_parent_id(candidate: "CandidateDoc") -> str:
    return str(candidate.metadata.get("parent_doc_id") or candidate.doc_id)


def _candidate_parent_groups(candidates: list["CandidateDoc"]) -> dict[str, list["CandidateDoc"]]:
    groups: dict[str, list["CandidateDoc"]] = {}
    for candidate in candidates:
        groups.setdefault(_candidate_parent_id(candidate), []).append(candidate)
    return groups


def _candidate_window_index(candidate: "CandidateDoc") -> int:
    value = candidate.metadata.get("window_index")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _candidate_window_count(candidate: "CandidateDoc") -> int:
    value = candidate.metadata.get("window_count")
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _candidate_query_match_score(candidate: "CandidateDoc") -> int:
    value = candidate.metadata.get("query_match_score")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _candidate_rank_score_key(candidate: "CandidateDoc") -> tuple[int, float, str]:
    score = candidate.score if candidate.score is not None else -1.0
    return (candidate.rank, -float(score), candidate.doc_id)


def _parent_candidate_card(candidates: list["CandidateDoc"], excerpt_chars: int) -> dict[str, Any]:
    first = candidates[0]
    parent_doc_id = _candidate_parent_id(first)
    best = min(
        candidates,
        key=lambda candidate: (
            candidate.rank,
            -(candidate.score if candidate.score is not None else -1.0),
            candidate.doc_id,
        ),
    )
    seen_queries: set[str] = set()
    top_queries: list[str] = []
    for candidate in candidates:
        for query in candidate.queries:
            if query not in seen_queries:
                seen_queries.add(query)
                top_queries.append(query)
    top_queries = top_queries[:5]
    total_window_count = max(int(candidate.metadata.get("window_count") or 1) for candidate in candidates)
    card = {
        "doc_id": parent_doc_id,
        "candidate_type": "parent_document",
        "source": first.source,
        "title": _truncate(first.title, 180),
        "locator": _truncate(first.locator, 180),
        "best_rank": min(candidate.rank for candidate in candidates),
        "best_score": max((candidate.score for candidate in candidates if candidate.score is not None), default=None),
        "matched_window_count": len(candidates),
        "total_window_count": total_window_count,
        "top_queries": [_truncate(query, 100) for query in top_queries],
        "representative_window_id": best.doc_id,
    }
    snippet = str(best.metadata.get("snippet") or best.excerpt).strip()
    if excerpt_chars > 0 and snippet:
        card["snippet"] = _truncate(snippet, excerpt_chars)
    memory_candidates = [candidate for candidate in candidates if candidate.metadata.get("memory_seed")]
    if memory_candidates:
        card["memory_seed"] = True
        supported_hops: list[int] = []
        supported_answers: list[str] = []
        for candidate in memory_candidates:
            for hop_number in candidate.metadata.get("supported_hops") or []:
                if isinstance(hop_number, int) and hop_number not in supported_hops:
                    supported_hops.append(hop_number)
            for answer in candidate.metadata.get("supported_answers") or []:
                answer_text = str(answer or "").strip()
                if answer_text and answer_text not in supported_answers:
                    supported_answers.append(answer_text)
        if supported_hops:
            card["supported_hops"] = sorted(supported_hops)
        if supported_answers:
            card["supported_answers"] = [_truncate(answer, 80) for answer in supported_answers[-3:]]
    return card


def _candidate_cards_for_prompt(
    candidates: list["CandidateDoc"],
    preferred_excerpt_chars: int,
    *,
    max_cards: int | None = None,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    parent_items = list(_candidate_parent_groups(candidates).items())
    if max_cards is not None and int(max_cards) > 0:
        parent_items = parent_items[: int(max_cards)]
    target_snippet_chars = 4000
    excerpt_chars = min(
        max(int(preferred_excerpt_chars), 0),
        max(16, target_snippet_chars // max(len(parent_items), 1)),
    )
    return [_parent_candidate_card(group, excerpt_chars) for _parent_id, group in parent_items]


def _reader_results_for_resolver(results: list["ReaderResult"]) -> list["ReaderResult"]:
    positives = [result for result in results if result.can_answer and result.proposed_answer.strip()]
    recent = results[-12:]
    ordered: list["ReaderResult"] = []
    seen: set[str] = set()
    for result in [*positives[-12:], *recent]:
        key = result.doc_id
        if key not in seen:
            seen.add(key)
            ordered.append(result)
    return ordered[-24:]


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
        snippet = str(self.metadata.get("snippet") or self.excerpt)
        card = {
            "doc_id": self.doc_id,
            "source": self.source,
            "title": self.title,
            "best_rank": self.rank,
            "hit_count": len(self.queries),
            "top_queries": self.queries[:5],
            "snippet": _truncate(snippet, chooser_chars),
        }
        parent_doc_id = self.metadata.get("parent_doc_id")
        if parent_doc_id and parent_doc_id != self.doc_id:
            card["parent_doc_id"] = parent_doc_id
        for key in (
            "evidence_window_id",
            "window_index",
            "window_count",
            "window_char_start",
            "window_char_end",
            "retrieval_context_mode",
            "retrieval_chunk_chars",
            "retrieval_chunk_overlap_chars",
            "memory_seed",
            "supported_hops",
            "supported_answers",
        ):
            if key in self.metadata:
                card[key] = self.metadata[key]
        return card

    def as_reader_input(self, excerpt_chars: int) -> dict[str, Any]:
        document = {
            "doc_id": self.doc_id,
            "source": self.source,
            "title": self.title,
            "locator": self.locator,
            "queries": self.queries,
            "excerpt": self.excerpt,
        }
        parent_doc_id = self.metadata.get("parent_doc_id")
        if parent_doc_id and parent_doc_id != self.doc_id:
            document["parent_doc_id"] = parent_doc_id
        for key in (
            "evidence_window_id",
            "window_index",
            "window_count",
            "window_char_start",
            "window_char_end",
            "retrieval_context_mode",
            "retrieval_chunk_chars",
            "retrieval_chunk_overlap_chars",
        ):
            if key in self.metadata:
                document[key] = self.metadata[key]
        return document

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
    parent_doc_id: str | None = None
    evidence_window_id: str | None = None
    window_index: int | None = None
    window_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UsefulDocumentMemory:
    parent_doc_id: str
    source: str
    title: str
    locator: str
    supported_hops: list[int] = field(default_factory=list)
    supported_answers: list[str] = field(default_factory=list)
    justifications: list[str] = field(default_factory=list)
    window_doc_ids: list[str] = field(default_factory=list)
    last_used_hop: int = 0

    def remember(
        self,
        hop_number: int,
        answer: str,
        justification: str,
        window_doc_id: str | None,
    ) -> None:
        if hop_number not in self.supported_hops:
            self.supported_hops.append(hop_number)
            self.supported_hops.sort()
        answer = str(answer or "").strip()
        if answer and answer not in self.supported_answers:
            self.supported_answers.append(answer)
        justification = str(justification or "").strip()
        if justification and justification not in self.justifications:
            self.justifications.append(justification)
        if window_doc_id and window_doc_id not in self.window_doc_ids:
            self.window_doc_ids.append(window_doc_id)
        self.last_used_hop = max(self.last_used_hop, hop_number)

    def to_prompt_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "doc_id": self.parent_doc_id,
            "source": self.source,
            "title": _truncate(self.title, 160),
            "locator": _truncate(self.locator, 180),
            "supported_hops": list(self.supported_hops),
            "supported_answers": [_truncate(answer, 80) for answer in self.supported_answers[-3:]],
        }
        if self.justifications:
            payload["support"] = _truncate(self.justifications[-1], 180)
        if self.window_doc_ids:
            payload["available_window_count"] = len(self.window_doc_ids)
        return payload


@dataclass
class HopState:
    hop_number: int
    question: str
    canonical_answer: str = ""
    accepted_answer_variants: list[str] = field(default_factory=list)
    alternate_valid_answers: list[str] = field(default_factory=list)
    gold_doc_id: str | None = None
    gold_doc_ref: str | None = None
    hop_type: str | None = None
    status: str = "pending"
    attempts: int = 0
    answer: str | None = None
    answer_for_context: str | None = None
    matched_accepted_variant: bool = False
    justification: str | None = None
    confidence: float = 0.0
    resolution_reason: str | None = None
    search_history: list[dict[str, Any]] = field(default_factory=list)
    selected_doc_ids: list[str] = field(default_factory=list)
    reader_results: list[dict[str, Any]] = field(default_factory=list)
    candidate_doc_ids: list[str] = field(default_factory=list)
    candidate_cards: list[dict[str, Any]] = field(default_factory=list)
    gold_parent_retrieved_before_trim: bool = False
    gold_parent_retrieved: bool = False
    gold_parent_read: bool = False
    gold_candidate_doc_ids: list[str] = field(default_factory=list)

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


class PrivacyHopQAProtocolError(RuntimeError):
    """Model/protocol failure during a rollout.

    These are bad model outputs or prompt/protocol failures, not infrastructure
    failures. Training should stop the rollout and score the outcome as zero.
    """


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
        self.hop_states = []
        for hop in hops:
            canonical_answer = str(hop.get("answer") or "").strip()
            self.hop_states.append(
                HopState(
                    hop_number=int(hop["hop_number"]),
                    question=str(hop.get("question") or ""),
                    canonical_answer=canonical_answer,
                    accepted_answer_variants=_dedupe_strings(
                        [
                            canonical_answer,
                            *(hop.get("accepted_answer_variants") or []),
                            *(hop.get("alternate_valid_answers") or []),
                        ]
                    ),
                    alternate_valid_answers=_dedupe_strings(hop.get("alternate_valid_answers") or []),
                    gold_doc_id=(str(hop.get("doc_id")).strip() if hop.get("doc_id") else None),
                    gold_doc_ref=(str(hop.get("doc_ref")).strip() if hop.get("doc_ref") else None),
                    hop_type=(str(hop.get("hop_type")).strip() if hop.get("hop_type") else None),
                )
            )
        self.helper_client = helper_client
        self.static_local_index = static_local_index
        self.browsecomp_tool = browsecomp_tool
        self.action_records: list[RetrievalActionRecord] = []
        self.timeline_events: list[dict[str, Any]] = []
        self.error_records: list[dict[str, Any]] = []
        self._errors_by_stage: Counter[str] = Counter()
        self._errors_by_kind: Counter[str] = Counter()
        self._errors_by_stage_kind: dict[str, Counter[str]] = defaultdict(Counter)
        self._retrieval_stats: dict[str, int] = {}
        self._document_memory: dict[str, UsefulDocumentMemory] = {}
        self._candidate_cache_by_doc_id: dict[str, CandidateDoc] = {}
        self._candidate_doc_ids_by_parent: dict[str, list[str]] = defaultdict(list)
        self.prompt_dir = self.run_root / "prompts"
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        self._run_started_at = _current_timestamp()

    def _uses_legacy_retrieval_context(self) -> bool:
        return _uses_legacy_retrieval_context(self.settings.retrieval_context_mode)

    def _browsecomp_request_chars(self) -> int:
        if self._uses_legacy_retrieval_context():
            return self.settings.browsecomp_max_chars
        if self.settings.retrieval_source_chars <= 0:
            return 0
        return max(self.settings.browsecomp_max_chars, self.settings.retrieval_source_chars)

    def _reader_excerpt_chars(self) -> int:
        if self._uses_legacy_retrieval_context():
            return self.settings.reader_excerpt_chars
        return max(self.settings.reader_excerpt_chars, self.settings.reader_window_chars)

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
        if text.startswith("resolver_wipe:"):
            return "resolver_wipe"
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
        if hop_number is not None and iteration is not None:
            captured_call_index = self.llm_adapter.last_captured_call_index_by_key.get(
                (stage, hop_number, iteration)
            )
            if captured_call_index is not None:
                record["captured_call_index"] = int(captured_call_index)
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

    def _inc_retrieval_stat(self, key: str, value: int = 1) -> None:
        self._retrieval_stats[key] = int(self._retrieval_stats.get(key, 0)) + int(value)

    def _max_retrieval_stat(self, key: str, value: int) -> None:
        self._retrieval_stats[key] = max(int(self._retrieval_stats.get(key, 0)), int(value))

    def _retrieval_summary(self) -> dict[str, Any]:
        gold_hops = [hop for hop in self.hop_states if hop.gold_doc_id]
        gold_hop_count = len(gold_hops)
        gold_parent_retrieved_before_trim = sum(1 for hop in gold_hops if hop.gold_parent_retrieved_before_trim)
        gold_parent_retrieved = sum(1 for hop in gold_hops if hop.gold_parent_retrieved)
        gold_parent_read = sum(1 for hop in gold_hops if hop.gold_parent_read)

        def _rate(numerator: int, denominator: int) -> float:
            return float(numerator) / float(denominator) if denominator else 0.0

        selected_windows_read = int(self._retrieval_stats.get("selected_windows_read", 0))
        selected_parent_windows_available = int(self._retrieval_stats.get("selected_parent_windows_available", 0))
        large_parent_windows_read = int(self._retrieval_stats.get("large_parent_windows_read", 0))
        large_parent_windows_available = int(self._retrieval_stats.get("large_parent_windows_available", 0))
        summary = {
            **{key: int(value) for key, value in sorted(self._retrieval_stats.items())},
            "gold_parent_hops": gold_hop_count,
            "gold_parent_retrieved_before_trim_hops": gold_parent_retrieved_before_trim,
            "gold_parent_retrieved_hops": gold_parent_retrieved,
            "gold_parent_read_hops": gold_parent_read,
            "gold_parent_retrieved_before_trim_rate": _rate(gold_parent_retrieved_before_trim, gold_hop_count),
            "gold_parent_retrieved_rate": _rate(gold_parent_retrieved, gold_hop_count),
            "gold_parent_read_rate": _rate(gold_parent_read, gold_hop_count),
            "gold_parent_missed_after_trim_hops": sum(
                1 for hop in gold_hops if hop.gold_parent_retrieved_before_trim and not hop.gold_parent_retrieved
            ),
            "gold_parent_retrieved_not_read_hops": sum(
                1 for hop in gold_hops if hop.gold_parent_retrieved and not hop.gold_parent_read
            ),
            "selected_window_read_ratio": _rate(selected_windows_read, selected_parent_windows_available),
            "large_parent_window_read_ratio": _rate(large_parent_windows_read, large_parent_windows_available),
        }
        return summary

    def _answer_matches_accepted_variant(self, hop: HopState, answer: str | None) -> bool:
        predicted = normalize_answer(answer)
        if not predicted:
            return False
        return any(predicted == normalize_answer(value) for value in hop.accepted_answer_variants)

    def _context_answer_for_raw(self, hop: HopState, answer: str | None) -> str:
        raw = str(answer or "").strip()
        if self._answer_matches_accepted_variant(hop, raw) and hop.canonical_answer:
            return hop.canonical_answer
        return raw

    def _answer_for_context(self, hop: HopState) -> str:
        return str(hop.answer_for_context or hop.answer or "").strip()

    def _current_answers_so_far(self) -> str:
        lines: list[str] = []
        for hop in self.hop_states:
            if hop.answer:
                context_answer = self._answer_for_context(hop)
                lines.append(f"({hop.hop_number}) {hop.answer}")
                if context_answer and _normalize_answer(context_answer) != _normalize_answer(hop.answer):
                    lines.append(f"    For references to ({hop.hop_number}), use: {context_answer}")
                if hop.justification:
                    lines.append(f"    Justification: {_truncate(hop.justification, 420)}")
        return "\n".join(lines) if lines else "None yet."

    def _materialize_question(self, question: str) -> str:
        resolved = {str(hop.hop_number): self._answer_for_context(hop) for hop in self.hop_states if hop.answer}

        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            return resolved.get(key, match.group(0))

        return re.sub(r"\((\d+)\)", replace, question)

    def _current_hop(self) -> HopState | None:
        for hop in self.hop_states:
            if not hop.answer:
                return hop
        return None

    def _document_memory_prompt_cards(self) -> list[dict[str, Any]]:
        # Deduped parent-doc memory from earlier answered hops. This is a hint,
        # not evidence for the current hop until one of these docs is read again.
        entries = sorted(
            self._document_memory.values(),
            key=lambda entry: (entry.last_used_hop, entry.parent_doc_id),
            reverse=True,
        )
        return [entry.to_prompt_dict() for entry in entries]

    def _cache_candidates(self, candidates: list[CandidateDoc]) -> None:
        # Candidate objects hold the full evidence window text. Keep the windows
        # around so a later hop can retry a previously useful parent document.
        for candidate in candidates:
            self._candidate_cache_by_doc_id[candidate.doc_id] = candidate
            parent_doc_id = _candidate_parent_id(candidate)
            parent_doc_ids = self._candidate_doc_ids_by_parent[parent_doc_id]
            if candidate.doc_id not in parent_doc_ids:
                parent_doc_ids.append(candidate.doc_id)

    def _clone_memory_candidate(self, candidate: CandidateDoc, memory: UsefulDocumentMemory) -> CandidateDoc:
        metadata = dict(candidate.metadata)
        metadata.update(
            {
                "memory_seed": True,
                "supported_hops": list(memory.supported_hops),
                "supported_answers": list(memory.supported_answers[-3:]),
            }
        )
        queries = ["previously useful document"]
        for query in candidate.queries:
            if query not in queries:
                queries.append(query)
        return CandidateDoc(
            doc_id=candidate.doc_id,
            source=candidate.source,
            title=candidate.title,
            locator=candidate.locator,
            excerpt=candidate.excerpt,
            score=candidate.score,
            rank=0,
            queries=queries,
            search_action_ids=[f"memory_hops_{'_'.join(str(hop) for hop in memory.supported_hops)}"],
            metadata=metadata,
        )

    def _memory_seed_candidates(self, hop: HopState) -> list[CandidateDoc]:
        seeds: list[CandidateDoc] = []
        seen_doc_ids: set[str] = set()
        for memory in sorted(self._document_memory.values(), key=lambda entry: entry.last_used_hop, reverse=True):
            for doc_id in memory.window_doc_ids:
                if doc_id in hop.selected_doc_ids or doc_id in seen_doc_ids:
                    continue
                candidate = self._candidate_cache_by_doc_id.get(doc_id)
                if candidate is None:
                    continue
                seeds.append(self._clone_memory_candidate(candidate, memory))
                seen_doc_ids.add(doc_id)
        if seeds:
            self._inc_retrieval_stat("memory_seed_windows", len(seeds))
            self._inc_retrieval_stat("memory_seed_parent_docs", len({_candidate_parent_id(seed) for seed in seeds}))
        return seeds

    def _recent_search_history(self, hop: HopState) -> list[dict[str, Any]]:
        return hop.search_history[-self.settings.max_history_entries :]

    def _recent_reader_results(self, hop: HopState) -> list[dict[str, Any]]:
        return [
            _compact_reader_result_dict(result)
            for result in hop.reader_results[-self.settings.max_history_entries :]
            if isinstance(result, dict)
        ]

    def _unread_candidates(self, hop: HopState, candidates: list[CandidateDoc]) -> list[CandidateDoc]:
        return [candidate for candidate in candidates if candidate.doc_id not in hop.selected_doc_ids]

    def _mark_selected_documents(self, hop: HopState, selected: list[CandidateDoc]) -> None:
        for doc in selected:
            if doc.doc_id not in hop.selected_doc_ids:
                hop.selected_doc_ids.append(doc.doc_id)
            if hop.gold_doc_id and _candidate_parent_id(doc) == hop.gold_doc_id:
                hop.gold_parent_read = True
                if doc.doc_id not in hop.gold_candidate_doc_ids:
                    hop.gold_candidate_doc_ids.append(doc.doc_id)

    def _select_windows_for_parent_read(self, group: list[CandidateDoc]) -> list[CandidateDoc]:
        if not group:
            return []
        ordered_by_index = sorted(group, key=lambda candidate: (_candidate_window_index(candidate), candidate.doc_id))
        policy = str(self.settings.large_parent_window_policy or "top_with_neighbors").strip().lower()
        full_window_count = max(_candidate_window_count(candidate) for candidate in group)
        read_all_threshold = max(0, int(self.settings.read_all_windows_max_count))
        if policy in {"all", "read_all", "legacy"}:
            return ordered_by_index
        if read_all_threshold <= 0 or full_window_count <= read_all_threshold or len(group) <= read_all_threshold:
            return ordered_by_index

        top_windows = max(1, int(self.settings.large_parent_top_windows))
        neighbor_windows = max(0, int(self.settings.large_parent_neighbor_windows))
        by_index = {_candidate_window_index(candidate): candidate for candidate in group}
        ranked = sorted(
            group,
            key=lambda candidate: (
                -_candidate_query_match_score(candidate),
                candidate.rank,
                -(candidate.score if candidate.score is not None else -1.0),
                _candidate_window_index(candidate),
                candidate.doc_id,
            ),
        )
        keep_indices: set[int] = set()
        for candidate in ranked[:top_windows]:
            center = _candidate_window_index(candidate)
            for index in range(center - neighbor_windows, center + neighbor_windows + 1):
                if index in by_index:
                    keep_indices.add(index)
        if not keep_indices:
            keep_indices = {_candidate_window_index(candidate) for candidate in ranked[:top_windows]}
        selected = [by_index[index] for index in sorted(keep_indices)]
        max_read_windows = int(self.settings.large_parent_max_read_windows)
        if max_read_windows > 0 and len(selected) > max_read_windows:
            selected = sorted(
                selected,
                key=lambda candidate: (
                    -_candidate_query_match_score(candidate),
                    *_candidate_rank_score_key(candidate),
                    _candidate_window_index(candidate),
                ),
            )[:max_read_windows]
            selected = sorted(selected, key=lambda candidate: (_candidate_window_index(candidate), candidate.doc_id))
        return selected

    def _expand_selected_parent_documents(
        self,
        parent_groups: dict[str, list[CandidateDoc]],
        selected_parent_ids: list[str],
        *,
        hop: HopState,
        iteration: int,
        selection_stage: str,
    ) -> tuple[list[CandidateDoc], dict[str, Any]]:
        selected: list[CandidateDoc] = []
        per_parent: list[dict[str, Any]] = []
        read_all_threshold = max(0, int(self.settings.read_all_windows_max_count))
        for parent_id in selected_parent_ids:
            group = parent_groups.get(parent_id) or []
            if not group:
                continue
            full_window_count = max(_candidate_window_count(candidate) for candidate in group)
            chosen = self._select_windows_for_parent_read(group)
            selected.extend(chosen)
            is_large_parent = full_window_count > read_all_threshold and str(self.settings.large_parent_window_policy).lower() not in {"all", "read_all", "legacy"}
            record = {
                "parent_doc_id": parent_id,
                "available_windows": len(group),
                "total_windows": full_window_count,
                "selected_windows": len(chosen),
                "selected_window_ids": [candidate.doc_id for candidate in chosen],
                "selected_window_indices": [_candidate_window_index(candidate) for candidate in chosen],
                "large_parent_window_limited": bool(is_large_parent and len(chosen) < len(group)),
            }
            per_parent.append(record)
            self._inc_retrieval_stat("selected_parent_docs", 1)
            self._inc_retrieval_stat("selected_parent_windows_available", len(group))
            self._inc_retrieval_stat("selected_windows_read", len(chosen))
            self._max_retrieval_stat("max_selected_parent_windows_available", len(group))
            self._max_retrieval_stat("max_selected_parent_windows_read", len(chosen))
            if is_large_parent:
                self._inc_retrieval_stat("large_parent_docs_selected", 1)
                self._inc_retrieval_stat("large_parent_windows_available", len(group))
                self._inc_retrieval_stat("large_parent_windows_read", len(chosen))
                if len(chosen) < len(group):
                    self._inc_retrieval_stat("large_parent_docs_limited", 1)

        meta = {
            "hop_number": hop.hop_number,
            "iteration": iteration,
            "selection_stage": selection_stage,
            "window_policy": str(self.settings.large_parent_window_policy),
            "read_all_windows_max_count": int(self.settings.read_all_windows_max_count),
            "large_parent_top_windows": int(self.settings.large_parent_top_windows),
            "large_parent_neighbor_windows": int(self.settings.large_parent_neighbor_windows),
            "selected_parent_doc_ids": selected_parent_ids,
            "selected_doc_ids": [doc.doc_id for doc in selected],
            "selected_parent_window_details": per_parent,
            "selected_parent_windows_available": sum(item["available_windows"] for item in per_parent),
            "selected_windows_read": len(selected),
            "large_parent_docs_selected": sum(1 for item in per_parent if item["large_parent_window_limited"]),
        }
        return selected, meta

    def _next_iteration_cap(self) -> int:
        total_hops = max(len(self.hop_states), 1)
        per_hop_budget = max(int(self.settings.iteration_budget_per_hop), 1)
        return min(self.settings.max_iterations, per_hop_budget * total_hops)

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
                current_answers_so_far=self._current_answers_so_far(),
                previously_useful_documents=self._document_memory_prompt_cards(),
                search_history=self._recent_search_history(hop),
                recent_reader_results=self._recent_reader_results(hop),
                company_name=self.company_name,
                company_description=self.company_description,
                max_parallel_retrieval_actions=self.settings.max_parallel_retrieval_actions,
                no_web=self.settings.no_web,
                retry_guidance=retry_guidance,
                planner_privacy_prompt=self.settings.planner_privacy_prompt,
            )
            prompt_name = f"hop_plan_iter{iteration}_hop{hop.hop_number}"
            if plan_attempt > 0:
                prompt_name = f"{prompt_name}_attempt{plan_attempt + 1}"
            self._save_prompt(prompt_name, prompt)
            try:
                response = await self.llm_adapter.generate_text(
                    prompt,
                    log_name="hop_plan",
                    max_tokens=self.settings.hop_plan_max_tokens,
                    max_context_tokens=self.settings.generation_context_limit_tokens,
                    context_margin_tokens=self.settings.generation_context_margin_tokens,
                    hop_number=hop.hop_number,
                    iteration=iteration,
                )
                payload = extract_json(response)
            except Exception as exc:
                if is_llm_infrastructure_error(exc):
                    raise
                logger.warning("hop planner failed for hop %s iteration %s attempt %s: %s", hop.hop_number, iteration, attempt_count, exc)
                self._record_error(stage="hop_plan", hop_number=hop.hop_number, iteration=iteration, exc=exc)
                raise PrivacyHopQAProtocolError(f"hop_plan protocol error: {exc}") from exc

            raw_actions = payload if isinstance(payload, list) else []
            records: list[RetrievalActionRecord] = []
            seen_keys: set[tuple[str, str]] = set()
            seen_recent = {
                (entry.get("type", ""), _normalize_query(entry.get("query", "")))
                for entry in self._recent_search_history(hop)
            }
            tried_local_for_hop = any(entry.get("type") == "local_document_search" for entry in hop.search_history)
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
            if (
                self.settings.enable_local_search_bootstrap
                and records
                and not tried_local_for_hop
                and not any(record.type == "local_document_search" for record in records)
            ):
                local_query = self._materialize_question(hop.question).strip()
                local_key = ("local_document_search", _normalize_query(local_query))
                if local_query and local_key not in seen_recent:
                    bootstrap_local = RetrievalActionRecord(
                        id=f"iter{iteration}_local_bootstrap",
                        type="local_document_search",
                        description=f"Search local company documents for hop {hop.hop_number}",
                        parameters={"query": local_query},
                        priority=1.0,
                        expected_output="Candidate local evidence for the current hop",
                    )
                    records = [bootstrap_local, *records[: self.settings.max_parallel_retrieval_actions - 1]]
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
                max_chars=self._browsecomp_request_chars(),
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
        excerpt = str(hit.get("text") or "")
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
            metadata={
                "source": hit.get("source"),
                "raw_docid": hit.get("raw_docid"),
                "parent_doc_id": doc_id or f"web_missing_{action_id}_{rank}",
                "retrieval_context_mode": "legacy_prefix",
            },
        )

    def _candidates_from_web_hit(self, hit: dict[str, Any], *, query: str, action_id: str, rank: int) -> list[CandidateDoc]:
        if self._uses_legacy_retrieval_context():
            return [self._candidate_from_web_hit(hit, query=query, action_id=action_id, rank=rank)]

        parent_doc_id = str(hit.get("docid") or hit.get("doc_id") or "") or f"web_missing_{action_id}_{rank}"
        url = str(hit.get("url") or "")
        title = url or parent_doc_id or "web document"
        source_text = str(hit.get("text") or "")
        score = hit.get("score")
        numeric_score = float(score) if score is not None else None
        spans = _window_spans(
            source_text,
            window_chars=self.settings.reader_window_chars,
            overlap_chars=self.settings.reader_window_overlap_chars,
        )
        total_windows = len(spans)
        retrieval_spans = _window_spans(
            source_text,
            window_chars=self.settings.retrieval_chunk_chars,
            overlap_chars=self.settings.retrieval_chunk_overlap_chars,
        )
        window_scores = [0 for _span in spans]
        for chunk_start, chunk_end in retrieval_spans:
            chunk_score = _query_match_score(query, source_text[chunk_start:chunk_end])
            for index, (start, end) in enumerate(spans):
                if chunk_start < end and chunk_end > start:
                    window_scores[index] = max(window_scores[index], chunk_score)
        scored_spans = [
            (window_scores[index], index, start, end)
            for index, (start, end) in enumerate(spans)
        ]
        max_windows = int(self.settings.max_windows_per_parent)
        if max_windows > 0 and len(scored_spans) > max_windows:
            keep = {
                index
                for _score, index, _start, _end in sorted(scored_spans, key=lambda item: (-item[0], item[1]))[
                    :max_windows
                ]
            }
            scored_spans = [item for item in scored_spans if item[1] in keep]

        candidates: list[CandidateDoc] = []
        for match_score, index, start, end in scored_spans:
            evidence_id = f"{parent_doc_id}#w{index + 1:02d}of{total_windows:02d}"
            window_text = source_text[start:end]
            candidates.append(
                CandidateDoc(
                    doc_id=evidence_id,
                    source="web",
                    title=title,
                    locator=url,
                    excerpt=window_text,
                    score=(numeric_score if numeric_score is not None else 0.0) + (match_score / 1000.0),
                    rank=rank,
                    queries=[query],
                    search_action_ids=[action_id],
                    metadata={
                        "source": hit.get("source"),
                        "raw_docid": hit.get("raw_docid"),
                        "parent_doc_id": parent_doc_id,
                        "evidence_window_id": evidence_id,
                        "retrieval_context_mode": str(self.settings.retrieval_context_mode),
                        "window_index": index + 1,
                        "window_count": total_windows,
                        "window_char_start": start,
                        "window_char_end": end,
                        "source_text_chars": len(source_text),
                        "query_match_score": match_score,
                        "retrieval_chunk_chars": self.settings.retrieval_chunk_chars,
                        "retrieval_chunk_overlap_chars": self.settings.retrieval_chunk_overlap_chars,
                        "snippet": _query_snippet(
                            window_text,
                            query,
                            max(self.settings.chooser_excerpt_chars * 4, self.settings.chooser_excerpt_chars),
                        ),
                    },
                )
            )
        return candidates

    def _candidate_from_local_hit(self, hit: dict[str, Any], *, query: str, action_id: str, rank: int) -> CandidateDoc:
        metadata = dict(hit.get("metadata") or {})
        file_path = str(metadata.get("file_path") or metadata.get("original_path") or "")
        title = Path(file_path).name if file_path else str(hit.get("doc_id") or "local document")
        score = hit.get("similarity_score", hit.get("relevance_score"))
        excerpt = str(hit.get("content") or hit.get("preview") or "")
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
            metadata={
                "file_path": file_path,
                "folder_path": metadata.get("folder_path"),
                "parent_doc_id": str(hit.get("doc_id") or f"local_missing_{action_id}_{rank}"),
            },
        )

    async def _execute_retrieval_action(
        self, action: RetrievalActionRecord, hop: HopState, iteration: int
    ) -> tuple[RetrievalActionRecord, list[CandidateDoc], dict[str, Any]]:
        query = str(action.parameters.get("query") or "")
        start_s = self._elapsed_s()
        try:
            if action.type == "web_search":
                raw_hits = await self._web_hits(query)
                candidates = []
                for rank, hit in enumerate(raw_hits, start=1):
                    candidates.extend(self._candidates_from_web_hit(hit, query=query, action_id=action.id, rank=rank))
                output = {
                    "tool": "browsecomp_search",
                    "query": query,
                    "success": True,
                    "data_retrieved": bool(raw_hits),
                    "results_count": len(raw_hits),
                    "candidate_count": len(candidates),
                    "retrieval_context_mode": str(self.settings.retrieval_context_mode),
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
                "top_parent_doc_ids": [
                    str(candidate.metadata.get("parent_doc_id") or candidate.doc_id) for candidate in candidates[:3]
                ],
            }
        except Exception as exc:
            if is_helper_infrastructure_error(exc):
                raise PrivacyHopQAHelperInfrastructureError(
                    f"Retrieval helper failure for {action.type} action {action.id}: {exc}"
                ) from exc
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
                "top_parent_doc_ids": history_entry.get("top_parent_doc_ids", []),
            },
        )
        return action, candidates, history_entry

    def _finalize_candidate_pool(self, hop: HopState, candidates: list[CandidateDoc]) -> list[CandidateDoc]:
        candidates_by_doc: dict[str, CandidateDoc] = {}
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
        max_parent_docs = int(self.settings.max_candidate_cards)
        if max_parent_docs > 0:
            parent_order: list[str] = []
            seen_parents: set[str] = set()
            for candidate in ordered:
                parent_doc_id = str(candidate.metadata.get("parent_doc_id") or candidate.doc_id)
                if parent_doc_id not in seen_parents:
                    seen_parents.add(parent_doc_id)
                    parent_order.append(parent_doc_id)
            kept_parents = set(parent_order[:max_parent_docs])
            trimmed = [
                candidate
                for candidate in ordered
                if str(candidate.metadata.get("parent_doc_id") or candidate.doc_id) in kept_parents
            ]
        else:
            trimmed = ordered

        self._cache_candidates(trimmed)
        hop.candidate_doc_ids = [candidate.doc_id for candidate in trimmed]
        hop.candidate_cards = [candidate.as_card(self.settings.chooser_excerpt_chars) for candidate in trimmed]
        before_trim_parent_ids = {_candidate_parent_id(candidate) for candidate in ordered}
        after_trim_parent_ids = {_candidate_parent_id(candidate) for candidate in trimmed}
        if hop.gold_doc_id:
            if hop.gold_doc_id in before_trim_parent_ids:
                hop.gold_parent_retrieved_before_trim = True
            if hop.gold_doc_id in after_trim_parent_ids:
                hop.gold_parent_retrieved = True
            for candidate in trimmed:
                if _candidate_parent_id(candidate) == hop.gold_doc_id and candidate.doc_id not in hop.gold_candidate_doc_ids:
                    hop.gold_candidate_doc_ids.append(candidate.doc_id)
        self._inc_retrieval_stat("candidate_windows_before_parent_trim", len(ordered))
        self._inc_retrieval_stat("candidate_windows_after_parent_trim", len(trimmed))
        self._inc_retrieval_stat("candidate_parent_docs_before_parent_trim", len(before_trim_parent_ids))
        self._inc_retrieval_stat("candidate_parent_docs_after_parent_trim", len(after_trim_parent_ids))
        return trimmed

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

        candidates: list[CandidateDoc] = self._memory_seed_candidates(hop)
        for action, action_candidates, history_entry in results:
            hop.search_history.append(history_entry)
            candidates.extend(action_candidates)
        return self._finalize_candidate_pool(hop, candidates)

    async def _choose_documents(self, hop: HopState, candidates: list[CandidateDoc], iteration: int) -> list[CandidateDoc]:
        if not candidates:
            return []
        parent_groups = _candidate_parent_groups(candidates)
        parent_ids = list(parent_groups)
        if len(parent_ids) <= self.settings.choose_top_k:
            selected_parent_ids = parent_ids[: self.settings.choose_top_k]
            selected, selection_meta = self._expand_selected_parent_documents(
                parent_groups,
                selected_parent_ids,
                hop=hop,
                iteration=iteration,
                selection_stage="initial_auto",
            )
            now = self._elapsed_s()
            self._record_event(
                stage="doc_choose",
                lane="choose",
                label=f"H{hop.hop_number} choose parent docs (skipped)",
                start_s=now,
                end_s=now,
                meta={
                    "hop_number": hop.hop_number,
                    "iteration": iteration,
                    "candidate_count": len(candidates),
                    "parent_candidate_count": len(parent_ids),
                    **selection_meta,
                    "skipped": True,
                },
            )
            self._mark_selected_documents(hop, selected)
            return selected
        prompt = build_doc_choose_prompt(
            numbered_questions=self.numbered_questions,
            current_hop_number=hop.hop_number,
            current_hop_question=self._materialize_question(hop.question),
            current_answers_so_far=self._current_answers_so_far(),
            previously_useful_documents=self._document_memory_prompt_cards(),
            candidate_cards=_candidate_cards_for_prompt(candidates, self.settings.chooser_excerpt_chars),
            choose_top_k=self.settings.choose_top_k,
            read_all_windows_max_count=self.settings.read_all_windows_max_count,
            large_parent_top_windows=self.settings.large_parent_top_windows,
            large_parent_neighbor_windows=self.settings.large_parent_neighbor_windows,
        )
        self._save_prompt(f"doc_choose_iter{iteration}_hop{hop.hop_number}", prompt)
        started = self._elapsed_s()
        try:
            response = await self.llm_adapter.generate_text(
                prompt,
                log_name="doc_choose",
                max_tokens=self.settings.doc_choose_max_tokens,
                max_context_tokens=self.settings.generation_context_limit_tokens,
                context_margin_tokens=self.settings.generation_context_margin_tokens,
                hop_number=hop.hop_number,
                iteration=iteration,
            )
            payload = extract_json(response)
        except Exception as exc:
            if is_llm_infrastructure_error(exc):
                raise
            logger.warning("doc chooser failed for hop %s iteration %s: %s", hop.hop_number, iteration, exc)
            self._record_error(stage="doc_choose", hop_number=hop.hop_number, iteration=iteration, exc=exc)
            raise PrivacyHopQAProtocolError(f"doc_choose protocol error: {exc}") from exc
        selected_ids = payload.get("selected_doc_ids") if isinstance(payload, dict) else None
        valid_parent_ids = set(parent_ids)
        normalized_ids: list[str] = []
        if isinstance(selected_ids, list):
            for raw_id in selected_ids:
                doc_id = str(raw_id)
                if doc_id in valid_parent_ids and doc_id not in normalized_ids:
                    normalized_ids.append(doc_id)
                    continue
                for candidate in candidates:
                    if candidate.doc_id == doc_id:
                        parent_id = _candidate_parent_id(candidate)
                        if parent_id not in normalized_ids:
                            normalized_ids.append(parent_id)
                        break
        if not normalized_ids:
            normalized_ids = parent_ids[: self.settings.choose_top_k]
        selected_parent_ids = normalized_ids[: self.settings.choose_top_k]
        selected, selection_meta = self._expand_selected_parent_documents(
            parent_groups,
            selected_parent_ids,
            hop=hop,
            iteration=iteration,
            selection_stage="initial_model",
        )
        ended = self._elapsed_s()
        self._record_event(
            stage="doc_choose",
            lane="choose",
            label=f"H{hop.hop_number} choose parent docs",
            start_s=started,
            end_s=ended,
            meta={
                "hop_number": hop.hop_number,
                "iteration": iteration,
                "candidate_count": len(candidates),
                "parent_candidate_count": len(parent_ids),
                **selection_meta,
            },
        )
        self._mark_selected_documents(hop, selected)
        return selected

    async def _read_document(self, hop: HopState, candidate: CandidateDoc, iteration: int) -> ReaderResult:
        prompt = build_doc_reader_prompt(
            numbered_questions=self.numbered_questions,
            current_hop_number=hop.hop_number,
            current_hop_question=self._materialize_question(hop.question),
            current_answers_so_far=self._current_answers_so_far(),
            document=candidate.as_reader_input(self._reader_excerpt_chars()),
        )
        self._save_prompt(f"doc_read_iter{iteration}_hop{hop.hop_number}_{candidate.doc_id.replace('/', '_')}", prompt)
        start_s = self._elapsed_s()
        try:
            response = await self.llm_adapter.generate_text(
                prompt,
                log_name="doc_read",
                max_tokens=self.settings.doc_read_max_tokens,
                max_context_tokens=self.settings.generation_context_limit_tokens,
                context_margin_tokens=self.settings.generation_context_margin_tokens,
                hop_number=hop.hop_number,
                iteration=iteration,
            )
            payload = extract_json(response)
            proposed_answer = str(payload.get("proposed_answer") or "").strip()
            justification = str(payload.get("justification") or "").strip()
            confidence = _clamp_confidence(payload.get("confidence"), default=0.0)
            missing_information = str(payload.get("missing_information") or "").strip()
            can_answer = bool(payload.get("can_answer")) and bool(proposed_answer)
            if can_answer and confidence < self.settings.reader_min_answer_confidence:
                can_answer = False
                missing_information = (
                    missing_information
                    or f"Reader confidence {confidence:.2f} is below the required answer threshold "
                    f"{self.settings.reader_min_answer_confidence:.2f}."
                )
            if can_answer and missing_information:
                can_answer = False
            if can_answer and justification and f"[DOC:{candidate.doc_id}]" not in justification:
                justification = f"{justification} [DOC:{candidate.doc_id}]"
            result = ReaderResult(
                doc_id=candidate.doc_id,
                source=candidate.source,
                title=candidate.title,
                locator=candidate.locator,
                can_answer=can_answer,
                proposed_answer=proposed_answer,
                justification=justification,
                confidence=confidence,
                missing_information=missing_information,
                parent_doc_id=str(candidate.metadata.get("parent_doc_id") or candidate.doc_id),
                evidence_window_id=str(candidate.metadata.get("evidence_window_id") or candidate.doc_id),
                window_index=candidate.metadata.get("window_index"),
                window_count=candidate.metadata.get("window_count"),
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
            raise PrivacyHopQAProtocolError(f"doc_read protocol error: {exc}") from exc
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
                "parent_doc_id": str(candidate.metadata.get("parent_doc_id") or candidate.doc_id),
                "window_index": candidate.metadata.get("window_index"),
                "window_count": candidate.metadata.get("window_count"),
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

    def _stored_reader_results(self, hop: HopState) -> list[ReaderResult]:
        results: list[ReaderResult] = []
        for item in hop.reader_results:
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("doc_id") or "").strip()
            if not doc_id:
                continue
            results.append(
                ReaderResult(
                    doc_id=doc_id,
                    source=str(item.get("source") or ""),
                    title=str(item.get("title") or ""),
                    locator=str(item.get("locator") or ""),
                    can_answer=bool(item.get("can_answer")),
                    proposed_answer=str(item.get("proposed_answer") or ""),
                    justification=str(item.get("justification") or ""),
                    confidence=_clamp_confidence(item.get("confidence"), default=0.0),
                    missing_information=str(item.get("missing_information") or ""),
                    parent_doc_id=str(item.get("parent_doc_id") or "") or None,
                    evidence_window_id=str(item.get("evidence_window_id") or "") or None,
                    window_index=item.get("window_index") if isinstance(item.get("window_index"), int) else None,
                    window_count=item.get("window_count") if isinstance(item.get("window_count"), int) else None,
                )
            )
        return results

    def _supporting_reader_results(
        self,
        resolution: dict[str, Any],
        reader_results: list[ReaderResult],
    ) -> list[ReaderResult]:
        positive = [result for result in reader_results if result.can_answer and result.proposed_answer.strip()]
        if not positive:
            return []
        justification = str(resolution.get("justification") or "")
        cited = [
            result
            for result in positive
            if result.doc_id in justification or (result.parent_doc_id and result.parent_doc_id in justification)
        ]
        if cited:
            return cited
        answer_key = normalize_answer(str(resolution.get("answer") or ""))
        matching = [result for result in positive if normalize_answer(result.proposed_answer) == answer_key]
        if matching:
            return [max(matching, key=lambda result: result.confidence)]
        return positive[:1]

    def _remember_useful_documents(
        self,
        hop: HopState,
        resolution: dict[str, Any],
        reader_results: list[ReaderResult],
    ) -> None:
        # Store only docs that the agent used as positive evidence. Retrieved
        # distractors are deliberately excluded from cross-hop memory.
        supporting_results = self._supporting_reader_results(resolution, reader_results)
        for result in supporting_results:
            parent_doc_id = result.parent_doc_id or result.doc_id
            if not parent_doc_id:
                continue
            entry = self._document_memory.get(parent_doc_id)
            if entry is None:
                entry = UsefulDocumentMemory(
                    parent_doc_id=parent_doc_id,
                    source=result.source,
                    title=result.title,
                    locator=result.locator,
                )
                self._document_memory[parent_doc_id] = entry
            if not entry.title and result.title:
                entry.title = result.title
            if not entry.locator and result.locator:
                entry.locator = result.locator
            if not entry.source and result.source:
                entry.source = result.source
            # If a selected parent was expanded into multiple windows, keep all
            # cached windows for that parent. A later hop may need a different
            # part of the same useful document.
            cached_doc_ids = self._candidate_doc_ids_by_parent.get(parent_doc_id) or [result.doc_id]
            for doc_id in cached_doc_ids:
                entry.remember(
                    hop.hop_number,
                    str(resolution.get("answer") or ""),
                    str(resolution.get("justification") or result.justification or ""),
                    doc_id,
                )
        if supporting_results:
            self._max_retrieval_stat("document_memory_parent_docs", len(self._document_memory))

    def _apply_resolution_to_hop(self, hop: HopState, resolution: dict[str, Any]) -> bool:
        if not (resolution.get("answered") and resolution.get("answer")):
            return False
        raw_answer = str(resolution.get("answer") or "").strip()
        hop.answer = raw_answer
        hop.matched_accepted_variant = self._answer_matches_accepted_variant(hop, raw_answer)
        hop.answer_for_context = self._context_answer_for_raw(hop, raw_answer)
        hop.justification = str(resolution.get("justification") or "").strip()
        hop.confidence = _clamp_confidence(resolution.get("confidence"), default=0.0)
        hop.status = "answered"
        return True

    def _should_stop_after_incorrect_hop(self, hop: HopState, iteration: int) -> bool:
        if not self.settings.stop_after_incorrect_hop or hop.matched_accepted_variant:
            return False
        now_s = self._elapsed_s()
        self._record_event(
            stage="stop",
            lane="stop",
            label=f"H{hop.hop_number} incorrect; stop rollout",
            start_s=now_s,
            end_s=now_s,
            meta={
                "hop_number": hop.hop_number,
                "iteration": iteration,
                "answer": hop.answer,
                "matched_accepted_variant": hop.matched_accepted_variant,
                "reason": "stop_after_incorrect_hop",
            },
        )
        return True

    def _select_followup_documents(
        self,
        hop: HopState,
        unread_candidates: list[CandidateDoc],
        resolution: dict[str, Any],
    ) -> list[CandidateDoc]:
        if not unread_candidates:
            return []
        parent_groups = _candidate_parent_groups(unread_candidates)
        valid_parent_ids = set(parent_groups)
        raw_ids = resolution.get("selected_doc_ids")
        normalized_ids: list[str] = []
        if isinstance(raw_ids, list):
            for raw_id in raw_ids:
                doc_id = str(raw_id)
                if doc_id in valid_parent_ids and doc_id not in normalized_ids:
                    normalized_ids.append(doc_id)
                    continue
                for candidate in unread_candidates:
                    if candidate.doc_id == doc_id:
                        parent_id = _candidate_parent_id(candidate)
                        if parent_id not in normalized_ids:
                            normalized_ids.append(parent_id)
                        break
        if not normalized_ids:
            normalized_ids = list(parent_groups)[: self.settings.choose_top_k]
        selected_parent_ids = normalized_ids[: self.settings.choose_top_k]
        selected, _selection_meta = self._expand_selected_parent_documents(
            parent_groups,
            selected_parent_ids,
            hop=hop,
            iteration=-1,
            selection_stage="followup_model",
        )
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
            numbered_questions=self.numbered_questions,
            current_hop_number=hop.hop_number,
            current_hop_question=self._materialize_question(hop.question),
            current_answers_so_far=self._current_answers_so_far(),
            search_history=self._recent_search_history(hop),
            reader_results=[_resolver_reader_result(result) for result in _reader_results_for_resolver(reader_results)],
            unread_candidate_cards=_candidate_cards_for_prompt(
                unread_candidates,
                self.settings.chooser_excerpt_chars,
                max_cards=80,
            ),
            choose_top_k=self.settings.choose_top_k,
        )
        self._save_prompt(f"hop_resolve_iter{iteration}_hop{hop.hop_number}", prompt)
        started = self._elapsed_s()
        try:
            response = await self.llm_adapter.generate_text(
                prompt,
                log_name="hop_resolve",
                max_tokens=self.settings.hop_resolve_max_tokens,
                max_context_tokens=self.settings.generation_context_limit_tokens,
                context_margin_tokens=self.settings.generation_context_margin_tokens,
                hop_number=hop.hop_number,
                iteration=iteration,
            )
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
            raise PrivacyHopQAProtocolError(f"hop_resolve protocol error: {exc}") from exc
        if resolution.get("answered"):
            positive_read_ids = {
                result.doc_id for result in reader_results if result.can_answer and result.proposed_answer.strip()
            }
            if not resolution.get("answer") or not positive_read_ids:
                wipe_reason = (
                    "empty_answer" if not resolution.get("answer") else "no_positive_reader"
                )
                self._record_error(
                    stage="hop_resolve",
                    hop_number=hop.hop_number,
                    iteration=iteration,
                    exc=f"resolver_wipe:{wipe_reason}",
                    label=f"hop_resolve resolver_wipe ({wipe_reason})",
                )
                raise PrivacyHopQAProtocolError(f"hop_resolve resolver_wipe:{wipe_reason}")
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
            positive_readers = [reader for reader in reader_results if reader.can_answer and reader.proposed_answer.strip()]
            if not cited_doc_ids and positive_readers:
                resolution["justification"] = f"{resolution['justification']} [DOC:{positive_readers[0].doc_id}]".strip()
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
        cumulative_reader_results: list[ReaderResult] = self._stored_reader_results(hop)
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
        stopped_after_incorrect_hop = False
        for iteration in range(max_iterations):
            hop = self._current_hop()
            if hop is None:
                break
            iterations_used = iteration + 1
            hop.attempts += 1
            retrieval_actions = await self._plan_retrieval_actions(hop, iteration)
            if not retrieval_actions:
                no_plan_reason = "No usable retrieval plan was produced for this hop in this iteration."
                prior_reader_results = self._stored_reader_results(hop)
                prior_resolution_reason = ""
                memory_candidates = self._finalize_candidate_pool(hop, self._memory_seed_candidates(hop))
                if memory_candidates:
                    resolution = await self._work_candidate_pool(hop, iteration, memory_candidates)
                    hop.resolution_reason = resolution.get("reason") or ""
                    if self._apply_resolution_to_hop(hop, resolution):
                        self._remember_useful_documents(hop, resolution, self._stored_reader_results(hop))
                        stalled = False
                        if self._should_stop_after_incorrect_hop(hop, iteration):
                            stopped_after_incorrect_hop = True
                            break
                        continue
                if prior_reader_results:
                    resolution = await self._resolve_hop(hop, iteration, prior_reader_results, [])
                    prior_resolution_reason = str(resolution.get("reason") or "").strip()
                    hop.resolution_reason = prior_resolution_reason
                    if self._apply_resolution_to_hop(hop, resolution):
                        self._remember_useful_documents(hop, resolution, prior_reader_results)
                        stalled = False
                        if self._should_stop_after_incorrect_hop(hop, iteration):
                            stopped_after_incorrect_hop = True
                            break
                        continue
                hop.status = "pending"
                if prior_resolution_reason:
                    hop.resolution_reason = f"{prior_resolution_reason} {no_plan_reason}"
                else:
                    hop.resolution_reason = no_plan_reason
                stalled = True
                continue
            candidates = await self._run_retrieval_actions(retrieval_actions, hop, iteration)
            resolution = await self._work_candidate_pool(hop, iteration, candidates)
            hop.resolution_reason = resolution.get("reason") or ""
            if self._apply_resolution_to_hop(hop, resolution):
                self._remember_useful_documents(hop, resolution, self._stored_reader_results(hop))
                stalled = False
                if self._should_stop_after_incorrect_hop(hop, iteration):
                    stopped_after_incorrect_hop = True
                    break
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
        retrieval_summary = self._retrieval_summary()
        summary = {
            "task_id": self.task_id,
            "chain_id": self.llm_adapter.chain_id,
            "iterations_used": iterations_used,
            "resolved_hops": sum(1 for hop in self.hop_states if hop.answer),
            "total_hops": len(self.hop_states),
            "searches_total": len(self.action_records),
            "docs_read": sum(len(hop.selected_doc_ids) for hop in self.hop_states),
            "document_memory_parent_docs": len(self._document_memory),
            "duration_s": self._elapsed_s(),
            "stalled": stalled,
            "llm_calls_total": self.llm_adapter.total_calls,
            "prompt_tokens": self.llm_adapter.total_prompt_tokens,
            "output_tokens": self.llm_adapter.total_output_tokens,
            "total_tokens": self.llm_adapter.total_prompt_tokens + self.llm_adapter.total_output_tokens,
            "retrieval_summary": retrieval_summary,
            "stopped_after_incorrect_hop": stopped_after_incorrect_hop,
            **error_summary,
        }
        trace = {
            "summary": summary,
            "hop_states": [hop.to_dict() for hop in self.hop_states],
            "actions": [action.to_dict() for action in self.action_records],
            "document_memory": self._document_memory_prompt_cards(),
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
            "retrieval_summary": retrieval_summary,
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
