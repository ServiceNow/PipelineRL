"""Prompt builders for the privacy_hopqa domain."""

import json
from typing import Any, Iterable


def _json_block(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def build_hop_plan_prompt(
    *,
    numbered_questions: str,
    current_hop_number: int,
    current_hop_question: str,
    resolved_answers: list[dict[str, Any]],
    search_history: list[dict[str, Any]],
    recent_reader_results: list[dict[str, Any]],
    company_name: str | None,
    company_description: str | None,
    max_parallel_retrieval_actions: int,
    no_web: bool,
    retry_guidance: str | None = None,
) -> str:
    company_lines = []
    if company_name:
        company_lines.append(f"Company: {company_name}")
    if company_description:
        company_lines.append(f"Company Description: {company_description}")
    task_context = "\n".join(company_lines)
    if task_context:
        task_context = f"Task Context:\n{task_context}\n"

    action_types = ["local_document_search"] if no_web else ["web_search", "local_document_search"]
    initial_retrieval_guidance = (
        "Recent search history and document-reading results are empty, so return at least one retrieval action. "
        "Returning [] would leave the agent with no candidate documents to read."
        if not (search_history or recent_reader_results)
        else "Return [] only if the existing search history or document-reading results are enough for the next step and no additional retrieval is needed."
    )
    retry_block = ""
    if retry_guidance:
        retry_block = f"""
Previous Planning Attempt Was Unusable:
{retry_guidance}

Please try again and return a usable retrieval plan for the current hop.
"""
    return f"""You are solving a multihop QA chain one hop at a time.

Full Numbered Questions:
{numbered_questions}

{task_context}Current Hop: {current_hop_number}
Current Hop Question:
{current_hop_question}

Resolved Earlier Hops:
{_json_block(resolved_answers or [])}

Recent Search History For This Hop:
{_json_block(search_history or [])}

Recent Document-Reading Results For This Hop:
{_json_block(recent_reader_results or [])}
{retry_block}

Plan up to {max_parallel_retrieval_actions} retrieval actions that would best help answer the CURRENT hop.
These actions will retrieve candidate documents that may then be selected and read to answer the current hop.
Think first about how to search for the necessary information, then return the retrieval actions.

- It is good to try different phrasings in parallel when useful
- Avoid exact repeats of recent searches unless you are intentionally refining them
- Use only these action types: {', '.join(action_types)}
- web_search searches online web pages
- local_document_search searches local company files
- Do not plan analysis, URL fetches, downloads, or enterprise tools

Return a JSON array of actions in this format:
[
  {{
    "type": "web_search",
    "description": "Search for ...",
    "parameters": {{"query": "..."}},
    "priority": 0.8,
    "expected_output": "Candidate evidence for the current hop"
  }}
]

{initial_retrieval_guidance}
Return valid JSON only.
"""


def build_doc_choose_prompt(
    *,
    current_hop_number: int,
    current_hop_question: str,
    resolved_answers: list[dict[str, Any]],
    candidate_cards: list[dict[str, Any]],
    choose_top_k: int,
) -> str:
    return f"""You are selecting which retrieved documents are worth reading closely for the current hop.

Current Hop: {current_hop_number}
Current Hop Question:
{current_hop_question}

Resolved Earlier Hops:
{_json_block(resolved_answers or [])}

Candidate Documents:
{_json_block(candidate_cards)}

Select up to {choose_top_k} document IDs to read next.
- It is okay to choose fewer than {choose_top_k}
- If one document looks decisive, choosing just that one is fine
- Prefer documents most likely to directly answer the current hop
- Avoid documents that look redundant with each other
- `best_rank` means the best retrieval rank this doc achieved across the search batch
- `hit_count` means how many different retrieval queries returned this doc
- `top_queries` are example phrasings that retrieved the doc and may help indicate why it matched

Return JSON in this format:
{{
  "selected_doc_ids": ["doc_id_1", "doc_id_2"]
}}

Return valid JSON only.
"""


def build_doc_reader_prompt(
    *,
    current_hop_number: int,
    current_hop_question: str,
    resolved_answers: list[dict[str, Any]],
    document: dict[str, Any],
) -> str:
    return f"""You are reading one candidate document excerpt to see if it can answer the current hop.

Current Hop: {current_hop_number}
Current Hop Question:
{current_hop_question}

Resolved Earlier Hops:
{_json_block(resolved_answers or [])}

Document:
{_json_block(document)}

Return JSON in this format:
{{
  "can_answer": true,
  "proposed_answer": "short answer",
  "justification": "1-2 sentences using only this document and citing [DOC:{document['doc_id']}]",
  "confidence": 0.82,
  "missing_information": ""
}}

If the excerpt is insufficient, set "can_answer" to false and explain what is missing.
Do not use knowledge outside the provided document excerpt.
Return valid JSON only.
"""


def build_hop_resolve_prompt(
    *,
    current_hop_number: int,
    current_hop_question: str,
    resolved_answers: list[dict[str, Any]],
    search_history: list[dict[str, Any]],
    reader_results: list[dict[str, Any]],
    unread_candidate_cards: list[dict[str, Any]],
    choose_top_k: int,
) -> str:
    return f"""Decide whether the current hop can now be answered.

Current Hop: {current_hop_number}
Current Hop Question:
{current_hop_question}

Resolved Earlier Hops:
{_json_block(resolved_answers or [])}

Recent Search History:
{_json_block(search_history or [])}

Document-Reading Results:
{_json_block(reader_results or [])}

Unread Candidate Documents From The Current Retrieval Round:
{_json_block(unread_candidate_cards or [])}

Return JSON in this format:
{{
  "answered": true,
  "answer": "short answer",
  "justification": "1-2 sentences citing the best supporting [DOC:...] references",
  "confidence": 0.9,
  "reason": "why this is enough, or what is still missing",
  "next_step": "done",
  "selected_doc_ids": []
}}

If you would like to mark this hop as answered, set:
- "answered" to true
- "next_step" to "done"
- "selected_doc_ids" to []

If you would like to read more documents from the unread candidate list above, set:
- "answered" to false
- "next_step" to "read_more"
- "selected_doc_ids" to up to {choose_top_k} unread doc IDs

If you would like the agent to search again instead of reading more from this batch, set:
- "answered" to false
- "next_step" to "search_more"
- "selected_doc_ids" to []

Return valid JSON only.
"""


def build_final_summary_lines(
    hop_answers: Iterable[dict[str, Any]],
    *,
    final_answer: str,
    final_justification: str,
) -> str:
    lines: list[str] = []
    for hop in hop_answers:
        hop_number = hop["hop_number"]
        lines.append(f"ANSWER_{hop_number}: {hop['answer']}")
        lines.append(f"JUSTIFICATION_{hop_number}: {hop['justification']}")
        lines.append("")
    lines.append(f"ANSWER_FINAL: {final_answer}")
    lines.append(f"JUSTIFICATION_FINAL: {final_justification}")
    return "\n".join(lines).strip() + "\n"
