"""Prompt builders for the privacy_hopqa domain."""

import json
from typing import Any, Iterable


def _json_block(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


ANSWER_FORMAT_GUIDANCE = """Answer Format Guidance:
- Answer only the current hop, not later hops
- Your answer will be string-matched against accepted answer variants, so output the answer unit only
- Use fewer than 5 words whenever possible; only exceed this for a longer proper name or required title
- Give the minimum words necessary to answer the question
- Include units or descriptors only if the question asks for them or they are needed to avoid ambiguity
- Do not answer with a full sentence
- Example: for "What percentage of river miles had bacteria exceeding EPA's recreational benchmark?", answer "20%", not "20% of river miles had bacteria exceeding EPA's recreational benchmark"
- If this hop's answer will be used as input to a later hop, prefer the form that can be directly substituted into that later question"""


def build_hop_plan_prompt(
    *,
    numbered_questions: str,
    current_hop_number: int,
    current_hop_question: str,
    current_answers_so_far: str,
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

Current Answers So Far:
{current_answers_so_far}

{task_context}Current Hop: {current_hop_number}
Current Hop Question:
{current_hop_question}

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
- If the hop depends on company-specific, internal, operational, or task-context facts, include a local_document_search
- Do not spend an entire early hop on web_search only when local company files may contain the answer
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
    numbered_questions: str,
    current_hop_number: int,
    current_hop_question: str,
    current_answers_so_far: str,
    candidate_cards: list[dict[str, Any]],
    choose_top_k: int,
) -> str:
    return f"""You are selecting which retrieved evidence windows are worth reading closely for the current hop.
Each candidate ID may refer to a window from a larger parent document. Multiple windows can share the same parent_doc_id.

Full Numbered Questions:
{numbered_questions}

Current Answers So Far:
{current_answers_so_far}

Current Hop: {current_hop_number}
Current Hop Question:
{current_hop_question}

Candidate Evidence Windows:
{_json_block(candidate_cards)}

Select up to {choose_top_k} candidate doc_id values to read next.
- It is okay to choose fewer than {choose_top_k}
- If one evidence window looks decisive, choosing just that one is fine
- Prefer evidence windows most likely to directly answer the current hop
- Avoid windows that look redundant with each other
- `parent_doc_id` identifies the original document when the candidate is a window
- `window_index` and `window_count` show where a window sits inside its parent document
- `best_rank` means the best retrieval rank this candidate achieved across the search batch
- `hit_count` means how many different retrieval queries returned this candidate
- `top_queries` are example phrasings that retrieved the candidate and may help indicate why it matched

Return JSON in this format:
{{
  "selected_doc_ids": ["doc_id_1", "doc_id_2"]
}}

Return valid JSON only.
"""


def build_doc_reader_prompt(
    *,
    numbered_questions: str,
    current_hop_number: int,
    current_hop_question: str,
    current_answers_so_far: str,
    document: dict[str, Any],
) -> str:
    return f"""You are reading one candidate evidence window to see if it can answer the current hop.

Full Numbered Questions:
{numbered_questions}

Current Answers So Far:
{current_answers_so_far}

Current Hop: {current_hop_number}
Current Hop Question:
{current_hop_question}

Evidence Window:
{_json_block(document)}

{ANSWER_FORMAT_GUIDANCE}

Return JSON in this format:
{{
  "can_answer": true,
  "proposed_answer": "short answer",
  "justification": "1-2 sentences using only this document and citing [DOC:{document['doc_id']}]",
  "confidence": 0.82,
  "missing_information": ""
}}

If the evidence window is insufficient, set "can_answer" to false and explain what is missing.
Do not use knowledge outside the provided evidence window.
Return valid JSON only.
"""


def build_hop_resolve_prompt(
    *,
    numbered_questions: str,
    current_hop_number: int,
    current_hop_question: str,
    current_answers_so_far: str,
    search_history: list[dict[str, Any]],
    reader_results: list[dict[str, Any]],
    unread_candidate_cards: list[dict[str, Any]],
    choose_top_k: int,
) -> str:
    return f"""Decide whether the current hop can now be answered.

Full Numbered Questions:
{numbered_questions}

Current Answers So Far:
{current_answers_so_far}

Current Hop: {current_hop_number}
Current Hop Question:
{current_hop_question}

Recent Search History:
{_json_block(search_history or [])}

Document-Reading Results:
{_json_block(reader_results or [])}

Unread Candidate Evidence Windows From The Current Retrieval Round:
{_json_block(unread_candidate_cards or [])}

{ANSWER_FORMAT_GUIDANCE}

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
- Only do this when at least one Document-Reading Result has "can_answer": true and directly supports the answer

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
