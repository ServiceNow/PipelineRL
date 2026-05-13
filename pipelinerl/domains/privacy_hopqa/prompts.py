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
    previously_useful_documents: list[dict[str, Any]] | None = None,
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
        if not (search_history or recent_reader_results or previously_useful_documents)
        else "Return [] only if the existing search history, document-reading results, or previously useful documents are enough for the next step and no additional retrieval is needed."
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

Previously Useful Documents From Earlier Hops:
{_json_block(previously_useful_documents or [])}

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
- Previously useful documents were helpful on earlier hops and may be retried by the harness, but they are not evidence for the current hop until read for this hop
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
    read_all_windows_max_count: int,
    large_parent_top_windows: int,
    large_parent_neighbor_windows: int,
    previously_useful_documents: list[dict[str, Any]] | None = None,
) -> str:
    return f"""You are selecting which retrieved parent documents are worth reading closely for the current hop.
Each candidate ID is a parent document ID. If you select a parent document with at most {read_all_windows_max_count} available evidence windows, all of those windows will be read.
For larger parent documents, the harness reads the {large_parent_top_windows} highest-scoring evidence windows plus {large_parent_neighbor_windows} neighboring window on each side first. If those windows are insufficient, request more reads from that same parent document in the resolver step.

Full Numbered Questions:
{numbered_questions}

Current Answers So Far:
{current_answers_so_far}

Previously Useful Documents From Earlier Hops:
{_json_block(previously_useful_documents or [])}

Current Hop: {current_hop_number}
Current Hop Question:
{current_hop_question}

Candidate Parent Documents:
{_json_block(candidate_cards)}

Select up to {choose_top_k} parent document doc_id values to read next.
- It is okay to choose fewer than {choose_top_k}
- If one parent document looks decisive, choosing just that one is fine
- Prefer parent documents most likely to directly answer the current hop
- Avoid parent documents that look redundant with each other
- `matched_window_count` is how many retrieved windows are available from this parent document
- `total_window_count` is how many evidence windows the full parent document was split into
- `best_rank` means the best retrieval rank this candidate achieved across the search batch
- `top_queries` are example phrasings that retrieved the candidate and may help indicate why it matched
- Candidates marked `memory_seed` were useful for earlier hops; prefer them only when they look relevant to the current hop

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

If the evidence is in a table, list, or compact row, first match the requested row/entity/date and the requested column or quantity type. Do not propose an adjacent value that answers a nearby but different quantity.
Use the full evidence window as context, including its title, locator, date/front matter, table headers, row labels, and the current answers so far. These context fields count as evidence.
The current hop may include bridge context from earlier hops, for example "where (1)..." or "after identifying (2)...". If that bridge context is already provided by Current Answers So Far, do not require this evidence window to independently restate it. Use the bridge context to choose the requested target, then answer the new fact from this evidence window.
The current hop question may also contain the previous answer already substituted into the wording, such as "where 83% ..." or "in the plan targeting 90% ...". Treat those bridge clauses as context selectors. Do not reject an evidence window solely because it does not repeat the bridge clause in the same sentence as the requested answer.
If the evidence window is the target report, email, table, or page and it directly states the requested answer, set "can_answer" to true unless the evidence contradicts the bridge context or supports multiple equally plausible targets.
Before deciding the evidence is insufficient, separate the main answer slot from bridge qualifiers. For example, in a question like "what percentage/year/company ... when/where/after [bridge context]?", the main answer slot is the requested percentage, year, company, etc. If the evidence directly states that main slot for the target entity/report/page, answer it. Treat a bridge qualifier as missing only when it is needed to choose between multiple plausible rows inside this evidence window.
Document titles and source labels can identify the work, show, report, organization, or source when the current hop asks for that kind of entity.
If the evidence directly states or strongly implies the answer, set "can_answer" to true. This includes extracting a component from a stated value, date, range, amount, or entity when the current hop asks for that component.
Do not refuse merely because the evidence uses a synonym, an abbreviation, title context, or does not repeat every word in the question.
In the justification, explicitly include the date/time/entity context that proves the answer is for the requested target, using title, date/front matter, or locator context when needed.
Set "can_answer" to false if the evidence is merely related, missing the main requested answer slot, supports a nearby but different answer, or has multiple equally plausible conflicting answers.
Use high confidence only for directly supported answers. If confidence would be below 0.75 because the evidence is genuinely ambiguous or missing the target, set "can_answer" to false and explain what is missing.

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

The document reader has already seen each selected evidence window in full.
Use the compact Document-Reading Results below as the reader's extracted answer, confidence, and citation evidence.
If those results are insufficient or conflict, ask to read more unread parent documents rather than guessing.
When reader results conflict, prefer the result whose justification most directly matches the current hop's requested entity, date, and quantity type. Do not select a high-confidence result if its justification answers a nearby but different question.
A negative reader result from a different evidence window is not a conflict if it only says that window lacks the answer. Prefer a positive reader result when its justification directly supports the current hop.
If exactly one positive reader result directly supports the current hop, answer from that result even if other windows are negative or unrelated.
If multiple positive reader results disagree, compare their justifications against the current hop's requested row/entity/date and quantity type; do not prefer a nearby value just because it appears in more windows.
Never create an answer from a negative reader result. If every reader result has can_answer=false, either read more or search more; if no more evidence is available, report that the answer is not supported.
For large parent documents, the unread list may include the same parent document again because only its most relevant evidence windows were read first.
If the unread candidate list is empty, no more documents can be read from the current retrieval batch. In that terminal case, do not set "next_step" to "read_more"; either answer from the best directly supported positive reader result or set "next_step" to "search_more" with a concrete reason for what evidence is still missing.

Full Numbered Questions:
{numbered_questions}

Current Answers So Far:
{current_answers_so_far}

Current Hop: {current_hop_number}
Current Hop Question:
{current_hop_question}

Recent Search History:
{_json_block(search_history or [])}

Compact Document-Reading Results:
{_json_block(reader_results or [])}

Unread Candidate Parent Documents From The Current Retrieval Round (compact cards; select parent doc_id values to read more):
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
- "selected_doc_ids" to up to {choose_top_k} unread parent document IDs
- Do not choose this option when the unread candidate list is empty

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
