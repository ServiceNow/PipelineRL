"""Report helpers for privacy_hopqa."""

from __future__ import annotations

import re
from typing import Any

from .prompts import build_final_summary_lines


def parse_answer_lines(report_text: str) -> dict[str, str]:
    answers: dict[str, str] = {}
    for line in report_text.splitlines():
        line = line.strip()
        match = re.match(r"^ANSWER_(\w+):\s*(.+)$", line)
        if match:
            answers[match.group(1)] = match.group(2).strip()
    return answers


def build_deterministic_report(
    *,
    numbered_questions: str,
    hop_states: list[dict[str, Any]],
) -> str:
    hop_answers: list[dict[str, Any]] = []
    for hop in hop_states:
        answer = str(hop.get("answer") or "NOT_FOUND")
        justification = str(hop.get("justification") or "Insufficient evidence was found for this hop.")
        hop_answers.append(
            {
                "hop_number": int(hop["hop_number"]),
                "answer": answer,
                "justification": justification,
            }
        )

    final_hop = hop_answers[-1] if hop_answers else {"answer": "NOT_FOUND", "justification": "No hops were available."}
    resolved_prefix = [f"H{hop['hop_number']}={hop['answer']}" for hop in hop_answers if hop["answer"] != "NOT_FOUND"]
    if final_hop["answer"] == "NOT_FOUND":
        final_justification = "The final hop could not be resolved from the available evidence."
    elif len(resolved_prefix) <= 1:
        final_justification = final_hop["justification"]
    elif resolved_prefix:
        final_justification = (
            f"The chain resolves through {', '.join(resolved_prefix[:-1])} to the final hop answer. "
            f"{final_hop['justification']}"
        ).strip()
    else:
        final_justification = final_hop["justification"]

    body = build_final_summary_lines(
        hop_answers,
        final_answer=final_hop["answer"],
        final_justification=final_justification,
    )
    return f"# Research Findings\n\nOriginal Questions:\n{numbered_questions}\n\n{body}"
