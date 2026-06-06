"""JSON response parsing for privacy_hopqa prompts."""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def extract_json(text: str) -> Any:
    """Extract JSON from an LLM response.

    This matches the upstream DRBench behavior closely enough for the planning and
    tool-selection prompts used by the concise-QA workflow.
    """

    candidate = text.strip()
    candidate = re.sub(r"<think>.*?</think>", "", candidate, flags=re.DOTALL).strip()

    fenced_match = re.search(r"```json\s*(.*?)\s*```", candidate, re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    candidate = re.sub(r"^```json\s*|\s*```$", "", candidate).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as direct_parse_error:
        first_error = direct_parse_error

    for end_idx in range(len(candidate) - 1, -1, -1):
        closing = candidate[end_idx]
        if closing not in ("]", "}"):
            continue
        opening = "[" if closing == "]" else "{"
        for start_idx in range(end_idx, -1, -1):
            if candidate[start_idx] != opening:
                continue
            try:
                return json.loads(candidate[start_idx : end_idx + 1])
            except json.JSONDecodeError:
                continue

    logger.warning("No valid JSON found in response; response_chars=%s", len(candidate))
    raise json.JSONDecodeError("No valid JSON found in response", first_error.doc, first_error.pos)
