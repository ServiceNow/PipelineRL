"""Small DRBench utility subset used by privacy_agent.

The local agent copy only needs JSON extraction plus the optional report-to-insights
helper. Insight extraction still delegates to the upstream DRBench helper because the
privacy_agent rollout path disables it during training.
"""


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
    except json.JSONDecodeError:
        pass

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
        break

    raise json.JSONDecodeError("No valid JSON found in response", candidate, 0)


def break_report_to_insights(
    report_text: str,
    model: str = "together_ai/meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    llm_adapter=None,
    save_json: bool = False,
    max_retries: int = 3,
    **kwargs,
):
    """Split a report into citation-backed insight records."""

    del kwargs
    if llm_adapter is None:
        raise ValueError("Insight extraction requires an llm_adapter.")

    prompt = f"""
    Please break down the following report text into insight claims. Each insight claim should be:
    1. A single insight, that might include multiple statements and claims
    2. Independent and self-contained
    3. Each claim can have more than one sentence, but should be focused on a single insight
    4. Support each insight with citations from the report text
    5. Return JSON in the format [{{"claim": "...", "citations": ["..."]}}]
    6. If no insights are found, return []

    <START OF REPORT>
    {report_text}
    <END OF REPORT>
    """

    for attempt in range(max_retries):
        try:
            response = llm_adapter.generate_text(prompt, model=model, log_name="report_to_insights")
            parsed = extract_json(response)
            if not isinstance(parsed, list):
                raise ValueError("Insight extraction did not return a JSON list.")

            validated = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                claim = item.get("claim")
                citations = item.get("citations")
                if not isinstance(claim, str) or len(claim.strip()) < 10:
                    continue
                if not isinstance(citations, list):
                    citations = []
                validated.append({"claim": claim.strip(), "citations": citations})

            if save_json:
                with open("insight_claims.json", "w", encoding="utf-8") as handle:
                    json.dump(validated, handle, indent=2, ensure_ascii=False)
            return validated
        except Exception as exc:
            logger.warning("Insight extraction attempt %s failed: %s", attempt + 1, exc)

    logger.error("Failed to extract insights after %s attempts", max_retries)
    return []
