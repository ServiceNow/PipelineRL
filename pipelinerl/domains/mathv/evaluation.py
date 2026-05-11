import re
from typing import Optional


def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Numeric answers within ``max_relative_change`` are correct; otherwise
    fall back to case-insensitive exact match (handles letter answers like
    "A"/"B"/"C"/"D" and short strings).
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith("%"):
                return float(text.rstrip("%")) / 100.0
            return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    return prediction.strip().lower() == target.strip().lower()


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{} format."""
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, text, re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


def evaluate_answer(predicted: str, ground_truth: str) -> str:
    """
    Evaluate math-visual answer and return status.

    Returns:
        - "correct": Answer is correct
        - "wrong": Answer is incorrect
        - "no_answer": No \\boxed{} found
        - "unparsable": Could not parse answer
    """
    try:
        boxed_answer = extract_boxed_answer(predicted)
        if not boxed_answer:
            return "no_answer"
        if relaxed_correctness(ground_truth, boxed_answer):
            return "correct"
        return "wrong"
    except Exception:
        return "unparsable"
