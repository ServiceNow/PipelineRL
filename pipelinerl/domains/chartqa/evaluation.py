import re
import string
from typing import Optional

def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    "Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct."

    Args:
        target: Target string.
        prediction: Predicted string.
        max_relative_change: Maximum relative change.

    Returns:
        Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith("%"):
                # Convert percentages to floats.
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{} format."""
    # Look for \\boxed{answer} pattern
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, text, re.IGNORECASE)
    if matches:
        return matches[-1].strip()  # Take the last boxed answer
    return None


def evaluate_answer(predicted: str, ground_truth: str) -> str:
    """
    Evaluate ChartQA answer and return status.
    
    Returns:
        - "correct": Answer is correct
        - "wrong": Answer is incorrect
        - "no_answer": No answer provided
        - "unparsable": Could not parse answer
    """
    try:
        # First try to extract answer from \\boxed{} format
        boxed_answer = extract_boxed_answer(predicted)
        if not boxed_answer:
            return "no_answer"
        if relaxed_correctness(ground_truth, boxed_answer):
            return "correct"
        else:
            return "wrong"
    except Exception:
        return "unparsable"