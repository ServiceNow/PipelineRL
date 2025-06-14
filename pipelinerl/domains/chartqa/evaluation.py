import re
import string
from typing import Union


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if not isinstance(answer, str):
        answer = str(answer)
    
    # Remove extra whitespace
    answer = answer.strip()
    
    # Convert to lowercase
    answer = answer.lower()
    
    # Remove punctuation
    answer = answer.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra spaces
    answer = ' '.join(answer.split())
    
    return answer


def extract_boxed_answer(text: str) -> Union[str, None]:
    """Extract answer from \\boxed{} format."""
    # Look for \\boxed{answer} pattern
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, text, re.IGNORECASE)
    if matches:
        return matches[-1].strip()  # Take the last boxed answer
    return None


def extract_numerical_answer(text: str) -> Union[float, None]:
    """Extract numerical answer from text."""
    # First try to extract from boxed format
    boxed_answer = extract_boxed_answer(text)
    if boxed_answer:
        try:
            return float(boxed_answer)
        except ValueError:
            pass
    
    # Look for patterns like "the answer is X", "answer: X", etc.
    patterns = [
        r'(?:the\s+)?answer\s*(?:is|:)\s*([+-]?\d*\.?\d+)',
        r'(?:^|\s)([+-]?\d*\.?\d+)(?:\s|$)',
        r'([+-]?\d+(?:\.\d+)?)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1])  # Take the last match
            except ValueError:
                continue
    
    return None


def is_numerical_match(pred: str, gold: str, tolerance: float = 1e-3) -> bool:
    """Check if numerical answers match within tolerance."""
    pred_num = extract_numerical_answer(pred)
    gold_num = extract_numerical_answer(gold)
    
    if pred_num is None or gold_num is None:
        return False
    
    return abs(pred_num - gold_num) <= tolerance


def evaluate_answer(predicted: str, ground_truth: str) -> str:
    """
    Evaluate ChartQA answer and return status.
    
    Returns:
        - "correct": Answer is correct
        - "wrong": Answer is incorrect
        - "no_answer": No answer provided
        - "unparsable": Could not parse answer
    """
    if not predicted or predicted.strip() == "":
        return "no_answer"
    
    try:
        # First try to extract answer from \\boxed{} format
        boxed_answer = extract_boxed_answer(predicted)
        if boxed_answer:
            predicted_clean = boxed_answer
        else:
            predicted_clean = predicted
        
        # Normalize both answers for comparison
        pred_norm = normalize_answer(predicted_clean)
        gold_norm = normalize_answer(ground_truth)
        
        # Direct string match
        if pred_norm == gold_norm:
            return "correct"
        
        # Try numerical matching for numeric answers
        if is_numerical_match(predicted_clean, ground_truth):
            return "correct"
        
        # Check if predicted answer is contained in ground truth or vice versa
        if pred_norm in gold_norm or gold_norm in pred_norm:
            return "correct"
        
        return "wrong"
        
    except Exception:
        return "unparsable"