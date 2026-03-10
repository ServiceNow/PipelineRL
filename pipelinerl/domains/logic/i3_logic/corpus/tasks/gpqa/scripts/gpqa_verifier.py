import re

from .....base import Data, Verifier


def _fuzzy_match(prediction: str, reference: str) -> bool:
    prediction = prediction.strip()
    reference = reference.strip()
    if prediction == reference:
        return True
    if len(prediction) == 3 and prediction[0] == "(" and (prediction[-1] == ")"):
        return prediction[1].lower() == reference.lower()
    if len(reference) == 3 and reference[0] == "(" and (reference[-1] == ")"):
        return reference[1].lower() == prediction.lower()
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True
    if prediction.endswith("?") and prediction[:-1] == reference:
        return True
    return False


def _preprocess_answer(answer: str) -> str:
    if not answer:
        return ""
    processed = answer.strip().lower()
    last_box_index = processed.rfind("\\boxed{")
    if last_box_index == -1:
        return ""
    processed = processed[last_box_index:]
    box_pattern = "\\\\boxed\\{([^}]*)\\}"
    match = re.search(box_pattern, processed)
    if match:
        return match.group(1).strip()
    return ""


class GPQAVerifier(Verifier):
    def verify(self, data: Data, test_solution: str) -> float:
        try:
            test_answer = self.extract_answer(test_solution)
            ground_truth = data.answer.strip().lower()
            correct = _fuzzy_match(test_answer, ground_truth)
            return 1.0 if correct else 0.0
        except Exception:
            return 0.0

    def extract_answer(self, test_solution: str) -> str:
        return _preprocess_answer(test_solution)


if __name__ == "__main__":
    test_cases = [
        ("<answer>Alright! The final answer is: 2, 3, 4</answer>", "2,3,4"),
        ("blah blah The final answer is: <answer>2, 3, 4</answer>", "2,3,4"),
        ("Ok The answer is: <answer>(A)</answer>", "a"),
        ("Ok The answer is: (A)", "b"),
        ("Ok The answer is: **<answer>25</answer>**\nHere's why.", "25.0"),
        ("Ok The answer is: **<answer>25</answer>**\nHere's why.", "26.0"),
    ]
    verifier = GPQAVerifier()
    for test_input, reference in test_cases:
        result = verifier.extract_answer(test_input)
