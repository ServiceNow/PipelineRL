import re

from .....base import Data, Verifier


def _search_validity(answer_str):
    strict_match = re.search(r"(?<=[the|The] answer is )(.*)(?=.)", answer_str)
    if strict_match:
        result = strict_match.group(1).strip()

        validity_match = re.search(r"\b(valid|invalid)\b", result, re.IGNORECASE)
        if validity_match:
            return validity_match.group(1).lower()

    flexible_match = re.search(r"\b(valid|invalid)\b", answer_str, re.IGNORECASE)
    if flexible_match:
        return flexible_match.group(1).lower()

    return ""


class BBHFormalFallaciesVerifier(Verifier):
    def verify(self, data: Data, test_solution: str):
        try:
            test_answer = self.extract_answer(test_solution)
            ground_truth = data.answer

            test_answer = test_answer.lower()
            ground_truth = ground_truth.lower()

            correct = test_answer == ground_truth
            if correct:
                acc_score = 1.0
            else:
                acc_score = 0

            return acc_score
        except Exception:
            return False

    def extract_answer(self, test_solution: str):
        answer_str = test_solution
        return _search_validity(answer_str=answer_str)


if __name__ == "__main__":
    test_cases = [
        "invalid The answer is valid.",
        "The answer is invalid.",
        "This argument is valid.",
        "The argument is Invalid.",
        "valid",
        "INVALID",
    ]
