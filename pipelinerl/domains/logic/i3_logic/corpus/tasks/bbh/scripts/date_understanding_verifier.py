import re

from .....base import Data, Verifier


def _search_multiple_choice(answer_str):
    strict_match = re.search("(?<=[the|The] answer is )(.*)(?=.)", answer_str)
    if strict_match:
        result = strict_match.group(1).strip()
        choice_match = re.search("\\(([A-Z])\\)", result)
        if choice_match:
            return f"({choice_match.group(1)})"
    flexible_match = re.search("\\(([A-Z])\\)", answer_str)
    if flexible_match:
        return f"({flexible_match.group(1)})"
    return ""


class BBHDateUnderstandingVerifier(Verifier):
    def verify(self, data: Data, test_solution: str):
        try:
            test_answer = self.extract_answer(test_solution)
            ground_truth = data.answer
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
        return _search_multiple_choice(answer_str=answer_str)


if __name__ == "__main__":
    answer_str = "The answer is (B) 01/02/2020."
    answer_str = ": (A)"
    answer_str = "The answer is (B)"
