import re

from .....base import Data, Verifier


def _search_yes_no(answer_str):
    strict_pattern = r"(?<=[the|The] answer is )(.*)(?=.)"
    strict_match = re.search(strict_pattern, answer_str)
    if strict_match:
        answer = strict_match.group(1).strip().lower()
        if answer in ["yes", "no"]:
            return answer.capitalize()

    flexible_pattern = r"\b(Yes|No|yes|no)\b"
    flexible_match = re.search(flexible_pattern, answer_str)
    if flexible_match:
        return flexible_match.group(1).capitalize()

    return ""


class BBHCausalJudgementVerifier(Verifier):
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
        return _search_yes_no(answer_str=answer_str)


if __name__ == "__main__":
    test_cases = [
        "The answer is Yes.",
        "The answer is no.",
        "Yes, this is correct.",
        "No, that's not right.",
        "The final answer is Yes.",
        "I think No is the answer.",
        "No.",
    ]

    for case in test_cases:
        pass
