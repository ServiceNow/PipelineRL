import re

from .....base import Data, Verifier


def _search_plausibility(answer_str):
    strict_match = re.search(r"(?<=[the|The] answer is )(.*)(?=.)", answer_str)
    if strict_match:
        result = strict_match.group(1).strip()

        no_match = re.search(r"\b(no|not plausible)\b", result, re.IGNORECASE)
        if no_match:
            return "no"

        yes_match = re.search(r"\b(yes|plausible)\b", result, re.IGNORECASE)
        if yes_match:
            return "yes"

    flexible_no_match = re.search(r"\b(no|not plausible)\b", answer_str, re.IGNORECASE)
    if flexible_no_match:
        return "no"

    flexible_yes_match = re.search(r"\b(yes|plausible)\b", answer_str, re.IGNORECASE)
    if flexible_yes_match:
        return "yes"

    return ""


class BBHSportsUnderstandingVerifier(Verifier):
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
        return _search_plausibility(answer_str=answer_str)


if __name__ == "__main__":
    test_cases = [
        "no The answer is yes.",
        "The answer is plausible.",
        "The answer is no.",
        "The answer is not plausible.",
        "Yes, this makes sense.",
        "No, this is impossible.",
        "This statement is plausible.",
        "This scenario is not plausible.",
    ]
