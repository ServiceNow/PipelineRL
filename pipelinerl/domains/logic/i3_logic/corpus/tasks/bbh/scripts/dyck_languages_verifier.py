import re

from .....base import Data, Verifier


def _search_brackets(answer_str):
    pattern = r"(?<=[the|The] answer is )(.*)(?=.)"
    match = re.search(pattern, answer_str)
    if match:
        brackets = match.group(1).strip()
        return brackets
    return answer_str.strip()


class BBHDyckLanguagesVerifier(Verifier):
    def verify(self, data: Data, test_solution: str):
        try:
            test_answer = self.extract_answer(test_solution)
            ground_truth = data.answer

            test_answer = " ".join(test_answer.split())
            ground_truth = " ".join(ground_truth.split())

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
        return _search_brackets(answer_str=answer_str)


if __name__ == "__main__":
    test_cases = [
        "The answer is ] } ].",
        "The answer is ] ) ).",
        "The answer is } ] >.",
        "] } ]",
        "Final answer: ] ) )",
    ]
