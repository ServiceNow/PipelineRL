import re

from .....base import Data, Verifier


def _extract_number(answer_str):
    strict_pattern = r"(?<=[the|The] answer is )([-0-9]+)(?=.)"
    match = re.search(strict_pattern, answer_str)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass

    number_pattern = r"([-0-9]+)"
    match = re.search(number_pattern, answer_str)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass

    return answer_str.strip()


class BBHMultistepArithmeticVerifier(Verifier):
    def verify(self, data: Data, test_solution: str):
        try:
            test_answer = self.extract_answer(test_solution)
            ground_truth = data.answer

            if isinstance(test_answer, str):
                try:
                    test_answer = int(test_answer)
                except ValueError:
                    return False

            if isinstance(ground_truth, str):
                try:
                    ground_truth = int(ground_truth)
                except ValueError:
                    return False

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
        return _extract_number(answer_str=answer_str)


if __name__ == "__main__":
    test_cases = [
        "The answer is 42.",
        "The answer is -15.",
        "The final result is 100",
        "42",
        "-15",
        "The sum equals 100.",
    ]
