import collections
import re

from .....base import Data, Verifier


def _extract_sorted_words(answer_str, original_words):
    word_pattern = "|".join([f"\\b{w}\\b" for w in original_words])
    regex = re.compile(word_pattern)

    strict_match = re.search(r"(?<=[the|The] answer is )(.*)(?=.)", answer_str)
    if strict_match:
        result = strict_match.group(1).strip()

        strict_matches = regex.findall(result)
        if strict_matches:
            strict_matches.reverse()
            ordered_words = reversed(collections.OrderedDict(zip(strict_matches, [None] * len(strict_matches))))
            return " ".join(ordered_words)

    flexible_matches = regex.findall(answer_str)
    if flexible_matches:
        flexible_matches.reverse()
        ordered_words = reversed(collections.OrderedDict(zip(flexible_matches, [None] * len(flexible_matches))))
        return " ".join(ordered_words)

    return ""


class BBHWordSortingVerifier(Verifier):
    def verify(self, data: Data, test_solution: str):
        try:
            if "List:" not in data.question:
                return False
            parts = data.question.split("List:")
            if len(parts) < 2:
                return False
            original_words = parts[1].split("\n")[0].strip().split()
            if not original_words:
                return False

            test_answer = self.extract_answer(test_solution, original_words)
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

    def extract_answer(self, test_solution: str, original_words):
        answer_str = test_solution
        return _extract_sorted_words(answer_str=answer_str, original_words=original_words)


if __name__ == "__main__":
    original_words = ["apple", "banana", "orange"]
    test_cases = [
        "The answer is banana apple.",
    ]
