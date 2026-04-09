import re

from .....base import Data, Verifier

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"


def _extract_answer(text):
    pattern = "<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None


def _extract_solution_with_thought(solution_str):
    model_output = solution_str
    if THOUGHT_DELIMITER_END in solution_str:
        model_output = solution_str.split(THOUGHT_DELIMITER_END)[1]
    predict_answer = _extract_answer(model_output)
    if predict_answer is not None:
        return predict_answer
    else:
        return model_output


def _search_boolean(answer_str):
    strict_match = re.search("(?<=[the|The] answer is )(.*)(?=.)", answer_str)
    if strict_match:
        result = strict_match.group(1).strip()
        bool_match = re.search("\\b(true|false)\\b", result, re.IGNORECASE)
        if bool_match:
            return bool_match.group(0)
    flexible_match = re.search("\\b(true|false)\\b", answer_str, re.IGNORECASE)
    if flexible_match:
        return flexible_match.group(0)
    return ""


class BBHBooleanExpressionsVerifier(Verifier):
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
        answer_str = _extract_solution_with_thought(solution_str=test_solution)
        return _search_boolean(answer_str=answer_str)


if __name__ == "__main__":
    answer_str = "true"
