from abc import ABC, abstractmethod

from .data import Data


class Verifier(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def verify(self, data: Data, test_answer: str):
        raise NotImplementedError("Verifier.verify() is not implemented")

    @abstractmethod
    def extract_answer(self, test_solution: str):
        raise NotImplementedError("Verifier.extract_answer() is not implemented")


import re

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
        return ""


class ExactMatchVerifier(Verifier):
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
        return _extract_solution_with_thought(solution_str=test_solution)
