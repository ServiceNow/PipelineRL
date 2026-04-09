import json
import re

from .....base.data import Data
from .....base.verifier import Verifier


class TimeSequenceVerifier(Verifier):
    def verify(self, data: Data, test_solution: str):
        try:
            test_answer = self.extract_answer(test_solution)
            metadata = data.metadata
            true_answers = metadata["records"]["answers"]
            try:
                test_list = json.loads(test_answer.replace("，", ","))
            except Exception:
                return False
            try:
                if not isinstance(test_list, list) or len(test_list) < 2:
                    return False
                if test_list[0] != true_answers["answer_maxLen"]:
                    return False
                if test_list[1] != true_answers["answer_nums"]:
                    return False
            except Exception:
                return False
            return True
        except Exception:
            return False

    def extract_answer(self, test_solution: str):
        if not test_solution:
            return ""
        matrix_pattern = "\\[.*?\\]"
        matrix_matches = re.findall(matrix_pattern, test_solution, re.DOTALL)
        if matrix_matches:
            return matrix_matches[-1].strip()
        return ""
