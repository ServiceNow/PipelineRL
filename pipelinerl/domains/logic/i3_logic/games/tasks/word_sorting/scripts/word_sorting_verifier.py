import re

from .....base.data import Data
from .....base.verifier import Verifier


class WordSortingVerifier(Verifier):
    def str2list(self, answer_str):
        answer_str = answer_str.replace("，", ",").replace(" ", "")
        return [w.strip() for w in answer_str.split(",")]

    def verify(self, data: Data, test_answer: str):
        try:
            ground_truth = self.str2list(data.answer)
            parsed_answer = self.str2list(self.extract_answer(test_answer))
            if parsed_answer is None:
                return False
            return parsed_answer == ground_truth
        except Exception:
            return False

    def extract_answer(self, answer_str):
        last_box_index = answer_str.rfind("\\boxed{")
        if last_box_index == -1:
            return None
        last_box_substring = answer_str[last_box_index:]
        box_pattern = "\\\\boxed\\{([^}]*)\\}"
        match = re.search(box_pattern, last_box_substring)
        if match:
            return match.group(1).strip()
        return None
