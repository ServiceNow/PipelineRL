import re

from .....base.data import Data
from .....base.verifier import Verifier


class WordSortingMistakeVerifier(Verifier):
    def verify(self, data: Data, test_answer: str):
        try:
            ground_truth = data.answer if data.answer is not None else "No"
            parsed_answer = self.extract_answer(test_answer)
            if parsed_answer is None:
                return False
            if parsed_answer.isdigit():
                try:
                    return int(parsed_answer) == int(ground_truth)
                except Exception:
                    return False
            else:
                return parsed_answer.lower() == ground_truth.lower()
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
