import re

from .....base.data import Data
from .....base.verifier import Verifier


class BooleanExpressionsVerifier(Verifier):
    def verify(self, data: Data, test_answer: str):
        try:
            test_answer = self.extract_answer(test_answer)
            if test_answer is None:
                return False
            test_answer_letters = re.findall("[a-zA-Z]", test_answer)
            ground_truth_letters = re.findall("[a-zA-Z]", data.answer)
            test_answer_letters = self.lower(test_answer_letters)
            ground_truth_letters = self.lower(ground_truth_letters)
            test_set = set(test_answer_letters)
            ground_truth_set = set(ground_truth_letters)
            return test_set == ground_truth_set
        except Exception:
            return False

    def lower(self, answer_list):
        return [answer.lower() for answer in answer_list]

    def extract_answer(self, answer_str):
        last_box_index = answer_str.rfind("\\boxed{")
        if last_box_index == -1:
            return None
        start_index = last_box_index + len("\\boxed{")
        bracket_stack = 1
        end_index = start_index
        while end_index < len(answer_str) and bracket_stack > 0:
            if answer_str[end_index] == "{":
                bracket_stack += 1
            elif answer_str[end_index] == "}":
                bracket_stack -= 1
            end_index += 1
        if bracket_stack != 0:
            return None
        latex_content = answer_str[start_index : end_index - 1].strip()
        return latex_content
