import math_verify
from .....base.data import Data
from .....base.verifier import Verifier


class OperationVerifier(Verifier):
    def verify(self, data: Data, test_answer: str):
        try:
            ground_truth = math_verify.parse(data.answer)
            parsed_answer = math_verify.parse(test_answer)
            if parsed_answer is None:
                return False
            return math_verify.verify(parsed_answer, ground_truth)
        except Exception:
            return False

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
