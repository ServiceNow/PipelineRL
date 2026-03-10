from .....base.data import Data
from .....base.verifier import Verifier


class SpaceReasoningVerifier(Verifier):
    def verify(self, data: Data, test_answer: str):
        test_answer = self.extract_answer(test_answer)
        if test_answer is None:
            return False
        return test_answer.lower() == data.answer.lower()

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
