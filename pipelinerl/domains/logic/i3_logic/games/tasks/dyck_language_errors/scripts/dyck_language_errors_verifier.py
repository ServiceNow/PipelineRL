import re

from .....base.data import Data
from .....base.verifier import Verifier


class DyckLanguageErrorsVerifier(Verifier):
    def verify(self, data: Data, test_answer: str):
        try:
            test_answer = self.extract_answer(test_solution=test_answer)

            if data.metadata["is_valid"]:
                correct_answer = "-1"
            else:
                correct_answer = str(data.metadata["first_error_pos"])

            test_answer = test_answer.strip()

            if correct_answer == "-1":
                if test_answer == "-1":
                    is_correct = True
                else:
                    is_correct = False
            else:
                try:
                    is_correct = int(test_answer) == int(correct_answer)
                except (ValueError, TypeError):
                    is_correct = False

            return is_correct

        except Exception:
            return False

    def extract_answer(self, test_solution: str):
        if test_solution is None:
            return ""

        answer_str = test_solution
        solution = test_solution.strip() if test_solution else ""
        numbers = re.findall(r"-?\d+", solution)
        if numbers:
            if "-1" in numbers:
                return "-1"
            for num in numbers:
                if num.isdigit() and int(num) >= 0:
                    return num
            if numbers:
                return numbers[0]

        if answer_str and any(keyword in answer_str.lower() for keyword in ["合法", "valid", "correct"]):
            return "-1"

        return answer_str if answer_str else ""
