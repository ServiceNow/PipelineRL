import re

from .....base.data import Data
from .....base.verifier import Verifier


class GameOf24Verifier(Verifier):
    def verify(self, data: Data, test_solution: str):
        try:
            test_answer = self.extract_answer(test_solution)
            numbers = data.metadata["numbers"]
            operators = data.metadata["operators"]
            target_result = data.metadata["result"]
            input_numbers = [str(num) for num in numbers]
            answer_numbers_str = re.sub("[^0-9]", " ", test_answer)
            answer_numbers = [num for num in answer_numbers_str.split() if num]
            answer_wo_numbers_str = re.sub("[0-9\\s]", "", test_answer)
            unknown_chars = []
            for c in answer_wo_numbers_str:
                if c in operators:
                    continue
                if c in ["(", ")"]:
                    continue
                unknown_chars.append(c)
            if len(unknown_chars) > 0:
                return False
            for num in answer_numbers:
                if num not in input_numbers:
                    return False
                if answer_numbers.count(num) > input_numbers.count(num):
                    return False
            if len(answer_numbers) != len(input_numbers):
                return False
            evaluated = eval(test_answer)
            return abs(float(evaluated) - float(target_result)) < 1e-10
        except Exception:
            return False

    def extract_answer(self, test_solution: str):
        regex_pattern = "```python.*?```"
        matches = re.findall(regex_pattern, test_solution, re.DOTALL)
        if matches:
            return matches[-1].replace("```python", "").replace("```", "").strip()
        else:
            return ""
