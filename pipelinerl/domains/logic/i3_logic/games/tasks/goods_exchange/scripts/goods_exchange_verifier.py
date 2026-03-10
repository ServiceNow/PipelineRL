import re

from .....base.data import Data
from .....base.verifier import Verifier


class GoodsExchangeVerifier(Verifier):
    def verify(self, data: Data, test_solution: str):
        try:
            test_answer = self.extract_answer(test_solution)
            correct_answer = data.metadata["owns_after"]
            model_ownership = self._parse_answer(test_answer)
            correct_ownership = self._parse_answer(correct_answer)
            is_correct = self._compare_answers(model_ownership, correct_ownership)
            return is_correct
        except Exception:
            return False

    def _parse_answer(self, answer_str):
        if not answer_str:
            return {}
        result = {}
        try:
            answer_str = answer_str.strip()
            pairs = eval(answer_str)
            if isinstance(pairs, tuple):
                for pair in pairs:
                    if isinstance(pair, tuple) and len(pair) == 2:
                        person, item = pair
                        result[person.strip()] = item.strip()
                return result
        except Exception:
            if answer_str.startswith("("):
                answer_str = answer_str[1:]
            if answer_str.endswith(")"):
                answer_str = answer_str[:-1]
            person_item_pairs = []
            current_pair = ""
            bracket_count = 0
            for char in answer_str:
                if char == "(":
                    bracket_count += 1
                    current_pair += char
                elif char == ")":
                    bracket_count -= 1
                    current_pair += char
                    if bracket_count == 0:
                        person_item_pairs.append(current_pair)
                        current_pair = ""
                elif char == "," and bracket_count == 0:
                    continue
                else:
                    current_pair += char
            for pair in person_item_pairs:
                pair = pair.strip()
                if pair.startswith("("):
                    pair = pair[1:]
                if pair.endswith(")"):
                    pair = pair[:-1]
                try:
                    parts = []
                    quote_count = 0
                    current = ""
                    for char in pair:
                        if char in "\"'" and (len(current) == 0 or current[-1] != "\\"):
                            quote_count = 1 - quote_count
                        if char == "," and quote_count == 0:
                            parts.append(current.strip())
                            current = ""
                        else:
                            current += char
                    if current:
                        parts.append(current.strip())
                    if len(parts) >= 2:
                        person = parts[0].strip().strip("'\"")
                        item = parts[1].strip().strip("'\"")
                        result[person] = item
                except Exception:
                    pass
        return result

    def _compare_answers(self, model_ownership, correct_ownership):
        if len(model_ownership) != len(correct_ownership):
            return False
        model_lower_to_original = {person.lower(): person for person in model_ownership}
        for person in correct_ownership:
            if person.lower() not in model_lower_to_original:
                return False
            model_person = model_lower_to_original[person.lower()]
            if model_ownership[model_person].lower() != correct_ownership[person].lower():
                return False
        return True

    def _print_difference(self, model_ownership, correct_ownership):
        pass

    def extract_answer(self, text):
        if not text:
            return ""
        code_block_pattern = "```python\\s*\\n(.*?)\\n```"
        code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
        if code_blocks:
            last_block = code_blocks[-1].strip()
            if last_block.startswith("(") and last_block.endswith(")"):
                return last_block
        return ""
