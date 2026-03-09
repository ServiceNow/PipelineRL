import re

from .....base.data import Data
from .....base.verifier import Verifier


class MathPathVerifier(Verifier):
    def verify(self, data: Data, test_answer: str):
        try:
            test_answer = self.extract_answer(test_solution=test_answer)
        except Exception:
            return False
        try:
            metadata = data.metadata
            # ref_expr = metadata["ref_expr"]
            query_expr = metadata["query_expr"]
            test_tmp = test_answer.replace(" ", "").strip()
            query_tmp = query_expr.replace(" ", "").strip()
            # ref_tmp = ref_expr.replace(" ", "").strip()
            query_nums = [x for x in query_tmp if "0" <= x <= "9" or x == "?"]
            test_nums = [x for x in test_tmp if "0" <= x <= "9"]
            if len(query_nums) != len(test_nums):
                return False
            else:
                for ind, x in enumerate(query_nums):
                    if x == "?":
                        continue
                    if x != test_nums[ind]:
                        return False
            query_symbols = [x for x in query_tmp if x in ["+", "-", "*", "/", "%"]]
            test_symbols = [x for x in test_tmp if x in ["+", "-", "*", "/", "%"]]
            if len(query_symbols) != len(test_symbols):
                return False
            else:
                for ind, x in enumerate(query_symbols):
                    if x != test_symbols[ind]:
                        return False
            try:
                tmp = test_tmp.replace("=", "==")
                if not eval(tmp):
                    return False
            except Exception:
                return False
            return True
        except Exception:
            return False

    def extract_answer(self, test_solution: str):
        if not test_solution:
            return ""
        code_block_pattern = "\\[\\[(.*?)\\]\\]"
        code_matches = re.findall(code_block_pattern, test_solution)
        if code_matches:
            operation_expression = code_matches[-1].strip()
            return operation_expression
        return ""
