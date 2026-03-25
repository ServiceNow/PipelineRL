import re

from .....base.data import Data
from .....base.verifier import Verifier


class WebOfLiesVerifier(Verifier):
    def verify(self, data: Data, test_solution: str):
        try:
            test_answer = self.extract_answer(test_solution)
            expected_answer = data.answer.lower()
            test_answer = test_answer.lower()
            expected_truths = self._parse_answer(expected_answer)
            test_truths = self._parse_answer(test_answer)
            if len(expected_truths) != len(test_truths):
                return False
            for i, (expected, actual) in enumerate(zip(expected_truths, test_truths)):
                if expected != actual:
                    return False
            return True
        except Exception:
            return False

    def _parse_answer(self, answer_str):
        yes_pattern = "yes|true|truth"
        no_pattern = "no|false|lie"
        cn_yes_pattern = "是|真话|真"
        cn_no_pattern = "否|假话|假|谎"
        yes_patterns = f"({yes_pattern}|{cn_yes_pattern})"
        no_patterns = f"({no_pattern}|{cn_no_pattern})"
        truths = []
        all_answers = re.findall(f"{yes_patterns}|{no_patterns}", answer_str)
        for match in all_answers:
            match_str = next((m for m in match if m), "")
            if re.search(yes_pattern, match_str) or re.search(cn_yes_pattern, match_str):
                truths.append(True)
            elif re.search(no_pattern, match_str) or re.search(cn_no_pattern, match_str):
                truths.append(False)
        return truths

    def extract_answer(self, test_solution: str) -> str:
        if not test_solution:
            return ""
        cn_patterns = ["答案是[：:]\\s*\\*\\*([^*]+)\\*\\*[.。]*$"]
        en_patterns = ["[Tt]he answer is[：:=]\\s*\\*\\*([^*]+)\\*\\*[.。]*$"]
        patterns = cn_patterns + en_patterns
        for pattern in patterns:
            matches = re.findall(pattern, test_solution, re.DOTALL)
            if matches:
                return matches[-1].strip()
        lines = test_solution.strip().split("\n")
        if lines:
            last_line = lines[-1].strip()
            bold_match = re.search("\\*\\*([^*]+)\\*\\*", last_line)
            if bold_match:
                return bold_match.group(1).strip()
            answer_match = re.search("(?:答案是|[Tt]he answer is)[：:=]?\\s*(.*?)(?:[.。]|$)", last_line)
            if answer_match:
                return answer_match.group(1).strip()
        yes_no_pattern = "(?:\\b(?:yes|no|是|否)\\b[,，\\s]*)+"
        matches = re.findall(yes_no_pattern, test_solution.lower())
        if matches:
            return matches[-1].strip()
        return ""
