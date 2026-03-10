import re

from .....base.data import Data
from .....base.verifier import Verifier


class BuggyTableVerifier(Verifier):
    def extract_answer(self, answer: str) -> str:
        return self._extract_answer(answer)

    def verify(self, data: Data, test_answer: str) -> bool:
        try:
            expected_answer = data.answer if data and hasattr(data, "answer") else ""
        except Exception as e:
            print(e)
            print("FUUUUUUCKKKKKK")
            return False
        if not expected_answer and (not test_answer):
            return True
        try:
            normalized_expected = self._extract_answer(expected_answer)
        except Exception as e:
            print(e)
            print("FUUUUUUCKKKKKK YOOOUUUU")
            return False
        try:
            normalized_test = self._extract_answer(test_answer)
        except Exception as e:
            print(e)
            print("FUUUUUUCKKKKKK YOOOUUUU A LOOOT")
            return False
        return normalized_expected == normalized_test

    def _is_raw_numeric_answer(self, value: str) -> bool:
        value = value.strip()
        import re

        return bool(re.match("^-?\\d+(\\.\\d+)?$", value))

    def _raw_has_exactly_two_decimals(self, value: str) -> bool:
        value = value.strip()
        parts = value.replace("-", "", 1).split(".")
        return len(parts) == 2 and len(parts[1]) == 2

    def _is_numeric(self, value: str) -> bool:
        value = value.strip()
        if value.startswith("-"):
            value = value[1:]
        return value.replace(".", "", 1).isdigit()

    def _has_exactly_two_decimals(self, value: str) -> bool:
        value = value.strip()
        if value.startswith("-"):
            value = value[1:]
        parts = value.split(".")
        if len(parts) != 2:
            return False
        return len(parts[1]) == 2

    def _extract_answer(self, answer: str) -> str:
        normalized = str(answer).strip() if answer is not None else ""
        exact_matches = re.findall("-?\\d+\\.\\d{2}\\b", normalized)
        if exact_matches:
            return exact_matches[-1]
        return normalized
