import re

from .....base.data import Data
from .....base.verifier import Verifier


class DyckLanguageReasoningErrorsVerifier(Verifier):
    def verify(self, data: Data, test_answer: str):
        try:
            test_answer = self.extract_answer(test_solution=test_answer)

            correct_indices = data.metadata["error_indices"]

            # expected_answer = self._format_answer(correct_indices)

            if "不确定" in test_answer or "不知道" in test_answer or "unclear" in test_answer.lower():
                return False

            cleaned_test_answer = self._standardize_answer(test_answer)

            if not correct_indices and (
                cleaned_test_answer == ""
                or cleaned_test_answer.lower()
                in ["无问题", "no", "无错误", "no error", "no errors", "no mistakes", "all correct"]
            ):
                is_correct = True
            else:
                test_error_indices = self._extract_error_indices(cleaned_test_answer)
                expected_error_indices = set(correct_indices)

                is_correct = test_error_indices == expected_error_indices

            return is_correct

        except Exception:
            return False

    def _standardize_answer(self, answer: str) -> str:
        if not answer or answer.strip() == "":
            return ""

        if answer.lower() in ["无问题", "no", "无错误", "no error", "no errors", "no mistakes", "all correct"]:
            return ""

        answer = answer.replace("，", ",")

        answer = re.sub(r"[^0-9,]", "", answer)

        return answer

    def _extract_error_indices(self, answer: str) -> set:
        if not answer or answer.strip() == "":
            return set()

        answer = answer.replace("，", ",")

        try:
            return set(int(idx) for idx in answer.split(",") if idx and idx.strip().isdigit())
        except ValueError:
            return set()

    def _format_answer(self, error_indices: list) -> str:
        if not error_indices:
            return ""

        sorted_indices = sorted(error_indices)

        return ",".join(map(str, sorted_indices))

    def extract_answer(self, test_solution: str) -> str:
        answer_str = test_solution
        return answer_str


if __name__ == "__main__":
    test_response = ""
    metadata = {
        "trace_id": "77db72eb-a9db-46cd-96ea-5a49eba78792",
        "dyck_sequence": "{}[][][]{}{}([])",
        "thoughts": [
            "Thought 1: 我们应该逐个处理输入并跟踪栈的配置。",
            "Thought 2: 栈: 空",
            "Thought 3: { ; 栈: {[",
            "Thought 4: } ; 栈: 空",
            "Thought 5: [ ; 栈: [",
            "Thought 6: ] ; 栈: 空",
            "Thought 7: [ ; 栈: [",
            "Thought 8: ] ; 栈: 空",
            "Thought 9: [ ; 栈: [",
            "Thought 10: ] ; 栈: ]",
            "Thought 11: { ; 栈: {",
            "Thought 12: } ; 栈: 空",
            "Thought 13: { ; 栈: {",
            "Thought 14: } ; 栈: 空",
            "Thought 15: ( ; 栈: (",
            "Thought 16: [ ; 栈: ([",
            "Thought 17: ] ; 栈: (",
            "Thought 18: ) ; 栈: 空",
            "Thought 19: 现在，我们已经到达结尾。最终栈是空的。",
        ],
        "error_indices": [3, 10],
        "n_types": 3,
        "total_length": 15,
        "n_errors": 2,
    }
    test_data = Data(question="", answer="", metadata=metadata)
    test_verifier = DyckLanguageReasoningErrorsVerifier()
    extracted_answer = test_verifier.extract_answer(test_response)
