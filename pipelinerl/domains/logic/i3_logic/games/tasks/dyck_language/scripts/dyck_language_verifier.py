from .....base.data import Data
from .....base.verifier import Verifier


class DyckLanguageVerifier(Verifier):
    def verify(self, data: Data, test_answer: str) -> bool:
        try:
            full_sequence = data.metadata["full_sequence"]

            extracted_answer = self.extract_answer(test_answer)

            is_correct = extracted_answer == full_sequence

            return is_correct

        except Exception:
            return False

    def extract_answer(self, test_solution: str) -> str:
        if not test_solution:
            return ""

        def clean_text(text: str) -> str:
            text = "".join(text.split())

            text = text.replace("\\n", "")
            text = text.replace("\\t", "")
            text = text.replace("\\r", "")
            text = text.replace("\\\\", "\\")

            if len(text) >= 2:
                if text.startswith('"') and text.endswith('"'):
                    text = text[1:-1]
                elif text.startswith("'") and text.endswith("'"):
                    text = text[1:-1]

            return text

        return clean_text(test_solution)


if __name__ == "__main__":
    test_response = ""
    metadata = {
        "trace_id": "38aeede4-d5d7-4863-91d2-df1fd99f491b",
        "full_sequence": "([])({})([()])",
        "question_sequence": "([])({})([(",
        "n_types": 3,
        "total_length": 14,
        "fill_length": 3,
        "nesting_depth": 0,
    }
    test_data = Data(question="", answer="", metadata=metadata)
    test_verifier = DyckLanguageVerifier()
    extracted_answer = test_verifier.extract_answer(test_response)
