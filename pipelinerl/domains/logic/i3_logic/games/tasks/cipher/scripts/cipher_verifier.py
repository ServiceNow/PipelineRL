from .....base.data import Data
from .....base.verifier import Verifier


class CipherVerifier(Verifier):
    def verify(self, data: Data, test_solution: str):
        try:
            answer = self.extract_answer(test_solution)
            if not answer:
                return 0.0
            answer = answer.split("```python")[-1].split("```")[0].strip()
            answer = eval(answer)
            gold_answer = data.answer
            if isinstance(gold_answer, str) and len(gold_answer) >= 4:
                gold_answer = [[gold_answer[2:-2]]]
            else:
                gold_answer = [[gold_answer]]
            if answer == gold_answer:
                return 1.0
            else:
                return 0.0
        except Exception:
            return 0.0

    def extract_answer(self, test_solution: str):
        return test_solution
