from abc import abstractmethod

from ...base import Data, Verifier


class Corpus:
    def __init__(self, name: str, verifier: Verifier):
        self.name = name
        self.verifier = verifier()

    @abstractmethod
    def generate(self, num_of_questions: int = 100, max_attempts: int = 100):
        raise NotImplementedError("Corpus.generate() is not implemented")

    def verify(self, corpus_data: Data, test_solution: str):
        test_answer = self.extract_answer(test_solution)
        return self.verifier.verify(corpus_data, test_answer)

    @abstractmethod
    def extract_answer(self, test_solution: str):
        raise NotImplementedError("Corpus.extract_answer() is not implemented")
