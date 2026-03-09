from abc import ABC, abstractmethod

from ...base import Data, Verifier


class Game(ABC):
    def __init__(self, name: str, verifier: Verifier):
        self.name = name
        self.verifier = verifier()

    @abstractmethod
    def generate(self, num_of_questions: int = 100, max_attempts: int = 100):
        raise NotImplementedError("Game.generate() is not implemented")

    def verify(self, data: Data, test_solution: str):
        return self.verifier.verify(data, test_solution)

    @abstractmethod
    def extract_answer(self, test_solution: str):
        raise NotImplementedError("Game.extract_answer() is not implemented")
