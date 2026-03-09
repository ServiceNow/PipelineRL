import json

from .....base.data import Data


class WordscapesData(Data):
    def __init__(self, idx, grid, across_words, down_words, solution):
        self.idx = idx
        self.grid = grid
        self.across_words = across_words
        self.down_words = down_words
        self.solution = solution
        self.answer = self._format_answer(solution)
        self.question = self._format_question()

    def _format_question(self):
        grid_str = "\n".join([" ".join(row) for row in self.grid])
        across_str = "Across: " + ", ".join(self.across_words)
        down_str = "Down: " + ", ".join(self.down_words)
        return f"Grid:\n{grid_str}\n\n{across_str}\n{down_str}"

    def _format_answer(self, solution):
        return json.dumps(solution)

    def to_dict(self):
        return {
            "idx": self.idx,
            "grid": self.grid,
            "across_words": self.across_words,
            "down_words": self.down_words,
            "solution": self.solution,
            "question": self.question,
            "answer": self.answer,
        }

    @classmethod
    def from_dict(cls, data_dict):
        instance = cls(
            data_dict["idx"],
            data_dict["grid"],
            data_dict["across_words"],
            data_dict["down_words"],
            data_dict["solution"],
        )
        instance.question = data_dict.get("question", instance.question)
        instance.answer = data_dict.get("answer", instance.answer)
        return instance
