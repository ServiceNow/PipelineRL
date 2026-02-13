import argparse
import itertools
import json
import pathlib
import random
import re
import uuid

from .....base.data import Data
from .....games.base.game import Game
from .....games.tasks.game_of_24.scripts.game_of_24_prompt import prompt_game_of_24
from .....games.tasks.game_of_24.scripts.game_of_24_verifier import GameOf24Verifier


class GameOf24(Game):
    def __init__(
        self,
        num_of_numbers: int = 4,
        result: int = 24,
        candidates: list[int] = list(range(1, 10)),
        operators: list[str] = ["+", "-", "*", "/"],
    ):
        super().__init__("Game of 24", GameOf24Verifier)
        self.num_of_numbers = num_of_numbers
        self.result = result
        self.candidates = candidates
        self.operators = operators

    def generate(self, num_of_questions: int = 100, max_attempts: int = 1000):
        game_data_list = []
        nums_combinations = set()
        for _ in range(num_of_questions):
            for attempt_idx in range(max_attempts):
                numbers = [random.choice(self.candidates) for _ in range(self.num_of_numbers)]
                numbers = sorted(numbers)
                if tuple(numbers) in nums_combinations:
                    continue
                nums_combinations.add(tuple(numbers))
                solutions = self.find_all_solutions(numbers, self.operators)
                if not solutions:
                    continue
                game_data = Data(
                    question=prompt_game_of_24(numbers, self.operators, self.result),
                    answer="",
                    metadata={
                        "trace_id": str(uuid.uuid4()),
                        "numbers": numbers,
                        "solutions": solutions,
                        "candidates": self.candidates,
                        "operators": self.operators,
                        "result": self.result,
                        "num_of_numbers": self.num_of_numbers,
                    },
                )
                game_data_list.append(game_data)
                break
            if attempt_idx == max_attempts - 1:
                return []
        return game_data_list

    def extract_answer(self, test_solution: str):
        regex_pattern = "[Tt]he answer is"
        indices = [m.end() for m in re.finditer(regex_pattern, test_solution)]
        if len(indices) == 0:
            return ""
        else:
            answer = test_solution[indices[-1] :].strip()
            answer = answer.replace(".", "")
            answer = answer.replace("$", "")
            answer = answer.replace("\\(", "")
            answer = answer.replace("\\)", "")
            return answer

    def find_all_solutions(self, numbers, operators):
        solutions = set()
        for nums in itertools.permutations(numbers):
            for ops in itertools.product(operators, repeat=len(nums) - 1):
                cur_nums = list(nums)
                cur_ops = list(ops)
                while cur_ops:
                    op = cur_ops.pop(0)
                    cur_num1 = cur_nums.pop(0)
                    cur_num2 = cur_nums.pop(0)
                    cur_nums.insert(0, eval(f"{cur_num1} {op} {cur_num2}"))
                if abs(cur_nums[0] - self.result) < 1e-10:
                    solutions.add(f"nums: {nums}, ops: {ops}")
        return list(solutions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_data", type=int, default=100)
    parser.add_argument("--max_attempts", type=int, default=1000)
    parser.add_argument("--num_of_numbers", type=int, default=4)
    parser.add_argument("--result", type=int, default=24)
    parser.add_argument("--operators", type=str, nargs="+", default=["+", "-", "*", "/"])
    parser.add_argument("--candidates", type=int, nargs="+", default=list(range(1, 10)))
    args = parser.parse_args()
    valid_operators = {"+": "a", "-": "b", "*": "c", "/": "d", "%": "e"}
    data_dir = pathlib.Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    operators_str = ""
    for operator in args.operators:
        if operator in valid_operators:
            operators_str += valid_operators[operator]
        else:
            raise ValueError(f"Invalid operator: {operator}")
    min_candidate = min(args.candidates)
    max_candidate = max(args.candidates)
    output_file = (
        data_dir
        / "hyperparameter_search"
        / f"num_of_numbers_{args.num_of_numbers}"
        / f"operators_{operators_str}"
        / f"candidates_{min_candidate}_{max_candidate}"
        / f"sum_{args.result}"
        / f"game_of_24_{args.num_of_data}.jsonl"
    )
    game = GameOf24(
        num_of_numbers=args.num_of_numbers,
        result=args.result,
        operators=list(args.operators),
        candidates=list(args.candidates),
    )
    game_data_list = game.generate(args.num_of_data, args.max_attempts)
    if len(game_data_list) == 0:
        exit(1)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for game_data in game_data_list:
            f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")
