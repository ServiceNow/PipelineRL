import argparse
import json
import pathlib
import random
import re
import uuid

from .....base.data import Data
from .....games.base.game import Game
from .....games.tasks.kukurasu.scripts.kukurasu_prompt import prompt_kukurasu
from .....games.tasks.kukurasu.scripts.kukurasu_verifier import KukurasuVerifier


class Kukurasu(Game):
    def __init__(self, n: int = 4, m: int = 4, ones_probability: float = 0.3):
        super().__init__("Kukurasu", KukurasuVerifier)
        self.n = n
        self.m = m
        self.ones_probability = ones_probability

    def generate(self, num_of_questions: int = 100, max_attempts: int = 1000):
        game_data_list = []
        puzzle_hashes = set()

        for _ in range(num_of_questions):
            for attempt_idx in range(max_attempts):
                solution_grid = self._generate_random_solution()

                row_sums, col_sums = self._calculate_sums(solution_grid)

                empty_grid = [["X" for _ in range(self.m)] for _ in range(self.n)]

                puzzle_hash = hash(tuple(map(tuple, solution_grid)))
                if puzzle_hash in puzzle_hashes:
                    continue

                puzzle_hashes.add(puzzle_hash)

                if self._is_valid_puzzle(row_sums, col_sums):
                    game_data = Data(
                        question=prompt_kukurasu(empty_grid, row_sums, col_sums),
                        answer="",
                        metadata={
                            "trace_id": str(uuid.uuid4()),
                            "grid": empty_grid,
                            "row_sums": row_sums,
                            "col_sums": col_sums,
                            "solution": solution_grid,
                            "n": self.n,
                            "m": self.m,
                            "ones_probability": self.ones_probability,
                        },
                    )
                    game_data_list.append(game_data)
                    break

        return game_data_list

    def _generate_random_solution(self):
        grid = []
        for i in range(self.n):
            row = []
            for j in range(self.m):
                if random.random() < self.ones_probability:
                    row.append("1")
                else:
                    row.append("X")
            grid.append(row)

        return grid

    def _calculate_sums(self, grid):
        row_sums = []
        col_sums = []

        for i, row in enumerate(grid):
            row_sum = 0
            for j, cell in enumerate(row):
                if cell == "1":
                    row_sum += j + 1
            row_sums.append(row_sum)

        for j in range(self.m):
            col_sum = 0
            for i in range(self.n):
                if grid[i][j] == "1":
                    col_sum += i + 1
            col_sums.append(col_sum)

        return row_sums, col_sums

    def _is_valid_puzzle(self, row_sums, col_sums):
        if 0 in row_sums or 0 in col_sums:
            return False

        if sum(row_sums) != sum(col_sums):
            return False

        return True

    def extract_answer(self, response: str):
        grid_pattern = (
            r'\[\s*\[(?:\s*"[X1]"\s*,\s*)*\s*"[X1]"\s*\]\s*(?:,\s*\[(?:\s*"[X1]"\s*,\s*)*\s*"[X1]"\s*\]\s*)*\]'
        )
        match = re.search(grid_pattern, response)

        if match:
            try:
                grid_str = match.group(0)
                return json.loads(grid_str)
            except json.JSONDecodeError:
                pass

        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_data", type=int, default=100)
    parser.add_argument("--max_attempts", type=int, default=1000)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--ones_probability", type=float, default=0.3)
    args = parser.parse_args()

    data_dir = (
        pathlib.Path(__file__).parent.parent / "data" / f"n_{args.n}" / f"m_{args.m}" / f"p_{args.ones_probability}"
    )
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    output_file = data_dir / f"kukurasu_{args.n}x{args.m}_p{args.ones_probability}_{args.num_of_data}.jsonl"

    game = Kukurasu(n=args.n, m=args.m, ones_probability=args.ones_probability)
    game_data_list = game.generate(args.num_of_data, args.max_attempts)

    if len(game_data_list) == 0:
        exit(1)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for game_data in game_data_list:
            f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")
