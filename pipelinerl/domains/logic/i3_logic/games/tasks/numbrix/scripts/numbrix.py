import argparse
import json
import math
import pathlib
import random
import uuid

from .....base.data import Data
from .....games.base.game import Game
from .....games.tasks.numbrix.scripts.numbrix_prompt import prompt_numbrix
from .....games.tasks.numbrix.scripts.numbrix_verifier import NumbrixVerifier


class Numbrix(Game):
    def __init__(self, n: int = 4, fill_rate: float = 0.3):
        super().__init__("Numbrix", NumbrixVerifier)
        self.n = n
        self.fill_rate = fill_rate

    def _generate_solution(self):
        n = self.n
        max_attempts = 100

        for _ in range(max_attempts):
            grid = [[0 for _ in range(n)] for _ in range(n)]

            directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

            row, col = random.randrange(n), random.randrange(n)
            grid[row][col] = 1

            current_num = 2
            steps_without_progress = 0
            max_steps_without_progress = n * n * 2

            while current_num <= n * n and steps_without_progress < max_steps_without_progress:
                possible_moves = []
                for dr, dc in directions:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < n and 0 <= new_col < n and grid[new_row][new_col] == 0:
                        possible_moves.append((new_row, new_col))

                if not possible_moves:
                    steps_without_progress += 1
                    continue

                next_row, next_col = random.choice(possible_moves)
                grid[next_row][next_col] = current_num
                row, col = next_row, next_col
                current_num += 1
                steps_without_progress = 0

            if current_num > n * n:
                return grid

        return self._generate_snake_pattern()

    def _generate_snake_pattern(self):
        n = self.n
        grid = [[0 for _ in range(n)] for _ in range(n)]

        num = 1
        for i in range(n):
            if i % 2 == 0:
                for j in range(n):
                    grid[i][j] = num
                    num += 1
            else:
                for j in range(n - 1, -1, -1):
                    grid[i][j] = num
                    num += 1

        return grid

    def generate(self, num_of_questions: int = 100, max_attempts: int = 1000):
        game_data_list = []
        grid_combinations = set()

        for _ in range(num_of_questions):
            success = False

            for attempt_idx in range(max_attempts):
                solution_grid = self._generate_solution()

                flattened = [num for row in solution_grid for num in row]
                if sorted(flattened) != list(range(1, self.n * self.n + 1)):
                    continue

                puzzle_grid = self._create_puzzle_from_solution(solution_grid)

                puzzle_tuple = tuple(map(tuple, puzzle_grid))
                if puzzle_tuple in grid_combinations:
                    continue

                grid_combinations.add(puzzle_tuple)

                game_data = Data(
                    question=prompt_numbrix(puzzle_grid),
                    answer="",
                    metadata={
                        "trace_id": str(uuid.uuid4()),
                        "grid": puzzle_grid,
                        "solution": solution_grid,
                        "n": self.n,
                        "fill_rate": self.fill_rate,
                    },
                )
                game_data_list.append(game_data)
                success = True
                break

            if not success:
                break

        return game_data_list

    def _create_puzzle_from_solution(self, solution_grid):
        puzzle = [row[:] for row in solution_grid]
        total_cells = self.n * self.n
        cells_to_remove = int(total_cells * (1 - self.fill_rate))

        all_cells = [
            (r, c)
            for r in range(self.n)
            for c in range(self.n)
            if solution_grid[r][c] != 1 and solution_grid[r][c] != self.n * self.n
        ]

        cells_to_remove = min(cells_to_remove, len(all_cells))
        remove_cells = random.sample(all_cells, cells_to_remove)

        for r, c in remove_cells:
            puzzle[r][c] = "X"

        return puzzle

    def _solve_numbrix(self, puzzle_grid):
        grid = [row[:] for row in puzzle_grid]

        known_positions = {}
        for r in range(self.n):
            for c in range(self.n):
                if grid[r][c] != "X":
                    known_positions[grid[r][c]] = (r, c)

        def backtrack():
            if all(grid[r][c] != "X" for r in range(self.n) for c in range(self.n)):
                return True

            for r in range(self.n):
                for c in range(self.n):
                    if grid[r][c] == "X":
                        for num in range(1, self.n * self.n + 1):
                            if self._is_valid_placement(grid, r, c, num):
                                grid[r][c] = num
                                if backtrack():
                                    return True
                                grid[r][c] = "X"
                        return False

            return True

        backtrack()
        return grid

    def _is_valid_placement(self, grid, row, col, num):
        for r in range(self.n):
            for c in range(self.n):
                if grid[r][c] == num:
                    return False

        # directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        if num > 1:
            found_prev = False
            for r in range(self.n):
                for c in range(self.n):
                    if grid[r][c] == num - 1:
                        if abs(r - row) + abs(c - col) == 1:
                            found_prev = True
                        else:
                            return False

            if not found_prev and any(grid[r][c] == num - 1 for r in range(self.n) for c in range(self.n)):
                return False

        if num < self.n * self.n:
            found_next = False
            for r in range(self.n):
                for c in range(self.n):
                    if grid[r][c] == num + 1:
                        if abs(r - row) + abs(c - col) == 1:
                            found_next = True
                        else:
                            return False

            if not found_next and any(grid[r][c] == num + 1 for r in range(self.n) for c in range(self.n)):
                return False

        return True

    def extract_answer(self, test_solution: str, strict=False):
        try:
            import ast
            import re

            pattern = r"\[\s*\[\s*\d+.*?\]\s*\]"
            matches = re.finditer(pattern, test_solution, re.DOTALL)
            match = None

            for m in matches:
                match = m
            if not match:
                return None

            grid_text = match.group(0)

            grid_text = grid_text.replace("'", "").replace('"', "")

            grid = ast.literal_eval(grid_text)

            if not isinstance(grid, list) or not all(isinstance(row, list) for row in grid):
                return None

            if not all(isinstance(cell, int) for row in grid for cell in row):
                return None

            return grid
        except Exception:
            return None


def generate_puzzles_in_range(n_min, n_max, fill_rate_min, fill_rate_max, total_puzzles, max_attempts):
    all_game_data = []

    n_values = list(range(n_min, n_max + 1))

    if fill_rate_min == fill_rate_max:
        fill_rates = [fill_rate_min]
    else:
        fill_rates = [round(fill_rate_min + i * (fill_rate_max - fill_rate_min) / 9, 1) for i in range(10)]

    total_configs = len(n_values) * len(fill_rates)

    puzzles_per_config = math.ceil(total_puzzles / total_configs)

    puzzles_generated = 0
    for n in n_values:
        for fill_rate in fill_rates:
            remaining_puzzles = total_puzzles - puzzles_generated
            if remaining_puzzles <= 0:
                break

            current_batch_size = min(puzzles_per_config, remaining_puzzles)

            game = Numbrix(n=n, fill_rate=fill_rate)
            game_data_list = game.generate(current_batch_size, max_attempts)

            all_game_data.extend(game_data_list)
            puzzles_generated += len(game_data_list)

            if puzzles_generated >= total_puzzles:
                break

        if puzzles_generated >= total_puzzles:
            break

    return all_game_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_data", type=int, default=40, help="要生成的谜题总数")
    parser.add_argument("--max_attempts", type=int, default=1000, help="每个谜题的最大尝试次数")
    parser.add_argument("--n_min", type=int, default=6, help="最小网格大小")
    parser.add_argument("--n_max", type=int, default=9, help="最大网格大小")
    parser.add_argument("--fill_rate_min", type=float, default=0.3, help="最小填充率 (0-1)")
    parser.add_argument("--fill_rate_max", type=float, default=0.3, help="最大填充率 (0-1)")
    parser.add_argument("--output_file", type=str, default=None, help="输出文件路径")
    args = parser.parse_args()

    if args.n_min > args.n_max:
        exit(1)

    if args.fill_rate_min > args.fill_rate_max:
        exit(1)

    if args.fill_rate_min < 0 or args.fill_rate_max > 1:
        exit(1)

    all_game_data = generate_puzzles_in_range(
        args.n_min, args.n_max, args.fill_rate_min, args.fill_rate_max, args.num_of_data, args.max_attempts
    )

    if args.output_file:
        output_file = pathlib.Path(args.output_file)
    else:
        data_dir = pathlib.Path(__file__).parent.parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        if args.n_min == args.n_max:
            n_part = f"n{args.n_min}"
        else:
            n_part = f"n{args.n_min}-{args.n_max}"

        if args.fill_rate_min == args.fill_rate_max:
            fill_part = f"fill{args.fill_rate_min}"
        else:
            fill_part = f"fill{args.fill_rate_min}-{args.fill_rate_max}"

        output_file = data_dir / f"numbrix_{n_part}_{fill_part}_{len(all_game_data)}.jsonl"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for game_data in all_game_data:
            f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")
