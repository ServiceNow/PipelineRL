import argparse
import copy
import json
import pathlib
import random
import re
import time
import uuid

from .....base.data import Data
from .....games.base.game import Game
from .....games.tasks.sudoku.scripts.sudoku_prompt import prompt_sudoku
from .....games.tasks.sudoku.scripts.sudoku_verifier import SudokuVerifier


class Sudoku(Game):
    def __init__(self):
        super().__init__("Sudoku", SudokuVerifier)

    def generate(
        self, num_of_questions: int = 100, max_attempts: int = 100, difficulty: int = 3, unique_solution: bool = True
    ):
        if difficulty < 1 or difficulty > 4:
            raise ValueError("难度级别必须在1-4之间")

        game_data_list = []
        attempts = 0

        while len(game_data_list) < num_of_questions and attempts < max_attempts:
            try:
                attempts += 1

                complete_sudoku = self._generate_complete_sudoku()

                masked_sudoku = self._mask_sudoku_by_difficulty(complete_sudoku, difficulty, unique_solution)

                is_chinese = random.choice([True, False])
                question = prompt_sudoku(masked_sudoku, is_chinese)

                answer_str = str(tuple(tuple(row) for row in complete_sudoku))

                game_data = Data(
                    question=question,
                    answer=answer_str,
                    difficulty=difficulty,
                    metadata={
                        "trace_id": str(uuid.uuid4()),
                        "original_sudoku": masked_sudoku,
                        "complete_sudoku": complete_sudoku,
                        "difficulty": difficulty,
                        "unique_solution": unique_solution,
                    },
                )

                game_data_list.append(game_data)

            except Exception:
                continue

        return game_data_list

    def _generate_complete_sudoku(self):
        grid = [[0 for _ in range(9)] for _ in range(9)]

        if self._solve_sudoku(grid):
            return grid
        else:
            raise RuntimeError("无法生成有效的数独解答")

    def _is_valid_placement(self, grid, row, col, num):
        for x in range(9):
            if grid[row][x] == num:
                return False

        for x in range(9):
            if grid[x][col] == num:
                return False

        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                if grid[i + start_row][j + start_col] == num:
                    return False

        return True

    def _solve_sudoku(self, grid):
        empty_cell = self._find_empty_cell(grid)
        if not empty_cell:
            return True

        row, col = empty_cell

        nums = list(range(1, 10))
        random.shuffle(nums)

        for num in nums:
            if self._is_valid_placement(grid, row, col, num):
                grid[row][col] = num

                if self._solve_sudoku(grid):
                    return True

                grid[row][col] = 0

        return False

    def _find_empty_cell(self, grid):
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 0:
                    return (i, j)
        return None

    def _mask_sudoku_by_difficulty(self, complete_sudoku, difficulty, unique_solution=False):
        masked_sudoku = copy.deepcopy(complete_sudoku)

        cells_to_keep_range = {
            1: (70, 80),
            2: (55, 70),
            3: (40, 55),
            4: (25, 40),
        }

        min_cells, max_cells = cells_to_keep_range.get(difficulty, (20, 25))

        cells_to_keep = random.randint(min_cells, max_cells)

        all_cells = [(i, j) for i in range(9) for j in range(9)]

        random.shuffle(all_cells)

        if not unique_solution:
            cells_to_keep_coords = all_cells[:cells_to_keep]

            for i in range(9):
                for j in range(9):
                    if (i, j) not in cells_to_keep_coords:
                        masked_sudoku[i][j] = "X"

            return masked_sudoku

        current_sudoku = copy.deepcopy(complete_sudoku)
        cells_to_remove = len(all_cells) - cells_to_keep
        removed_cells = 0

        for i, j in all_cells:
            temp_value = current_sudoku[i][j]
            current_sudoku[i][j] = "X"

            if self._has_unique_solution(current_sudoku, complete_sudoku):
                removed_cells += 1
                if removed_cells >= cells_to_remove:
                    break
            else:
                current_sudoku[i][j] = temp_value

        return current_sudoku

    def _has_unique_solution(self, masked_sudoku, expected_solution):
        sudoku_copy = [[0 if masked_sudoku[i][j] == "X" else masked_sudoku[i][j] for j in range(9)] for i in range(9)]

        solutions = []

        def solve_all(grid, i=0, j=0):
            if len(solutions) > 1:
                return

            if i == 9:
                if all(grid[i][j] == expected_solution[i][j] for i in range(9) for j in range(9)):
                    solutions.append(True)
                else:
                    solutions.append(False)
                return

            next_i, next_j = (i, j + 1) if j < 8 else (i + 1, 0)

            if grid[i][j] != 0:
                solve_all(grid, next_i, next_j)
                return

            for num in range(1, 10):
                if self._is_valid_placement(grid, i, j, num):
                    grid[i][j] = num
                    solve_all(grid, next_i, next_j)

                    if len(solutions) > 1:
                        return
                    grid[i][j] = 0

        solve_all(sudoku_copy)

        return len(solutions) == 1 and solutions[0] == True

    def extract_answer(self, test_solution: str):
        if not test_solution:
            return ""

        code_block_pattern = r"```python\s*([\s\S]*?)\s*```"
        matches = re.findall(code_block_pattern, test_solution)

        if matches:
            python_code = matches[-1].strip()
            return python_code

        tuple_pattern = r"\(\s*\(\s*\d+\s*,.*?\)\s*\)"
        matches = re.findall(tuple_pattern, test_solution, re.DOTALL)

        if matches:
            return matches[-1].strip()

        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数独游戏生成器")
    parser.add_argument("--num_of_data", type=int, default=100, help="生成的题目数量")
    parser.add_argument("--max_attempts", type=int, default=1000, help="每个题目的最大尝试次数")
    parser.add_argument("--difficulty", type=int, default=3, help="难度级别（1-4）")
    parser.add_argument("--unique_solution", action="store_true", help="是否要求数独有唯一解（默认：否）")
    args = parser.parse_args()

    data_dir = pathlib.Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    output_dir = (
        data_dir / f"unique_solution_{args.unique_solution}/difficulty_{args.difficulty}/num_of_data_{args.num_of_data}"
    )
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "data.jsonl"

    game = Sudoku()

    start_time = time.time()

    game_data_list = game.generate(
        num_of_questions=args.num_of_data,
        max_attempts=args.max_attempts,
        difficulty=args.difficulty,
        unique_solution=args.unique_solution,
    )

    end_time = time.time()
    generation_time = end_time - start_time

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for game_data in game_data_list:
                f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")
    except Exception:
        pass
