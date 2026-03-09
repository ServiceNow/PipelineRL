import argparse
import ast
import copy
import itertools
import json
import pathlib
import random
import re
import uuid

from .....base.data import Data
from .....games.base.game import Game
from .....games.tasks.skyscraper_puzzle.scripts.skyscraper_puzzle_prompt import (
    generate_prompts,
    prompt_skyscraper_puzzle,
)
from .....games.tasks.skyscraper_puzzle.scripts.skyscraper_puzzle_verifier import SkyscraperPuzzleVerifier


class SkyscraperPuzzle(Game):
    def __init__(self, n=4):
        super().__init__("Skyscraper Puzzle", SkyscraperPuzzleVerifier)
        self.n = n

    def generate(self, num_of_questions: int = 100, max_attempts: int = 1000):
        game_data_list = []

        while len(game_data_list) < num_of_questions and max_attempts > 0:
            n = self.n

            solved_grid, top, bottom, left, right = self._generate_valid_skyscraper(n)

            if solved_grid:
                initial_grid = [["X" for _ in range(n)] for _ in range(n)]

                question = prompt_skyscraper_puzzle(n, initial_grid, top, bottom, left, right)

                all_prompts = generate_prompts(n, initial_grid, top, bottom, left, right)

                game_data = Data(
                    question=question,
                    answer="",
                    metadata={
                        "trace_id": str(uuid.uuid4()),
                        "n": n,
                        "solved_grid": solved_grid,
                        "top": top,
                        "bottom": bottom,
                        "left": left,
                        "right": right,
                        "all_prompts": all_prompts,
                    },
                )

                game_data_list.append(game_data)
            else:
                max_attempts -= 1

        return game_data_list

    def extract_answer(self, test_solution: str):
        try:
            n = self.n

            code_block_pattern = r"```python\s*\n([\s\S]*?)\n\s*```"
            code_blocks = re.findall(code_block_pattern, test_solution)

            if code_blocks:
                code_block = code_blocks[0].strip()
                try:
                    grid = ast.literal_eval(code_block)

                    if (
                        isinstance(grid, list)
                        and len(grid) == n
                        and all(isinstance(row, list) and len(row) == n for row in grid)
                    ):
                        return grid
                except Exception:
                    code_without_comments = re.sub(r"#.*$", "", code_block, flags=re.MULTILINE)
                    try:
                        grid = ast.literal_eval(code_without_comments.strip())
                        if (
                            isinstance(grid, list)
                            and len(grid) == n
                            and all(isinstance(row, list) and len(row) == n for row in grid)
                        ):
                            return grid
                    except Exception:
                        pass

            return test_solution
        except Exception:
            return test_solution

    def _generate_valid_skyscraper(self, n):
        for _ in range(100):
            try:
                rows = []
                numbers = list(range(1, n + 1))

                first_row = copy.deepcopy(numbers)
                random.shuffle(first_row)
                rows.append(first_row)

                for i in range(1, n):
                    available_perms = list(itertools.permutations(numbers))
                    random.shuffle(available_perms)

                    valid_perm_found = False
                    for perm in available_perms:
                        valid = True
                        for col in range(n):
                            col_values = [rows[r][col] for r in range(i)]
                            if perm[col] in col_values:
                                valid = False
                                break

                        if valid:
                            rows.append(list(perm))
                            valid_perm_found = True
                            break

                    if not valid_perm_found:
                        raise ValueError("无法找到有效行排列")

                grid = rows
                top = []
                bottom = []
                left = []
                right = []

                for j in range(n):
                    top.append(self._count_visible_skyscrapers([grid[i][j] for i in range(n)]))

                for j in range(n):
                    bottom.append(self._count_visible_skyscrapers([grid[i][j] for i in range(n - 1, -1, -1)]))

                for i in range(n):
                    left.append(self._count_visible_skyscrapers(grid[i]))

                for i in range(n):
                    right.append(self._count_visible_skyscrapers(grid[i][::-1]))

                return grid, top, bottom, left, right

            except Exception:
                continue

        return None, None, None, None, None

    def _count_visible_skyscrapers(self, heights):
        visible_count = 0
        max_height = 0

        for height in heights:
            if height > max_height:
                visible_count += 1
                max_height = height

        return visible_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="摩天楼游戏生成器")
    parser.add_argument("--num_of_data", type=int, default=100, help="生成的题目数量")
    parser.add_argument("--max_attempts", type=int, default=1000, help="每个题目的最大尝试次数")
    parser.add_argument("--n", type=int, default=4, help="网格大小")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    args = parser.parse_args()

    if args.output_dir:
        output_dir = pathlib.Path(args.output_dir)
    else:
        output_dir = pathlib.Path(__file__).parent.parent / "data"

    nested_dir = output_dir / "hyperparameter_search" / f"n_{args.n}" / f"num_of_data_{args.num_of_data}"

    if not nested_dir.exists():
        nested_dir.mkdir(parents=True, exist_ok=True)

    nested_output_file = nested_dir / f"num_of_data_{args.num_of_data}.jsonl"

    direct_output_file = output_dir / f"skyscraper_puzzle_{args.num_of_data}.jsonl"

    game = SkyscraperPuzzle(n=args.n)

    game_data_list = game.generate(args.num_of_data, args.max_attempts)

    try:
        with open(nested_output_file, "w", encoding="utf-8") as f:
            for game_data in game_data_list:
                f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")

        with open(direct_output_file, "w", encoding="utf-8") as f:
            for game_data in game_data_list:
                f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")
    except Exception:
        pass
