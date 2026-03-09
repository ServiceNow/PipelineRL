import argparse
import json
import pathlib
import random
import uuid
from typing import List, Set, Tuple

from .....base.data import Data
from .....games.base.game import Game
from .....games.tasks.campsite.scripts.campsite_prompt import prompt_campsite
from .....games.tasks.campsite.scripts.campsite_verifier import CampsiteVerifier


class Campsite(Game):
    def __init__(self, n: int = 5, m: int = 10, tree_density: float = 0.2):
        super().__init__("Campsite", CampsiteVerifier)
        self.n = n
        self.m = m
        self.tree_density = min(max(tree_density, 0.1), 0.4)

    def generate(self, num_of_questions: int = 100, max_attempts: int = 1000):
        game_data_list = []
        puzzle_hashes = set()

        for _ in range(num_of_questions):
            for attempt_idx in range(max_attempts):
                grid = [["X" for _ in range(self.m)] for _ in range(self.n)]

                tree_count = int(self.n * self.m * self.tree_density)
                self._place_trees(grid, tree_count)

                grid_hash = self._hash_grid(grid)
                if grid_hash in puzzle_hashes:
                    continue

                solution, is_solvable = self._solve(grid)
                if not is_solvable:
                    continue

                row_constraints, col_constraints = self._extract_constraints(solution)

                puzzle_hashes.add(grid_hash)

                game_data = Data(
                    question=prompt_campsite(grid, row_constraints, col_constraints),
                    answer="",
                    metadata={
                        "trace_id": str(uuid.uuid4()),
                        "grid": grid,
                        "solution": solution,
                        "row_constraints": row_constraints,
                        "col_constraints": col_constraints,
                        "n": self.n,
                        "m": self.m,
                        "tree_density": self.tree_density,
                    },
                )
                game_data_list.append(game_data)
                break

        return game_data_list

    def _place_trees(self, grid: List[List[str]], tree_count: int) -> None:
        positions = [(i, j) for i in range(self.n) for j in range(self.m)]
        random.shuffle(positions)

        trees_placed = 0
        for i, j in positions:
            if trees_placed >= tree_count:
                break

            if self._can_place_tree(grid, i, j):
                grid[i][j] = "T"
                trees_placed += 1

    def _can_place_tree(self, grid: List[List[str]], i: int, j: int) -> bool:
        if grid[i][j] == "T":
            return False

        has_space = False
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.n and 0 <= nj < self.m and grid[ni][nj] == "X":
                has_space = True
                break

        return has_space

    def _solve(self, grid: List[List[str]]) -> Tuple[List[List[str]], bool]:
        solution = [row[:] for row in grid]

        trees = []
        for i in range(self.n):
            for j in range(self.m):
                if grid[i][j] == "T":
                    trees.append((i, j))

        is_solvable = self._backtrack(solution, trees, set(), set())

        return solution, is_solvable

    def _backtrack(
        self,
        solution: List[List[str]],
        trees: List[Tuple[int, int]],
        used_trees: Set[Tuple[int, int]],
        tent_positions: Set[Tuple[int, int]],
    ) -> bool:
        if len(used_trees) == len(trees):
            return True

        for tree in trees:
            if tree in used_trees:
                continue

            tree_i, tree_j = tree

            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                tent_i, tent_j = tree_i + di, tree_j + dj

                if (
                    0 <= tent_i < self.n
                    and 0 <= tent_j < self.m
                    and solution[tent_i][tent_j] == "X"
                    and (tent_i, tent_j) not in tent_positions
                    and not self._has_adjacent_tent(tent_i, tent_j, tent_positions)
                ):
                    solution[tent_i][tent_j] = "C"
                    tent_positions.add((tent_i, tent_j))
                    used_trees.add(tree)

                    if self._backtrack(solution, trees, used_trees, tent_positions):
                        return True

                    solution[tent_i][tent_j] = "X"
                    tent_positions.remove((tent_i, tent_j))
                    used_trees.remove(tree)

            return False

        return True

    def _has_adjacent_tent(self, i: int, j: int, tent_positions: Set[Tuple[int, int]]) -> bool:
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if (ni, nj) in tent_positions:
                    return True
        return False

    def _extract_constraints(self, solution: List[List[str]]) -> Tuple[List[int], List[int]]:
        row_constraints = []
        for i in range(self.n):
            row_count = sum(1 for j in range(self.m) if solution[i][j] == "C")
            row_constraints.append(row_count)

        col_constraints = []
        for j in range(self.m):
            col_count = sum(1 for i in range(self.n) if solution[i][j] == "C")
            col_constraints.append(col_count)

        return row_constraints, col_constraints

    def _hash_grid(self, grid: List[List[str]]) -> str:
        return "".join("".join(row) for row in grid)

    def extract_answer(self, test_solution: str):
        import re

        grid_pattern = r"\[\s*\[.*?\]\s*\]"
        match = re.search(grid_pattern, test_solution, re.DOTALL)
        if match:
            try:
                grid_str = match.group(0)
                import ast

                return ast.literal_eval(grid_str)
            except Exception:
                pass
        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_data", type=int, default=100)
    parser.add_argument("--max_attempts", type=int, default=1000)
    parser.add_argument("--n", type=int, default=5, help="Number of rows in the grid")
    parser.add_argument("--m", type=int, default=10, help="Number of columns in the grid")
    parser.add_argument("--tree_density", type=float, default=0.2, help="Density of trees in the grid (0-1)")
    args = parser.parse_args()

    data_dir = pathlib.Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        data_dir
        / "hyperparameter_search"
        / f"n_{args.n}"
        / f"m_{args.m}"
        / f"tree_density_{args.tree_density:.2f}"
        / f"campsite_{args.num_of_data}.jsonl"
    )

    game = Campsite(n=args.n, m=args.m, tree_density=args.tree_density)
    game_data_list = game.generate(args.num_of_data, args.max_attempts)

    if len(game_data_list) == 0:
        exit(1)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for game_data in game_data_list:
            f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")
