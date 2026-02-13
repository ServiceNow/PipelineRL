import argparse
import json
import math
import pathlib
import random
import re
import uuid
from collections import defaultdict
from typing import List, Tuple

from .....base.data import Data
from .....games.base.game import Game
from .....games.tasks.minesweeper.scripts.minesweeper_prompt import prompt_minesweeper
from .....games.tasks.minesweeper.scripts.minesweeper_verifier import MinesweeperVerifier


class Minesweeper(Game):
    def __init__(self, n: int = 8, mine_den: float = 0.2, reveal_frac: float = 0.4):
        super().__init__("Minesweeper", MinesweeperVerifier)
        self.n = n
        self.mine_den = mine_den
        self.reveal_frac = reveal_frac

    def generate(self, num_of_questions: int = 100, max_attempts: int = 5000):
        game_data_list = []
        puzzle_hashes = set()
        full_grid_hashes = set()

        for _ in range(num_of_questions):
            # valid_puzzle_found = False
            for attempt_idx in range(max_attempts):
                full_grid = self._generate_full_grid()

                full_grid_hash = hash(tuple(map(tuple, full_grid)))
                if full_grid_hash in full_grid_hashes:
                    continue
                full_grid_hashes.add(full_grid_hash)

                revealed_grid, revealed_positions = self._create_revealed_grid(full_grid)

                puzzle_hash = hash(tuple(map(tuple, revealed_grid)))
                if puzzle_hash in puzzle_hashes:
                    continue

                current_mines, reasoning_steps = self._find_deterministic_mines(
                    full_grid, revealed_grid, revealed_positions
                )

                if len(current_mines) > 0:
                    if self._verify_unique_solution(revealed_grid, current_mines):
                        puzzle_hashes.add(puzzle_hash)

                        game_data = Data(
                            question=prompt_minesweeper(revealed_grid),
                            answer="",
                            metadata={
                                "trace_id": str(uuid.uuid4()),
                                "full_grid": full_grid,
                                "revealed_grid": revealed_grid,
                                "current_mines": list(current_mines),
                                "reasoning_steps": reasoning_steps,
                                "n": self.n,
                                "mine_den": self.mine_den,
                                "reveal_frac": self.reveal_frac,
                            },
                        )
                        game_data_list.append(game_data)
                        # valid_puzzle_found = True
                        break

        return game_data_list

    def _generate_full_grid(self):
        total_cells = self.n * self.n
        num_mines = int(total_cells * self.mine_den)

        mine_positions = random.sample([(i, j) for i in range(self.n) for j in range(self.n)], num_mines)

        full_grid = [["0" for _ in range(self.n)] for _ in range(self.n)]

        for i, j in mine_positions:
            full_grid[i][j] = "M"

        for i in range(self.n):
            for j in range(self.n):
                if full_grid[i][j] != "M":
                    mine_count = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.n and 0 <= nj < self.n and full_grid[ni][nj] == "M":
                                mine_count += 1
                    full_grid[i][j] = str(mine_count)

        return full_grid

    def _create_revealed_grid(self, full_grid):
        non_mine_cells = []
        for i in range(self.n):
            for j in range(self.n):
                if full_grid[i][j] != "M":
                    non_mine_cells.append((i, j))

        num_to_reveal = int(len(non_mine_cells) * self.reveal_frac)
        num_to_reveal = max(num_to_reveal, 1)

        revealed_positions = random.sample(non_mine_cells, num_to_reveal)

        revealed_grid = [["X" for _ in range(self.n)] for _ in range(self.n)]
        for i, j in revealed_positions:
            revealed_grid[i][j] = full_grid[i][j]

        return revealed_grid, revealed_positions

    def _get_adjacent_cells(self, i, j):
        adjacent_cells = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.n and 0 <= nj < self.n:
                    adjacent_cells.append((ni, nj))
        return adjacent_cells

    def _find_deterministic_mines(self, full_grid, revealed_grid, revealed_positions):
        deterministic_mines = set()
        reasoning_steps = []

        cell_to_constraints = defaultdict(list)
        for i, j in revealed_positions:
            if revealed_grid[i][j] not in ["0", "X", "M"]:
                for ni, nj in self._get_adjacent_cells(i, j):
                    if revealed_grid[ni][nj] == "X":
                        cell_to_constraints[(ni, nj)].append((i, j))

        new_mines_found = True
        iteration = 0

        while new_mines_found:
            iteration += 1
            new_mines_found = False

            for i, j in revealed_positions:
                if revealed_grid[i][j] not in ["0", "X", "M"]:
                    adjacent_cells = self._get_adjacent_cells(i, j)

                    adjacent_unrevealed = [
                        (ni, nj)
                        for ni, nj in adjacent_cells
                        if revealed_grid[ni][nj] == "X" and (ni, nj) not in deterministic_mines
                    ]

                    known_mines = [(ni, nj) for ni, nj in adjacent_cells if (ni, nj) in deterministic_mines]

                    if len(adjacent_unrevealed) + len(known_mines) == int(revealed_grid[i][j]) and adjacent_unrevealed:
                        step_mines = []
                        for mine_pos in adjacent_unrevealed:
                            if mine_pos not in deterministic_mines:
                                deterministic_mines.add(mine_pos)
                                step_mines.append(mine_pos)
                                new_mines_found = True

                        if step_mines:
                            reasoning_steps.append(
                                {
                                    "iteration": iteration,
                                    "source_cell": (i, j),
                                    "source_value": revealed_grid[i][j],
                                    "found_mines": step_mines,
                                    "logic": "All unrevealed cells must be mines because their count plus known mines equals the cell value",
                                }
                            )

            if not new_mines_found:
                break

        for i, j in deterministic_mines:
            assert full_grid[i][j] == "M", f"Cell ({i},{j}) was determined to be a mine but is not in the full grid"

        return deterministic_mines, reasoning_steps

    def _verify_unique_solution(self, revealed_grid, current_mines):
        test_grid = [row[:] for row in revealed_grid]

        for i, j in current_mines:
            test_grid[i][j] = "M"

        remaining_unrevealed = []
        for i in range(self.n):
            for j in range(self.n):
                if test_grid[i][j] == "X":
                    remaining_unrevealed.append((i, j))

        if not remaining_unrevealed:
            return True

        return len(current_mines) > 0

    def extract_answer(self, response: str) -> List[Tuple[int, int]]:
        patterns = [
            r"\[\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)(?:\s*,\s*\(\s*\d+\s*,\s*\d+\s*\))*\s*\]",
            r"\[\s*\[\s*(\d+)\s*,\s*(\d+)\s*\](?:\s*,\s*\[\s*\d+\s*,\s*\d+\s*\])*\s*\]",
            r"\(\s*(\d+)\s*,\s*(\d+)\s*\)(?:\s*,\s*\(\s*\d+\s*,\s*\d+\s*\))*",
        ]

        for pattern in patterns:
            coords = []
            for match in re.finditer(pattern, response):
                try:
                    coord_pattern = r"(?:\(|\[)\s*(\d+)\s*,\s*(\d+)\s*(?:\)|\])"
                    for coord_match in re.finditer(coord_pattern, match.group(0)):
                        i, j = int(coord_match.group(1)), int(coord_match.group(2))
                        coords.append((i, j))
                except Exception:
                    continue

            if coords:
                return coords

        number_pairs = re.findall(r"(\d+)[^\d]+(\d+)", response)
        if number_pairs:
            return [(int(i), int(j)) for i, j in number_pairs]

        return []


def generate_puzzles_in_range(
    n_min, n_max, mine_den_min, mine_den_max, reveal_frac_min, reveal_frac_max, total_puzzles, max_attempts
):
    all_game_data = []

    n_values = list(range(n_min, n_max + 1))

    if mine_den_min == mine_den_max:
        mine_densities = [mine_den_min]
    else:
        mine_densities = [round(mine_den_min + i * (mine_den_max - mine_den_min) / 9, 2) for i in range(10)]

    if reveal_frac_min == reveal_frac_max:
        reveal_fractions = [reveal_frac_min]
    else:
        reveal_fractions = [round(reveal_frac_min + i * (reveal_frac_max - reveal_frac_min) / 9, 2) for i in range(10)]

    total_configs = len(n_values) * len(mine_densities) * len(reveal_fractions)

    puzzles_per_config = math.ceil(total_puzzles / total_configs)

    puzzles_generated = 0
    for n in n_values:
        for mine_den in mine_densities:
            for reveal_frac in reveal_fractions:
                remaining_puzzles = total_puzzles - puzzles_generated
                if remaining_puzzles <= 0:
                    break

                current_batch_size = min(puzzles_per_config, remaining_puzzles)

                game = Minesweeper(n=n, mine_den=mine_den, reveal_frac=reveal_frac)
                game_data_list = game.generate(current_batch_size, max_attempts)

                all_game_data.extend(game_data_list)
                puzzles_generated += len(game_data_list)

                if puzzles_generated >= total_puzzles:
                    break

            if puzzles_generated >= total_puzzles:
                break

        if puzzles_generated >= total_puzzles:
            break

    return all_game_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_data", type=int, default=10, help="要生成的谜题总数")
    parser.add_argument("--max_attempts", type=int, default=5000, help="每个谜题的最大尝试次数")
    parser.add_argument("--n_min", type=int, default=8, help="最小网格大小")
    parser.add_argument("--n_max", type=int, default=8, help="最大网格大小")
    parser.add_argument("--mine_den_min", type=float, default=0.2, help="最小地雷密度 (0-1)")
    parser.add_argument("--mine_den_max", type=float, default=0.2, help="最大地雷密度 (0-1)")
    parser.add_argument("--reveal_frac_min", type=float, default=0.4, help="最小揭示比例 (0-1)")
    parser.add_argument("--reveal_frac_max", type=float, default=0.4, help="最大揭示比例 (0-1)")
    parser.add_argument("--output_file", type=str, default=None, help="输出文件路径")
    args = parser.parse_args()

    if args.n_min > args.n_max:
        exit(1)

    if args.mine_den_min > args.mine_den_max:
        exit(1)

    if args.reveal_frac_min > args.reveal_frac_max:
        exit(1)

    if args.mine_den_min < 0 or args.mine_den_max > 1:
        exit(1)

    if args.reveal_frac_min < 0 or args.reveal_frac_max > 1:
        exit(1)

    all_game_data = generate_puzzles_in_range(
        args.n_min,
        args.n_max,
        args.mine_den_min,
        args.mine_den_max,
        args.reveal_frac_min,
        args.reveal_frac_max,
        args.num_of_data,
        args.max_attempts,
    )

    if args.output_file:
        output_file = pathlib.Path(args.output_file)
    else:
        data_dir = pathlib.Path(__file__).parent.parent / "data"

        if args.n_min == args.n_max:
            n_part = f"n_{args.n_min}"
        else:
            n_part = f"n_{args.n_min}-{args.n_max}"

        if args.mine_den_min == args.mine_den_max:
            mine_part = f"mine_den_{args.mine_den_min}"
        else:
            mine_part = f"mine_den_{args.mine_den_min}-{args.mine_den_max}"

        if args.reveal_frac_min == args.reveal_frac_max:
            reveal_part = f"reveal_{args.reveal_frac_min}"
        else:
            reveal_part = f"reveal_{args.reveal_frac_min}-{args.reveal_frac_max}"

        data_dir = data_dir / n_part / mine_part / reveal_part
        data_dir.mkdir(parents=True, exist_ok=True)

        output_file = data_dir / f"minesweeper_{n_part}_{mine_part}_{reveal_part}_{len(all_game_data)}.jsonl"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for game_data in all_game_data:
            f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")
