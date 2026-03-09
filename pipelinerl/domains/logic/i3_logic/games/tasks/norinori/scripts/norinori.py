import argparse
import json
import pathlib
import random
import re
import string
import uuid
from collections import defaultdict, deque

from .....base.data import Data
from .....games.base.game import Game
from .....games.tasks.norinori.scripts.norinori_prompt import prompt_norinori
from .....games.tasks.norinori.scripts.norinori_verifier import NorinoriVerifier
from tqdm import tqdm


class Norinori(Game):
    def __init__(self, n: int = 6, region_nums_range: tuple = (6, 10), shadow_ratio: float = 0.05):
        super().__init__("Norinori", NorinoriVerifier)
        self.n = n
        self.region_nums_range = region_nums_range
        self.shadow_ratio = shadow_ratio

    def generate(self, n_samples: int = 100, max_attempts: int = 1000):
        game_data_list = []
        generated_grids = set()

        pbar = tqdm(total=n_samples, desc="生成 Norinori 谜题")

        for _ in range(n_samples):
            for attempt_idx in range(max_attempts):
                region_nums = random.randint(self.region_nums_range[0], self.region_nums_range[1])

                region_grid, region_map = self._generate_regions(region_nums)

                region_grid = self._add_shadows(region_grid)

                grid_str = "".join(["".join(row) for row in region_grid])
                if grid_str in generated_grids:
                    continue

                solution = self._solve(region_grid, region_map)
                if not solution:
                    continue

                generated_grids.add(grid_str)

                game_data = Data(
                    question=prompt_norinori(region_grid),
                    answer="",
                    metadata={
                        "trace_id": str(uuid.uuid4()),
                        "region_grid": region_grid,
                        "solution": solution,
                        "n": self.n,
                        "region_nums": region_nums,
                        "shadow_ratio": self.shadow_ratio,
                    },
                )
                game_data_list.append(game_data)

                pbar.update(1)
                break

        pbar.close()

        return game_data_list

    def _generate_regions(self, region_nums):
        grid = [[None for _ in range(self.n)] for _ in range(self.n)]
        region_map = {}

        region_labels = list(string.ascii_uppercase[:region_nums])

        for label in region_labels:
            region_size = random.randint(2, 4)
            region_map[label] = []

            while True:
                if all(all(cell is not None for cell in row) for row in grid):
                    return self._generate_regions(region_nums)

                start_i = random.randint(0, self.n - 1)
                start_j = random.randint(0, self.n - 1)
                if grid[start_i][start_j] is None:
                    break

            grid[start_i][start_j] = label
            region_map[label].append((start_i, start_j))

            queue = deque([(start_i, start_j)])
            while queue and len(region_map[label]) < region_size:
                i, j = queue.popleft()

                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                random.shuffle(directions)

                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.n and 0 <= nj < self.n and grid[ni][nj] is None:
                        grid[ni][nj] = label
                        region_map[label].append((ni, nj))
                        queue.append((ni, nj))
                        if len(region_map[label]) >= region_size:
                            break

        for i in range(self.n):
            for j in range(self.n):
                if grid[i][j] is None:
                    neighbors = []
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.n and 0 <= nj < self.n and grid[ni][nj] is not None:
                            neighbors.append(grid[ni][nj])

                    if neighbors:
                        label = random.choice(neighbors)
                        grid[i][j] = label
                        region_map[label].append((i, j))
                    else:
                        label = random.choice(region_labels)
                        grid[i][j] = label
                        region_map[label].append((i, j))

        return grid, region_map

    def _add_shadows(self, grid):
        shadow_count = int(self.n * self.n * self.shadow_ratio)

        region_counts = defaultdict(int)
        for i in range(self.n):
            for j in range(self.n):
                if grid[i][j] != "X":
                    region_counts[grid[i][j]] += 1

        for _ in range(shadow_count):
            attempts = 0
            while attempts < 100:
                i = random.randint(0, self.n - 1)
                j = random.randint(0, self.n - 1)
                region = grid[i][j]

                if region != "X" and region_counts[region] > 2:
                    grid[i][j] = "X"
                    region_counts[region] -= 1
                    break

                attempts += 1

            if attempts >= 100:
                break

        return grid

    def _solve(self, grid, region_map):
        board = [row[:] for row in grid]
        n = len(board)

        dominoes = []

        covered = [[False for _ in range(n)] for _ in range(n)]

        shadow_cells = []
        for i in range(n):
            for j in range(n):
                if board[i][j] == "X":
                    shadow_cells.append((i, j))

        region_to_cover = {}
        for region in region_map:
            if region != "X":
                region_to_cover[region] = 2

        def is_valid_placement(i1, j1, i2, j2):
            if not (0 <= i1 < n and 0 <= j1 < n and 0 <= i2 < n and 0 <= j2 < n):
                return False

            if covered[i1][j1] or covered[i2][j2]:
                return False

            for i, j in [(i1, j1), (i2, j2)]:
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if (
                        0 <= ni < n
                        and 0 <= nj < n
                        and covered[ni][nj]
                        and (ni, nj) != (i1, j1)
                        and (ni, nj) != (i2, j2)
                    ):
                        return False

            for i, j in [(i1, j1), (i2, j2)]:
                if 0 <= i < n and 0 <= j < n and board[i][j] != "X":
                    region = board[i][j]
                    if region_to_cover.get(region, 0) <= 0:
                        return False

            return True

        def update_coverage(i1, j1, i2, j2, add=True):
            for i, j in [(i1, j1), (i2, j2)]:
                if 0 <= i < n and 0 <= j < n:
                    covered[i][j] = add
                    if board[i][j] != "X" and add:
                        region_to_cover[board[i][j]] -= 1
                    elif board[i][j] != "X" and not add:
                        region_to_cover[board[i][j]] += 1

        def all_conditions_met():
            if not all(covered[i][j] for i, j in shadow_cells):
                return False

            for region, remaining in region_to_cover.items():
                if remaining != 0:
                    return False

            return True

        def backtrack():
            if all_conditions_met():
                return [[(i + 1, j + 1), (i2 + 1, j2 + 1)] for (i, j), (i2, j2) in dominoes]

            for i, j in shadow_cells:
                if not covered[i][j]:
                    if j + 1 < n and not covered[i][j + 1] and is_valid_placement(i, j, i, j + 1):
                        update_coverage(i, j, i, j + 1, True)
                        dominoes.append(((i, j), (i, j + 1)))
                        result = backtrack()
                        if result:
                            return result
                        dominoes.pop()
                        update_coverage(i, j, i, j + 1, False)

                    if i + 1 < n and not covered[i + 1][j] and is_valid_placement(i, j, i + 1, j):
                        update_coverage(i, j, i + 1, j, True)
                        dominoes.append(((i, j), (i + 1, j)))
                        result = backtrack()
                        if result:
                            return result
                        dominoes.pop()
                        update_coverage(i, j, i + 1, j, False)

                    if j - 1 >= 0 and not covered[i][j - 1] and is_valid_placement(i, j - 1, i, j):
                        update_coverage(i, j - 1, i, j, True)
                        dominoes.append(((i, j - 1), (i, j)))
                        result = backtrack()
                        if result:
                            return result
                        dominoes.pop()
                        update_coverage(i, j - 1, i, j, False)

                    if i - 1 >= 0 and not covered[i - 1][j] and is_valid_placement(i - 1, j, i, j):
                        update_coverage(i - 1, j, i, j, True)
                        dominoes.append(((i - 1, j), (i, j)))
                        result = backtrack()
                        if result:
                            return result
                        dominoes.pop()
                        update_coverage(i - 1, j, i, j, False)

                    return None

            for region, remaining in list(region_to_cover.items()):
                if remaining > 0:
                    cells = [(i, j) for i, j in region_map[region] if not covered[i][j]]

                    for idx, (i, j) in enumerate(cells):
                        if j + 1 < n and (i, j + 1) in cells and is_valid_placement(i, j, i, j + 1):
                            update_coverage(i, j, i, j + 1, True)
                            dominoes.append(((i, j), (i, j + 1)))
                            result = backtrack()
                            if result:
                                return result
                            dominoes.pop()
                            update_coverage(i, j, i, j + 1, False)

                        if i + 1 < n and (i + 1, j) in cells and is_valid_placement(i, j, i + 1, j):
                            update_coverage(i, j, i + 1, j, True)
                            dominoes.append(((i, j), (i + 1, j)))
                            result = backtrack()
                            if result:
                                return result
                            dominoes.pop()
                            update_coverage(i, j, i + 1, j, False)

                    if remaining == 2:
                        return None

            if all_conditions_met():
                return [[(i + 1, j + 1), (i2 + 1, j2 + 1)] for (i, j), (i2, j2) in dominoes]

            for i in range(n):
                for j in range(n):
                    if not covered[i][j]:
                        if j + 1 < n and not covered[i][j + 1] and is_valid_placement(i, j, i, j + 1):
                            update_coverage(i, j, i, j + 1, True)
                            dominoes.append(((i, j), (i, j + 1)))
                            result = backtrack()
                            if result:
                                return result
                            dominoes.pop()
                            update_coverage(i, j, i, j + 1, False)

                        if i + 1 < n and not covered[i + 1][j] and is_valid_placement(i, j, i + 1, j):
                            update_coverage(i, j, i + 1, j, True)
                            dominoes.append(((i, j), (i + 1, j)))
                            result = backtrack()
                            if result:
                                return result
                            dominoes.pop()
                            update_coverage(i, j, i + 1, j, False)

            return None

        return backtrack()

    def extract_answer(self, test_solution: str, strict=False):
        answer_patterns = [
            r"\[\s*\[\s*\(\s*\d+\s*,\s*\d+\s*\)\s*,\s*\(\s*\d+\s*,\s*\d+\s*\)\s*\]",
            r"答案是\s*(.*?)\s*$",
            r"answer is\s*(.*?)\s*$",
            r"solution is\s*(.*?)\s*$",
        ]

        for pattern in answer_patterns:
            matches = re.findall(pattern, test_solution, re.IGNORECASE | re.DOTALL)
            if matches:
                return matches[-1]

        return test_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100, help="生成的谜题数量")
    parser.add_argument("--max_attempts", type=int, default=1000, help="每个谜题的最大尝试次数")
    parser.add_argument("--n", type=int, default=5, help="网格大小")
    parser.add_argument("--min_regions", type=int, default=3, help="最小区域数量")
    parser.add_argument("--max_regions", type=int, default=5, help="最大区域数量")
    parser.add_argument("--shadow_ratio", type=float, default=0.1, help="阴影格子比率")
    args = parser.parse_args()

    region_range = (args.min_regions, args.max_regions)
    data_dir = (
        pathlib.Path(__file__).parent.parent
        / "data"
        / f"n_{args.n}"
        / f"regions_{region_range[0]}-{region_range[1]}"
        / f"shadow_{args.shadow_ratio}"
    )
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        data_dir
        / f"norinori_n{args.n}_regions{region_range[0]}-{region_range[1]}_shadow{args.shadow_ratio}_{args.n_samples}.jsonl"
    )

    game = Norinori(n=args.n, region_nums_range=region_range, shadow_ratio=args.shadow_ratio)
    game_data_list = game.generate(args.n_samples, args.max_attempts)

    if len(game_data_list) == 0:
        exit(1)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for game_data in game_data_list:
            f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")
