import argparse
import json
import pathlib
import random
import re
import uuid
from collections import deque

from .....base.data import Data
from .....games.base.game import Game
from .....games.tasks.number_wall.scripts.number_wall_prompt import prompt_number_wall
from .....games.tasks.number_wall.scripts.number_wall_verifier import NumberWallVerifier


class NumberWall(Game):
    def __init__(self, n: int = 5, number_rate: float = 0.2):
        super().__init__("NumberWall", NumberWallVerifier)
        self.n = n
        self.number_rate = number_rate
        self.min_number = 1
        self.max_number = n
        self.failed_attempts_cache = set()

    def generate(self, num_of_questions: int = 100, max_attempts: int = 1000):
        game_data_list = []
        puzzle_hashes = set()

        for _ in range(num_of_questions):
            for attempt_idx in range(max_attempts):
                grid, solution = self._generate_simple_puzzle()

                if grid is None or solution is None:
                    continue

                puzzle_hash = hash(str(grid))
                if puzzle_hash in puzzle_hashes or puzzle_hash in self.failed_attempts_cache:
                    continue

                if self._is_valid_puzzle(grid, solution):
                    puzzle_hashes.add(puzzle_hash)
                    game_data = Data(
                        question=prompt_number_wall(grid),
                        answer="",
                        metadata={
                            "trace_id": str(uuid.uuid4()),
                            "grid": grid,
                            "solution": solution,
                            "n": self.n,
                            "number_rate": self.number_rate,
                        },
                    )
                    game_data_list.append(game_data)
                    break
                else:
                    self.failed_attempts_cache.add(puzzle_hash)

        return game_data_list

    def _generate_simple_puzzle(self):
        grid = [["X" for _ in range(self.n)] for _ in range(self.n)]

        num_numbers = max(1, int(self.n * self.n * self.number_rate * 0.8))

        solution = [["A" for _ in range(self.n)] for _ in range(self.n)]

        placed_numbers = []

        all_positions = [(i, j) for i in range(self.n) for j in range(self.n)]
        random.shuffle(all_positions)

        for row, col in all_positions:
            if len(placed_numbers) >= num_numbers:
                break

            too_close = False
            for p_row, p_col, _ in placed_numbers:
                if abs(row - p_row) + abs(col - p_col) < 2:
                    too_close = True
                    break

            if too_close:
                continue

            number = random.randint(self.min_number, self.max_number)

            island_cells = self._create_simple_island(solution, row, col, number)
            if island_cells:
                grid[row][col] = number
                solution[row][col] = number
                placed_numbers.append((row, col, number))

                for r, c in island_cells:
                    if (r, c) != (row, col):
                        solution[r][c] = "X"

        if not placed_numbers:
            return None, None

        self._fix_wall_blocks(solution)

        self._fix_diagonal_borders(solution)

        return grid, solution

    def _create_simple_island(self, solution, row, col, number):
        island_cells = set([(row, col)])

        if number == 1:
            return island_cells

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)

        cells_needed = number - 1

        for dr, dc in directions:
            if cells_needed <= 0:
                break

            r, c = row, col
            for _ in range(min(cells_needed, 2)):
                r += dr
                c += dc

                if 0 <= r < self.n and 0 <= c < self.n:
                    island_cells.add((r, c))
                    cells_needed -= 1
                else:
                    break

        if cells_needed > 0:
            for dr, dc in directions:
                if cells_needed <= 0:
                    break

                r, c = row, col
                r += dr
                c += dc

                if 0 <= r < self.n and 0 <= c < self.n and (r, c) not in island_cells:
                    island_cells.add((r, c))
                    cells_needed -= 1

        if cells_needed > 0:
            return None

        return island_cells

    def _fix_wall_blocks(self, solution):
        for i in range(self.n - 1):
            for j in range(self.n - 1):
                if (
                    solution[i][j] == "A"
                    and solution[i][j + 1] == "A"
                    and solution[i + 1][j] == "A"
                    and solution[i + 1][j + 1] == "A"
                ):
                    r, c = random.choice([(i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)])
                    solution[r][c] = "X"

    def _fix_diagonal_borders(self, solution):
        island_map = {}
        island_id = 0

        for i in range(self.n):
            for j in range(self.n):
                if solution[i][j] != "A" and (i, j) not in island_map:
                    queue = deque([(i, j)])
                    while queue:
                        r, c = queue.popleft()
                        island_map[(r, c)] = island_id

                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = r + dr, c + dc
                            if (
                                0 <= nr < self.n
                                and 0 <= nc < self.n
                                and solution[nr][nc] != "A"
                                and (nr, nc) not in island_map
                            ):
                                queue.append((nr, nc))

                    island_id += 1

        for i in range(self.n - 1):
            for j in range(self.n - 1):
                if (
                    solution[i][j] != "A"
                    and solution[i + 1][j + 1] != "A"
                    and solution[i][j + 1] == "A"
                    and solution[i + 1][j] == "A"
                ):
                    if island_map.get((i, j)) != island_map.get((i + 1, j + 1)):
                        if random.choice([True, False]):
                            solution[i][j] = "A"
                        else:
                            solution[i + 1][j + 1] = "A"

                if (
                    solution[i][j + 1] != "A"
                    and solution[i + 1][j] != "A"
                    and solution[i][j] == "A"
                    and solution[i + 1][j + 1] == "A"
                ):
                    if island_map.get((i, j + 1)) != island_map.get((i + 1, j)):
                        if random.choice([True, False]):
                            solution[i][j + 1] = "A"
                        else:
                            solution[i + 1][j] = "A"

    def _is_valid_puzzle(self, grid, solution):
        if solution is None:
            return False

        for i in range(self.n):
            for j in range(self.n):
                if isinstance(grid[i][j], int) and grid[i][j] != solution[i][j]:
                    return False

        for i in range(self.n - 1):
            for j in range(self.n - 1):
                if (
                    solution[i][j] == "A"
                    and solution[i][j + 1] == "A"
                    and solution[i + 1][j] == "A"
                    and solution[i + 1][j + 1] == "A"
                ):
                    return False

        visited = set()

        for i in range(self.n):
            for j in range(self.n):
                if (i, j) not in visited and solution[i][j] != "A":
                    island_cells = []
                    island_number = None
                    queue = deque([(i, j)])
                    visited.add((i, j))

                    while queue:
                        r, c = queue.popleft()
                        island_cells.append((r, c))

                        if isinstance(solution[r][c], int):
                            if island_number is not None:
                                return False
                            island_number = solution[r][c]

                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = r + dr, c + dc
                            if (
                                0 <= nr < self.n
                                and 0 <= nc < self.n
                                and (nr, nc) not in visited
                                and solution[nr][nc] != "A"
                            ):
                                queue.append((nr, nc))
                                visited.add((nr, nc))

                    if island_number is None:
                        return False

                    if len(island_cells) != island_number:
                        return False

        if self._has_diagonal_borders(solution):
            return False

        return True

    def _has_diagonal_borders(self, solution):
        island_map = {}
        island_id = 0

        for i in range(self.n):
            for j in range(self.n):
                if solution[i][j] != "A" and (i, j) not in island_map:
                    queue = deque([(i, j)])
                    while queue:
                        r, c = queue.popleft()
                        island_map[(r, c)] = island_id

                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = r + dr, c + dc
                            if (
                                0 <= nr < self.n
                                and 0 <= nc < self.n
                                and solution[nr][nc] != "A"
                                and (nr, nc) not in island_map
                            ):
                                queue.append((nr, nc))

                    island_id += 1

        for i in range(self.n - 1):
            for j in range(self.n - 1):
                if (
                    solution[i][j] != "A"
                    and solution[i + 1][j + 1] != "A"
                    and solution[i][j + 1] == "A"
                    and solution[i + 1][j] == "A"
                ):
                    if island_map.get((i, j)) != island_map.get((i + 1, j + 1)):
                        return True

                if (
                    solution[i][j + 1] != "A"
                    and solution[i + 1][j] != "A"
                    and solution[i][j] == "A"
                    and solution[i + 1][j + 1] == "A"
                ):
                    if island_map.get((i, j + 1)) != island_map.get((i + 1, j)):
                        return True

        return False

    def extract_answer(self, response: str):
        grid_pattern = r'\[\s*\[(?:\s*(?:"[XA]"|\'[XA]\'|[0-9]+|"[0-9]+"|\'[0-9]+\')\s*,\s*)*\s*(?:"[XA]"|\'[XA]\'|[0-9]+|"[0-9]+"|\'[0-9]+\')\s*\]\s*(?:,\s*\[(?:\s*(?:"[XA]"|\'[XA]\'|[0-9]+|"[0-9]+"|\'[0-9]+\')\s*,\s*)*\s*(?:"[XA]"|\'[XA]\'|[0-9]+|"[0-9]+"|\'[0-9]+\')\s*\]\s*)*\]'
        matches = re.findall(grid_pattern, response)

        if matches:
            grid_str = matches[-1]

            try:
                cleaned_grid_str = grid_str.replace("\n", "").replace("\r", "").strip()
                grid = json.loads(cleaned_grid_str)

                for i in range(len(grid)):
                    for j in range(len(grid[i])):
                        if isinstance(grid[i][j], str) and grid[i][j].isdigit():
                            grid[i][j] = int(grid[i][j])

                return grid
            except json.JSONDecodeError:
                try:
                    import ast

                    grid = ast.literal_eval(cleaned_grid_str)

                    for i in range(len(grid)):
                        for j in range(len(grid[i])):
                            if isinstance(grid[i][j], str) and grid[i][j].isdigit():
                                grid[i][j] = int(grid[i][j])

                    return grid
                except Exception:
                    pass
        else:
            pass

        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_data", type=int, default=100)
    parser.add_argument("--max_attempts", type=int, default=1000)
    parser.add_argument("--n_min", type=int, default=3)
    parser.add_argument("--n_max", type=int, default=6)
    parser.add_argument("--number_rate_min", type=float, default=0.15)
    parser.add_argument("--number_rate_max", type=float, default=0.3)
    args = parser.parse_args()

    data_dir = (
        pathlib.Path(__file__).parent.parent
        / "data"
        / f"n_{args.n_min}_to_{args.n_max}"
        / f"rate_{args.number_rate_min}_to_{args.number_rate_max}"
    )
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    output_file = data_dir / f"number_wall_varied_{args.num_of_data}.jsonl"

    game_data_list = []
    puzzles_generated = 0

    while puzzles_generated < args.num_of_data:
        n = random.randint(args.n_min, args.n_max)
        number_rate = round(random.uniform(args.number_rate_min, args.number_rate_max), 2)

        game = NumberWall(n=n, number_rate=number_rate)

        puzzle_data = game.generate(1, args.max_attempts)

        if puzzle_data:
            game_data_list.extend(puzzle_data)
            puzzles_generated += len(puzzle_data)

    if len(game_data_list) == 0:
        exit(1)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for game_data in game_data_list:
            f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")
