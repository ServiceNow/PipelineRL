import argparse
import ast
import json
import pathlib
import random
import re
import string
import uuid

from .....base.data import Data
from .....games.base.game import Game
from .....games.tasks.star_placement_puzzle.scripts.star_placement_puzzle_prompt import (
    generate_prompts,
    prompt_star_placement_puzzle,
)
from .....games.tasks.star_placement_puzzle.scripts.star_placement_puzzle_verifier import (
    StarPlacementPuzzleVerifier,
)


class StarPlacementPuzzle(Game):
    def __init__(self, n=4, k=1):
        super().__init__("Star Placement Puzzle", StarPlacementPuzzleVerifier)
        self.n = n
        self.k = k

    def generate(self, num_of_questions: int = 100, max_attempts: int = 1000):
        game_data_list = []
        total_attempts = 0
        max_total_attempts = max(max_attempts, num_of_questions * 10)

        n = self.n
        k = self.k

        while len(game_data_list) < num_of_questions and total_attempts < max_total_attempts:
            puzzle_data = self._generate_new_puzzle(n, k)
            total_attempts += 1

            if puzzle_data:
                region_grid, solution = puzzle_data

                question = prompt_star_placement_puzzle(n, k, region_grid)

                all_prompts = generate_prompts(n, k, region_grid)

                formatted_solution = self._format_solution(solution, region_grid)

                game_data = Data(
                    question=question,
                    answer=formatted_solution,
                    metadata={
                        "trace_id": str(uuid.uuid4()),
                        "n": n,
                        "k": k,
                        "region_nums": n,
                        "region_grid": region_grid,
                        "solution": None,
                        "all_prompts": all_prompts,
                    },
                )

                game_data_list.append(game_data)
            else:
                pass
        return game_data_list

    def extract_answer(self, test_solution: str):
        try:
            python_match = re.search(r"```python\s*\n(.*?)\n\s*```", test_solution, re.DOTALL)
            if not python_match:
                return None

            code_content = python_match.group(1)

            try:
                dict_match = re.search(r"\{[^{}]*\}", code_content, re.DOTALL)
                if dict_match:
                    dict_str = dict_match.group(0)
                    try:
                        coords_dict = ast.literal_eval(dict_str)

                        if isinstance(coords_dict, dict):
                            result = {}
                            for region, coords in coords_dict.items():
                                result[region] = [(row - 1, col - 1) for row, col in coords]
                            return result
                    except (ValueError, SyntaxError):
                        pass

                assign_match = re.search(r"(\w+)\s*=\s*(\{[^{}]*\})", code_content, re.DOTALL)
                if assign_match:
                    dict_str = assign_match.group(2)
                    try:
                        coords_dict = ast.literal_eval(dict_str)

                        if isinstance(coords_dict, dict):
                            result = {}
                            for region, coords in coords_dict.items():
                                result[region] = [(row - 1, col - 1) for row, col in coords]
                            return result
                    except (ValueError, SyntaxError):
                        pass
            except Exception:
                pass

            return None

        except Exception:
            return None

    def _generate_new_puzzle(self, n, k):
        solution = self._generate_valid_star_placement(n, k)
        if not solution:
            return None

        region_grid = self._create_regions_based_on_stars(n, k, solution)
        if not region_grid:
            return None

        return region_grid, solution

    def _generate_valid_star_placement(self, n, k):
        if k == 1:
            if n == 4:
                return {(0, 0), (1, 3), (2, 1), (3, 2)}

            stars = set()
            for i in range(n):
                stars.add((i, (i * 2) % n))

            if self._verify_star_placement(n, k, stars):
                return stars

            for offset in range(1, n):
                stars = set()
                for i in range(n):
                    stars.add((i, (i + offset) % n))

                if self._verify_star_placement(n, k, stars):
                    return stars

        elif k > 1:
            if n == 4 and k == 2:
                return {(0, 0), (0, 3), (1, 1), (1, 2), (2, 0), (2, 3), (3, 1), (3, 2)}

            for attempt in range(50):
                stars = set()

                for row in range(n):
                    row_stars = 0
                    possible_cols = list(range(n))
                    random.shuffle(possible_cols)

                    for col in possible_cols:
                        if row_stars >= k:
                            break

                        conflict = False
                        for sr, sc in stars:
                            if abs(sr - row) <= 1 and abs(sc - col) <= 1:
                                conflict = True
                                break
                            if sc == col:
                                if sum(1 for r, c in stars if c == col) >= k:
                                    conflict = True
                                    break

                        if not conflict:
                            stars.add((row, col))
                            row_stars += 1

                if self._verify_star_placement(n, k, stars):
                    return stars

        return None

    def _create_regions_based_on_stars(self, n, k, stars):
        region_grid = [["" for _ in range(n)] for _ in range(n)]

        region_ids = list(string.ascii_uppercase[:n])

        stars_list = list(stars)
        random.shuffle(stars_list)

        region_to_stars = {region_id: [] for region_id in region_ids}

        for i, star in enumerate(stars_list):
            region_idx = i // k
            if region_idx < len(region_ids):
                region_id = region_ids[region_idx]
                region_to_stars[region_id].append(star)
                region_grid[star[0]][star[1]] = region_id

        for region_id, region_stars in region_to_stars.items():
            frontier = list(region_stars)
            visited = set(region_stars)

            total_cells_needed = n * n // n
            current_cells = len(region_stars)

            while frontier and current_cells < total_cells_needed:
                r, c = frontier.pop(0)

                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc

                    if 0 <= nr < n and 0 <= nc < n:
                        if region_grid[nr][nc] == "" and (nr, nc) not in visited:
                            region_grid[nr][nc] = region_id
                            frontier.append((nr, nc))
                            visited.add((nr, nc))
                            current_cells += 1

                            if current_cells >= total_cells_needed:
                                break

        unassigned = []
        for r in range(n):
            for c in range(n):
                if region_grid[r][c] == "":
                    unassigned.append((r, c))

        for r, c in unassigned:
            min_dist = float("inf")
            nearest_region = None

            for region_id in region_ids:
                region_cells = sum(1 for i in range(n) for j in range(n) if region_grid[i][j] == region_id)

                if region_cells < n * n // n:
                    for i in range(n):
                        for j in range(n):
                            if region_grid[i][j] == region_id:
                                dist = abs(r - i) + abs(c - j)
                                if dist < min_dist:
                                    min_dist = dist
                                    nearest_region = region_id

            if nearest_region:
                region_grid[r][c] = nearest_region
            else:
                region_grid[r][c] = random.choice(region_ids)

        if self._verify_region_grid(n, k, region_grid, stars):
            return region_grid

        region_grid = [["" for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                region_grid[i][j] = region_ids[i % len(region_ids)]

        if self._verify_region_grid(n, k, region_grid, stars):
            return region_grid

        return None

    def _verify_star_placement(self, n, k, stars):
        if len(stars) != n * k:
            return False

        for i in range(n):
            row_stars = sum(1 for r, c in stars if r == i)
            if row_stars != k:
                return False

            col_stars = sum(1 for r, c in stars if c == i)
            if col_stars != k:
                return False

        for r1, c1 in stars:
            for r2, c2 in stars:
                if (r1, c1) != (r2, c2) and abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1:
                    return False

        return True

    def _verify_region_grid(self, n, k, region_grid, stars):
        regions = {}
        for r in range(n):
            for c in range(n):
                region = region_grid[r][c]
                if region not in regions:
                    regions[region] = []
                regions[region].append((r, c))

        for region, cells in regions.items():
            region_stars = sum(1 for r, c in cells if (r, c) in stars)
            if region_stars != k:
                return False

        for r in range(n):
            for c in range(n):
                if not region_grid[r][c]:
                    return False

        return True

    def _format_solution(self, solution, region_grid):
        regions_stars = {}
        for r, c in solution:
            region = region_grid[r][c]
            if region not in regions_stars:
                regions_stars[region] = []

            regions_stars[region].append((r + 1, c + 1))

        formatted_parts = []
        for region in sorted(regions_stars.keys()):
            coords = sorted(regions_stars[region])
            coords_str = "".join([f"({r},{c})" for r, c in coords])
            formatted_parts.append(f"{region}{coords_str}")

        return "[[" + "\n\n".join(formatted_parts) + "]]"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="星星放置游戏生成器")
    parser.add_argument("--num_of_data", type=int, default=10, help="生成的题目数量")
    parser.add_argument("--max_attempts", type=int, default=1000, help="每个题目的最大尝试次数")
    parser.add_argument("--n", type=int, default=4, help="网格大小")
    parser.add_argument("--k", type=int, default=1, help="每行/列/区域的星星数量")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    args = parser.parse_args()

    if args.output_dir:
        base_output_dir = pathlib.Path(args.output_dir)
    else:
        base_output_dir = pathlib.Path(__file__).parent.parent / "data"

    output_dir = base_output_dir / "star_placement_puzzle" / f"n_{args.n}" / f"k_{args.k}"

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"num_of_data_{args.num_of_data}.jsonl"

    game = StarPlacementPuzzle(n=args.n, k=args.k)

    game_data_list = game.generate(args.num_of_data, args.max_attempts)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for game_data in game_data_list:
                f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")
    except Exception:
        pass
