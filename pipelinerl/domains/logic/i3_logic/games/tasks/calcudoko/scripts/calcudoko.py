import json
import os
import random
import re
from typing import Dict, List, Tuple

import numpy as np
from .....games.tasks.calcudoko.scripts.calcudoko_prompt import prompt_calcudoko
from tqdm import tqdm


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class CalcudokoGenerator:
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.operators = ["+", "-", "*", "÷"]
        self.grid = None
        self.regions = []

    @staticmethod
    def extract_answer(response: str) -> List[List[int]]:
        pattern = "\\[\\[(.*?)\\]\\]"
        match = re.search(pattern, response)
        if not match:
            raise ValueError("No answer found in the response")
        answer_str = match.group(1)
        rows = answer_str.split(",")
        grid = []
        for row in rows:
            numbers = [int(n) for n in row.strip().split()]
            grid.append(numbers)
        return grid

    def generate_sudoku_grid(self) -> np.ndarray:
        def is_valid(grid: np.ndarray, row: int, col: int, num: int) -> bool:
            if num in grid[row]:
                return False
            if num in grid[:, col]:
                return False
            return True

        def solve(grid: np.ndarray) -> bool:
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    if grid[row][col] == 0:
                        for num in range(1, self.grid_size + 1):
                            if is_valid(grid, row, col, num):
                                grid[row][col] = num
                                if solve(grid):
                                    return True
                                grid[row][col] = 0
                        return False
            return True

        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        grid[0] = np.random.permutation(range(1, self.grid_size + 1))
        solve(grid)
        return grid

    def create_regions(self) -> List[Dict]:
        coords = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        random.shuffle(coords)
        regions = []
        used_coords = set()
        max_attempts = 1000
        while len(used_coords) < len(coords):
            attempts = 0
            while attempts < max_attempts:
                available_coords = [c for c in coords if c not in used_coords]
                if not available_coords:
                    break
                start_coord = random.choice(available_coords)
                max_size = min(4, len(available_coords))
                if max_size < 2:
                    if regions:
                        last_region = regions[-1]
                        for coord in available_coords:
                            last_region["cells"].append(f"({coord[0] + 1},{coord[1] + 1})")
                            used_coords.add(coord)
                    break
                region_size = random.randint(2, max_size)
                region_coords = [start_coord]
                current_coord = start_coord
                for _ in range(region_size - 1):
                    neighbors = []
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = (current_coord[0] + dx, current_coord[1] + dy)
                        if (
                            0 <= nx < self.grid_size
                            and 0 <= ny < self.grid_size
                            and ((nx, ny) in available_coords)
                            and ((nx, ny) not in region_coords)
                        ):
                            neighbors.append((nx, ny))
                    if not neighbors:
                        break
                    next_coord = random.choice(neighbors)
                    region_coords.append(next_coord)
                    current_coord = next_coord
                if len(region_coords) < 2:
                    attempts += 1
                    continue
                numbers = [self.grid[r][c] for r, c in region_coords]
                if len(set(numbers)) != len(numbers):
                    attempts += 1
                    continue
                if len(region_coords) == 2:
                    operator = random.choice(["-", "÷"])
                else:
                    operator = random.choice(["+", "*"])
                if operator == "+":
                    target = sum(numbers)
                elif operator == "-":
                    target = abs(numbers[0] - numbers[1])
                elif operator == "*":
                    target = np.prod(numbers)
                else:
                    target = max(numbers[0] // numbers[1], numbers[1] // numbers[0])
                regions.append(
                    {"cells": [f"({r + 1},{c + 1})" for r, c in region_coords], "target": target, "operator": operator}
                )
                for coord in region_coords:
                    used_coords.add(coord)
                break
            if attempts >= max_attempts:
                remaining_coords = [c for c in coords if c not in used_coords]
                if remaining_coords and len(remaining_coords) >= 2:
                    while len(remaining_coords) >= 2:
                        pair = remaining_coords[:2]
                        numbers = [self.grid[r][c] for r, c in pair]
                        operator = random.choice(["-", "÷"])
                        if operator == "-":
                            target = abs(numbers[0] - numbers[1])
                        else:
                            target = max(numbers[0] // numbers[1], numbers[1] // numbers[0])
                        regions.append(
                            {"cells": [f"({r + 1},{c + 1})" for r, c in pair], "target": target, "operator": operator}
                        )
                        for coord in pair:
                            used_coords.add(coord)
                        remaining_coords = remaining_coords[2:]
                elif remaining_coords and regions:
                    last_region = regions[-1]
                    for coord in remaining_coords:
                        last_region["cells"].append(f"({coord[0] + 1},{coord[1] + 1})")
                        used_coords.add(coord)
        if len(used_coords) == len(coords) and regions:
            return regions
        return None

    def generate(self) -> Tuple[str, np.ndarray]:
        self.grid = self.generate_sudoku_grid()
        while True:
            self.regions = self.create_regions()
            if self.regions:
                break
        question = prompt_calcudoko(self.grid_size, self.regions)
        answer = "[[" + ",".join([" ".join(map(str, row)) for row in self.grid]) + "]]"
        return (question, answer, self.regions)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate Calcudoko puzzles")
    parser.add_argument("--num_of_data", type=int, default=100, help="Number of puzzles to generate")
    parser.add_argument("--grid_size", type=int, default=4, help="Size of the grid (N)")
    args = parser.parse_args()
    calcudoko_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(calcudoko_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    output_dir = os.path.join(data_dir, f"grid_size_{args.grid_size}")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"puzzles_grid{args.grid_size}_n{args.num_of_data}.jsonl"
    output_path = os.path.join(output_dir, output_filename)
    generator = CalcudokoGenerator(args.grid_size)
    with open(output_path, "w") as f:
        for i in tqdm(range(args.num_of_data), desc="生成进度"):
            question, answer, regions = generator.generate()
            data = {
                "id": i + 1,
                "question": question.replace(chr(10), "\\n"),
                "answer": answer,
                "metadata": {"grid_size": args.grid_size, "regions": regions},
            }
            json_str = json.dumps(data, ensure_ascii=False, cls=NumpyEncoder)
            f.write(json_str + "\n")


if __name__ == "__main__":
    main()
