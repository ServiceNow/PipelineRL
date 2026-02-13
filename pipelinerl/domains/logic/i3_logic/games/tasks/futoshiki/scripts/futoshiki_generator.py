import argparse
import random
from typing import Dict, List, Tuple

import numpy as np
from .....games.tasks.futoshiki.scripts.futoshiki_prompt import get_prompt


class FutoshikiGenerator:
    def generate_valid_grid(self, grid_size: int) -> np.ndarray:
        grid = np.zeros((grid_size, grid_size), dtype=int)
        first_row = list(range(1, grid_size + 1))
        random.shuffle(first_row)
        grid[0] = first_row
        for row in range(1, grid_size):
            available = list(range(1, grid_size + 1))
            for col in range(grid_size):
                used_in_col = set(grid[:row, col])
                valid_numbers = [n for n in available if n not in used_in_col]
                if not valid_numbers:
                    return self.generate_valid_grid(grid_size)
                num = random.choice(valid_numbers)
                grid[row, col] = num
                available.remove(num)
        return grid

    def generate_inequality_constraints(
        self, grid_size: int, num_constraints: int
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int], str]]:
        constraints = []
        possible_pairs = []
        for i in range(grid_size):
            for j in range(grid_size):
                if j < grid_size - 1:
                    possible_pairs.append(((i, j), (i, j + 1)))
                if i < grid_size - 1:
                    possible_pairs.append(((i, j), (i + 1, j)))
        selected_pairs = random.sample(possible_pairs, min(num_constraints, len(possible_pairs)))
        for pair in selected_pairs:
            sign = random.choice([">", "<"])
            constraints.append((pair[0], pair[1], sign))
        return constraints

    def select_prefilled_coordinates(self, grid_size: int, num_prefilled: int) -> List[Tuple[int, int]]:
        all_coords = [(i, j) for i in range(grid_size) for j in range(grid_size)]
        return random.sample(all_coords, min(num_prefilled, len(all_coords)))

    def generate_prompt(
        self,
        grid: np.ndarray,
        prefilled_coords: List[Tuple[int, int]],
        constraints: List[Tuple[Tuple[int, int], Tuple[int, int], str]],
        grid_size: int,
        is_chinese: bool = False,
    ) -> str:
        display_grid = np.full((grid_size, grid_size), "X")
        for row, col in prefilled_coords:
            display_grid[row, col] = str(grid[row, col])
        formatted_grid = ""
        for row in display_grid:
            formatted_row = " ".join(row)
            formatted_grid += formatted_row + "\n"
        formatted_grid = formatted_grid.rstrip()
        formatted_constraints = ""
        for coord1, coord2, sign in constraints:
            row1, col1 = coord1
            row2, col2 = coord2
            formatted_constraints += f"({row1 + 1},{col1 + 1}) {sign} ({row2 + 1},{col2 + 1})\n"
        formatted_constraints = formatted_constraints.rstrip()
        return get_prompt(formatted_grid, formatted_constraints, grid_size, is_chinese)

    def generate_sample(
        self, grid_size: int, num_inequality_signs: int, num_prefilled_coords: int, is_chinese: bool = False
    ) -> Dict:
        grid = self.generate_valid_grid(grid_size)
        constraints = self.generate_inequality_constraints(grid_size, num_inequality_signs)
        prefilled_coords = self.select_prefilled_coordinates(grid_size, num_prefilled_coords)
        prompt = self.generate_prompt(grid, prefilled_coords, constraints, grid_size, is_chinese)
        answer = [[int(x) for x in row] for row in grid.tolist()]
        return {
            "question": prompt,
            "answer": answer,
            "metadata": {
                "grid_size": grid_size,
                "num_inequality_signs": num_inequality_signs,
                "num_prefilled_coords": num_prefilled_coords,
                "prefilled_coords": prefilled_coords,
                "constraints": constraints,
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Generate Futoshiki puzzle samples")
    parser.add_argument("--grid_size", type=int, default=4, help="Size of the grid (default: 4)")
    parser.add_argument("--num_inequality_signs", type=int, default=4, help="Number of inequality signs (default: 4)")
    parser.add_argument(
        "--num_prefilled_coords", type=int, default=2, help="Number of pre-filled coordinates (default: 2)"
    )
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate (default: 10)")
    parser.add_argument("--is_chinese", action="store_true", help="Generate Chinese prompts (default: False)")
    parser.add_argument("--output", type=str, help="Specify the output file name (optional)")
    args = parser.parse_args()
    generator = FutoshikiGenerator()
    samples = []
    for i in range(args.num_samples):
        sample = generator.generate_sample(
            grid_size=args.grid_size,
            num_inequality_signs=args.num_inequality_signs,
            num_prefilled_coords=args.num_prefilled_coords,
            is_chinese=args.is_chinese,
        )
        samples.append(sample)
    import json
    import os
    from datetime import datetime

    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    if args.output:
        output_file = os.path.join(output_dir, args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"futoshiki_samples_{timestamp}.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            json_line = json.dumps(sample, ensure_ascii=False)
            f.write(json_line + "\n")


if __name__ == "__main__":
    main()
