import re

import numpy as np
from .....base.data import Data
from .....base.verifier import Verifier


class CalcudokoVerifier(Verifier):
    def extract_answer(self, test_solution: str):
        if not test_solution:
            return ""
        code_block_pattern = "```python\\s*([\\s\\S]*?)\\s*```"
        matches = re.findall(code_block_pattern, test_solution)
        if matches:
            python_code = matches[-1].strip()
            return python_code
        tuple_pattern = "\\(\\s*\\(\\s*\\d+\\s*,.*?\\)\\s*\\)"
        matches = re.findall(tuple_pattern, test_solution, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return test_solution

    def parse_grid_from_answer(self, test_solution: str, data: Data) -> np.ndarray:
        try:
            grid_size = data.metadata.get("grid_size", 4)
            answer = self.extract_answer(test_solution)
            bracket_pattern = "\\[\\[(.*?)\\]\\]"
            bracket_match = re.search(bracket_pattern, test_solution)
            if bracket_match:
                answer = bracket_match.group(1)
            answer = "".join((c for c in answer if c.isdigit() or c in "[]," or c.isspace()))
            answer = answer.replace("，", ",")
            answer = answer.strip("[]")
            rows = [row.strip() for row in answer.split(",")]
            grid = []
            for row in rows:
                numbers = [int(n) for n in row.split() if n.strip()]
                if len(numbers) != grid_size:
                    if len(numbers) < grid_size:
                        row_patterns = [
                            "第\\s*\\d+\\s*行\\s*[:：]\\s*([\\d\\s]+)",
                            "行\\s*\\d+\\s*[:：]\\s*([\\d\\s]+)",
                            "Row\\s*\\d+\\s*[:：]\\s*([\\d\\s]+)",
                            "- 第\\d+行[:：]?\\s*([\\d\\s]+)",
                            "- 行\\d+[:：]?\\s*([\\d\\s]+)",
                            "- Row\\d+[:：]?\\s*([\\d\\s]+)",
                        ]
                        found_rows = []
                        for pattern in row_patterns:
                            matches = re.findall(pattern, test_solution)
                            if matches:
                                found_rows.extend(matches)
                        if len(found_rows) == grid_size:
                            grid = []
                            for found_row in found_rows:
                                numbers = [int(n) for n in found_row.split() if n.strip()]
                                if len(numbers) != grid_size:
                                    if len(numbers) < grid_size:
                                        numbers.extend([1] * (grid_size - len(numbers)))
                                    else:
                                        numbers = numbers[:grid_size]
                                grid.append(numbers)
                            break
                    if len(numbers) < grid_size:
                        numbers.extend([1] * (grid_size - len(numbers)))
                    else:
                        numbers = numbers[:grid_size]
                grid.append(numbers)
            if len(grid) != grid_size:
                if len(grid) < grid_size:
                    for _ in range(grid_size - len(grid)):
                        grid.append([1] * grid_size)
                else:
                    grid = grid[:grid_size]
            return np.array(grid)
        except Exception:
            try:
                bracket_pattern = "\\[\\[(.*?)\\]\\]"
                bracket_match = re.search(bracket_pattern, test_solution)
                if bracket_match:
                    answer_str = bracket_match.group(1)
                    rows = answer_str.split(",")
                    grid = []
                    for row in rows:
                        numbers = [int(n) for n in row.split() if n.strip()]
                        if len(numbers) != grid_size:
                            if len(numbers) < grid_size:
                                numbers.extend([1] * (grid_size - len(numbers)))
                            else:
                                numbers = numbers[:grid_size]
                        grid.append(numbers)
                    if len(grid) != grid_size:
                        if len(grid) < grid_size:
                            for _ in range(grid_size - len(grid)):
                                grid.append([1] * grid_size)
                        else:
                            grid = grid[:grid_size]
                    return np.array(grid)
            except Exception:
                pass
            return np.array([[i % grid_size + 1 for i in range(grid_size)] for _ in range(grid_size)])

    def verify(self, data: Data, test_answer: str) -> bool:
        try:
            grid_size = data.metadata.get("grid_size", 4)
            regions = data.metadata.get("regions", [])
            try:
                grid = self.parse_grid_from_answer(test_answer, data)
            except Exception:
                return False
            for i, row in enumerate(grid):
                if len(set(row)) != grid_size:
                    return False
                if not all((1 <= n <= grid_size for n in row)):
                    return False
            for i, col in enumerate(grid.T):
                if len(set(col)) != grid_size:
                    return False
                if not all((1 <= n <= grid_size for n in col)):
                    return False
            for i, region in enumerate(regions):
                cells = region["cells"]
                numbers = []
                for cell in cells:
                    try:
                        if isinstance(cell, str):
                            match = re.search("\\((\\d+)\\s*,\\s*(\\d+)\\)", cell)
                            if match:
                                r, c = (int(match.group(1)), int(match.group(2)))
                        r, c = (r - 1, c - 1)
                        if r < 0 or r >= grid_size or c < 0 or (c >= grid_size):
                            return False
                        numbers.append(grid[r][c])
                    except Exception:
                        return False
                if not all((1 <= n <= grid_size for n in numbers)):
                    return False
                if len(set(numbers)) != len(numbers):
                    return False
                target = region["target"]
                operator = region["operator"]
                try:
                    if operator == "+":
                        result = sum(numbers)
                        valid = result == target
                        if not valid:
                            return False
                    elif operator == "-":
                        if len(numbers) != 2:
                            return False
                        result1 = numbers[0] - numbers[1]
                        result2 = numbers[1] - numbers[0]
                        valid = result1 == target or result2 == target
                        if not valid:
                            return False
                    elif operator == "*":
                        result = np.prod(numbers)
                        valid = result == target
                        if not valid:
                            return False
                    else:
                        if len(numbers) != 2:
                            return False
                        if numbers[0] == 0 or numbers[1] == 0:
                            return False
                        result1 = numbers[0] / numbers[1]
                        result2 = numbers[1] / numbers[0]
                        valid = abs(result1 - target) < 0.001 or abs(result2 - target) < 0.001
                        if not valid:
                            return False
                except Exception:
                    return False
            return True
        except Exception:
            return False


def main():
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description="Verify Calcudoko solutions")
    parser.add_argument("--grid_size", type=int, default=4, help="Size of the grid (N)")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output_file", type=str, help="Path to save valid samples (optional)")
    args = parser.parse_args()
    verifier = CalcudokoVerifier()
    total_puzzles = 0
    valid_answers = 0
    invalid_answers = 0
    valid_samples = []

    def parse_regions_from_text(text):
        regions = []
        regions_text = re.search("Regions:\\n(.*?)(?:\\n\\n|$)", text, re.DOTALL)
        if regions_text:
            for line in regions_text.group(1).strip().split("\n"):
                if line.strip():
                    parts = line.split(":")
                    if len(parts) < 2:
                        continue
                    cells_part, target = parts[0], parts[1]
                    if not target or len(target) == 0:
                        continue
                    cells = re.findall("\\((\\d+),(\\d+)\\)", cells_part)
                    cells = [tuple(map(int, pair)) for pair in cells]
                    target_value, operator = (target[:-1], target[-1])
                    regions.append({"cells": cells, "target": int(target_value), "operator": operator})
        return regions

    try:
        with open(args.input_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    total_puzzles += 1
                    regions = None
                    if "regions" in data:
                        regions = data["regions"]
                    elif "regions" in data["metadata"]:
                        regions = data["metadata"]["regions"]
                    elif "question_data" in data:
                        question_data = json.loads(data["question_data"])
                        if "regions" in question_data:
                            regions = question_data["regions"]
                    elif "question" in data:
                        regions = parse_regions_from_text(data["question"])
                    if not regions:
                        continue
                    answer = data.get("answer", "")
                    if not answer:
                        continue
                    data_obj = Data(
                        question=data["question"],
                        answer=answer,
                        metadata={"grid_size": args.grid_size, "regions": regions},
                    )
                    is_valid = verifier.verify(data_obj, answer)
                    if is_valid:
                        valid_answers += 1
                        valid_samples.append(data)
                    else:
                        invalid_answers += 1
                except Exception:
                    invalid_answers += 1
    except Exception:
        return
    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir and (not os.path.exists(output_dir)):
            os.makedirs(output_dir)
        with open(args.output_file, "w", encoding="utf-8") as f:
            for sample in valid_samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write("\n")


if __name__ == "__main__":
    main()
