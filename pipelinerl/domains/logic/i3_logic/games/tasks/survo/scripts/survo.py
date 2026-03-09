import argparse
import json
import pathlib
import random
import re
import uuid
from typing import List, Tuple

import numpy as np
from .....base.data import Data
from .....games.base.game import Game
from .....games.tasks.survo.scripts.survo_prompt import prompt_survo
from .....games.tasks.survo.scripts.survo_verifier import SurvoVerifier


class Survo(Game):
    def __init__(self):
        super().__init__("Survo", SurvoVerifier)

    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        n: int = 4,
        x: int = 3,
        min_num: int = 1,
        max_num: int = 9,
    ):
        if n <= 3:
            raise ValueError("矩阵维度n必须大于3")
        if x <= 0 or x > (n - 1) * (n - 1):
            raise ValueError(f"待填充数字数量x必须在1到{(n - 1) * (n - 1)}之间")
        if min_num >= max_num:
            raise ValueError("最小值必须小于最大值")

        game_data_list = []
        generated_matrices = set()
        attempts = 0

        while len(game_data_list) < num_of_questions and attempts < max_attempts:
            try:
                attempts += 1

                matrix, original_matrix, candidate_numbers = self._generate_valid_matrix(n, x, min_num, max_num)

                matrix_str = str(original_matrix.tolist())
                if matrix_str in generated_matrices:
                    continue

                generated_matrices.add(matrix_str)

                is_chinese = random.choice([True, False])

                question = prompt_survo(original_matrix, candidate_numbers, n, is_chinese)

                game_data = Data(
                    question=question,
                    answer=str(matrix.tolist()),
                    metadata={
                        "trace_id": str(uuid.uuid4()),
                        "original_matrix": original_matrix.tolist(),
                        "filled_matrix": matrix.tolist(),
                        "candidate_numbers": candidate_numbers,
                        "n": n,
                        "x": x,
                        "min_num": min_num,
                        "max_num": max_num,
                    },
                )

                game_data_list.append(game_data)

            except Exception:
                continue

        return game_data_list

    def _generate_valid_matrix(
        self, n: int, x: int, min_num: int, max_num: int
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        matrix = np.zeros((n, n), dtype=int)

        for i in range(n - 1):
            for j in range(n - 1):
                matrix[i, j] = random.randint(min_num, max_num)

        for i in range(n - 1):
            row_sum = sum(matrix[i, 0 : n - 1])
            matrix[i, n - 1] = row_sum

            col_sum = sum(matrix[0 : n - 1, i])
            matrix[n - 1, i] = col_sum

        matrix[n - 1, n - 1] = sum(matrix[0 : n - 1, n - 1])

        filled_matrix = matrix.copy()

        positions = [(i, j) for i in range(n - 1) for j in range(n - 1)]
        selected_positions = random.sample(positions, x)

        candidate_numbers = []
        for i, j in selected_positions:
            if not isinstance(matrix[i, j], (int, np.integer)):
                raise TypeError(f"矩阵位置 ({i}, {j}) 的值 {matrix[i, j]} 不是整数类型")
            candidate_numbers.append(int(matrix[i, j]))
            matrix[i, j] = 0

        return filled_matrix, matrix, candidate_numbers

    def extract_answer(self, test_solution: str):
        if not test_solution:
            return ""

        code_block_pattern = r"```python\s*([\s\S]*?)\s*```"
        code_matches = re.findall(code_block_pattern, test_solution)

        if code_matches:
            matrix_str = code_matches[-1].strip()
            return matrix_str

        general_code_block = r"```([\s\S]*?)```"
        general_matches = re.findall(general_code_block, test_solution)

        if general_matches:
            matrix_str = general_matches[-1].strip()
            return matrix_str

        matrix_pattern = r"\[\s*\[.*?\]\s*\]"
        matrix_matches = re.findall(matrix_pattern, test_solution, re.DOTALL)

        if matrix_matches:
            return matrix_matches[-1].strip()

        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Survo矩阵填充游戏生成器")
    parser.add_argument("--num_of_data", type=int, default=100, help="生成的题目数量")
    parser.add_argument("--max_attempts", type=int, default=1000, help="每个题目的最大尝试次数")
    parser.add_argument("--n", type=int, default=4, help="矩阵的维度")
    parser.add_argument("--x", type=int, default=3, help="待填充数字的数量")
    parser.add_argument("--min_num", type=int, default=1, help="矩阵中数字的最小值")
    parser.add_argument("--max_num", type=int, default=9, help="矩阵中数字的最大值")
    args = parser.parse_args()

    base_data_dir = pathlib.Path(__file__).parent.parent / "data"
    if not base_data_dir.exists():
        base_data_dir.mkdir(parents=True, exist_ok=True)

    game = Survo()

    game_data_list = game.generate(
        num_of_questions=args.num_of_data,
        max_attempts=args.max_attempts,
        n=args.n,
        x=args.x,
        min_num=args.min_num,
        max_num=args.max_num,
    )

    nested_dir = (
        base_data_dir
        / f"num_of_data_{args.num_of_data}"
        / f"n_{args.n}"
        / f"x_{args.x}"
        / f"min_num_{args.min_num}_max_num_{args.max_num}"
    )
    if not nested_dir.exists():
        nested_dir.mkdir(parents=True, exist_ok=True)

    output_file = nested_dir / f"survo_{args.num_of_data}.jsonl"

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for game_data in game_data_list:
                f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")
    except Exception:
        pass
