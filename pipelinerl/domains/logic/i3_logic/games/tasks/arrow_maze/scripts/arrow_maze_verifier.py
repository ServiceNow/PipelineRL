import json
import re
from typing import List

from .....base.data import Data
from .....base.verifier import Verifier


class ArrowMazeVerifier(Verifier):
    VALID_ARROWS = {"↑", "↓", "←", "→", "↖", "↗", "↘", "↙"}

    ARROWS_DIRECTIONS = {
        "↑": (-1, 0),
        "↓": (1, 0),
        "←": (0, -1),
        "→": (0, 1),
        "↖": (-1, -1),
        "↗": (-1, 1),
        "↘": (1, 1),
        "↙": (1, -1),
    }

    def verify(self, data: Data, test_solution_str: str) -> bool:
        test_answer_str = self.extract_answer(test_solution_str)
        if not test_answer_str:
            return False

        try:
            test_answer = json.loads(test_answer_str)

            question_grid = data.metadata["maze"]

            if not self._verify_grid_size(test_answer, question_grid):
                return False

            if not self._verify_number_positions(test_answer, question_grid):
                return False

            if not self._verify_all_blanks_filled(test_answer, question_grid):
                return False

            if not self._verify_arrow_symbols(test_answer):
                return False

            if not self._verify_prefilled_arrows(test_answer, question_grid):
                return False

            if not self._verify_arrow_rays(test_answer):
                return False

            if not self._verify_number_rays(test_answer):
                return False

            return True

        except Exception:
            return False

    def _verify_grid_size(self, test_answer: List[List[str]], question_grid: List[List[str]]) -> bool:
        if len(test_answer) != len(question_grid):
            return False

        for i in range(len(test_answer)):
            if len(test_answer[i]) != len(question_grid[i]):
                return False

        return True

    def _verify_number_positions(self, test_answer: List[List[str]], question_grid: List[List[str]]) -> bool:
        for i in range(len(question_grid)):
            for j in range(len(question_grid[i])):
                if question_grid[i][j].isdigit():
                    if test_answer[i][j] != question_grid[i][j]:
                        return False
        return True

    def _verify_all_blanks_filled(self, test_answer: List[List[str]], question_grid: List[List[str]]) -> bool:
        for i in range(len(question_grid)):
            for j in range(len(question_grid[i])):
                if question_grid[i][j] == "X" and test_answer[i][j] == "X":
                    return False
        return True

    def _verify_arrow_symbols(self, test_answer: List[List[str]]) -> bool:
        for i in range(len(test_answer)):
            for j in range(len(test_answer[i])):
                cell = test_answer[i][j]
                if not cell.isdigit() and cell != "X" and cell not in self.VALID_ARROWS:
                    return False
        return True

    def _verify_prefilled_arrows(self, test_answer: List[List[str]], question_grid: List[List[str]]) -> bool:
        for i in range(len(question_grid)):
            for j in range(len(question_grid[i])):
                cell = question_grid[i][j]
                if not cell.isdigit() and cell != "X":
                    if test_answer[i][j] != cell:
                        return False
        return True

    def _verify_arrow_rays(self, test_answer: List[List[str]]) -> bool:
        n = len(test_answer)
        m = len(test_answer[0]) if n > 0 else 0

        covered = [[False for _ in range(m)] for _ in range(n)]

        for i in range(n):
            for j in range(m):
                if test_answer[i][j].isdigit():
                    covered[i][j] = True

        for i in range(n):
            for j in range(m):
                if test_answer[i][j].isdigit():
                    for arrow_symbol, (di, dj) in self.ARROWS_DIRECTIONS.items():
                        ni, nj = i + di, j + dj

                        while 0 <= ni < n and 0 <= nj < m and test_answer[ni][nj] == arrow_symbol:
                            covered[ni][nj] = True
                            ni += di
                            nj += dj

        for i in range(n):
            for j in range(m):
                if test_answer[i][j] in self.VALID_ARROWS and not covered[i][j]:
                    return False

        return True

    def _verify_number_rays(self, test_answer: List[List[str]]) -> bool:
        n = len(test_answer)
        m = len(test_answer[0]) if n > 0 else 0

        for i in range(n):
            for j in range(m):
                if test_answer[i][j].isdigit():
                    number = int(test_answer[i][j])
                    arrow_count = self._count_arrow_rays(test_answer, i, j)
                    if arrow_count != number:
                        return False

        return True

    def _count_arrow_rays(self, grid: List[List[str]], i: int, j: int) -> int:
        n = len(grid)
        m = len(grid[0]) if n > 0 else 0
        count = 0

        for arrow_symbol, (di, dj) in self.ARROWS_DIRECTIONS.items():
            ni, nj = i + di, j + dj
            ray_length = 0

            while 0 <= ni < n and 0 <= nj < m and grid[ni][nj] == arrow_symbol:
                ray_length += 1
                ni += di
                nj += dj

            count += ray_length

        return count

    def extract_answer(self, test_solution: str) -> str:
        if not test_solution:
            return ""

        code_block_patterns = [
            r"```python\s*\n(.*?\[.*?\].*?)\n```",
            r"```\s*\n(.*?\[.*?\].*?)\n```",
            r"```(.*?\[.*?\].*?)```",
        ]

        for pattern in code_block_patterns:
            matches = re.findall(pattern, test_solution, re.DOTALL)
            if matches:
                code_block = matches[-1].strip()
                try:
                    grid = eval(code_block)

                    if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
                        return json.dumps(grid)
                except Exception:
                    continue

        list_pattern = r"\[\s*\[.*?\]\s*\]"
        matches = re.findall(list_pattern, test_solution, re.DOTALL)
        if matches:
            try:
                grid = eval(matches[-1])

                if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
                    return json.dumps(grid)
            except Exception:
                pass

        return ""
