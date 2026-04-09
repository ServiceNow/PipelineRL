import json
import re

from .....base.data import Data
from .....base.verifier import Verifier


class KukurasuVerifier(Verifier):
    def verify(self, data: Data, test_solution: str, **kwargs):
        try:
            grid = self.extract_answer(test_solution)
            row_sums = data.metadata["row_sums"]
            col_sums = data.metadata["col_sums"]
            n = data.metadata["n"]
            m = data.metadata["m"]
            if len(grid) != n:
                return False
            for row in grid:
                if len(row) != m:
                    return False
                for cell in row:
                    if cell not in ["1", "X"]:
                        return False
            calculated_row_sums = []
            for i, row in enumerate(grid):
                row_sum = 0
                for j, cell in enumerate(row):
                    if cell == "1":
                        row_sum += j + 1
                calculated_row_sums.append(row_sum)
            calculated_col_sums = []
            for j in range(m):
                col_sum = 0
                for i in range(n):
                    if grid[i][j] == "1":
                        col_sum += i + 1
                calculated_col_sums.append(col_sum)
            if calculated_row_sums != row_sums:
                return False
            if calculated_col_sums != col_sums:
                return False
            return True
        except Exception:
            return False

    def extract_answer(self, response: str):
        grid_pattern = '\\[\\s*\\[(?:\\s*"[X1]"\\s*,\\s*)*\\s*"[X1]"\\s*\\]\\s*(?:,\\s*\\[(?:\\s*"[X1]"\\s*,\\s*)*\\s*"[X1]"\\s*\\]\\s*)*\\]'
        match = re.search(grid_pattern, response)
        if match:
            try:
                grid_str = match.group(0)
                return json.loads(grid_str)
            except json.JSONDecodeError:
                pass
        return ""
