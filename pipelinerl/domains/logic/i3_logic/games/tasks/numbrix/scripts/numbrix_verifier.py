from .....base.data import Data
from .....base.verifier import Verifier


class NumbrixVerifier(Verifier):
    def verify(self, data: Data, test_solution: str):
        try:
            test_grid = self.extract_answer(test_solution)
            if not test_grid:
                return False
            original_grid = data.metadata["grid"]
            n = len(original_grid)
            n_squared = n * n
            if len(test_grid) != n or any((len(row) != n for row in test_grid)):
                return False
            flattened_grid = [cell for row in test_grid for cell in row]
            if sorted(flattened_grid) != list(range(1, n_squared + 1)):
                return False
            for i in range(n):
                for j in range(n):
                    if original_grid[i][j] != "X" and test_grid[i][j] != original_grid[i][j]:
                        return False
            for num in range(1, n_squared):
                current_pos = None
                next_pos = None
                for i in range(n):
                    for j in range(n):
                        if test_grid[i][j] == num:
                            current_pos = (i, j)
                        elif test_grid[i][j] == num + 1:
                            next_pos = (i, j)
                if current_pos is None or next_pos is None:
                    return False
                i1, j1 = current_pos
                i2, j2 = next_pos
                manhattan_distance = abs(i1 - i2) + abs(j1 - j2)
                if manhattan_distance != 1:
                    return False
            return True
        except Exception:
            return False

    def extract_answer(self, test_solution: str, strict=False):
        try:
            import ast
            import re

            pattern = "\\[\\s*\\[\\s*\\d+.*?\\]\\s*\\]"
            matches = re.finditer(pattern, test_solution, re.DOTALL)
            match = None
            for m in matches:
                match = m
            if not match:
                return None
            grid_text = match.group(0)
            grid_text = grid_text.replace("'", "").replace('"', "")
            grid = ast.literal_eval(grid_text)
            if not isinstance(grid, list) or not all((isinstance(row, list) for row in grid)):
                return None
            if not all((isinstance(cell, int) for row in grid for cell in row)):
                return None
            return grid
        except Exception:
            return None
