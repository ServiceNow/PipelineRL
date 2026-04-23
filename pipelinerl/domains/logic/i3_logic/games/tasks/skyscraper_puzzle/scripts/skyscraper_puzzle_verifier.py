import ast
import re

from .....base.data import Data
from .....base.verifier import Verifier


class SkyscraperPuzzleVerifier(Verifier):
    def verify(self, data: Data, test_solution: str):
        try:
            metadata = data.metadata
            n = metadata["n"]
            top = metadata["top"]
            bottom = metadata["bottom"]
            left = metadata["left"]
            right = metadata["right"]

            self.n = n
            test_answer = self.extract_answer(test_solution)

            grid = test_answer

            if isinstance(grid, str):
                return False

            if len(grid) != n or any(len(row) != n for row in grid):
                return False

            for i in range(n):
                for j in range(n):
                    if not isinstance(grid[i][j], int) or grid[i][j] < 1 or grid[i][j] > n:
                        return False

            for i in range(n):
                if len(set(grid[i])) != n:
                    return False

            for j in range(n):
                column = [grid[i][j] for i in range(n)]
                if len(set(column)) != n:
                    return False

            for j in range(n):
                visible_count = self._count_visible_skyscrapers([grid[i][j] for i in range(n)])
                if visible_count != top[j]:
                    return False

            for j in range(n):
                visible_count = self._count_visible_skyscrapers([grid[i][j] for i in range(n - 1, -1, -1)])
                if visible_count != bottom[j]:
                    return False

            for i in range(n):
                visible_count = self._count_visible_skyscrapers(grid[i])
                if visible_count != left[i]:
                    return False

            for i in range(n):
                visible_count = self._count_visible_skyscrapers(grid[i][::-1])
                if visible_count != right[i]:
                    return False

            return True

        except Exception:
            return False

    def _count_visible_skyscrapers(self, heights):
        visible_count = 0
        max_height = 0

        for height in heights:
            if height > max_height:
                visible_count += 1
                max_height = height

        return visible_count

    def extract_answer(self, test_solution: str):
        try:
            n = self.n

            code_block_pattern = r"```python\s*\n([\s\S]*?)\n\s*```"
            code_blocks = re.findall(code_block_pattern, test_solution)

            if code_blocks:
                code_block = code_blocks[0].strip()
                try:
                    grid = ast.literal_eval(code_block)

                    if (
                        isinstance(grid, list)
                        and len(grid) == n
                        and all(isinstance(row, list) and len(row) == n for row in grid)
                    ):
                        return grid
                except Exception:
                    code_without_comments = re.sub(r"#.*$", "", code_block, flags=re.MULTILINE)
                    try:
                        grid = ast.literal_eval(code_without_comments.strip())
                        if (
                            isinstance(grid, list)
                            and len(grid) == n
                            and all(isinstance(row, list) and len(row) == n for row in grid)
                        ):
                            return grid
                    except Exception:
                        pass

            return test_solution
        except Exception:
            return test_solution
