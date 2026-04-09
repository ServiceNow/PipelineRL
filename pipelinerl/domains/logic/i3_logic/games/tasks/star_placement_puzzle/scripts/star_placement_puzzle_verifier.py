import ast
import re

from .....base.data import Data
from .....base.verifier import Verifier


class StarPlacementPuzzleVerifier(Verifier):
    def verify(self, data: Data, test_solution: str):
        try:
            star_coords = self.extract_answer(test_solution)

            metadata = data.metadata
            n = metadata["n"]
            k = metadata["k"]
            region_grid = metadata["region_grid"]

            if not star_coords:
                return False

            star_grid = [[0 for _ in range(n)] for _ in range(n)]
            for region, coords in star_coords.items():
                for coord in coords:
                    row, col = coord
                    if row < 0 or row >= n or col < 0 or col >= n:
                        return False
                    star_grid[row][col] = 1

            for i in range(n):
                stars_in_row = sum(star_grid[i])
                if stars_in_row != k:
                    return False

            for j in range(n):
                stars_in_col = sum(star_grid[i][j] for i in range(n))
                if stars_in_col != k:
                    return False

            regions = {}
            for i in range(n):
                for j in range(n):
                    region = region_grid[i][j]
                    if region not in regions:
                        regions[region] = []
                    regions[region].append((i, j))

            for region, cells in regions.items():
                stars_in_region = sum(star_grid[i][j] for i, j in cells)
                if stars_in_region != k:
                    return False

            for i in range(n):
                for j in range(n):
                    if star_grid[i][j] == 1:
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                ni, nj = i + di, j + dj
                                if 0 <= ni < n and 0 <= nj < n and star_grid[ni][nj] == 1:
                                    return False

            return True

        except Exception:
            return False

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
