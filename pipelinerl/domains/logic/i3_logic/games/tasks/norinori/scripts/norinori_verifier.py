import re
from collections import defaultdict

from .....base.data import Data
from .....base.verifier import Verifier


class NorinoriVerifier(Verifier):
    def __init__(self):
        super().__init__()

    def verify(self, data: Data, test_solution: str):
        try:
            region_grid = data.metadata["region_grid"]
            n = len(region_grid)
            dominoes = self._parse_answer(test_solution)
            if dominoes is None:
                return False
            if not self._check_domino_shapes(dominoes):
                return False
            covered = [[False for _ in range(n)] for _ in range(n)]
            for domino in dominoes:
                for i, j in domino:
                    i -= 1
                    j -= 1
                    if i < 0 or i >= n or j < 0 or (j >= n):
                        return False
                    if covered[i][j]:
                        return False
                    covered[i][j] = True
            if not self._check_domino_adjacency(dominoes, n):
                return False
            region_coverage = defaultdict(int)
            for i in range(n):
                for j in range(n):
                    if covered[i][j] and region_grid[i][j] != "X":
                        region_coverage[region_grid[i][j]] += 1
            for region, count in region_coverage.items():
                if count != 2:
                    return False
            for i in range(n):
                for j in range(n):
                    if region_grid[i][j] == "X" and (not covered[i][j]):
                        return False
            return True
        except Exception:
            return False

    def _parse_answer(self, test_solution: str):
        try:
            pattern = "\\[\\((\\d+),\\s*(\\d+)\\),\\s*\\((\\d+),\\s*(\\d+)\\)\\]"
            matches = re.findall(pattern, test_solution)
            if not matches:
                pattern = "\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)\\s*,\\s*\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)"
                matches = re.findall(pattern, test_solution)
            dominoes = []
            for match in matches:
                i1, j1, i2, j2 = map(int, match)
                dominoes.append([(i1, j1), (i2, j2)])
            return dominoes
        except Exception:
            return None

    def _check_domino_shapes(self, dominoes):
        for domino in dominoes:
            if len(domino) != 2:
                return False
            (i1, j1), (i2, j2) = domino
            if not (i1 == i2 and abs(j1 - j2) == 1 or (j1 == j2 and abs(i1 - i2) == 1)):
                return False
        return True

    def _check_domino_adjacency(self, dominoes, n):
        grid = [[-1 for _ in range(n + 2)] for _ in range(n + 2)]
        for idx, domino in enumerate(dominoes):
            for i, j in domino:
                grid[i][j] = idx
        for idx, domino in enumerate(dominoes):
            for i, j in domino:
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = (i + di, j + dj)
                    if 1 <= ni <= n and 1 <= nj <= n:
                        if grid[ni][nj] != -1 and grid[ni][nj] != idx:
                            return False
        return True

    def extract_answer(self, test_solution: str, strict=False):
        answer_patterns = [
            "\\[\\s*\\[\\s*\\(\\s*\\d+\\s*,\\s*\\d+\\s*\\)\\s*,\\s*\\(\\s*\\d+\\s*,\\s*\\d+\\s*\\)\\s*\\]",
            "答案是\\s*(.*?)\\s*$",
            "answer is\\s*(.*?)\\s*$",
            "solution is\\s*(.*?)\\s*$",
        ]
        for pattern in answer_patterns:
            matches = re.findall(pattern, test_solution, re.IGNORECASE | re.DOTALL)
            if matches:
                return matches[-1]
        return test_solution
