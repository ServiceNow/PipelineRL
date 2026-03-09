import re
from typing import List, Tuple

from .....base.data import Data
from .....base.verifier import Verifier


class MinesweeperVerifier(Verifier):
    def verify(self, data: Data, test_solution: str, **kwargs):
        try:
            predicted_mines = self.extract_answer(test_solution)
            expected_mines = data.metadata["current_mines"]
            if set((tuple(mine) for mine in predicted_mines)) == set((tuple(mine) for mine in expected_mines)):
                return True
            return False
        except Exception:
            return False

    def extract_answer(self, response: str) -> List[Tuple[int, int]]:
        patterns = [
            "\\[\\s*\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)(?:\\s*,\\s*\\(\\s*\\d+\\s*,\\s*\\d+\\s*\\))*\\s*\\]",
            "\\[\\s*\\[\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\](?:\\s*,\\s*\\[\\s*\\d+\\s*,\\s*\\d+\\s*\\])*\\s*\\]",
            "\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)(?:\\s*,\\s*\\(\\s*\\d+\\s*,\\s*\\d+\\s*\\))*",
        ]
        for pattern in patterns:
            coords = []
            for match in re.finditer(pattern, response):
                try:
                    coord_pattern = "(?:\\(|\\[)\\s*(\\d+)\\s*,\\s*(\\d+)\\s*(?:\\)|\\])"
                    for coord_match in re.finditer(coord_pattern, match.group(0)):
                        i, j = (int(coord_match.group(1)), int(coord_match.group(2)))
                        coords.append((i, j))
                except Exception:
                    continue
            if coords:
                return coords
        number_pairs = re.findall("(\\d+)[^\\d]+(\\d+)", response)
        if number_pairs:
            return [(int(i), int(j)) for i, j in number_pairs]
        return []
