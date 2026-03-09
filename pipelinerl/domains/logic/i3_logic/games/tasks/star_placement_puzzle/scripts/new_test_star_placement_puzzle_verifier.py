import unittest

from .....base.data import Data
from .....games.tasks.star_placement_puzzle.scripts.star_placement_puzzle_verifier import (
    StarPlacementPuzzleVerifier,
)


class NewTestStarPlacementPuzzleVerifier(unittest.TestCase):
    def setUp(self):
        self.verifier = StarPlacementPuzzleVerifier()
        self.test_region_grid = [
            ["B", "B", "A", "E", "E"],
            ["B", "A", "A", "A", "C"],
            ["B", "B", "A", "C", "C"],
            ["D", "D", "D", "D", "C"],
            ["E", "D", "E", "E", "C"],
        ]
        self.test_data = Data(
            question="在每个区域放置星星，满足每行每列每区域有1颗星星，且星星不相邻",
            answer="",
            metadata={"n": 5, "k": 1, "region_grid": self.test_region_grid},
        )

    def test_coordinates_standard_format(self):
        coords = {"A": [(0, 0), (2, 2), (4, 4)], "B": [(1, 3), (3, 1), (5, 5)], "C": [(0, 4), (2, 0), (4, 2)]}
        self.assertEqual(len(coords), 3, "应有3个区域的坐标")
        self.assertEqual(len(coords["A"]), 3, "A区域应有3个坐标")
        self.assertEqual(len(coords["B"]), 3, "B区域应有3个坐标")
        self.assertEqual(len(coords["C"]), 3, "C区域应有3个坐标")
        self.assertIn((0, 0), coords["A"])
        self.assertIn((2, 2), coords["A"])
        self.assertIn((4, 4), coords["A"])
        self.assertIn((1, 3), coords["B"])
        self.assertIn((3, 1), coords["B"])
        self.assertIn((5, 5), coords["B"])

    def test_coordinates_with_extra_data(self):
        coords = {"A": [(0, 1), (3, 4)], "B": [(1, 2), (4, 5)], "C": [(2, 0), (5, 3)]}
        self.assertEqual(len(coords), 3, "应有3个区域的坐标")
        self.assertEqual(len(coords["A"]), 2, "A区域应有2个坐标")
        self.assertEqual(len(coords["B"]), 2, "B区域应有2个坐标")
        self.assertEqual(len(coords["C"]), 2, "C区域应有2个坐标")

    def test_valid_5x5_solution(self):
        valid_coords = {"A": [(1, 2)], "B": [(0, 0)], "C": [(2, 4)], "D": [(3, 1)], "E": [(4, 3)]}
        result = self.verifier.verify(self.test_data, valid_coords)
        self.assertTrue(result, "应验证为有效解决方案")

    def test_invalid_row_constraint(self):
        invalid_coords = {"A": [(0, 2)], "B": [(0, 0)], "C": [(2, 4)], "D": [(3, 1)], "E": [(4, 3)]}
        result = self.verifier.verify(self.test_data, invalid_coords)
        self.assertFalse(result, "应验证为无效解决方案：违反行约束")

    def test_invalid_column_constraint(self):
        invalid_coords = {"A": [(1, 2)], "B": [(0, 0)], "C": [(2, 0)], "D": [(3, 1)], "E": [(4, 3)]}
        result = self.verifier.verify(self.test_data, invalid_coords)
        self.assertFalse(result, "应验证为无效解决方案：违反列约束")

    def test_invalid_region_constraint(self):
        invalid_coords = {"A": [(1, 2), (2, 2)], "B": [(0, 0)], "C": [(2, 4)], "D": [(3, 1)], "E": []}
        result = self.verifier.verify(self.test_data, invalid_coords)
        self.assertFalse(result, "应验证为无效解决方案：违反区域约束")

    def test_invalid_adjacency_constraint(self):
        invalid_coords = {"A": [(1, 2)], "B": [(0, 0)], "C": [(2, 4)], "D": [(3, 1)], "E": [(0, 1)]}
        result = self.verifier.verify(self.test_data, invalid_coords)
        self.assertFalse(result, "应验证为无效解决方案：存在相邻星星")

    def test_invalid_diagonal_adjacency(self):
        invalid_coords = {"A": [(1, 1)], "B": [(0, 0)], "C": [(2, 4)], "D": [(3, 2)], "E": [(4, 3)]}
        result = self.verifier.verify(self.test_data, invalid_coords)
        self.assertFalse(result, "应验证为无效解决方案：存在对角线相邻星星")

    def test_out_of_bounds_coordinates(self):
        invalid_coords = {"A": [(1, 2)], "B": [(0, 0)], "C": [(2, 4)], "D": [(3, 1)], "E": [(5, 3)]}
        result = self.verifier.verify(self.test_data, invalid_coords)
        self.assertFalse(result, "应验证为无效解决方案：坐标超出范围")

    def test_duplicate_coordinates(self):
        invalid_coords = {"A": [(1, 2)], "B": [(0, 0)], "C": [(1, 2)], "D": [(3, 1)], "E": [(4, 3)]}
        result = self.verifier.verify(self.test_data, invalid_coords)
        self.assertFalse(result, "应验证为无效解决方案：存在重复坐标")


if __name__ == "__main__":
    unittest.main()
