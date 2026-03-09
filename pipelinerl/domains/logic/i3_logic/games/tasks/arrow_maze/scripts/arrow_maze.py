import argparse
import copy
import json
import pathlib
import random
import uuid
from typing import List, Tuple

from .....base.data import Data
from .....games.base.game import Game
from .....games.tasks.arrow_maze.scripts.arrow_maze_prompt import prompt_arrow_maze
from .....games.tasks.arrow_maze.scripts.arrow_maze_verifier import ArrowMazeVerifier


class ArrowMaze(Game):
    ARROWS = {
        "↑": (-1, 0),
        "↓": (1, 0),
        "←": (0, -1),
        "→": (0, 1),
        "↖": (-1, -1),
        "↗": (-1, 1),
        "↘": (1, 1),
        "↙": (1, -1),
    }

    def __init__(self):
        super().__init__("Arrow Maze", ArrowMazeVerifier)

    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 10000,
        width: int = 8,
        height: int = 8,
        arrow_fill_rate_min: float = 0.0,
        arrow_fill_rate_max: float = 0.0,
    ):
        game_data_list = []
        attempts = 0

        while len(game_data_list) < num_of_questions and attempts < max_attempts:
            attempts += 1

            try:
                n = height
                m = width

                arrow_fill_rate = random.uniform(arrow_fill_rate_min, arrow_fill_rate_max)

                arrow_fill_rate = round(arrow_fill_rate, 1)

                max_numbers = m + n
                num_numbers = random.randint(4, min(max_numbers, n * m // 5))

                maze, solution = self._generate_maze(n, m, num_numbers, min_numbers=4)

                if arrow_fill_rate > 0:
                    fillable_positions = []
                    for i in range(n):
                        for j in range(m):
                            if maze[i][j] == "X" and solution[i][j] in self.ARROWS:
                                fillable_positions.append((i, j))

                    fill_count = int(len(fillable_positions) * arrow_fill_rate)
                    if fill_count > 0:
                        random.shuffle(fillable_positions)
                        for i, j in fillable_positions[:fill_count]:
                            maze[i][j] = solution[i][j]

                if not self._is_valid_maze(maze, solution):
                    continue

                question = prompt_arrow_maze(maze, n, m, random.choice([True, False]))

                answer = json.dumps(solution)

                maze_data = Data(
                    question=question,
                    answer=answer,
                    metadata={
                        "trace_id": str(uuid.uuid4()),
                        "maze": maze,
                        "solution": solution,
                        "height": n,
                        "width": m,
                        "arrow_fill_rate": arrow_fill_rate,
                    },
                )

                game_data_list.append(maze_data)

            except Exception:
                continue

        return game_data_list

    def _generate_maze(
        self, n: int, m: int, num_numbers: int, min_numbers: int = 4
    ) -> Tuple[List[List[str]], List[List[str]]]:
        maze = [["X" for _ in range(m)] for _ in range(n)]
        solution = [["X" for _ in range(m)] for _ in range(n)]

        actual_num_numbers = max(min_numbers, int(num_numbers * random.uniform(0.7, 0.9)))

        num_regions = 9

        region_rows_1 = n // 3
        region_rows_2 = 2 * n // 3
        region_cols_1 = m // 3
        region_cols_2 = 2 * m // 3

        regions = [
            (0, 0, region_rows_1, region_cols_1),
            (0, region_cols_1, region_rows_1, region_cols_2),
            (0, region_cols_2, region_rows_1, m),
            (region_rows_1, 0, region_rows_2, region_cols_1),
            (region_rows_1, region_cols_1, region_rows_2, region_cols_2),
            (region_rows_1, region_cols_2, region_rows_2, m),
            (region_rows_2, 0, n, region_cols_1),
            (region_rows_2, region_cols_1, n, region_cols_2),
            (region_rows_2, region_cols_2, n, m),
        ]

        base_numbers = actual_num_numbers // num_regions
        extra_numbers = actual_num_numbers % num_regions

        numbers_per_region = [base_numbers] * num_regions

        center_region_idx = 4
        numbers_per_region[center_region_idx] += min(2, extra_numbers)

        remaining_extra = max(0, extra_numbers - 2)
        for i in range(num_regions):
            if i != center_region_idx and remaining_extra > 0:
                numbers_per_region[i] += 1
                remaining_extra -= 1

        number_positions = []

        for region_idx, (start_i, start_j, end_i, end_j) in enumerate(regions):
            region_area = (end_i - start_i) * (end_j - start_j)
            if region_area == 0:
                continue

            region_positions = [(i, j) for i in range(start_i, end_i) for j in range(start_j, end_j)]
            random.shuffle(region_positions)

            if region_idx == center_region_idx:
                min_distance = max(1, min(n, m) // 6)
            else:
                min_distance = max(2, min(n, m) // 5)

            placed_in_region = 0
            for pos in region_positions:
                if placed_in_region >= numbers_per_region[region_idx]:
                    break

                i, j = pos

                too_close = False
                for ni, nj in number_positions:
                    if abs(ni - i) + abs(nj - j) < min_distance:
                        too_close = True
                        break

                if not too_close:
                    maze[i][j] = "1"
                    solution[i][j] = "1"
                    number_positions.append((i, j))
                    placed_in_region += 1

            if placed_in_region < numbers_per_region[region_idx] // 2:
                min_distance = max(1, min_distance // 2)
                for pos in region_positions:
                    if placed_in_region >= numbers_per_region[region_idx]:
                        break

                    i, j = pos
                    if (i, j) in number_positions:
                        continue

                    too_close = False
                    for ni, nj in number_positions:
                        if abs(ni - i) + abs(nj - j) < min_distance:
                            too_close = True
                            break

                    if not too_close:
                        maze[i][j] = "1"
                        solution[i][j] = "1"
                        number_positions.append((i, j))
                        placed_in_region += 1

        if len(number_positions) < min(min_numbers, min(n, m) // 2):
            center_positions = []
            for i in range(n // 3, 2 * n // 3):
                for j in range(m // 3, 2 * m // 3):
                    if (i, j) not in number_positions and solution[i][j] == "X":
                        center_positions.append((i, j))

            random.shuffle(center_positions)

            for i, j in center_positions:
                if len(number_positions) >= min(min_numbers, min(n, m) // 2):
                    break
                maze[i][j] = "1"
                solution[i][j] = "1"
                number_positions.append((i, j))

            if len(number_positions) < min(min_numbers, min(n, m) // 2):
                available_positions = [
                    (i, j)
                    for i in range(n)
                    for j in range(m)
                    if (i, j) not in number_positions and solution[i][j] == "X"
                ]
                random.shuffle(available_positions)

                while len(number_positions) < min(min_numbers, min(n, m) // 2) and available_positions:
                    i, j = available_positions.pop(0)
                    maze[i][j] = "1"
                    solution[i][j] = "1"
                    number_positions.append((i, j))

        covered = set()
        for i, j in number_positions:
            covered.add((i, j))

        center_i, center_j = n / 2, m / 2

        balanced_priority = lambda pos: (
            -0.5 * min(pos[0], pos[1], n - 1 - pos[0], m - 1 - pos[1])
            - 0.5 * (abs(pos[0] - center_i) + abs(pos[1] - center_j))
        )

        random.shuffle(number_positions)
        number_positions.sort(key=balanced_priority)

        for priority_idx, (i, j) in enumerate(number_positions):
            directions = list(self.ARROWS.items())
            random.shuffle(directions)

            in_center = (n // 3 <= i < 2 * n // 3) and (m // 3 <= j < 2 * m // 3)

            min_rays = min(1, len(directions))
            max_rays = min(len(directions), 2 + (priority_idx * 2) // max(1, len(number_positions)))
            selected_directions = directions[: random.randint(min_rays, max_rays)]

            total_arrows = 0
            ray_positions = []

            for arrow_symbol, (di, dj) in selected_directions:
                ni, nj = i + di, j + dj
                ray_length = 0

                min_length = 1

                while 0 <= ni < n and 0 <= nj < m:
                    if (ni, nj) in covered or solution[ni][nj].isdigit():
                        break

                    solution[ni][nj] = arrow_symbol
                    covered.add((ni, nj))
                    ray_positions.append((ni, nj))
                    ray_length += 1
                    total_arrows += 1

                    max_length = 2 + (priority_idx * 2) // max(1, len(number_positions))
                    if in_center:
                        max_length = min(max_length, 3)

                    if ray_length >= min_length and (ray_length >= max_length or random.random() < 0.4):
                        break

                    ni += di
                    nj += dj

            if total_arrows > 0:
                maze[i][j] = str(total_arrows)
                solution[i][j] = str(total_arrows)
            else:
                for arrow_symbol, (di, dj) in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < m and solution[ni][nj] == "X":
                        solution[ni][nj] = arrow_symbol
                        covered.add((ni, nj))
                        maze[i][j] = "1"
                        solution[i][j] = "1"
                        break

                if maze[i][j] == "1" and self._count_arrow_rays(solution, i, j) == 0:
                    maze[i][j] = "X"
                    solution[i][j] = "X"
                    number_positions[priority_idx] = None

        number_positions = [pos for pos in number_positions if pos is not None]

        remaining_positions = []
        for i in range(n):
            for j in range(m):
                if solution[i][j] == "X":
                    remaining_positions.append((i, j))

        while remaining_positions:
            if not remaining_positions:
                break

            remaining_positions.sort(
                key=lambda pos: min(abs(pos[0] - num_pos[0]) + abs(pos[1] - num_pos[1]) for num_pos in number_positions)
                if number_positions
                else 0
            )

            curr_pos = remaining_positions.pop(0)
            i, j = curr_pos

            add_number_prob = 0.15
            if len(number_positions) < min(n, m) // 3:
                add_number_prob = 0.25

            added_number = False
            if random.random() < add_number_prob:
                directions = list(self.ARROWS.items())
                random.shuffle(directions)

                for arrow_symbol, (di, dj) in directions:
                    ni, nj = i + di, j + dj

                    if 0 <= ni < n and 0 <= nj < m and solution[ni][nj] == "X":
                        too_close = False
                        for num_i, num_j in number_positions:
                            if abs(num_i - i) + abs(num_j - j) < 2:
                                too_close = True
                                break

                        if too_close:
                            continue

                        possible_arrows = 0
                        test_ni, test_nj = ni, nj
                        while 0 <= test_ni < n and 0 <= test_nj < m and solution[test_ni][test_nj] == "X":
                            possible_arrows += 1
                            test_ni += di
                            test_nj += dj
                            if possible_arrows >= 2:
                                break

                        if possible_arrows < 1:
                            continue

                        solution[i][j] = "1"

                        arrow_count = 0
                        for k in range(possible_arrows):
                            if 0 <= ni < n and 0 <= nj < m and solution[ni][nj] == "X":
                                solution[ni][nj] = arrow_symbol
                                if (ni, nj) in remaining_positions:
                                    remaining_positions.remove((ni, nj))
                                ni += di
                                nj += dj
                                arrow_count += 1

                        solution[i][j] = str(arrow_count)
                        maze[i][j] = str(arrow_count)

                        number_positions.append((i, j))
                        added_number = True
                        break

            if not added_number:
                nearest_number = None
                nearest_dist = float("inf")

                for ni, nj in number_positions:
                    dist = abs(ni - i) + abs(nj - j)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_number = (ni, nj)

                if nearest_number and nearest_dist < n + m:
                    ni, nj = nearest_number

                    di = 0 if i == ni else (1 if i < ni else -1)
                    dj = 0 if j == nj else (1 if j < nj else -1)

                    arrow_symbol = None
                    for arrow, (d_i, d_j) in self.ARROWS.items():
                        if (d_i, d_j) == (di, dj):
                            arrow_symbol = arrow
                            break

                    if arrow_symbol:
                        solution[i][j] = arrow_symbol
                    else:
                        solution[i][j] = random.choice(list(self.ARROWS.keys()))
                else:
                    solution[i][j] = random.choice(list(self.ARROWS.keys()))

        for i in range(n):
            for j in range(m):
                if solution[i][j].isdigit():
                    arrows_count = self._count_arrow_rays(solution, i, j)

                    if arrows_count > 0:
                        maze[i][j] = str(arrows_count)
                        solution[i][j] = str(arrows_count)
                    else:
                        added = False
                        for arrow_symbol, (di, dj) in self.ARROWS.items():
                            ni, nj = i + di, j + dj
                            if 0 <= ni < n and 0 <= nj < m and solution[ni][nj] == "X":
                                solution[ni][nj] = arrow_symbol
                                maze[i][j] = "1"
                                solution[i][j] = "1"
                                added = True
                                break

                        if not added:
                            maze[i][j] = "X"
                            solution[i][j] = random.choice(list(self.ARROWS.keys()))

        covered = [[False for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if solution[i][j].isdigit():
                    covered[i][j] = True
                    self._mark_covered_arrows(solution, covered, i, j)

        uncovered_arrows = []
        for i in range(n):
            for j in range(m):
                if solution[i][j] in self.ARROWS and not covered[i][j]:
                    uncovered_arrows.append((i, j))

        while uncovered_arrows:
            i, j = uncovered_arrows.pop(0)
            if covered[i][j]:
                continue

            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue

                    ni, nj = i + di, j + dj
                    if not (0 <= ni < n and 0 <= nj < m):
                        continue

                    if solution[ni][nj] in self.ARROWS or solution[ni][nj] == "X":
                        if solution[ni][nj] in self.ARROWS and covered[ni][nj]:
                            continue

                        direction = None
                        for arrow, (d_i, d_j) in self.ARROWS.items():
                            if (d_i, d_j) == (-di, -dj):
                                direction = arrow
                                break

                        if direction:
                            original_value = solution[ni][nj]

                            maze[ni][nj] = "1"
                            solution[ni][nj] = "1"

                            if solution[i][j] != direction:
                                solution[i][j] = direction

                            new_covered = [[False for _ in range(m)] for _ in range(n)]
                            for r in range(n):
                                for c in range(m):
                                    if solution[r][c].isdigit():
                                        new_covered[r][c] = True
                                        self._mark_covered_arrows(solution, new_covered, r, c)

                            if new_covered[i][j]:
                                covered = new_covered
                                break
                            else:
                                solution[ni][nj] = original_value
                                maze[ni][nj] = "X" if original_value in self.ARROWS else original_value

                if covered[i][j]:
                    break

        value_one_numbers = []
        for i in range(n):
            for j in range(m):
                if solution[i][j] == "1":
                    value_one_numbers.append((i, j))

        if len(value_one_numbers) > max(2, len(number_positions) // 3):
            random.shuffle(value_one_numbers)

            for i, j in value_one_numbers[: len(value_one_numbers) // 2]:
                arrow_positions = []
                for arrow_symbol, (di, dj) in self.ARROWS.items():
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < m and solution[ni][nj] == arrow_symbol:
                        arrow_positions.append((ni, nj, arrow_symbol))

                if arrow_positions:
                    ni, nj, arrow_symbol = arrow_positions[0]

                    nearby_numbers = []

                    for search_dist in [2, 3, 4]:
                        for di in range(-search_dist, search_dist + 1):
                            for dj in range(-search_dist, search_dist + 1):
                                if di == 0 and dj == 0:
                                    continue
                                new_i, new_j = i + di, j + dj
                                if (
                                    0 <= new_i < n
                                    and 0 <= new_j < m
                                    and solution[new_i][new_j].isdigit()
                                    and (new_i, new_j) != (i, j)
                                ):
                                    value = int(solution[new_i][new_j])
                                    distance = abs(di) + abs(dj)
                                    nearby_numbers.append((value, distance, new_i, new_j))

                        if nearby_numbers:
                            break

                    nearby_numbers.sort(key=lambda x: (-x[0], x[1]))

                    if nearby_numbers:
                        _, _, new_i, new_j = nearby_numbers[0]

                        dir_i = 1 if ni > new_i else (-1 if ni < new_i else 0)
                        dir_j = 1 if nj > new_j else (-1 if nj < new_j else 0)

                        new_arrow_symbol = None
                        for arrow, (d_i, d_j) in self.ARROWS.items():
                            if (d_i, d_j) == (dir_i, dir_j):
                                new_arrow_symbol = arrow
                                break

                        if new_arrow_symbol:
                            can_make_path = True
                            test_ni, test_nj = new_i, new_j

                            while test_ni != i or test_nj != j:
                                test_ni += dir_i
                                test_nj += dir_j

                                if (
                                    test_ni < 0
                                    or test_ni >= n
                                    or test_nj < 0
                                    or test_nj >= m
                                    or solution[test_ni][test_nj].isdigit()
                                ):
                                    if (test_ni, test_nj) != (i, j):
                                        can_make_path = False
                                        break

                            if can_make_path:
                                temp_maze = copy.deepcopy(maze)
                                temp_solution = copy.deepcopy(solution)

                                try:
                                    solution[i][j] = new_arrow_symbol
                                    maze[i][j] = "X"

                                    test_ni, test_nj = new_i + dir_i, new_j + dir_j
                                    while test_ni != i or test_nj != j:
                                        if 0 <= test_ni < n and 0 <= test_nj < m:
                                            solution[test_ni][test_nj] = new_arrow_symbol
                                        test_ni += dir_i
                                        test_nj += dir_j

                                    solution[ni][nj] = new_arrow_symbol

                                    new_count = self._count_arrow_rays(solution, new_i, new_j)
                                    solution[new_i][new_j] = str(new_count)
                                    maze[new_i][new_j] = str(new_count)

                                    check_covered = [[False for _ in range(m)] for _ in range(n)]
                                    for r in range(n):
                                        for c in range(m):
                                            if solution[r][c].isdigit():
                                                check_covered[r][c] = True
                                                self._mark_covered_arrows(solution, check_covered, r, c)

                                    all_covered = True
                                    for r in range(n):
                                        for c in range(m):
                                            if solution[r][c] in self.ARROWS and not check_covered[r][c]:
                                                all_covered = False
                                                break
                                        if not all_covered:
                                            break

                                    if not all_covered:
                                        maze = temp_maze
                                        solution = temp_solution
                                except Exception:
                                    maze = temp_maze
                                    solution = temp_solution

        value_one_numbers = []
        value_two_numbers = []
        for i in range(n):
            for j in range(m):
                if solution[i][j] == "1":
                    value_one_numbers.append((i, j))
                elif solution[i][j] == "2":
                    value_two_numbers.append((i, j))

        if len(value_one_numbers) > 1 and value_two_numbers:
            random.shuffle(value_one_numbers)

            removal_count = min(len(value_one_numbers) // 2, len(value_one_numbers) - 1)

            for i, j in value_one_numbers[:removal_count]:
                nearby_twos = []
                for ti, tj in value_two_numbers:
                    dist = abs(ti - i) + abs(tj - j)
                    if dist <= 4:
                        nearby_twos.append((dist, ti, tj))

                nearby_twos.sort()

                if nearby_twos:
                    _, ti, tj = nearby_twos[0]

                    one_arrow_pos = None
                    for arrow_symbol, (di, dj) in self.ARROWS.items():
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n and 0 <= nj < m and solution[ni][nj] == arrow_symbol:
                            one_arrow_pos = (ni, nj, arrow_symbol)
                            break

                    if one_arrow_pos:
                        ni, nj, one_arrow_symbol = one_arrow_pos

                        dir_i = 1 if ni > ti else (-1 if ni < ti else 0)
                        dir_j = 1 if nj > tj else (-1 if nj < tj else 0)

                        new_arrow_symbol = None
                        for arrow, (d_i, d_j) in self.ARROWS.items():
                            if (d_i, d_j) == (dir_i, dir_j):
                                new_arrow_symbol = arrow
                                break

                        if new_arrow_symbol:
                            temp_maze = copy.deepcopy(maze)
                            temp_solution = copy.deepcopy(solution)

                            try:
                                solution[i][j] = new_arrow_symbol
                                maze[i][j] = "X"

                                solution[ni][nj] = new_arrow_symbol

                                new_count = self._count_arrow_rays(solution, ti, tj)
                                solution[ti][tj] = str(new_count)
                                maze[ti][tj] = str(new_count)

                                check_covered = [[False for _ in range(m)] for _ in range(n)]
                                for r in range(n):
                                    for c in range(m):
                                        if solution[r][c].isdigit():
                                            check_covered[r][c] = True
                                            self._mark_covered_arrows(solution, check_covered, r, c)

                                all_covered = True
                                for r in range(n):
                                    for c in range(m):
                                        if solution[r][c] in self.ARROWS and not check_covered[r][c]:
                                            all_covered = False
                                            break
                                    if not all_covered:
                                        break

                                if not all_covered:
                                    maze = temp_maze
                                    solution = temp_solution
                            except Exception:
                                maze = temp_maze
                                solution = temp_solution

        for i in range(n):
            for j in range(m):
                if solution[i][j].isdigit():
                    arrows_count = self._count_arrow_rays(solution, i, j)
                    maze[i][j] = str(arrows_count)
                    solution[i][j] = str(arrows_count)

        if not self._is_valid_maze(maze, solution):
            return self._generate_basic_maze(n, m, num_numbers, min_numbers)

        return maze, solution

    def _generate_basic_maze(
        self, n: int, m: int, num_numbers: int, min_numbers: int = 4
    ) -> Tuple[List[List[str]], List[List[str]]]:
        maze = [["X" for _ in range(m)] for _ in range(n)]
        solution = [["X" for _ in range(m)] for _ in range(n)]

        actual_num_numbers = min(num_numbers, n * m // 4)

        actual_num_numbers = max(min_numbers, actual_num_numbers)

        positions = [(i, j) for i in range(n) for j in range(m)]
        random.shuffle(positions)

        number_positions = []
        for idx in range(min(actual_num_numbers, len(positions))):
            i, j = positions[idx]
            maze[i][j] = "1"
            solution[i][j] = "1"
            number_positions.append((i, j))

        for i, j in number_positions:
            directions = list(self.ARROWS.items())
            random.shuffle(directions)
            selected_directions = directions[: random.randint(1, min(2, len(directions)))]

            total_arrows = 0
            for arrow_symbol, (di, dj) in selected_directions:
                ni, nj = i + di, j + dj
                arrows_added = 0

                while 0 <= ni < n and 0 <= nj < m and arrows_added < 2:
                    if solution[ni][nj].isdigit():
                        break
                    if solution[ni][nj] != "X":
                        break

                    solution[ni][nj] = arrow_symbol
                    arrows_added += 1
                    total_arrows += 1

                    ni += di
                    nj += dj

            if total_arrows > 0:
                maze[i][j] = str(total_arrows)
                solution[i][j] = str(total_arrows)

        for i in range(n):
            for j in range(m):
                if solution[i][j] == "X":
                    solution[i][j] = "↓"

        covered = [[False for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if solution[i][j].isdigit():
                    covered[i][j] = True
                    self._mark_covered_arrows(solution, covered, i, j)

        for i in range(n):
            for j in range(m):
                if solution[i][j] in self.ARROWS and not covered[i][j]:
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < n and 0 <= nj < m and solution[ni][nj] in self.ARROWS and not covered[ni][nj]:
                                direction = None
                                for arrow, (d_i, d_j) in self.ARROWS.items():
                                    if (d_i, d_j) == (-di, -dj):
                                        direction = arrow
                                        break

                                if direction and solution[i][j] == direction:
                                    maze[ni][nj] = "1"
                                    solution[ni][nj] = "1"

                                    covered[ni][nj] = True
                                    self._mark_covered_arrows(solution, covered, ni, nj)
                                    break
                        if covered[i][j]:
                            break

                    if not covered[i][j]:
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                ni, nj = i + di, j + dj
                                if not (0 <= ni < n and 0 <= nj < m):
                                    continue
                                if solution[ni][nj] not in self.ARROWS:
                                    continue

                                direction = None
                                for arrow, (d_i, d_j) in self.ARROWS.items():
                                    if (d_i, d_j) == (-di, -dj):
                                        direction = arrow
                                        break

                                if direction:
                                    solution[i][j] = direction
                                    maze[ni][nj] = "1"
                                    solution[ni][nj] = "1"

                                    covered[ni][nj] = True
                                    self._mark_covered_arrows(solution, covered, ni, nj)
                                    break
                            if covered[i][j]:
                                break

        for i in range(n):
            for j in range(m):
                if solution[i][j].isdigit():
                    arrows_count = self._count_arrow_rays(solution, i, j)
                    maze[i][j] = str(arrows_count)
                    solution[i][j] = str(arrows_count)

        return maze, solution

    def _mark_covered_arrows(self, solution: List[List[str]], covered: List[List[bool]], i: int, j: int):
        n = len(solution)
        m = len(solution[0])

        for arrow_symbol, (di, dj) in self.ARROWS.items():
            ni, nj = i + di, j + dj

            while 0 <= ni < n and 0 <= nj < m and solution[ni][nj] == arrow_symbol:
                covered[ni][nj] = True
                ni += di
                nj += dj

    def _is_valid_maze(self, maze: List[List[str]], solution: List[List[str]]) -> bool:
        n = len(maze)
        m = len(maze[0])

        for i in range(n):
            for j in range(m):
                if maze[i][j] == "X" and solution[i][j] == "X":
                    return False

        for i in range(n):
            for j in range(m):
                if maze[i][j].isdigit() and maze[i][j] != solution[i][j]:
                    return False

        for i in range(n):
            for j in range(m):
                if solution[i][j] not in ["X"] + [str(k) for k in range(1, 10)] and solution[i][j] not in self.ARROWS:
                    return False

        covered = [[False for _ in range(m)] for _ in range(n)]

        for i in range(n):
            for j in range(m):
                if solution[i][j].isdigit():
                    covered[i][j] = True

        for i in range(n):
            for j in range(m):
                if maze[i][j].isdigit():
                    number = int(maze[i][j])
                    arrows_count = self._count_arrow_rays(solution, i, j)
                    if arrows_count != number:
                        return False

                    self._mark_covered_arrows(solution, covered, i, j)

        for i in range(n):
            for j in range(m):
                if solution[i][j] in self.ARROWS and not covered[i][j]:
                    return False

        return True

    def _count_arrow_rays(self, solution: List[List[str]], i: int, j: int) -> int:
        n = len(solution)
        m = len(solution[0])
        count = 0

        for arrow_symbol, (di, dj) in self.ARROWS.items():
            ni, nj = i + di, j + dj
            ray_length = 0

            while 0 <= ni < n and 0 <= nj < m and solution[ni][nj] == arrow_symbol:
                ray_length += 1
                ni += di
                nj += dj

            count += ray_length

        return count

    def extract_answer(self, test_solution: str) -> str:
        if not test_solution:
            return ""

        import re

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="箭头迷宫游戏生成器")
    parser.add_argument("--num_of_data", type=int, default=100, help="生成的题目数量")
    parser.add_argument("--max_attempts", type=int, default=10000, help="每个题目的最大尝试次数")
    parser.add_argument("--width", type=int, default=8, help="迷宫宽度")
    parser.add_argument("--height", type=int, default=8, help="迷宫高度")
    parser.add_argument("--arrow_fill_rate_min", type=float, default=0.0, help="预填箭头比例最小值（0.0-1.0之间）")
    parser.add_argument("--arrow_fill_rate_max", type=float, default=0.0, help="预填箭头比例最大值（0.0-1.0之间）")
    args = parser.parse_args()

    args.arrow_fill_rate_min = max(0.0, min(1.0, args.arrow_fill_rate_min))
    args.arrow_fill_rate_max = max(args.arrow_fill_rate_min, min(1.0, args.arrow_fill_rate_max))

    data_dir = pathlib.Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    game = ArrowMaze()

    game_data_list = game.generate(
        num_of_questions=args.num_of_data,
        max_attempts=args.max_attempts,
        width=args.width,
        height=args.height,
        arrow_fill_rate_min=args.arrow_fill_rate_min,
        arrow_fill_rate_max=args.arrow_fill_rate_max,
    )

    base_data_dir = pathlib.Path(__file__).parent.parent / "data"
    if not base_data_dir.exists():
        base_data_dir.mkdir(parents=True, exist_ok=True)

    nested_dir = (
        base_data_dir
        / f"num_of_data_{args.num_of_data}"
        / f"width_{args.width}_height_{args.height}"
        / f"fill_rate_{args.arrow_fill_rate_min:.1f}_{args.arrow_fill_rate_max:.1f}"
    )
    if not nested_dir.exists():
        nested_dir.mkdir(parents=True, exist_ok=True)

    nested_output_file = nested_dir / f"arrow_maze_{args.num_of_data}.jsonl"

    try:
        with open(nested_output_file, "w", encoding="utf-8") as f:
            for game_data in game_data_list:
                f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")
    except Exception:
        pass
