import argparse
import json
import random
import uuid
from pathlib import Path

import numpy as np
from english_words import get_english_words_set
from .....games.base.game import Game
from .....games.tasks.wordscapes.scripts.wordscapes_data import WordscapesData
from .....games.tasks.wordscapes.scripts.wordscapes_verifier import WordscapesVerifier


class WordscapesGame(Game):
    def __init__(self):
        super().__init__("Wordscapes", WordscapesVerifier)

    def generate(self, num_of_questions: int = 100, max_attempts: int = 100):
        game_data = []
        for i in range(num_of_questions):
            size = random.randint(4, 6)
            num_words = random.randint(4, 8)
            max_intersections = random.randint(5, 10)
            for _ in range(max_attempts):
                try:
                    grid, across_words, down_words, solution = self._generate_puzzle(size, num_words, max_intersections)
                    data = WordscapesData(str(i + 1), grid, across_words, down_words, solution)
                    game_data.append(data)
                    break
                except Exception:
                    continue
        return game_data

    def _generate_puzzle(self, size, num_words, max_intersections):
        english_words_set = get_english_words_set(["web2"], lower=True)
        grid = np.full((size, size), "0", dtype=str)
        solution_grid = np.full((size, size), " ", dtype=str)
        h_slots = []
        for r in range(size):
            if random.random() < 0.7:
                length = random.randint(3, size)
                if length > 2:
                    start_col = random.randint(0, size - length)
                    h_slots.append((r, start_col, length))
        v_slots = []
        for c in range(size):
            if random.random() < 0.7:
                length = random.randint(3, size)
                if length > 2:
                    start_row = random.randint(0, size - length)
                    v_slots.append((start_row, c, length))
        random.shuffle(h_slots)
        random.shuffle(v_slots)
        h_slots = h_slots[: num_words // 2 + 1]
        v_slots = v_slots[: num_words - len(h_slots)]
        intersection_points = set()
        for r, c, length in h_slots:
            for i in range(length):
                grid[r][c + i] = "X"
        for r, c, length in v_slots:
            for i in range(length):
                if grid[r + i][c] == "X":
                    intersection_points.add((r + i, c))
                grid[r + i][c] = "X"
        if len(intersection_points) > max_intersections:
            return self._generate_puzzle(size, num_words, max_intersections)
        across_words = []
        down_words = []
        valid_words = {}
        for length in range(2, size + 1):
            valid_words[length] = [word for word in english_words_set if len(word) == length]
        for r, c, length in h_slots:
            if not valid_words.get(length, []):
                for i in range(length):
                    grid[r][c + i] = "0"
                continue
            constraints = {}
            for i in range(length):
                pos = (r, c + i)
                if pos in intersection_points:
                    for row in range(size):
                        if row != r and grid[row][c + i] == "X":
                            constraints[i] = None
            word = random.choice(valid_words[length])
            across_words.append(word)
            for i, letter in enumerate(word):
                solution_grid[r][c + i] = letter
        for r, c, length in v_slots:
            if not valid_words.get(length, []):
                for i in range(length):
                    grid[r + i][c] = "0"
                continue
            constraints = {}
            for i in range(length):
                pos = (r + i, c)
                if pos in intersection_points:
                    h_index = None
                    for idx, (hr, hc, hl) in enumerate(h_slots):
                        if hr == r + i and hc <= c < hc + hl:
                            h_index = idx
                            h_pos = c - hc
                            break
                    if h_index is not None and h_index < len(across_words):
                        constraints[i] = across_words[h_index][h_pos]
            candidates = []
            for word in valid_words[length]:
                matches = True
                for pos, letter in constraints.items():
                    if letter and word[pos] != letter:
                        matches = False
                        break
                if matches:
                    candidates.append(word)
            if candidates:
                word = random.choice(candidates)
                down_words.append(word)
                for i, letter in enumerate(word):
                    solution_grid[r + i][c] = letter
            else:
                return self._generate_puzzle(size, num_words, max_intersections)
        grid_list = [list(row) for row in grid]
        solution_list = [list(row) for row in solution_grid]
        return (grid_list, across_words, down_words, solution_list)

    def extract_answer(self, test_solution: str):
        return self.verifier.extract_answer(test_solution)


def main():
    parser = argparse.ArgumentParser(description="Generate Wordscapes puzzles")
    parser.add_argument("--num-questions", type=int, default=100, help="Number of puzzles to generate (default: 100)")
    parser.add_argument(
        "--max-attempts", type=int, default=100, help="Maximum attempts to generate a valid puzzle (default: 100)"
    )
    parser.add_argument("--size", type=int, default=None, help="Grid size (default: random between 4-6)")
    parser.add_argument(
        "--num-words", type=int, default=None, help="Number of words per puzzle (default: random between 4-8)"
    )
    parser.add_argument(
        "--max-intersections",
        type=int,
        default=None,
        help="Maximum letter intersections (default: random between 5-10)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Custom output file path (overrides default structured path)"
    )
    args = parser.parse_args()
    game = WordscapesGame()
    original_generate_puzzle = game._generate_puzzle
    if args.size is not None or args.num_words is not None or args.max_intersections is not None:

        def custom_generate_puzzle(size, num_words, max_intersections):
            custom_size = args.size if args.size is not None else size
            custom_num_words = args.num_words if args.num_words is not None else num_words
            custom_max_intersections = (
                args.max_intersections if args.max_intersections is not None else max_intersections
            )
            return original_generate_puzzle(custom_size, custom_num_words, custom_max_intersections)

        game._generate_puzzle = custom_generate_puzzle
    game_data = game.generate(args.num_questions, args.max_attempts)
    if args.output is None:
        data_dir = Path(__file__).parent.parent / "data"
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
        size_str = f"size_{args.size}" if args.size is not None else "size_random"
        words_str = f"words_{args.num_words}" if args.num_words is not None else "words_random"
        intersect_str = (
            f"intersections_{args.max_intersections}" if args.max_intersections is not None else "intersections_random"
        )
        output_file = (
            data_dir
            / "hyperparameter_search"
            / size_str
            / words_str
            / intersect_str
            / f"wordscapes_{args.num_questions}.jsonl"
        )
    else:
        output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for data in game_data:
            json_data = {
                "id": data.id,
                "grid": data.grid,
                "across_words": data.across_words,
                "down_words": data.down_words,
                "solution": data.solution,
                "metadata": {
                    "trace_id": str(uuid.uuid4()),
                    "size": args.size,
                    "num_words": args.num_words,
                    "max_intersections": args.max_intersections,
                },
            }
            f.write(json.dumps(json_data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
