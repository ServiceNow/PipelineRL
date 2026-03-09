import argparse
import json
import uuid
from pathlib import Path

from .....base.data import Data
from .....games.tasks.wordscapes.scripts.wordscapes_game import WordscapesGame
from .....games.tasks.wordscapes.scripts.wordscapes_prompt import prompt_wordscapes


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
    puzzles = game.generate(args.num_questions, args.max_attempts)
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
    game_data_list = []
    for puzzle in puzzles:
        question = prompt_wordscapes(puzzle.across_words, puzzle.down_words, puzzle.grid)
        game_data = Data(
            question=question,
            answer=puzzle.solution,
            metadata={
                "trace_id": str(uuid.uuid4()),
                "id": puzzle.idx,
                "grid": puzzle.grid,
                "across_words": puzzle.across_words,
                "down_words": puzzle.down_words,
                "solution": puzzle.solution,
                "size": args.size,
                "num_words": args.num_words,
                "max_intersections": args.max_intersections,
            },
        )
        game_data_list.append(game_data)
    with open(output_file, "w") as f:
        for game_data in game_data_list:
            f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
