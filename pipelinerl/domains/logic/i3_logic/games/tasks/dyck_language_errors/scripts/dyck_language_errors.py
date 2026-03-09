import argparse
import json
import pathlib
import random
import uuid
from typing import List, Tuple

from .....base.data import Data
from .....games.base.game import Game
from .....games.tasks.dyck_language_errors.scripts.dyck_language_errors_prompt import prompt_dyck_language_errors
from .....games.tasks.dyck_language_errors.scripts.dyck_language_errors_verifier import DyckLanguageErrorsVerifier


class DyckLanguageErrors(Game):
    BRACKET_PAIRS = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]

    def __init__(self):
        super().__init__("Dyck Language Errors", DyckLanguageErrorsVerifier)

    def generate(self, num_of_questions: int = 100, max_attempts: int = 100, n_types: int = 3, total_length: int = 20):
        if n_types < 1 or n_types > 4:
            raise ValueError("括号对的种类数量必须在1到4之间")
        if total_length < 2:
            raise ValueError("括号序列的总长度必须至少为2")

        game_data_list = []
        generated_strings = set()
        attempts = 0

        while len(game_data_list) < num_of_questions and attempts < max_attempts:
            try:
                attempts += 1

                bracket_pairs = random.sample(self.BRACKET_PAIRS, n_types)

                is_valid = random.random() < 0.2

                if is_valid:
                    bracket_string, first_error_pos = self._generate_valid_brackets(bracket_pairs, total_length)

                    if first_error_pos is not None:
                        is_valid = False
                        answer = str(first_error_pos)
                    else:
                        answer = "-1"
                else:
                    bracket_string, first_error_pos = self._generate_invalid_brackets(bracket_pairs, total_length)
                    answer = str(first_error_pos)

                if bracket_string in generated_strings:
                    continue

                generated_strings.add(bracket_string)

                is_chinese = random.random() < 0.5
                question = prompt_dyck_language_errors(bracket_string, n_types, total_length, is_chinese)

                game_data = Data(
                    question=question,
                    answer=answer,
                    metadata={
                        "trace_id": str(uuid.uuid4()),
                        "bracket_string": bracket_string,
                        "first_error_pos": first_error_pos,
                        "is_valid": is_valid,
                        "n_types": n_types,
                        "total_length": total_length,
                        "bracket_pairs": [{"open": open_b, "close": close_b} for open_b, close_b in bracket_pairs],
                    },
                )

                game_data_list.append(game_data)

            except Exception:
                continue

        return game_data_list

    def _generate_valid_brackets(self, bracket_pairs: List[Tuple[str, str]], total_length: int) -> Tuple[str, None]:
        max_even_length = (total_length // 2) * 2

        bracket_string = self._generate_balanced_brackets(bracket_pairs, max_even_length // 2)

        if total_length % 2 != 0 and max_even_length < total_length:
            random_brackets = [b for pair in bracket_pairs for b in pair]
            bracket_string += random.choice(random_brackets)

        bracket_list = list(bracket_string)
        while len(bracket_list) > total_length:
            bracket_list.pop()

        while len(bracket_list) < total_length:
            random_brackets = [b for pair in bracket_pairs for b in pair]
            bracket_list.append(random.choice(random_brackets))

        bracket_string = "".join(bracket_list)

        error_pos = self._find_first_error_position(list(bracket_string), bracket_pairs)
        return bracket_string, error_pos

    def _generate_balanced_brackets(self, bracket_pairs: List[Tuple[str, str]], pairs_count: int) -> str:
        if pairs_count == 0:
            return ""

        result = []
        stack = []

        open_bracket, close_bracket = random.choice(bracket_pairs)
        result.append(open_bracket)
        stack.append((open_bracket, close_bracket))

        for _ in range(pairs_count * 2 - 1):
            if stack and random.random() < 0.5:
                open_b, close_b = stack.pop()
                result.append(close_b)
            else:
                if len(stack) < pairs_count:
                    open_b, close_b = random.choice(bracket_pairs)
                    result.append(open_b)
                    stack.append((open_b, close_b))
                else:
                    if stack:
                        open_b, close_b = stack.pop()
                        result.append(close_b)

        while stack:
            open_b, close_b = stack.pop()
            result.append(close_b)

        return "".join(result)

    def _generate_invalid_brackets(self, bracket_pairs: List[Tuple[str, str]], total_length: int) -> Tuple[str, int]:
        valid_sequence, _ = self._generate_valid_brackets(bracket_pairs, total_length)

        sequence_list = list(valid_sequence)

        error_type = random.choice(["mismatch", "unclosed", "extra_closing"])

        if error_type == "mismatch":
            closing_indices = [
                i for i, char in enumerate(sequence_list) if any(char == close_b for _, close_b in bracket_pairs)
            ]

            if closing_indices:
                error_index = random.choice(closing_indices)
                current_close = sequence_list[error_index]

                other_close_brackets = [close_b for _, close_b in bracket_pairs if close_b != current_close]

                if other_close_brackets:
                    sequence_list[error_index] = random.choice(other_close_brackets)
                    first_error_pos = error_index + 1
                else:
                    open_brackets = [open_b for open_b, _ in bracket_pairs]
                    sequence_list[error_index] = random.choice(open_brackets)
                    first_error_pos = error_index + 1
            else:
                error_type = "extra_closing"

        if error_type == "unclosed":
            closing_indices = [
                i for i, char in enumerate(sequence_list) if any(char == close_b for _, close_b in bracket_pairs)
            ]

            if closing_indices:
                error_index = random.choice(closing_indices)

                del sequence_list[error_index]

                random_brackets = [b for pair in bracket_pairs for b in pair]
                sequence_list.append(random.choice(random_brackets))

                first_error_pos = self._find_first_error_position(sequence_list, bracket_pairs)
            else:
                error_type = "extra_closing"

        if error_type == "extra_closing":
            close_brackets = [close_b for _, close_b in bracket_pairs]
            extra_close = random.choice(close_brackets)

            insert_pos = random.randint(0, len(sequence_list) - 1)
            sequence_list.insert(insert_pos, extra_close)

            while len(sequence_list) > total_length:
                sequence_list.pop()

            while len(sequence_list) < total_length:
                random_brackets = [b for pair in bracket_pairs for b in pair]
                sequence_list.append(random.choice(random_brackets))

            first_error_pos = self._find_first_error_position(sequence_list, bracket_pairs)

        invalid_sequence = "".join(sequence_list)

        return invalid_sequence, first_error_pos

    def _find_first_error_position(self, sequence_list: List[str], bracket_pairs: List[Tuple[str, str]]) -> int:
        stack = []
        opening_brackets = [open_b for open_b, _ in bracket_pairs]
        closing_brackets = [close_b for _, close_b in bracket_pairs]
        bracket_map = {close_b: open_b for open_b, close_b in bracket_pairs}

        for i, char in enumerate(sequence_list):
            if char in opening_brackets:
                stack.append(char)
            elif char in closing_brackets:
                if not stack:
                    return i + 1

                last_open = stack.pop()
                if last_open != bracket_map[char]:
                    return i + 1

        if stack:
            return len(sequence_list) + 1

        return None

    def extract_answer(self, test_solution: str):
        import re

        solution = test_solution.strip() if test_solution else ""

        numbers = re.findall(r"-?\d+", solution)
        if numbers:
            if "-1" in numbers:
                return "-1"

            for num in numbers:
                if num.isdigit() and int(num) >= 0:
                    return num

            return numbers[0]

        if any(keyword in solution.lower() for keyword in ["合法", "valid", "correct"]):
            return "-1"

        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="括号闭合的错误识别游戏生成器")
    parser.add_argument("--num_of_data", type=int, default=100, help="生成的题目数量")
    parser.add_argument("--max_attempts", type=int, default=1000, help="每个题目的最大尝试次数")
    parser.add_argument("--n_types", type=int, default=3, help="括号对的种类数量(1-4)")
    parser.add_argument("--total_length", type=int, default=20, help="括号序列的总长度")
    args = parser.parse_args()

    data_dir = pathlib.Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    game = DyckLanguageErrors()

    game_data_list = game.generate(
        num_of_questions=args.num_of_data,
        max_attempts=args.max_attempts,
        n_types=args.n_types,
        total_length=args.total_length,
    )

    nested_dir = (
        data_dir / f"num_of_data_{args.num_of_data}" / f"n_types_{args.n_types}" / f"total_length_{args.total_length}"
    )
    if not nested_dir.exists():
        nested_dir.mkdir(parents=True, exist_ok=True)

    nested_output_file = nested_dir / f"dyck_language_errors_{args.num_of_data}.jsonl"

    try:
        with open(nested_output_file, "w", encoding="utf-8") as f:
            for game_data in game_data_list:
                f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")
    except Exception:
        pass
