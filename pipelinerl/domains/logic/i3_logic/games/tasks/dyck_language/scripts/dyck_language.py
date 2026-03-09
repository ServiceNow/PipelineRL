import argparse
import json
import pathlib
import random
import uuid

from .....base.data import Data
from .....games.base.game import Game


class DyckLanguage(Game):
    def __init__(self):
        from .....games.tasks.dyck_language.scripts.dyck_language_verifier import DyckLanguageVerifier

        super().__init__("Dyck Language", DyckLanguageVerifier)

        self.brackets = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]

    def _check_unique_completion(self, sequence: str, cut_point: int) -> bool:
        prefix = sequence[:cut_point]

        stack = []
        bracket_pairs = {")": "(", "]": "[", "}": "{", ">": "<"}

        for char in prefix:
            if char in "([{<":
                stack.append(char)
            elif char in ")]}>":
                if stack and stack[-1] == bracket_pairs[char]:
                    stack.pop()

        remaining = sequence[cut_point:]
        required_closing = []

        while stack:
            char = stack.pop()
            for close, open in bracket_pairs.items():
                if open == char:
                    required_closing.append(close)
                    break

        return "".join(required_closing) == remaining

    def _generate_valid_sequence_with_unique_completion(
        self, total_length: int, cut_point: int, nesting_depth: int = 0
    ) -> str:
        max_attempts = 100

        for _ in range(max_attempts):
            try:
                if total_length % 2 != 0:
                    total_length -= 1

                sequence = []
                stack = []
                current_depth = 0
                max_depth = 0

                left_count = total_length // 2

                if nesting_depth > 0:
                    for _ in range(nesting_depth):
                        bracket = random.choice(self.used_brackets)
                        sequence.append(bracket[0])
                        stack.append(bracket)
                        current_depth += 1
                        max_depth = max(max_depth, current_depth)
                        left_count -= 1

                while left_count > 0 or stack:
                    if left_count > 0 and (len(stack) == 0 or random.random() < 0.5):
                        bracket = random.choice(self.used_brackets)
                        sequence.append(bracket[0])
                        stack.append(bracket)
                        current_depth += 1
                        max_depth = max(max_depth, current_depth)
                        left_count -= 1
                    elif stack:
                        bracket = stack.pop()
                        sequence.append(bracket[1])
                        current_depth -= 1

                if len(sequence) != total_length:
                    raise ValueError("生成的序列长度不符合要求")
                if nesting_depth > 0 and max_depth < nesting_depth:
                    raise ValueError("生成的序列嵌套深度不足")

                if len(sequence) < cut_point:
                    raise ValueError("截取点超出序列长度")

                result = "".join(sequence)

                if not self._check_unique_completion(result, cut_point):
                    raise ValueError("序列不具有唯一完成方式")

                return result

            except ValueError:
                continue

        raise ValueError("无法生成满足要求的序列")

    def _verify_unique_completion(self, prefix: str) -> bool:
        stack = []
        bracket_pairs = {")": "(", "]": "[", "}": "{", ">": "<"}

        for char in prefix:
            if char in "([{<":
                stack.append(char)
            elif char in ")]}>":
                if stack and stack[-1] == bracket_pairs[char]:
                    stack.pop()
                else:
                    return False

        balanced_pairs = 0
        for i in range(len(prefix) - 1):
            for left, right in self.brackets:
                if prefix[i] == left and prefix[i + 1] == right:
                    balanced_pairs += 1

        return balanced_pairs == 0 or len(stack) == 0

    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        n_types: int = 3,
        total_length: int = 0,
        to_fill_length: int = 0,
        nesting_depth: int = 0,
    ):
        from .....games.tasks.dyck_language.scripts.dyck_language_prompt import prompt_dyck_language

        if n_types < 1 or n_types > 6:
            raise ValueError("括号类型数量必须在1-6之间")
        if total_length < 0:
            raise ValueError("总长度不能为负数")
        if to_fill_length < 0:
            raise ValueError("填充长度不能为负数")
        if nesting_depth < 0:
            raise ValueError("嵌套深度不能为负数")
        if nesting_depth > 0 and total_length > 0 and nesting_depth > total_length // 2:
            raise ValueError("嵌套深度不能大于总长度的一半")

        self.used_brackets = self.brackets[:n_types]

        game_data_list = []
        generated_sequences = set()
        total_attempts = 0

        while len(game_data_list) < num_of_questions and total_attempts < max_attempts:
            try:
                total_attempts += 1

                current_total_length = total_length
                if current_total_length <= 0:
                    current_total_length = random.randint(4 * n_types, 8 * n_types) * 2
                elif current_total_length % 2 != 0:
                    current_total_length -= 1

                current_fill_length = to_fill_length
                if current_fill_length <= 0:
                    current_fill_length = random.randint(
                        max(1, int(current_total_length * 0.2)),
                        min(int(current_total_length * 0.5), current_total_length // 2),
                    )

                cut_point = current_total_length - current_fill_length

                sequence = self._generate_valid_sequence_with_unique_completion(
                    current_total_length, cut_point, nesting_depth
                )

                if sequence in generated_sequences:
                    continue

                generated_sequences.add(sequence)

                question_sequence = sequence[:cut_point]

                question = prompt_dyck_language(question_sequence, random.choice([True, False]))

                game_data = Data(
                    question=question,
                    answer=sequence,
                    metadata={
                        "trace_id": str(uuid.uuid4()),
                        "full_sequence": sequence,
                        "question_sequence": question_sequence,
                        "n_types": n_types,
                        "total_length": current_total_length,
                        "fill_length": current_fill_length,
                        "nesting_depth": nesting_depth,
                    },
                )

                game_data_list.append(game_data)

            except Exception:
                continue

        return game_data_list

    def extract_answer(self, test_solution: str) -> str:
        if not test_solution:
            return ""

        def clean_text(text: str) -> str:
            text = "".join(text.split())

            text = text.replace("\\n", "")
            text = text.replace("\\t", "")
            text = text.replace("\\r", "")
            text = text.replace("\\\\", "\\")

            if len(text) >= 2:
                if text.startswith('"') and text.endswith('"'):
                    text = text[1:-1]
                elif text.startswith("'") and text.endswith("'"):
                    text = text[1:-1]

            return text

        def is_valid_sequence(text: str) -> bool:
            if not text:
                return False

            bracket_chars = set("()[]{}<>")
            if not all(c in bracket_chars for c in text):
                return False

            stack = []
            bracket_pairs = {")": "(", "]": "[", "}": "{", ">": "<"}

            for char in text:
                if char in "([{<":
                    stack.append(char)
                elif char in ")]}>":
                    if stack and stack[-1] == bracket_pairs[char]:
                        stack.pop()
                    else:
                        return False

            return len(stack) == 0

        cleaned_text = clean_text(test_solution)

        max_length = 0
        best_sequence = ""

        for i in range(len(cleaned_text)):
            for j in range(i + 2, len(cleaned_text) + 1):
                potential_sequence = cleaned_text[i:j]
                if is_valid_sequence(potential_sequence) and len(potential_sequence) > max_length:
                    max_length = len(potential_sequence)
                    best_sequence = potential_sequence

        if best_sequence:
            return best_sequence

        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dyck Language游戏生成器")
    parser.add_argument("--num_of_data", type=int, default=100, help="生成的题目数量")
    parser.add_argument("--max_attempts", type=int, default=1000, help="每个题目的最大尝试次数")
    parser.add_argument("--n_types", type=int, default=3, help="使用的括号类型数量（1-6）")
    parser.add_argument("--total_length", type=int, default=0, help="完整序列的总长度，如果为0则随机生成")
    parser.add_argument("--to_fill_length", type=int, default=0, help="需要填充的长度，如果为0则随机生成")
    args = parser.parse_args()

    data_dir = pathlib.Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    game = DyckLanguage()

    game_data_list = game.generate(
        num_of_questions=args.num_of_data,
        max_attempts=args.max_attempts,
        n_types=args.n_types,
        total_length=args.total_length,
        to_fill_length=args.to_fill_length,
    )

    nested_dir = (
        data_dir
        / f"num_of_data_{args.num_of_data}"
        / f"n_types_{args.n_types}"
        / f"total_length_{args.total_length}"
        / f"to_fill_length_{args.to_fill_length}"
    )
    if not nested_dir.exists():
        nested_dir.mkdir(parents=True, exist_ok=True)

    output_file = nested_dir / f"dyck_language_{args.num_of_data}.jsonl"

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for game_data in game_data_list:
                f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")
    except Exception:
        pass
