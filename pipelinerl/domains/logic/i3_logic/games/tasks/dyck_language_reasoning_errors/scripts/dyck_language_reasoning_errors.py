import argparse
import json
import pathlib
import random
import re
import uuid
from typing import List, Tuple

from .....base.data import Data
from .....games.base.game import Game
from .....games.tasks.dyck_language_reasoning_errors.scripts.dyck_language_reasoning_errors_prompt import (
    prompt_dyck_language_reasoning_errors,
)
from .....games.tasks.dyck_language_reasoning_errors.scripts.dyck_language_reasoning_errors_verifier import (
    DyckLanguageReasoningErrorsVerifier,
)


class DyckLanguageReasoningErrors(Game):
    def __init__(self):
        super().__init__("Dyck Language Reasoning Errors", DyckLanguageReasoningErrorsVerifier)

        self.brackets = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]

    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        n_types: int = 3,
        total_length: int = 20,
        n_errors: int = None,
    ):
        if n_types < 1 or n_types > 4:
            raise ValueError("括号种类数量必须在1到4之间")
        if total_length < 2:
            raise ValueError("括号序列的总长度必须至少为2")
        if n_errors is not None and (n_errors < 0 or n_errors > 10):
            raise ValueError("错误思考过程的数量必须在0到10之间")

        game_data_list = []
        generated_sequences = set()
        attempts = 0

        while len(game_data_list) < num_of_questions and attempts < max_attempts:
            try:
                attempts += 1

                selected_brackets = random.sample(self.brackets[:n_types], n_types)

                dyck_sequence = self._generate_valid_dyck_sequence(selected_brackets, total_length)

                if dyck_sequence in generated_sequences:
                    continue

                generated_sequences.add(dyck_sequence)

                thoughts, error_indices = self._generate_thoughts_with_errors(
                    dyck_sequence, n_errors, selected_brackets
                )

                question = prompt_dyck_language_reasoning_errors(dyck_sequence, thoughts)

                answer = self._format_answer(error_indices)

                game_data = Data(
                    question=question,
                    answer=answer,
                    metadata={
                        "trace_id": str(uuid.uuid4()),
                        "dyck_sequence": dyck_sequence,
                        "thoughts": thoughts,
                        "error_indices": error_indices,
                        "n_types": n_types,
                        "total_length": total_length,
                        "n_errors": len(error_indices),
                    },
                )

                game_data_list.append(game_data)

            except Exception:
                continue

        return game_data_list

    def _generate_valid_dyck_sequence(self, selected_brackets: List[Tuple[str, str]], total_length: int) -> str:
        if total_length % 2 != 0:
            total_length += 1

        sequence = []
        stack = []

        remaining = total_length

        while remaining > 0:
            if not stack or (random.random() < 0.5 and remaining > len(stack) * 2):
                bracket_type = random.choice(selected_brackets)

                sequence.append(bracket_type[0])

                stack.append(bracket_type[1])
            else:
                sequence.append(stack.pop())

            remaining -= 1

        while stack:
            sequence.append(stack.pop())

        return "".join(sequence)

    def _generate_thoughts_with_errors(
        self, dyck_sequence: str, n_errors: int = None, selected_brackets: List[Tuple[str, str]] = None
    ) -> Tuple[List[str], List[int]]:
        if n_errors is None:
            n_errors = random.randint(1, 5)

        thoughts = []
        error_indices = []

        thoughts.append("我们应该逐个处理输入并跟踪栈的配置。")
        thoughts.append("栈: 空")

        correct_stack = []

        current_stack = []

        chars_with_indices = list(enumerate(dyck_sequence))

        if n_errors > 0 and n_errors <= len(chars_with_indices):
            error_positions = random.sample(range(len(chars_with_indices)), min(n_errors, len(chars_with_indices)))
        else:
            error_positions = []

        for i, char in enumerate(dyck_sequence):
            if char in "([{<":
                correct_stack.append(char)
            else:
                if correct_stack:
                    last_char = correct_stack.pop()

                    if not self._is_matching_bracket(last_char, char):
                        correct_stack.append(last_char)
                        correct_stack.append(char)

            should_add_error = i in error_positions

            current_stack = correct_stack.copy()

            if should_add_error:
                error_thought, current_stack = self._generate_error_thought(char, current_stack, i, selected_brackets)

                if self._format_stack(current_stack) != self._format_stack(correct_stack):
                    thoughts.append(f"{char} ; 栈: {self._format_stack(current_stack)}")
                    error_indices.append(len(thoughts) - 1)
                else:
                    thoughts.append(f"{char} ; 栈: {self._format_stack(correct_stack)}")
            else:
                thoughts.append(f"{char} ; 栈: {self._format_stack(correct_stack)}")

        if len(error_indices) < n_errors and random.random() < 0.5:
            if correct_stack:
                thoughts.append("现在，我们已经到达结尾。最终栈是空的。")
                error_indices.append(len(thoughts) - 1)
            else:
                thoughts.append("现在，我们已经到达结尾。最终栈是不为空的。")
                error_indices.append(len(thoughts) - 1)
        else:
            if correct_stack:
                thoughts.append("现在，我们已经到达结尾。最终栈是不为空的。")
            else:
                thoughts.append("现在，我们已经到达结尾。最终栈是空的。")

        if len(error_indices) == 0 and n_errors > 0:
            error_index = random.randint(2, len(thoughts) - 2)
            thoughts[error_index] = self._make_error_in_thought(thoughts[error_index])
            error_indices.append(error_index)
        elif len(error_indices) > n_errors:
            error_indices = random.sample(error_indices, n_errors)

        numbered_thoughts = []
        for i, thought in enumerate(thoughts):
            numbered_thoughts.append(f"Thought {i + 1}: {thought}")

        error_indices = [i + 1 for i in error_indices]

        return numbered_thoughts, error_indices

    def _generate_error_thought(
        self, char: str, stack: List[str], position: int, selected_brackets: List[Tuple[str, str]]
    ) -> Tuple[str, List[str]]:
        error_type = random.choice(["wrong_pop", "no_pop", "wrong_push", "stack_corruption"])

        original_stack = stack.copy()
        max_attempts = 5

        open_brackets = [bracket[0] for bracket in selected_brackets]
        close_brackets = [bracket[1] for bracket in selected_brackets]
        all_brackets = open_brackets + close_brackets

        for _ in range(max_attempts):
            modified_stack = original_stack.copy()

            if error_type == "wrong_pop" and modified_stack:
                modified_stack.pop()
                if len(modified_stack) > 0:
                    modified_stack[-1] = random.choice(open_brackets)

            elif error_type == "no_pop" and char in close_brackets:
                if modified_stack:
                    modified_stack[-1] = random.choice(open_brackets)
                else:
                    modified_stack.append(random.choice(open_brackets))

            elif error_type == "wrong_push":
                wrong_char = random.choice(all_brackets)
                modified_stack.append(wrong_char)

            elif error_type == "stack_corruption":
                if modified_stack:
                    idx = random.randint(0, len(modified_stack) - 1)
                    modified_stack[idx] = random.choice(all_brackets)
                else:
                    modified_stack.append(random.choice(open_brackets))

            if self._format_stack(modified_stack) != self._format_stack(original_stack):
                return "引入错误", modified_stack

            error_type = random.choice(["wrong_pop", "no_pop", "wrong_push", "stack_corruption"])

        if original_stack:
            modified_stack = [random.choice(all_brackets) for _ in original_stack]
            return "强制引入错误", modified_stack
        else:
            modified_stack = [random.choice(all_brackets)]
            return "强制引入错误", modified_stack

    def _make_error_in_thought(self, thought: str) -> str:
        parts = thought.split(" ; 栈: ")

        if len(parts) != 2:
            return thought

        char, stack_str = parts

        open_brackets = [bracket[0] for bracket in self.brackets]
        close_brackets = [bracket[1] for bracket in self.brackets]
        all_brackets = open_brackets + close_brackets

        if stack_str == "空":
            new_stack = random.choice(open_brackets)
        elif stack_str == "":
            new_stack = random.choice(open_brackets)
        else:
            if random.random() < 0.5 and len(stack_str) > 0:
                pos = random.randint(0, len(stack_str) - 1)
                new_stack = stack_str[:pos] + stack_str[pos + 1 :]
            else:
                pos = random.randint(0, len(stack_str))
                new_stack = stack_str[:pos] + random.choice(all_brackets) + stack_str[pos:]

        return f"{char} ; 栈: {new_stack}"

    def _is_matching_bracket(self, open_bracket: str, close_bracket: str) -> bool:
        bracket_pairs = dict([self.brackets[i] for i in range(len(self.brackets))])
        return bracket_pairs.get(open_bracket) == close_bracket

    def _format_stack(self, stack: List[str]) -> str:
        if not stack:
            return "空"
        return "".join(stack)

    def _format_answer(self, error_indices: List[int]) -> str:
        if not error_indices:
            return ""

        error_indices.sort()

        return ",".join(map(str, error_indices))

    def extract_answer(self, test_solution: str) -> str:
        if not test_solution:
            return ""

        solution = test_solution.strip()

        if solution == "" or solution.lower() in ["无问题", "no", "无错误", "none", "no errors", "no mistakes"]:
            return ""

        if re.match(r"^[0-9,，]+$", solution):
            return solution.replace("，", ",")

        num_list_pattern = r"([0-9]+(?:[,，][0-9]+)+)"
        num_list_match = re.search(num_list_pattern, solution)
        if num_list_match:
            answer = num_list_match.group(1).strip()
            answer = answer.replace("，", ",")
            return answer

        single_num_pattern = r"(?<!\d)([0-9]+)(?!\d)"
        single_num_match = re.search(single_num_pattern, solution)
        if single_num_match:
            return single_num_match.group(1)

        if len(solution) < 20:
            cleaned = re.sub(r"[^0-9,，]", "", solution)
            if cleaned:
                return cleaned.replace("，", ",")

        all_nums = re.findall(r"(?<!\d)([1-9]\d*)(?!\d)", solution)
        if all_nums:
            unique_nums = sorted(set(map(int, all_nums)))
            return ",".join(map(str, unique_nums))

        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dyck语言推理错误识别游戏生成器")
    parser.add_argument("--num_of_data", type=int, default=100, help="生成的题目数量")
    parser.add_argument("--max_attempts", type=int, default=1000, help="每个题目的最大尝试次数")
    parser.add_argument("--n_types", type=int, default=3, help="括号种类数量 (1-4)")
    parser.add_argument("--total_length", type=int, default=20, help="括号序列的总长度")
    parser.add_argument("--n_errors", type=int, default=None, help="错误思考过程的数量，默认随机1-5个")
    args = parser.parse_args()

    data_dir = pathlib.Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    game = DyckLanguageReasoningErrors()

    game_data_list = game.generate(
        num_of_questions=args.num_of_data,
        max_attempts=args.max_attempts,
        n_types=args.n_types,
        total_length=args.total_length,
        n_errors=args.n_errors,
    )

    base_data_dir = pathlib.Path(__file__).parent.parent / "data"
    if not base_data_dir.exists():
        base_data_dir.mkdir(parents=True, exist_ok=True)

    n_errors_part = f"n_errors_{args.n_errors}" if args.n_errors is not None else "n_errors_random"
    nested_dir = (
        base_data_dir
        / f"num_of_data_{args.num_of_data}"
        / f"n_types_{args.n_types}"
        / f"total_length_{args.total_length}"
        / n_errors_part
    )
    if not nested_dir.exists():
        nested_dir.mkdir(parents=True, exist_ok=True)

    nested_output_file = nested_dir / f"dyck_language_reasoning_errors_{args.num_of_data}.jsonl"

    try:
        with open(nested_output_file, "w", encoding="utf-8") as f:
            for game_data in game_data_list:
                f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")
    except Exception:
        pass
