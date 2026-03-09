import argparse
import json
import operator
import pathlib
import random
import re
import string
import uuid
from typing import Dict, List, Tuple

from .....base.data import Data
from .....games.base.game import Game
from .....games.tasks.cryptarithm.scripts.cryptarithm_prompt import prompt_cryptarithm
from .....games.tasks.cryptarithm.scripts.cryptarithm_verifier import CryptarithmVerifier


class Cryptarithm(Game):
    def __init__(self):
        super().__init__("Cryptarithm", CryptarithmVerifier)
        self.operators = {"+": operator.add, "-": operator.sub, "*": operator.mul}
        self.operator_symbols = {1: ["+", "-"], 2: ["+", "-"], 3: ["+", "-", "*"]}

    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 1000,
        num_letter: int = 8,
        operator_num: int = 1,
        operator_level: int = 1,
    ):
        if num_letter < 1 or num_letter > 9:
            raise ValueError("不同字母的数量必须在1-9之间")
        if operator_num < 1:
            raise ValueError("操作符数量必须为正整数")
        if operator_level < 1 or operator_level > 3:
            raise ValueError("操作符难度必须在1-3之间")

        game_data_list = []
        attempts = 0

        while len(game_data_list) < num_of_questions and attempts < max_attempts:
            try:
                attempts += 1

                digits = random.sample(range(10), min(num_letter, 10))

                equation_data = self._generate_valid_equation(digits, operator_num, operator_level, max_attempts)

                if not equation_data:
                    continue

                numbers, operators, used_digits_list = equation_data

                all_digits_str = []
                for num in numbers:
                    all_digits_str.extend(list(str(num)))

                unique_digits = set(all_digits_str)
                if len(unique_digits) != num_letter:
                    continue

                if not self._verify_unique_solution(numbers, operators):
                    continue

                letter_words, digit_map = self._convert_to_letters(numbers)

                is_chinese = random.choice([True, False])
                question = self._generate_prompt(letter_words, operators, is_chinese)

                answer = self._construct_answer(numbers, operators)

                game_data = Data(
                    question=question,
                    answer=answer,
                    metadata={
                        "trace_id": str(uuid.uuid4()),
                        "numbers": numbers,
                        "letter_words": letter_words,
                        "operators": operators,
                        "digit_map": digit_map,
                        "num_letter": num_letter,
                        "operator_num": operator_num,
                        "operator_level": operator_level,
                    },
                )

                game_data_list.append(game_data)

            except Exception:
                continue

        return game_data_list

    def _generate_valid_equation(
        self, digits: List[int], operator_num: int, operator_level: int, max_attempts: int = 100
    ) -> Tuple:
        local_attempts = 0

        while local_attempts < max_attempts:
            local_attempts += 1

            available_operators = self.operator_symbols[operator_level]

            operators = [random.choice(available_operators) for _ in range(operator_num)]

            digit_usage_count = {digit: 0 for digit in digits}
            used_digits_list = []

            numbers = []

            for i in range(operator_num + 1):
                if random.random() < 0.1:
                    num_length = random.randint(1, 2)
                else:
                    num_length = random.randint(3, 6)

                number_digits = []

                non_zero_digits = [d for d in digits if d != 0]
                if not non_zero_digits:
                    continue

                first_digit = random.choice(non_zero_digits)
                number_digits.append(first_digit)
                digit_usage_count[first_digit] += 1
                used_digits_list.append(first_digit)

                remaining_length = num_length - 1
                if remaining_length > 0:
                    for _ in range(remaining_length):
                        digit = random.choice(digits)
                        number_digits.append(digit)
                        digit_usage_count[digit] += 1
                        used_digits_list.append(digit)

                number = int("".join(map(str, number_digits)))
                numbers.append(number)

            left_expr = f"{numbers[0]}"
            for i in range(operator_num):
                left_expr += f" {operators[i]} {numbers[i + 1]}"

            try:
                result = self._calculate_equation(numbers[: operator_num + 1], operators)
            except Exception:
                continue

            if result <= 0:
                continue

            result_digits = [int(d) for d in str(result)]

            valid_result = True
            for d in result_digits:
                if d not in digits:
                    valid_result = False
                    break
                digit_usage_count[d] += 1
                used_digits_list.append(d)

            if not valid_result:
                continue

            numbers.append(result)

            # equation = left_expr + f" = {result}"

            unique_used_digits = set(used_digits_list)
            if len(unique_used_digits) < min(len(digits), 4):
                continue

            return (numbers, operators, used_digits_list)

        return None

    def _convert_to_letters(self, numbers: List[int]) -> Tuple[List[str], Dict[str, int]]:
        all_digits = set()
        for num in numbers:
            for digit in str(num):
                all_digits.add(int(digit))

        alphabet = random.sample(string.ascii_uppercase, len(all_digits))

        digit_to_letter = {digit: letter for digit, letter in zip(all_digits, alphabet)}

        letter_words = []
        for num in numbers:
            word = "".join(digit_to_letter[int(d)] for d in str(num))
            letter_words.append(word)

        letter_to_digit = {letter: digit for digit, letter in digit_to_letter.items()}

        return letter_words, letter_to_digit

    def _generate_prompt(self, letter_words: List[str], operators: List[str], is_chinese: bool) -> str:
        return prompt_cryptarithm(letter_words, operators, is_chinese)

    def _verify_unique_solution(self, numbers: List[int], operators: List[str]) -> bool:
        result = self._calculate_equation(numbers[:-1], operators)
        if result != numbers[-1]:
            return False

        all_digits_str = []
        for num in numbers:
            all_digits_str.extend(list(str(num)))

        unique_letters = set(all_digits_str)
        letter_list = list(unique_letters)
        letter_count = len(letter_list)

        first_positions = {}
        for num in numbers:
            num_str = str(num)
            if len(num_str) > 1:
                first_char = num_str[0]
                if first_char not in first_positions:
                    first_positions[first_char] = True

        def evaluate_number(num_str, digit_map):
            result = 0
            for c in num_str:
                result = result * 10 + digit_map[c]
            return result

        def is_equation_valid(digit_map):
            equation_numbers = []
            for num in numbers:
                num_str = str(num)
                equation_numbers.append(evaluate_number(num_str, digit_map))

            result = self._calculate_equation(equation_numbers[:-1], operators)
            return result == equation_numbers[-1]

        def backtrack(index, used_digits, digit_map):
            nonlocal valid_mappings

            if len(valid_mappings) > 1:
                return

            if index == letter_count:
                if is_equation_valid(digit_map):
                    valid_mappings.append(digit_map.copy())
                return

            current_letter = letter_list[index]

            if current_letter in first_positions:
                start_digit = 1
            else:
                start_digit = 0

            for digit in range(start_digit, 10):
                if digit not in used_digits:
                    digit_map[current_letter] = digit
                    used_digits.add(digit)

                    backtrack(index + 1, used_digits, digit_map)

                    used_digits.remove(digit)
                    digit_map.pop(current_letter)

        valid_mappings = []

        backtrack(0, set(), {})

        return len(valid_mappings) == 1

    def _calculate_equation(self, operands: List[int], operators: List[str]) -> int:
        if len(operands) != len(operators) + 1:
            raise ValueError("操作数数量必须比操作符数量多1")

        ops = operators.copy()
        nums = operands.copy()

        i = 0
        while i < len(ops):
            if ops[i] == "*":
                nums[i] = nums[i] * nums[i + 1]

                nums.pop(i + 1)

                ops.pop(i)
            else:
                i += 1

        result = nums[0]
        for i in range(len(ops)):
            if ops[i] == "+":
                result += nums[i + 1]
            elif ops[i] == "-":
                result -= nums[i + 1]

        return result

    def _construct_answer(self, numbers: List[int], operators: List[str]) -> str:
        equation = str(numbers[0])

        for i, op in enumerate(operators):
            equation += f" {op} {numbers[i + 1]}"

        equation += f" = {numbers[-1]}"

        return equation

    def extract_answer(self, test_solution: str):
        if not test_solution:
            return ""

        test_solution = test_solution.replace("THE ANSWER IS", "The answer is")
        test_solution = test_solution.replace("答案是：", "答案是:")
        test_solution = test_solution.replace("答案：", "答案:")

        equation_patterns = [
            r"(\d+(?:\s*(?:\+|\-|\*)\s*\d+)+\s*=\s*-?\d+)",
            r"(\d+(?:\s*(?:\+|\-|\*)\s*\d+)+\s*=\s*-?\d+)[.。]*$",
            r"(\d+\s*(?:\+|\-|\*)\s*\d+\s*=\s*-?\d+)",
        ]

        for pattern in equation_patterns:
            matches = re.findall(pattern, test_solution)
            if matches:
                return matches[-1].strip()

        cn_patterns = [
            r"答案是[：:]\s*([0-9\s\+\-\*=]+)[.。]*$",
            r"答案[：:]\s*([0-9\s\+\-\*=]+)[.。]*$",
            r"我的答案是[：:]\s*([0-9\s\+\-\*=]+)[.。]*$",
            r"正确答案[：:]\s*([0-9\s\+\-\*=]+)[.。]*$",
            r"数字等式是[：:]\s*([0-9\s\+\-\*=]+)[.。]*$",
            r"数字等式为[：:]\s*([0-9\s\+\-\*=]+)[.。]*$",
            r"等式为[：:]\s*([0-9\s\+\-\*=]+)[.。]*$",
            r"等式是[：:]\s*([0-9\s\+\-\*=]+)[.。]*$",
            r"结果是[：:]\s*([0-9\s\+\-\*=]+)[.。]*$",
            r"结果为[：:]\s*([0-9\s\+\-\*=]+)[.。]*$",
        ]

        en_patterns = [
            r"[Tt]he answer is[：:=]\s*([0-9\s\+\-\*=]+)[.。]*$",
            r"[Tt]he answer[：:=]\s*([0-9\s\+\-\*=]+)[.。]*$",
            r"[Aa]nswer[：:=]\s*([0-9\s\+\-\*=]+)[.。]*$",
            r"[Mm]y answer is[：:=]\s*([0-9\s\+\-\*=]+)[.。]*$",
            r"[Tt]he final answer is[：:=]\s*([0-9\s\+\-\*=]+)[.。]*$",
            r"[Tt]he equation is[：:=]\s*([0-9\s\+\-\*=]+)[.。]*$",
            r"[Tt]he result is[：:=]\s*([0-9\s\+\-\*=]+)[.。]*$",
            r"[Tt]he numeric equation is[：:=]\s*([0-9\s\+\-\*=]+)[.。]*$",
            r"[Tt]herefore,\s*([0-9\s\+\-\*=]+)[.。]*$",
            r"[Ss]o,\s*([0-9\s\+\-\*=]+)[.。]*$",
        ]

        patterns = cn_patterns + en_patterns

        for pattern in patterns:
            matches = re.findall(pattern, test_solution, re.DOTALL)
            if matches:
                answer = matches[-1].strip()

                answer = answer.replace("$", "").replace("。", "").replace(".", "")

                if re.match(r"\d+(?:\s*(?:\+|\-|\*)\s*\d+)+\s*=\s*-?\d+", answer):
                    return answer

        lines = test_solution.strip().split("\n")
        for line in reversed(lines):
            equation_match = re.search(r"\d+(?:\s*(?:\+|\-|\*)\s*\d+)+\s*=\s*-?\d+", line)
            if equation_match:
                return equation_match.group(0)

        general_equation_pattern = r"\d+\s*(?:\+|\-|\*)\s*\d+\s*=\s*-?\d+"
        all_equations = re.findall(general_equation_pattern, test_solution)
        if all_equations:
            return all_equations[-1]

        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="密码算术游戏生成器")
    parser.add_argument("--num_of_data", type=int, default=100, help="生成的题目数量")
    parser.add_argument("--max_attempts", type=int, default=1000, help="每个题目的最大尝试次数")
    parser.add_argument("--num_letter", type=int, default=8, help="题目中不同字母的数量(1-9)")
    parser.add_argument("--operator_num", type=int, default=1, help="等式中的计算次数")
    parser.add_argument("--operator_level", type=int, default=1, help="计算字符难度(1=加减,2=加减,3=加减乘)")
    args = parser.parse_args()

    data_dir = pathlib.Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    output_dir = (
        data_dir
        / f"num_letter_{args.num_letter}/operator_num_{args.operator_num}/operator_level_{args.operator_level}/num_of_data_{args.num_of_data}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "data.jsonl"

    game = Cryptarithm()

    game_data_list = game.generate(
        num_of_questions=args.num_of_data,
        max_attempts=args.max_attempts,
        num_letter=args.num_letter,
        operator_num=args.operator_num,
        operator_level=args.operator_level,
    )

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for game_data in game_data_list:
                f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")
    except Exception:
        pass
