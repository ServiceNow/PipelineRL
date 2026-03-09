import argparse
import json
import pathlib
import random
import re
import uuid
from typing import List, Optional

from .....base.data import Data
from .....games.base.game import Game
from .....games.tasks.math_path.scripts.math_path_prompt import prompt_mathPath
from .....games.tasks.math_path.scripts.math_path_verifier import MathPathVerifier


class MathPath(Game):
    def __init__(self):
        super().__init__("Math_Path", MathPathVerifier)

    def generate(self, num_of_questions: int = 100, max_attempts: int = 10000, n: int = 4, x_prob: int = 100):
        if n <= 2:
            raise ValueError("运算符号数量必须大于2")
        if x_prob <= 0 or x_prob > 100:
            raise ValueError("x_prob难度系数必须大于0，小于等于100.")

        game_data_list = []
        generated_matrices = set()
        attempts = 0

        while len(game_data_list) < num_of_questions and attempts < max_attempts:
            try:
                attempts += 1

                ref_expr, query_expr = self._generate_valid_expr(n=n, x_prob=x_prob)

                if ref_expr in generated_matrices:
                    continue
                generated_matrices.add(ref_expr)

                is_chinese = random.choice([True, False])

                question = prompt_mathPath(query_expr=query_expr, is_chinese=is_chinese)

                game_data = Data(
                    question=question,
                    answer=ref_expr,
                    metadata={
                        "trace_id": str(uuid.uuid4()),
                        "ref_expr": ref_expr,
                        "query_expr": query_expr,
                        "n": n,
                        "x_prob": x_prob,
                    },
                )

                game_data_list.append(game_data)

            except Exception:
                continue

        return game_data_list

    def _generate_valid_expr(self, n: int = 3, x_prob: int = 100):
        class ParserError(Exception):
            pass

        def generate_equation(num_operations: int) -> str:
            while True:
                numbers = [str(random.randint(0, 9)) for _ in range(num_operations + 1)]
                operators = [random.choice(["+", "-", "*", "/", "%"]) for _ in range(num_operations)]
                expr_parts = [numbers[0]]
                for op, num in zip(operators, numbers[1:]):
                    expr_parts.extend([op, num])
                expr = " ".join(expr_parts)
                expr = add_parentheses(expr)
                try:
                    value = evaluate_expression(expr)
                    return expr, value
                except ParserError:
                    continue

        def add_parentheses(expr: str) -> str:
            tokens = tokenize(expr)
            for _ in range(random.randint(0, 2)):
                tokens = attempt_add_parentheses(tokens)
            return remove_redundant_parentheses(detokenize(tokens))

        def remove_redundant_parentheses(expr):
            if "((" in expr and "))" in expr:
                expr = expr.replace("((", "(")
                expr = expr.replace("))", ")")
                return expr
            return expr

        def attempt_add_parentheses(tokens: List[str]) -> List[str]:
            valid_ops = [i for i, t in enumerate(tokens) if t in "+-*/%" and i > 0 and i < len(tokens) - 1]
            if not valid_ops:
                return tokens
            pos = random.choice(valid_ops)
            left_pos = pos - 1
            right_pos = pos + 1
            if left_pos < 0 or right_pos >= len(tokens):
                return tokens
            if not tokens[left_pos].isdigit() or not tokens[right_pos].isdigit():
                return tokens
            new_tokens = tokens[:left_pos] + ["("] + tokens[left_pos : right_pos + 1] + [")"] + tokens[right_pos + 1 :]
            return new_tokens

        def tokenize(expr: str) -> List[str]:
            tokens = []
            current = []
            for c in expr:
                if c.isspace():
                    if current:
                        tokens.append("".join(current))
                        current = []
                elif c in "()+-*/%":
                    if current:
                        tokens.append("".join(current))
                        current = []
                    tokens.append(c)
                else:
                    current.append(c)
            if current:
                tokens.append("".join(current))
            return tokens

        def detokenize(tokens: List[str]) -> str:
            expr = []
            for i, token in enumerate(tokens):
                if token in "()":
                    expr.append(token)
                else:
                    if expr and expr[-1] not in "+-*/%(":
                        expr.append(" ")
                    expr.append(token)
            return "".join(expr).replace("( ", "(").replace(" )", ")")

        class Tokenizer:
            def __init__(self, tokens: List[str]):
                self.tokens = tokens
                self.pos = 0

            def next_token(self) -> Optional[str]:
                if self.pos < len(self.tokens):
                    token = self.tokens[self.pos]
                    self.pos += 1
                    return token
                return None

            def peek_token(self) -> Optional[str]:
                if self.pos < len(self.tokens):
                    return self.tokens[self.pos]
                return None

        def evaluate_expression(expr: str) -> int:
            tokens = tokenize(expr)
            tokenizer = Tokenizer(tokens)
            try:
                value = parse_expression(tokenizer)
                if tokenizer.peek_token() is not None:
                    raise ParserError("Unexpected tokens")
                return value
            except (ParserError, ValueError) as e:
                raise ParserError() from e

        def parse_expression(tokenizer: Tokenizer) -> int:
            value = parse_term(tokenizer)
            while True:
                op = tokenizer.peek_token()
                if op in ("+", "-"):
                    tokenizer.next_token()
                    right = parse_term(tokenizer)
                    value = value + right if op == "+" else value - right
                else:
                    break
            return value

        def parse_term(tokenizer: Tokenizer) -> int:
            value = parse_factor(tokenizer)
            while True:
                op = tokenizer.peek_token()
                if op in ("*", "/", "%"):
                    tokenizer.next_token()
                    right = parse_factor(tokenizer)
                    if op == "*":
                        value *= right
                    elif op == "/":
                        if right == 0:
                            raise ParserError("Division by zero")
                        if value % right != 0:
                            raise ParserError("Non-integer division")
                        value = value // right
                    elif op == "%":
                        if right == 0:
                            raise ParserError("Mod by zero")
                        value %= right
                else:
                    break
            return value

        def parse_factor(tokenizer: Tokenizer) -> int:
            token = tokenizer.next_token()
            if token is None:
                raise ParserError("Unexpected end of expression")
            if token == "(":
                value = parse_expression(tokenizer)
                if tokenizer.next_token() != ")":
                    raise ParserError("Missing closing parenthesis")
                return value
            elif token.isdigit():
                return int(token)
            else:
                raise ParserError(f"Unexpected token: {token}")

        def replace_num(expr, value, x_prob):
            x_prob /= 100
            replaced_expr = ""
            for s in expr:
                if s == " ":
                    continue
                elif "0" <= s <= "9":
                    if random.random() <= x_prob:
                        replaced_expr += "?"
                    else:
                        replaced_expr += s
                else:
                    replaced_expr += s
            return f"{expr} = {value}", f"{replaced_expr} = {value}"

        expr, value = generate_equation(n)
        ref_expr, query_expr = replace_num(expr, value, x_prob)
        return ref_expr, query_expr

    def extract_answer(self, test_solution: str):
        if not test_solution:
            return ""

        code_block_pattern = r"\[\[(.*?)\]\]"
        code_matches = re.findall(code_block_pattern, test_solution)

        if code_matches:
            operation_expression = code_matches[-1].strip()
            return operation_expression

        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Math_Path游戏生成器")
    parser.add_argument("--num_of_data", type=int, default=100, help="生成的题目数量")
    parser.add_argument("--max_attempts", type=int, default=10000, help="每个题目的最大尝试次数")
    parser.add_argument("--n", type=int, default=4, help="运算符号的数量")
    parser.add_argument("--x_prob", type=int, default=1, help="难度系数")
    args = parser.parse_args()

    base_data_dir = pathlib.Path(__file__).parent.parent / "data"
    if not base_data_dir.exists():
        base_data_dir.mkdir(parents=True, exist_ok=True)

    game = MathPath()

    game_data_list = game.generate(
        num_of_questions=args.num_of_data,
        max_attempts=args.max_attempts,
        n=args.n,
        x_prob=args.x_prob,
    )

    nested_dir = base_data_dir / f"num_of_data_{args.num_of_data}" / f"n_{args.n}" / f"x_{args.x_prob}"
    if not nested_dir.exists():
        nested_dir.mkdir(parents=True, exist_ok=True)

    output_file = nested_dir / f"math_path_{args.num_of_data}.jsonl"

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for game_data in game_data_list:
                f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")
    except Exception:
        pass
