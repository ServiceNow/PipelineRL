import argparse
import json
import pathlib
import random
import re

import sympy as sp
from .....base.data import Data
from .....games.base.game import Game
from .....games.tasks.operation.scripts.operation_prompt import chinese_prompts, english_prompts
from .....games.tasks.operation.scripts.operation_verifier import OperationVerifier
from tqdm import tqdm


class Operation(Game):
    def __init__(
        self,
        num_symbols_range=(1, 3),
        max_operands=5,
        condition_rate=0.5,
        bracket_rate=0.3,
        abs_rate=0.4,
        definition_num_symbols_max=4,
        language="mixed",
    ):
        super().__init__("Operation", OperationVerifier)
        self.condition_rate = condition_rate
        self.bracket_rate = bracket_rate
        self.abs_rate = abs_rate
        self.num_symbols_min = num_symbols_range[0]
        self.num_symbols_max = num_symbols_range[1]
        self.max_operands = max_operands
        self.language = language
        self.definition_num_symbols_max = definition_num_symbols_max

        self.symbols = ["△", "▽", "◇", "○", "☆", "◎", "□", "♡", "♢", "⊕", "⊗", "⊙"]

        self.base_operations = ["+", "-", "*", "/", "%", "**"]

        self.condition_types = [
            "x和y都是偶数",
            "x和y都是奇数",
            "x是偶数",
            "x是奇数",
            "y是偶数",
            "y是奇数",
            "x大于y",
            "x小于y",
            "x等于y",
            "x是3的倍数",
            "y是3的倍数",
            "x是5的倍数",
            "y是5的倍数",
            "x和y的和是偶数",
            "x和y的和是奇数",
        ]
        self.condition2english = {
            "x和y都是偶数": "x and y are both even",
            "x和y都是奇数": "x and y are both odd",
            "x是偶数": "x is even",
            "x是奇数": "x is odd",
            "y是偶数": "y is even",
            "y是奇数": "y is odd",
            "x大于y": "x is greater than y",
            "x小于y": "x is less than y",
            "x等于y": "x is equal to y",
            "x是3的倍数": "x is a multiple of 3",
            "y是3的倍数": "y is a multiple of 3",
            "x是5的倍数": "x is a multiple of 5",
            "y是5的倍数": "y is a multiple of 5",
            "x和y的和是偶数": "the sum of x and y is even",
            "x和y的和是奇数": "the sum of x and y is odd",
        }

        self.condition_checks = {
            "x和y都是偶数": lambda x, y: x % 2 == 0 and y % 2 == 0,
            "x和y都是奇数": lambda x, y: x % 2 != 0 and y % 2 != 0,
            "x是偶数": lambda x, y: x % 2 == 0,
            "x是奇数": lambda x, y: x % 2 != 0,
            "y是偶数": lambda x, y: y % 2 == 0,
            "y是奇数": lambda x, y: y % 2 != 0,
            "x大于y": lambda x, y: x > y,
            "x小于y": lambda x, y: x < y,
            "x等于y": lambda x, y: x == y,
            "x是3的倍数": lambda x, y: x % 3 == 0,
            "y是3的倍数": lambda x, y: y % 3 == 0,
            "x是5的倍数": lambda x, y: x % 5 == 0,
            "y是5的倍数": lambda x, y: y % 5 == 0,
            "x和y的和是偶数": lambda x, y: (x + y) % 2 == 0,
            "x和y的和是奇数": lambda x, y: (x + y) % 2 != 0,
        }

    def generate(self, num_of_questions=100, max_attempts=1000):
        outputs = []

        total_attempts = 0
        tqdm_bar = tqdm(total=num_of_questions, desc="生成题目")
        while len(outputs) < num_of_questions and total_attempts < max_attempts:
            total_attempts += 1

            result = self.generate_problem()

            if result is not None:
                try:
                    if abs(float(result["answer"])) < 100000:
                        outputs.append(
                            Data(
                                question=result["question"],
                                answer=result["answer"],
                                difficulty=1,
                                metadata=result["metadata"],
                            )
                        )
                        tqdm_bar.update(1)
                    else:
                        continue
                except Exception:
                    pass

        tqdm_bar.close()

        return outputs

    def generate_problem(self):
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("生成问题超时")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(3)

        try:
            num_symbols = random.randint(self.num_symbols_min, self.num_symbols_max)

            selected_symbols = random.sample(self.symbols, num_symbols)

            symbol_definitions = {}
            for symbol in selected_symbols:
                symbol_definitions[symbol] = self._create_symbol_definition(symbol_definitions)

            for i, symbol in enumerate(selected_symbols):
                precedence = random.randint(1, 5)
                symbol_definitions[symbol]["precedence"] = precedence

            expression, operands = self._generate_expression(list(symbol_definitions.keys()))

            result, simplified_expr = self._evaluate_expression(expression, symbol_definitions)

            if isinstance(result, str) and (result == "计算超时" or result == "计算错误"):
                raise TimeoutError("计算结果超时或错误")

            question = self._format_question(expression, symbol_definitions)

            signal.alarm(0)

            return {
                "question": question,
                "answer": str(result),
                "metadata": {
                    "expression": expression,
                    "symbol_definitions": symbol_definitions,
                    "result": result,
                    "simplified_expr": simplified_expr,
                },
            }
        except TimeoutError:
            signal.alarm(0)
            return None
        except Exception:
            signal.alarm(0)
            return None

    def _create_symbol_definition(self, symbol_definitions=None):
        if symbol_definitions is None:
            symbol_definitions = {}

        definition = {"conditions": [], "associativity": "left", "precedence": 0}
        if random.random() < self.condition_rate:
            selected_conditions = random.choice(self.condition_types)

            operation = self._generate_random_operation(symbol_definitions)
            definition["conditions"].append({"condition": selected_conditions, "operation": operation})

        definition["default_operation"] = self._generate_random_operation(symbol_definitions)
        return definition

    def _generate_random_operation(self, symbol_definitions=None):
        basic_ops = ["+", "-", "*", "/", "**"]
        weights = [0.3, 0.3, 0.25, 0.1, 0.05]

        if symbol_definitions is None:
            symbol_definitions = {}

        custom_symbols = list(symbol_definitions.keys())

        while True:
            num_variables = random.randint(2, self.definition_num_symbols_max)
            num_constants = random.randint(0, self.definition_num_symbols_max - num_variables)
            # total_elements = num_variables + num_constants

            elements = []
            for _ in range(num_variables):
                elements.append(random.choice(["x", "y"]))
            for _ in range(num_constants):
                elements.append(str(random.randint(1, 5)))

            random.shuffle(elements)

            operators = []

            is_custom = False
            for _ in range(len(elements) - 1 - len(operators)):
                if custom_symbols and random.random() < 0.2 and not is_custom:
                    is_custom = True
                    operators.append(random.choice(custom_symbols))
                else:
                    operators.append(random.choices(basic_ops, weights=weights)[0])
            random.shuffle(operators)

            operators = [""] + operators

            if random.random() < self.bracket_rate:
                position1 = random.choice(range(len(operators)))
                position2 = random.choice(range(len(operators)))
                left_position = min(position1, position2)
                right_position = max(position1, position2)
                elements[left_position] = "(" + elements[left_position]
                elements[right_position] = elements[right_position] + ")"
                if random.random() < self.abs_rate:
                    elements[left_position] = "abs" + elements[left_position]

            expression = ""
            for i in range(len(operators)):
                expression += f" {operators[i]} {elements[i]}"

            if is_custom:
                original_expression = expression
                expression = self._simplify_mix_expression(expression, symbol_definitions)

            x, y = sp.symbols("x y")

            expr_str = expression
            if "abs(" in expr_str:
                expr_str = expr_str.replace("abs(", "Abs(")

            expr = sp.sympify(expr_str)

            simplified = sp.simplify(expr)

            variables = simplified.free_symbols
            has_x_y = sp.Symbol("x") in variables and sp.Symbol("y") in variables

            is_constant = simplified.is_constant()

            if has_x_y and not is_constant:
                if is_custom:
                    return original_expression
                else:
                    return str(simplified)

    def check_number(self, token):
        try:
            return float(token)
        except Exception:
            return "Not number"

    def _simplify_mix_expression(self, expression, symbol_definitions):
        if not expression:
            return expression

        expression = expression.replace("abs(", "Abs(")

        tokens = re.findall(
            r"Abs\(|\*\*|\-?\d+\.\d+(?:[eE][\+\-]?\d+)?|\-?\d+(?:[eE][\+\-]?\d+)?|\(|\)|[a-zA-Z]+|[\+\-\*/\^%]|[^\s\w\+\-\*/\^%\(\)]+",
            expression,
        )

        precedence = {"+": 1, "-": 1, "*": 2, "/": 2, "%": 2, "^": 3, "**": 3}

        for symbol, definition in symbol_definitions.items():
            precedence[symbol] = definition["precedence"]

        operand_stack = []
        operator_stack = []

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if self.check_number(token) != "Not number" or token.isalpha():
                operand_stack.append(token)

            elif token == "Abs(" or token == "(":
                operator_stack.append(token)

            elif token == ")":
                while operator_stack and operator_stack[-1] not in ["(", "Abs("]:
                    self._process_top_operator(operand_stack, operator_stack, symbol_definitions)

                if operator_stack:
                    left_bracket = operator_stack.pop()

                    if left_bracket == "Abs(":
                        top_operand = operand_stack.pop()
                        operand_stack.append(f"Abs({top_operand})")
                else:
                    raise ValueError("括号不匹配")

            else:
                while (
                    operator_stack
                    and operator_stack[-1] not in ["(", "Abs("]
                    and (
                        token not in precedence
                        or (operator_stack[-1] in precedence and precedence[operator_stack[-1]] >= precedence[token])
                    )
                ):
                    self._process_top_operator(operand_stack, operator_stack, symbol_definitions)

                operator_stack.append(token)
            i += 1

        while operator_stack:
            if operator_stack[-1] in ["(", "Abs("]:
                operator_stack.pop()
            else:
                self._process_top_operator(operand_stack, operator_stack, symbol_definitions)

        if len(operand_stack) == 1:
            return operand_stack[0]
        else:
            raise ValueError(f"表达式解析错误: {operand_stack}")

    def _check_operator(self, operand, operators):
        for operator in operators:
            if operator in operand:
                return True
        return False

    def eval_operation(self, operation):
        try:
            return eval(operation)
        except Exception:
            return "Can't eval"

    def _check_custom_operator(self, operand, operators):
        for operator in operators:
            if operator in operand:
                return True
        return False

    def check_int(self, operand):
        try:
            return int(operand)
        except Exception:
            return "Not int"

    def _process_top_operator(self, operand_stack, operator_stack, symbol_definitions):
        operators = ["+", "-", "*", "/", "%", "^", "**"] + list(symbol_definitions.keys())
        if len(operator_stack) == 0 or len(operand_stack) < 2:
            return

        operator = operator_stack.pop()
        right_operand = operand_stack.pop()
        left_operand = operand_stack.pop()
        if self._check_operator(right_operand, operators):
            right_operand = f"({right_operand})"
        if self._check_operator(left_operand, operators):
            left_operand = f"({left_operand})"

        if operator in ["+", "-", "*", "/", "%", "^", "**"]:
            if operator == "+":
                result = f"{left_operand} + {right_operand}"
            elif operator == "-":
                result = f"{left_operand} - {right_operand}"
            elif operator == "*":
                result = f"{left_operand} * {right_operand}"
            elif operator == "/":
                result = f"{left_operand} / {right_operand}"
            elif operator == "%":
                result = f"{left_operand} % {right_operand}"
            elif operator in ["^", "**"]:
                result = f"{left_operand} ** {right_operand}"

        elif operator in symbol_definitions:
            definition = symbol_definitions[operator]
            if (
                self.eval_operation(left_operand) != "Can't eval"
                and self.eval_operation(right_operand) != "Can't eval"
                and definition["conditions"] != []
            ):
                left_operand = str(self.eval_operation(left_operand))
                right_operand = str(self.eval_operation(right_operand))
                condition = definition["conditions"][0]["condition"]
                if self.check_int(left_operand) == "Not int" or self.check_int(right_operand) == "Not int":
                    operation = definition["default_operation"]
                else:
                    if self._check_condition(int(left_operand), int(right_operand), condition):
                        operation = definition["conditions"][0]["operation"]
                    else:
                        operation = definition["default_operation"]
            else:
                operation = definition["default_operation"]

            result = operation.replace("x", "LEFT").replace("y", "RIGHT")
            result = result.replace("LEFT", left_operand).replace("RIGHT", right_operand)
            if self._check_custom_operator(result, symbol_definitions):
                result = self._simplify_mix_expression(result, symbol_definitions)

        result = f"({result})"

        operand_stack.append(result)

    def _generate_expression(self, symbols):
        num_operands = random.randint(len(symbols), len(symbols) + 3)

        operands = [random.randint(1, 10) for _ in range(num_operands)]

        components = []
        operators = symbols
        for i in range(num_operands - len(symbols) - 1):
            operators.append(random.choice(self.base_operations + symbols))
        random.shuffle(operators)

        for i in range(num_operands):
            components.append(str(operands[i]))

            if i < num_operands - 1:
                components.append(operators[i])

        expression = " ".join(components)

        return expression, operands

    def _evaluate_expression(self, expression, symbol_definitions):
        simplified_expr = self._simplify_mix_expression(expression, symbol_definitions)

        try:
            if "Abs(" in simplified_expr:
                simplified_expr = simplified_expr.replace("Abs(", "abs(")

            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("计算超时")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)

            try:
                result = eval(simplified_expr)

                signal.alarm(0)

                if isinstance(result, float) and result.is_integer():
                    return int(result), simplified_expr
                elif isinstance(result, float):
                    return result, simplified_expr
                else:
                    return result, simplified_expr

            except TimeoutError:
                return "计算超时", ""
            finally:
                signal.alarm(0)

        except Exception:
            return "计算错误", ""

    def _check_condition(self, x, y, condition):
        if condition in self.condition_checks:
            return self.condition_checks[condition](x, y)
        return False

    def _format_question(self, expression, symbol_definitions):
        if self.language == "mixed":
            language = random.choice(["chinese", "english"])
        else:
            language = self.language
        if language == "chinese":
            question = "定义 "
        else:
            question = "Define "

        for i, (symbol, definition) in enumerate(symbol_definitions.items()):
            if i > 0:
                if language == "chinese":
                    question += "和 "
                else:
                    question += "and "
            question += f"{symbol}"

            if language == "chinese":
                question += "，规则如下：\n"
            else:
                question += "，the rules are as follows:\n"

            for j, condition_def in enumerate(definition["conditions"]):
                condition = condition_def["condition"]
                operation = condition_def["operation"]
                if language == "chinese":
                    question += f"当{condition}时，x {symbol} y = {operation}；\n"
                else:
                    question += f"when {self.condition2english[condition]}, x {symbol} y = {operation}；\n"

            default_operation = definition["default_operation"]
            if len(definition["conditions"]) == 0:
                if language == "chinese":
                    question += f"实数域上，x {symbol} y = {default_operation}。\n"
                else:
                    question += f"on the real number field, x {symbol} y = {default_operation}。\n"
            else:
                if language == "chinese":
                    question += f"其他情况下，x {symbol} y = {default_operation}。\n"
                else:
                    question += f"otherwise, x {symbol} y = {default_operation}。\n"

        if len(symbol_definitions) > 0:
            all_operators = [
                {"symbol": "**", "precedence": 3},
                {"symbol": "*", "precedence": 2},
                {"symbol": "/", "precedence": 2},
                {"symbol": "%", "precedence": 2},
                {"symbol": "+", "precedence": 1},
                {"symbol": "-", "precedence": 1},
            ]

            for symbol, definition in symbol_definitions.items():
                all_operators.append({"symbol": symbol, "precedence": definition["precedence"]})

            all_operators.sort(key=lambda x: x["precedence"], reverse=True)

            if language == "chinese":
                question += "运算优先级："
            else:
                question += "The precedence of operations："
            current_precedence = all_operators[0]["precedence"]
            question += all_operators[0]["symbol"]

            for op in all_operators[1:]:
                if op["precedence"] == current_precedence:
                    question += " = " + op["symbol"]
                else:
                    question += " > " + op["symbol"]
                    current_precedence = op["precedence"]

            question += "。\n"

        if language == "chinese":
            question += "括号具有最高优先级，先计算括号内的表达式。\n"
        else:
            question += "Parentheses have the highest priority, and the expression inside the parentheses is calculated first.\n"

        if language == "chinese":
            question += f"请计算表达式的值: {expression}"
        else:
            question += f"Please calculate the value of the expression: {expression}"

        if language == "chinese":
            question = random.choice(chinese_prompts).format(question=question)
        else:
            question = random.choice(english_prompts).format(question=question)
        return question

    def extract_answer(self, test_answer):
        return self.verifier.extract_answer(test_answer)

    def verify(self, data, test_answer):
        return self.verifier.verify(data, test_answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成符号重定义游戏数据")
    parser.add_argument("--num_of_data", type=int, default=100, help="生成的题目数量")
    parser.add_argument(
        "--num_symbols_range", type=int, nargs=2, default=[4, 4], help="使用的符号数量范围 [最小值, 最大值]"
    )
    parser.add_argument("--definition_num_symbols_max", type=int, default=5, help="定义中使用的符号数量最大值")
    parser.add_argument("--max_operands", type=int, default=5, help="表达式中最大操作数数量")
    parser.add_argument("--condition_rate", type=float, default=0.5, help="条件定义的概率")
    parser.add_argument("--debug", action="store_true", help="运行调试模式")
    parser.add_argument("--output", type=str, default="symbol_redefinition_data.json", help="输出文件名")
    parser.add_argument("--max_attempts", type=int, default=3000, help="每个题目的最大尝试次数")
    parser.add_argument("--language", type=str, default="mixed", help="语言")
    args = parser.parse_args()

    game = Operation(
        num_symbols_range=args.num_symbols_range,
        max_operands=args.max_operands,
        definition_num_symbols_max=args.definition_num_symbols_max,
        condition_rate=args.condition_rate,
    )

    data_dir = pathlib.Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    filename = f"data_sl{args.num_symbols_range[0]}-{args.num_symbols_range[1]}_op{args.max_operands}_dnm{args.definition_num_symbols_max}_cr{args.condition_rate}_{args.language}_n{args.num_of_data}.jsonl"
    output_file = data_dir / filename
    game_data_list = game.generate(num_of_questions=args.num_of_data, max_attempts=args.max_attempts)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for game_data in game_data_list:
                f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")
    except Exception:
        pass
