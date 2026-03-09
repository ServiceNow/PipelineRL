import argparse
import json
import pathlib
import random
from typing import List, Tuple

from .....base.data import Data
from .....games.base.game import Game
from .....games.tasks.boolean_expressions.scripts.boolean_expressions_prompt import prompts_en, prompts_zh
from .....games.tasks.boolean_expressions.scripts.boolean_expressions_verifier import BooleanExpressionsVerifier


class BooleanExpressions(Game):
    def __init__(self, min_depth=2, max_depth=5, min_options=3, max_options=6):
        super().__init__("BooleanExpressions", BooleanExpressionsVerifier)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_options = min_options
        self.max_options = max_options

        self.TRUE_FACTS_EN = [
            "The capital of France is Paris.",
            "The capital of Japan is Tokyo.",
            "The capital of Nigeria is Abuja.",
            "Water boils at 100 degrees Celsius at sea level.",
            "The Earth rotates around the Sun.",
            "The chemical symbol for gold is Au.",
            "Oxygen is necessary for human survival.",
            "The human body has 206 bones.",
            "Mount Everest is the tallest mountain on Earth.",
            "The Great Wall of China is visible from space.",
            "Sound cannot travel through a vacuum.",
            "Diamonds are made of carbon.",
            "Antarctica is the coldest continent.",
            "A day on Earth is 24 hours.",
            "Humans have five fingers on each hand.",
        ]

        self.FALSE_FACTS_EN = [
            "The capital of India is Mumbai.",
            "The capital of Brazil is Rio de Janeiro.",
            "The capital of USA is New York.",
            "The Sun revolves around the Earth.",
            "Humans have three lungs.",
            "Pi equals exactly 3.",
            "Gold is a liquid at room temperature.",
            "The Great Wall of China was built in one year.",
            "Spiders are insects.",
            "Mars is the closest planet to the Sun.",
            "Sharks are mammals.",
            "Penguins can fly.",
            "The Sahara Desert is in Australia.",
            "Electricity flows faster than light.",
            "The human body has 150 bones.",
        ]

        self.TRUE_FACTS_CN = [
            "法国的首都是巴黎。",
            "日本的首都是东京。",
            "尼日利亚的首都是阿布贾。",
            "水在海平面上沸点是100摄氏度。",
            "地球绕太阳运转。",
            "黄金的化学符号是Au。",
            "氧气对人类生存是必需的。",
            "人体有206块骨头。",
            "珠穆朗玛峰是地球上最高的山脉。",
            "中国长城从太空可见。",
            "声音无法在真空中传播。",
            "钻石由碳元素组成。",
            "南极洲是最冷的大陆。",
            "地球上一天是24小时。",
            "人类每只手有五个手指。",
        ]

        self.FALSE_FACTS_CN = [
            "印度的首都是孟买。",
            "巴西的首都是里约热内卢。",
            "美国的首都是纽约。",
            "太阳绕地球运转。",
            "人类有三个肺。",
            "圆周率等于3。",
            "黄金在室温下是液态的。",
            "中国长城是在一年内建成的。",
            "蜘蛛是昆虫。",
            "火星是距离太阳最近的行星。",
            "鲨鱼是哺乳动物。",
            "企鹅能飞翔。",
            "撒哈拉沙漠在澳大利亚。",
            "电流比光速更快。",
            "人体有150块骨头。",
        ]

        self.TRUE_FACTS = self.TRUE_FACTS_EN
        self.FALSE_FACTS = self.FALSE_FACTS_EN

        self.TEXT_TO_OP_EN = {
            "is greater than": ">",
            "is less than": "<",
            "is greater than or equal to": ">=",
            "is less than or equal to": "<=",
            "is equal to": "==",
            "equals": "==",
            "plus": "+",
            "minus": "-",
            "multiplied by": "*",
            "times": "*",
            "divided by": "/",
            "divided into": "/",
            "is not equal to": "!=",
            "is not": "!=",
        }

        self.TEXT_TO_OP_CN = {
            "大于": ">",
            "小于": "<",
            "大于等于": ">=",
            "小于等于": "<=",
            "等于": "==",
            "加": "+",
            "减": "-",
            "乘以": "*",
            "乘": "*",
            "除以": "/",
            "不等于": "!=",
        }

        self.TEXT_TO_OP = self.TEXT_TO_OP_EN

        self.OPTION_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]

    def extract_answer(self, test_solution: str):
        return self.verifier.extract_answer(test_solution)

    def verify(self, data: Data, test_solution: str):
        return self.verifier.verify(data, test_solution)

    def generate_arithmetic_expr(self):
        ops = ["+", "-", "*"]
        text_ops = list(self.TEXT_TO_OP.keys())
        num1 = random.randint(-12, 12)
        num2 = random.randint(-12, 12)

        use_text_op = random.random() < 0

        if use_text_op:
            op = random.choice(text_ops)
        else:
            op = random.choice(ops)

        if random.random() < 0.3:
            num3 = random.randint(-9, 9)
            while num3 == 0:
                num3 = random.randint(-9, 9)
            expr = f"{num1} - ({num2} / {num3})"
        else:
            if random.random() < 0.6:
                num3 = random.randint(-9, 9)
                num4 = random.randint(-9, 9)

                if use_text_op and random.random() < 0.5:
                    op2 = random.choice(text_ops)
                    if "times" in self.TEXT_TO_OP:
                        expr = f"{num1} times {num2} {op2} {num3} times {num4}"
                    else:
                        expr = f"{num1} 乘以 {num2} {op2} {num3} 乘以 {num4}"
                else:
                    op2 = random.choice(ops)
                    expr = f"{num1} * {num2} {op2} {num3} * {num4}"
            else:
                if use_text_op:
                    expr = f"{num1} {op} {num2}"
                else:
                    expr = f"{num1} {op} {num2}"

        return expr

    def generate_comparison_expr(self):
        expr1 = self.generate_arithmetic_expr()
        expr2 = self.generate_arithmetic_expr()

        comparison_ops = list(self.TEXT_TO_OP.keys())
        op = random.choice(comparison_ops)

        return f"({expr1}) {op} ({expr2})"

    def generate_fact_expr(self):
        if random.random() < 0.5:
            return random.choice(self.TRUE_FACTS)
        else:
            return random.choice(self.FALSE_FACTS)

    def generate_boolean_expr(self, depth=5):
        if depth <= 0 or (depth <= 2 and random.random() < 0.3):
            choice = random.random()
            if choice < 0.4:
                return self.generate_comparison_expr()
            elif choice < 0.7:
                return self.generate_fact_expr()
            else:
                return random.choice(["True", "False"])

        choice = random.random()
        if choice < 0.35:
            sub_expr = self.generate_boolean_expr(depth - 1)
            prefix = "not " * random.randint(1, 3)
            return f"{prefix}({sub_expr})"
        elif choice < 0.7:
            left = self.generate_boolean_expr(depth - 1)
            right = self.generate_boolean_expr(depth - 1)

            if random.random() < 0.4 and depth > 2:
                middle = self.generate_boolean_expr(depth - 2)
                return f"({left}) and ({middle}) and ({right})"
            return f"({left}) and ({right})"
        else:
            left = self.generate_boolean_expr(depth - 1)
            right = self.generate_boolean_expr(depth - 1)

            if random.random() < 0.4 and depth > 2:
                middle = self.generate_boolean_expr(depth - 2)
                return f"({left}) or ({middle}) or ({right})"
            return f"({left}) or ({right})"

    def preprocess_expression(self, expr):
        for fact in self.TRUE_FACTS:
            expr = expr.replace(f'"{fact}"', "True")

            if fact in expr:
                expr = expr.replace(fact, "True")

        for fact in self.FALSE_FACTS:
            expr = expr.replace(f'"{fact}"', "False")

            if fact in expr:
                expr = expr.replace(fact, "False")

        sorted_text_ops = sorted(self.TEXT_TO_OP.keys(), key=len, reverse=True)

        for text in sorted_text_ops:
            op = self.TEXT_TO_OP[text]
            expr = expr.replace(text, op)

        return expr

    def evaluate_expression(self, expr):
        processed_expr = self.preprocess_expression(expr)

        try:
            result = eval(processed_expr)
        except Exception:
            return None
        return result

    def generate_expressions(self, num_expressions: int, depth: int, language: str) -> List[Tuple[str, bool]]:
        expressions = []
        for i in range(num_expressions):
            while True:
                expr = self.generate_boolean_expr(depth)
                value = self.evaluate_expression(expr)
                if value is not None:
                    break

            max_attempts = 5
            attempt = 0
            while (value is None or len(expr) > 500) and attempt < max_attempts:
                expr = self.generate_boolean_expr(depth)
                value = self.evaluate_expression(expr)
                attempt += 1

            if value is not None:
                expressions.append((expr, value))
        if not any(value for _, value in expressions):
            return self.generate_expressions(num_expressions, depth, language)

        return expressions

    def generate_problem(self, language="en"):
        num_options = random.randint(self.min_options, self.max_options)
        depth = random.randint(self.min_depth, self.max_depth)

        expressions = self.generate_expressions(num_options, depth, language)

        if len(expressions) < 2:
            return self.generate_problem(language)

        options = []
        true_options = []

        for i, (expr, value) in enumerate(expressions):
            if i >= len(self.OPTION_LABELS):
                break

            label = self.OPTION_LABELS[i]
            options.append(f"{label}. {expr}")

            if value:
                true_options.append(label)

        true_options.sort()
        answer = ",".join(true_options)

        if language == "cn":
            context = "请评估以下布尔表达式，选择其中为真（True）的选项：\n\n" + "\n".join(options)
            question = "哪些选项的表达式值为真？请列出所有为真的选项标识，用逗号分隔。"

            prompt_template = random.choice(prompts_zh)
        else:
            context = "Evaluate the following boolean expressions and select the ones that are true:\n\n" + "\n".join(
                options
            )
            question = "Which options are true? List all option labels that are true, separated by commas."

            prompt_template = random.choice(prompts_en)

        full_question = prompt_template.format(context=context, question=question)

        return full_question, answer

    def generate(self, num_of_data=100, language="mixed"):
        data_list = []
        for _ in range(num_of_data):
            if language == "mixed":
                curr_lang = random.choice(["cn", "en"])
                self.curr_lang = curr_lang
            else:
                curr_lang = language

            if curr_lang == "cn":
                self.TRUE_FACTS = self.TRUE_FACTS_CN
                self.FALSE_FACTS = self.FALSE_FACTS_CN
                self.TEXT_TO_OP = self.TEXT_TO_OP_CN
                # prompts = prompts_zh
            else:
                self.TRUE_FACTS = self.TRUE_FACTS_EN
                self.FALSE_FACTS = self.FALSE_FACTS_EN
                self.TEXT_TO_OP = self.TEXT_TO_OP_EN
                # prompts = prompts_en

            # prompt_template = random.choice(prompts)

            question, answer = self.generate_problem(curr_lang)

            data_list.append(Data(question=question, answer=answer))

        return data_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成布尔表达式推理游戏数据")
    parser.add_argument("--num_of_data", type=int, default=100, help="生成的题目数量")
    parser.add_argument("--min_depth", type=int, default=2, help="最小表达式深度")
    parser.add_argument("--max_depth", type=int, default=5, help="最大表达式深度")
    parser.add_argument("--min_options", type=int, default=3, help="每个问题的最小选项数量")
    parser.add_argument("--max_options", type=int, default=6, help="每个问题的最大选项数量")
    parser.add_argument("--language", type=str, default="mixed", help="语言：cn, en, mixed")
    args = parser.parse_args()

    data_dir = pathlib.Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    game = BooleanExpressions(
        min_depth=args.min_depth, max_depth=args.max_depth, min_options=args.min_options, max_options=args.max_options
    )

    filename = f"data_depth{args.min_depth}-{args.max_depth}_options{args.min_options}-{args.max_options}_{args.language}_n{args.num_of_data}.jsonl"
    output_file = data_dir / filename

    game_data_list = game.generate(num_of_data=args.num_of_data, language=args.language)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for game_data in game_data_list:
                f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")
    except Exception:
        pass
