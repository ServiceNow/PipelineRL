import argparse
import copy
import json
import pathlib
import random
import re
import uuid
from typing import Dict, List, Tuple

from .....base.data import Data
from .....games.base.game import Game
from .....games.tasks.goods_exchange.scripts.goods_exchange_prompt import (
    chinese_categories,
    chinese_colors,
    chinese_names,
    chinese_operations,
    english_categories,
    english_colors,
    english_names,
    english_operations,
    prompt_goods_exchange,
)
from .....games.tasks.goods_exchange.scripts.goods_exchange_verifier import GoodsExchangeVerifier


class GoodsExchange(Game):
    def __init__(self):
        super().__init__("Goods Exchange", GoodsExchangeVerifier)

    def generate(
        self, num_of_questions: int = 100, max_attempts: int = 100, num_people: int = 5, operator_num: int = 3
    ):
        if num_people <= 1:
            raise ValueError("人物数量必须大于1")
        if operator_num <= 0:
            raise ValueError("交换操作数量必须为正整数")

        game_data_list = []
        attempts = 0

        while len(game_data_list) < num_of_questions and attempts < max_attempts:
            try:
                attempts += 1

                is_chinese = random.choice([True, False])

                names = self._generate_names(num_people, is_chinese)

                objects = self._generate_objects(num_people, is_chinese)

                owns_before = self._initialize_ownership(names, objects)

                operations, owns_after = self._generate_operations(
                    names, objects, owns_before, operator_num, is_chinese
                )

                question = prompt_goods_exchange(
                    n=num_people,
                    names=names,
                    objects=objects,
                    owns_before=owns_before,
                    operations=operations,
                    is_chinese=is_chinese,
                )

                answer = self._format_ownership_as_answer(owns_after)

                game_data = Data(
                    question=question,
                    answer=answer,
                    metadata={
                        "trace_id": str(uuid.uuid4()),
                        "num_people": num_people,
                        "operator_num": operator_num,
                        "names": names,
                        "objects": objects,
                        "owns_before": owns_before,
                        "operations": operations,
                        "owns_after": answer,
                        "is_chinese": is_chinese,
                    },
                )

                game_data_list.append(game_data)

            except Exception:
                continue

        return game_data_list

    def _generate_names(self, num_people: int, is_chinese: bool = False) -> List[str]:
        name_pool = chinese_names if is_chinese else english_names
        return random.sample(name_pool, num_people)

    def _generate_objects(self, num_objects: int, is_chinese: bool = False) -> List[str]:
        colors = chinese_colors if is_chinese else english_colors
        categories = chinese_categories if is_chinese else english_categories

        color_range = min(len(colors), num_objects + random.randint(0, 2))
        category_range = min(len(categories), num_objects + random.randint(0, 2))

        selected_colors = random.sample(colors, color_range)
        selected_categories = random.sample(categories, category_range)

        objects = []
        for _ in range(num_objects):
            color = random.choice(selected_colors)
            category = random.choice(selected_categories)
            obj = f"{color}{category}" if is_chinese else f"{color} {category}"

            while obj in objects:
                color = random.choice(selected_colors)
                category = random.choice(selected_categories)
                obj = f"{color}{category}" if is_chinese else f"{color} {category}"

            objects.append(obj)

        return objects

    def _initialize_ownership(self, names: List[str], objects: List[str]) -> Dict[str, str]:
        shuffled_objects = random.sample(objects, len(objects))
        return {name: obj for name, obj in zip(names, shuffled_objects)}

    def _generate_operations(
        self,
        names: List[str],
        objects: List[str],
        initial_ownership: Dict[str, str],
        num_operations: int,
        is_chinese: bool = False,
    ) -> Tuple[List[str], Dict[str, str]]:
        operations = []
        current_ownership = copy.deepcopy(initial_ownership)

        person_to_item = current_ownership
        item_to_person = {item: person for person, item in current_ownership.items()}

        op_templates = chinese_operations if is_chinese else english_operations

        for _ in range(num_operations):
            operation_type = random.choice(list(op_templates.keys()))

            if operation_type == "operation1":
                name1, name2 = random.sample(names, 2)

                item1 = person_to_item[name1]
                item2 = person_to_item[name2]

                person_to_item[name1] = item2
                person_to_item[name2] = item1

                item_to_person[item1] = name2
                item_to_person[item2] = name1

                operations.append(op_templates[operation_type].format(name1=name1, name2=name2))

            elif operation_type == "operation2":
                obj1, obj2 = random.sample(objects, 2)

                owner1 = item_to_person[obj1]
                owner2 = item_to_person[obj2]

                person_to_item[owner1] = obj2
                person_to_item[owner2] = obj1

                item_to_person[obj1] = owner2
                item_to_person[obj2] = owner1

                operations.append(op_templates[operation_type].format(object1=obj1, object2=obj2))

            elif operation_type == "operation3":
                obj1 = random.choice(objects)
                name1 = random.choice([name for name in names if name != item_to_person[obj1]])

                owner1 = item_to_person[obj1]
                item1 = person_to_item[name1]

                person_to_item[owner1] = item1
                person_to_item[name1] = obj1

                item_to_person[obj1] = name1
                item_to_person[item1] = owner1

                operations.append(op_templates[operation_type].format(object1=obj1, name1=name1))

            elif operation_type == "operation4":
                name1 = random.choice(names)
                obj1 = person_to_item[name1]
                obj2 = random.choice([obj for obj in objects if obj != obj1])

                owner2 = item_to_person[obj2]

                person_to_item[name1] = obj2
                person_to_item[owner2] = obj1

                item_to_person[obj1] = owner2
                item_to_person[obj2] = name1

                operations.append(op_templates[operation_type].format(name1=name1, object1=obj1, object2=obj2))

            elif operation_type == "operation5":
                name1, name2 = random.sample(names, 2)
                obj1 = person_to_item[name1]
                obj2 = person_to_item[name2]

                person_to_item[name1] = obj2
                person_to_item[name2] = obj1

                item_to_person[obj1] = name2
                item_to_person[obj2] = name1

                operations.append(
                    op_templates[operation_type].format(name1=name1, name2=name2, object1=obj1, object2=obj2)
                )

            elif operation_type == "operation6":
                name1, name2 = random.sample(names, 2)
                obj1 = person_to_item[name1]
                obj2 = person_to_item[name2]

                person_to_item[name1] = obj2
                person_to_item[name2] = obj1

                item_to_person[obj1] = name2
                item_to_person[obj2] = name1

                operations.append(
                    op_templates[operation_type].format(name1=name1, name2=name2, object1=obj1, object2=obj2)
                )

            elif operation_type == "operation7":
                obj1, obj2 = random.sample(objects, 2)

                owner1 = item_to_person[obj1]
                owner2 = item_to_person[obj2]

                person_to_item[owner1] = obj2
                person_to_item[owner2] = obj1

                item_to_person[obj1] = owner2
                item_to_person[obj2] = owner1

                operations.append(op_templates[operation_type].format(object1=obj1, object2=obj2))

            elif operation_type in ["operation8", "operation9", "operation10"]:
                if operation_type == "operation8":
                    name1, name2 = random.sample(names, 2)
                    operations.append(op_templates[operation_type].format(name1=name1, name2=name2))
                elif operation_type == "operation9":
                    name1 = random.choice(names)
                    obj2 = random.choice([obj for obj in objects if obj != person_to_item[name1]])
                    operations.append(op_templates[operation_type].format(name1=name1, object2=obj2))
                elif operation_type == "operation10":
                    name1 = random.choice(names)
                    obj1 = random.choice([obj for obj in objects if obj != person_to_item[name1]])
                    operations.append(op_templates[operation_type].format(name1=name1, object1=obj1))

                elif operation_type == "operation11":
                    obj1 = random.choice(objects)
                    owner1 = item_to_person[obj1]

                    name1 = random.choice([name for name in names if name != owner1])
                    obj2 = person_to_item[name1]

                    person_to_item[owner1] = obj2
                    person_to_item[name1] = obj1

                    item_to_person[obj1] = name1
                    item_to_person[obj2] = owner1

                    operations.append(op_templates[operation_type].format(name1=name1, object1=obj1))

                elif operation_type == "operation13":
                    if len(names) >= 3:
                        name1, name2, name3 = random.sample(names, 3)

                        obj1 = person_to_item[name1]
                        obj2 = person_to_item[name2]
                        obj3 = person_to_item[name3]

                        person_to_item[name2] = obj1
                        person_to_item[name3] = obj2
                        person_to_item[name1] = obj3

                        item_to_person[obj1] = name2
                        item_to_person[obj2] = name3
                        item_to_person[obj3] = name1

                        operations.append(op_templates[operation_type].format(name1=name1, name2=name2, name3=name3))
                    else:
                        name1, name2 = random.sample(names, 2)

                        item1 = person_to_item[name1]
                        item2 = person_to_item[name2]

                        person_to_item[name1] = item2
                        person_to_item[name2] = item1

                        item_to_person[item1] = name2
                        item_to_person[item2] = name1

                        operations.append(op_templates["operation1"].format(name1=name1, name2=name2))

        return operations, person_to_item

    def _format_ownership_as_answer(self, ownership: Dict[str, str]) -> str:
        formatted_pairs = []
        for person, item in ownership.items():
            formatted_pairs.append(f"('{person}','{item}')")
        return f"({','.join(formatted_pairs)})"

    def extract_answer(self, text):
        if not text:
            return ""

        code_block_pattern = r"```python\s*\n(.*?)\n```"
        code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
        if code_blocks:
            last_block = code_blocks[-1].strip()
            if last_block.startswith("(") and last_block.endswith(")"):
                return last_block
        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="物品交换游戏生成器")
    parser.add_argument("--num_of_data", type=int, default=100, help="生成的题目数量")
    parser.add_argument("--max_attempts", type=int, default=1000, help="每个题目的最大尝试次数")
    parser.add_argument("--num_people", type=int, default=5, help="人物数量")
    parser.add_argument("--operator_num", type=int, default=3, help="交换操作数量")
    args = parser.parse_args()

    data_dir = pathlib.Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    output_dir = (
        data_dir / f"num_people_{args.num_people}/operator_num_{args.operator_num}/num_of_data_{args.num_of_data}"
    )
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "data.jsonl"

    game = GoodsExchange()

    game_data_list = game.generate(
        num_of_questions=args.num_of_data,
        max_attempts=args.max_attempts,
        num_people=args.num_people,
        operator_num=args.operator_num,
    )

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for game_data in game_data_list:
                f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")
    except Exception:
        pass
