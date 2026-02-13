import argparse
import json
import pathlib
import random
from typing import List, Optional, Tuple

from .....base.data import Data
from .....games.base.game import Game
from .....games.tasks.space_reasoning_tree.scripts.items_collection import get_random_items
from .....games.tasks.space_reasoning_tree.scripts.space_reasoning_tree_prompt import prompts_en, prompts_zh
from .....games.tasks.space_reasoning_tree.scripts.space_reasoning_tree_verifier import SpaceReasoningTreeVerifier


class TreeNode:
    def __init__(self, item: str = None):
        self.item = item
        self.parent = None
        self.children = []

    def add_child(self, node: "TreeNode"):
        self.children.append(node)
        node.parent = self

    def get_siblings(self) -> List["TreeNode"]:
        if not self.parent:
            return []
        return [node for node in self.parent.children if node != self]

    def get_cousins(self) -> List["TreeNode"]:
        if not self.parent or not self.parent.parent:
            return []

        cousins = []
        for uncle in self.parent.get_siblings():
            cousins.extend(uncle.children)

        return cousins

    def get_grandchildren(self) -> List["TreeNode"]:
        grandchildren = []
        for child in self.children:
            grandchildren.extend(child.children)
        return grandchildren

    def get_grandfather(self) -> Optional["TreeNode"]:
        if not self.parent or not self.parent.parent:
            return None
        return self.parent.parent


class SpaceReasoningTree(Game):
    def __init__(self, min_nodes=50, max_nodes=300):
        super().__init__("SpaceReasoningTree", SpaceReasoningTreeVerifier)
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes

    def extract_answer(self, test_solution: str):
        return self.verifier.extract_answer(test_solution)

    def verify(self, data: Data, test_solution: str):
        return self.verifier.verify(data, test_solution)

    def build_five_layer_tree(self, num_nodes: int, language: str = "cn") -> Tuple[TreeNode, List[TreeNode]]:
        nodes = [TreeNode() for _ in range(num_nodes)]

        items = get_random_items(num_nodes, language)

        for i, node in enumerate(nodes):
            node.item = items[i]

        root = nodes[0]
        remaining_nodes = nodes[1:]

        level2_count = min(random.randint(3, 6), len(remaining_nodes))
        level2_nodes = remaining_nodes[:level2_count]
        remaining_nodes = remaining_nodes[level2_count:]

        for node in level2_nodes:
            root.add_child(node)

        level3_nodes = []
        for parent in level2_nodes:
            child_count = min(random.randint(2, 5), len(remaining_nodes))
            if child_count == 0:
                continue

            children = remaining_nodes[:child_count]
            remaining_nodes = remaining_nodes[child_count:]

            for child in children:
                parent.add_child(child)
                level3_nodes.append(child)

        level4_nodes = []
        for parent in level3_nodes:
            if random.random() < 0.75 and remaining_nodes:
                child_count = min(random.randint(1, 4), len(remaining_nodes))
                if child_count == 0:
                    continue

                children = remaining_nodes[:child_count]
                remaining_nodes = remaining_nodes[child_count:]

                for child in children:
                    parent.add_child(child)
                    level4_nodes.append(child)

        for parent in level4_nodes:
            if random.random() < 0.6 and remaining_nodes:
                child_count = min(random.randint(1, 3), len(remaining_nodes))
                if child_count == 0:
                    continue

                children = remaining_nodes[:child_count]
                remaining_nodes = remaining_nodes[child_count:]

                for child in children:
                    parent.add_child(child)

        while remaining_nodes:
            potential_parents = level2_nodes + level3_nodes + level4_nodes
            if not potential_parents:
                break

            parent = random.choice(potential_parents)
            child = remaining_nodes.pop(0)
            parent.add_child(child)

        return root, nodes

    def find_valid_cousin_target(self, nodes: List[TreeNode]) -> Optional[TreeNode]:
        valid_targets = []

        for node in nodes:
            cousins = node.get_cousins()

            if cousins and node.parent and node.parent.parent:
                valid_targets.append(node)

        if valid_targets:
            return random.choice(valid_targets)
        return None

    def generate_tree_description(self, nodes: List[TreeNode], target_node: TreeNode, language: str = "en") -> str:
        descriptions = []

        grandfather = target_node.get_grandfather()
        if not grandfather:
            return "无效的目标节点，没有祖父节点"

        info_method = random.choice(["method1", "method2"])

        if info_method == "method1":
            grandchildren = grandfather.get_grandchildren()
            children_of_parent = target_node.parent.children

            grandchildren_items = [node.item for node in grandchildren]
            parent_children_items = [node.item for node in children_of_parent]

            if language == "cn":
                descriptions.append(f"{grandfather.item}的孙子是：{', '.join(grandchildren_items)}。")
            else:
                descriptions.append(
                    f"{grandfather.item} has {len(grandchildren_items)} grandchildren: {', '.join(grandchildren_items)}."
                )

            if language == "cn":
                descriptions.append(f"{target_node.parent.item}的孩子是：{', '.join(parent_children_items)}。")
            else:
                descriptions.append(
                    f"{target_node.parent.item} has {len(parent_children_items)} {'child' if len(parent_children_items) == 1 else 'children'}: {', '.join(parent_children_items)}."
                )

        else:
            grandfather_children = grandfather.children
            descriptions.append(
                f"{grandfather.item}的孩子是：{', '.join([node.item for node in grandfather_children])}。"
            )
            for uncle in grandfather_children:
                children_items = [node.item for node in uncle.children]

                if children_items:
                    if language == "cn":
                        descriptions.append(f"{uncle.item}的孩子是：{', '.join(children_items)}。")
                    else:
                        descriptions.append(
                            f"{uncle.item} has {len(children_items)} {'child' if len(children_items) == 1 else 'children'}: {', '.join(children_items)}."
                        )
                else:
                    if language == "cn":
                        descriptions.append(f"{uncle.item}没有孩子。")
                    else:
                        descriptions.append(f"{uncle.item} has no children.")

        additional_nodes = random.sample(
            [n for n in nodes if n != target_node and n != grandfather and n != target_node.parent],
            random.randint(len(nodes) // 3, len(nodes) // 2),
        )

        for node in additional_nodes:
            info_type = random.choice(["children", "grandchildren", "no_children", "no_grandchildren", "siblings"])

            if info_type == "children" and node.children:
                children_items = [child.item for child in node.children]
                if language == "cn":
                    descriptions.append(f"{node.item}有{len(children_items)}个孩子：{', '.join(children_items)}。")
                else:
                    descriptions.append(
                        f"{node.item} has {len(children_items)} {'child' if len(children_items) == 1 else 'children'}: {', '.join(children_items)}."
                    )

            elif info_type == "grandchildren":
                grandchildren = node.get_grandchildren()
                if grandchildren:
                    gc_items = [gc.item for gc in grandchildren]
                    if language == "cn":
                        descriptions.append(f"{node.item}有{len(gc_items)}个孙子：{', '.join(gc_items)}。")
                    else:
                        if len(gc_items) > 4:
                            gc_by_parent = {}
                            for child in node.children:
                                if child.children:
                                    gc_by_parent[child.item] = [gc.item for gc in child.children]

                            gc_descriptions = []
                            for parent, children in gc_by_parent.items():
                                gc_descriptions.append(
                                    f"{', '.join(children[:-1])}{' and ' if len(children) > 1 else ''}{children[-1]} (whose parent is {parent})"
                                )

                            descriptions.append(
                                f"{node.item} has {len(gc_items)} grandchildren: {' and '.join(gc_descriptions)}."
                            )
                        else:
                            descriptions.append(
                                f"{node.item} has {len(gc_items)} grandchildren: {', '.join(gc_items)}."
                            )

            elif info_type == "no_children" and not node.children:
                if language == "cn":
                    descriptions.append(f"{node.item}没有孩子。")
                else:
                    descriptions.append(f"{node.item} has no children.")

            elif info_type == "no_grandchildren" and not node.get_grandchildren():
                if language == "cn":
                    if node.children:
                        children_items = [child.item for child in node.children]
                        descriptions.append(
                            f"{node.item}没有孙子，但有{len(children_items)}个孩子：{', '.join(children_items)}。"
                        )
                    else:
                        descriptions.append(f"{node.item}没有孙子。")
                else:
                    if node.children:
                        children_items = [child.item for child in node.children]
                        descriptions.append(
                            f"{node.item} has no grandchildren but has {len(children_items)} {'child' if len(children_items) == 1 else 'children'}: {', '.join(children_items)}."
                        )
                    else:
                        descriptions.append(f"{node.item} has no grandchildren.")

            elif info_type == "siblings":
                siblings = node.get_siblings()
                if siblings:
                    sibling_items = [sib.item for sib in siblings]
                    if language == "cn":
                        descriptions.append(f"{node.item}与{', '.join(sibling_items)}是兄弟节点。")
                    else:
                        descriptions.append(
                            f"{node.item}{',' if len(sibling_items) > 1 else ''} and {', '.join(sibling_items[:-1])}{' and ' if len(sibling_items) > 1 else ''}{sibling_items[-1]} are siblings."
                        )

        random.shuffle(descriptions)

        return " ".join(descriptions)

    def generate_problem(self, language="en"):
        num_nodes = random.randint(self.min_nodes, self.max_nodes)

        root, nodes = self.build_five_layer_tree(num_nodes, language)

        target_node = self.find_valid_cousin_target(nodes)

        if not target_node:
            return self.generate_problem(language)

        tree_description = self.generate_tree_description(nodes, target_node, language)

        cousins = target_node.get_cousins()
        cousin_items = sorted([node.item for node in cousins])
        answer = ", ".join(cousin_items)

        if language == "cn":
            context = f"你被给定了一个有{num_nodes}个节点的树结构。{tree_description}"
            question = f"{target_node.item}的堂兄弟节点是什么？你的最终答案必须只包含物品名称。如果有多个物品，请将它们按字母顺序排列并用逗号分隔。"

            prompt_template = random.choice(prompts_zh)
        else:
            context = f"You have been given a tree structure with {num_nodes} nodes. {tree_description}"
            question = f"What is the cousin of the {target_node.item}? Your final answer must be only the object name (e.g., laptop). If there are multiple objects, provide them as a comma-separated list in alphabetical order (e.g., laptop, mug)."

            prompt_template = random.choice(prompts_en)

        full_question = prompt_template.format(context=context, question=question)

        return full_question, answer

    def generate(self, num_of_data=10, language="mixed"):
        outputs = []
        for i in range(num_of_data):
            if language == "mixed":
                now_language = random.choice(["cn", "en"])
            else:
                now_language = language

            question, answer = self.generate_problem(now_language)

            outputs.append(
                Data(
                    question=question,
                    answer=answer,
                    difficulty=2,
                    metadata={
                        "language": now_language,
                        "max_num_nodes": self.max_nodes,
                        "min_num_nodes": self.min_nodes,
                    },
                )
            )

        return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成空间推理树结构游戏数据")
    parser.add_argument("--num_of_data", type=int, default=100, help="生成的题目数量")
    parser.add_argument("--min_nodes", type=int, default=50, help="最小节点数量")
    parser.add_argument("--max_nodes", type=int, default=300, help="最大节点数量")
    parser.add_argument("--language", type=str, default="mixed", help="语言：cn, en, mixed")
    args = parser.parse_args()

    data_dir = pathlib.Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    game = SpaceReasoningTree(min_nodes=args.min_nodes, max_nodes=args.max_nodes)

    filename = f"data_nodes{args.min_nodes}-{args.max_nodes}_{args.language}_n{args.num_of_data}.jsonl"
    output_file = data_dir / filename

    game_data_list = game.generate(num_of_data=args.num_of_data, language=args.language)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for game_data in game_data_list:
                f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")
    except Exception:
        pass
