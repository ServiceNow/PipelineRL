import argparse
import json
import pathlib
import random
from typing import List, Optional, Tuple

from .....base.data import Data
from .....games.base.game import Game
from .....games.tasks.space_reasoning.scripts.items_collection import distribute_items_to_nodes
from .....games.tasks.space_reasoning.scripts.space_reasoning_prompt import prompts_en, prompts_zh
from .....games.tasks.space_reasoning.scripts.space_reasoning_verifier import SpaceReasoningVerifier


class Node:
    def __init__(self, item: str = None):
        self.item = item
        self.connections = {}

    def add_connection(self, direction: str, node: "Node"):
        self.connections[direction] = node


class SpaceReasoning(Game):
    def __init__(self, jump_ratio=0.5, n=5):
        super().__init__("SpaceReasoning", SpaceReasoningVerifier)
        self.jump_ratio = jump_ratio
        self.n = n

    def extract_answer(self, test_solution: str):
        return self.verifier.extract_answer(test_solution)

    def verify(self, data: Data, test_solution: str):
        return self.verifier.verify(data, test_solution)

    def build_square_network(self, n: int, language: str = "cn") -> List[Node]:
        if n <= 0:
            return []

        total_nodes = (n + 1) * (n + 1)

        nodes = [Node() for _ in range(total_nodes)]

        def get_node_index(row, col):
            return row * (n + 1) + col

        for row in range(n + 1):
            for col in range(n + 1):
                current_idx = get_node_index(row, col)

                if col < n:
                    right_idx = get_node_index(row, col + 1)
                    nodes[current_idx].add_connection("右" if language == "cn" else "right", nodes[right_idx])
                    nodes[right_idx].add_connection("左" if language == "cn" else "left", nodes[current_idx])

                if row < n:
                    down_idx = get_node_index(row + 1, col)
                    nodes[current_idx].add_connection("下" if language == "cn" else "down", nodes[down_idx])
                    nodes[down_idx].add_connection("上" if language == "cn" else "up", nodes[current_idx])

        distribute_items_to_nodes(nodes, language)

        if language == "cn":
            description = "你在一个N*N的正方形瓷砖上，每一行都有N个正方形，正方形每个顶点上都有一个独特的物品。你将沿着这些瓷砖的边移动，并在顶点看到物体。"
        else:
            description = "You are on an N*N square grid where each row has N squares. Each vertex of these squares has a unique item. You will move along the edges of these tiles and see objects at the vertices."

        return description, nodes

    def build_diamond_network(self, n: int, language: str = "cn") -> List[Node]:
        if n <= 0:
            return []

        total_nodes = (n + 1) * (n + 2) // 2 + n

        nodes = [Node() for _ in range(total_nodes)]

        row_indices = []
        node_index = 0
        for i in range(1, n + 2):
            row_indices.append(list(range(node_index, node_index + i)))
            node_index += i
        row_indices.append(range(node_index, node_index + n))

        for row in range(n + 1):
            row_nodes = row_indices[row]

            if row < n:
                next_row_nodes = row_indices[row + 1]

                for i in range(len(row_nodes)):
                    nodes[row_nodes[i]].add_connection(
                        "左下" if language == "cn" else "bottom-left", nodes[next_row_nodes[i]]
                    )
                    nodes[next_row_nodes[i]].add_connection(
                        "右上" if language == "cn" else "top-right", nodes[row_nodes[i]]
                    )

                    nodes[row_nodes[i]].add_connection(
                        "右下" if language == "cn" else "bottom-right", nodes[next_row_nodes[i + 1]]
                    )
                    nodes[next_row_nodes[i + 1]].add_connection(
                        "左上" if language == "cn" else "top-left", nodes[row_nodes[i]]
                    )

        for i in range(n + 1):
            if i != n:
                nodes[row_indices[-2][i]].add_connection(
                    "右下" if language == "cn" else "bottom-right", nodes[row_indices[-1][i]]
                )
                nodes[row_indices[-1][i]].add_connection(
                    "左上" if language == "cn" else "top-left", nodes[row_indices[-2][i]]
                )
            if i != 0:
                nodes[row_indices[-2][i]].add_connection(
                    "左下" if language == "cn" else "bottom-left", nodes[row_indices[-1][i - 1]]
                )
                nodes[row_indices[-1][i - 1]].add_connection(
                    "右上" if language == "cn" else "top-right", nodes[row_indices[-2][i]]
                )

        distribute_items_to_nodes(nodes, language)

        if language == "cn":
            description = "构建了一个N层的菱形网络，其中第1层有一个菱形，第2层有2个菱形，第3层有3个菱形，以此类推，第N层有N个菱形，每个顶点上都有一个独特的物品。你将沿着这些瓷砖的边移动，并在每个顶点看到物体。"
        else:
            description = "There is an N-layer diamond network where the 1st layer has one diamond, the 2nd layer has 2 diamonds, the 3rd layer has 3 diamonds, and so on. The Nth layer has N diamonds, with each vertex having a unique item. You will move along the edges and see objects at each vertex."

        return description, nodes

    def build_triangle_network(self, n: int, language: str = "cn") -> List[Node]:
        if n <= 0:
            return []

        total_nodes = (n + 1) * (n + 2) // 2

        nodes = [Node() for _ in range(total_nodes)]

        row_indices = []
        node_index = 0
        for i in range(1, n + 2):
            row_indices.append(list(range(node_index, node_index + i)))
            node_index += i

        for row in range(n + 1):
            row_nodes = row_indices[row]

            if row != 0:
                nodes[row_nodes[0]].add_connection("右" if language == "cn" else "right", nodes[row_nodes[1]])
                nodes[row_nodes[-1]].add_connection("左" if language == "cn" else "left", nodes[row_nodes[-2]])
                for i in range(1, len(row_nodes) - 1):
                    nodes[row_nodes[i]].add_connection("右" if language == "cn" else "right", nodes[row_nodes[i + 1]])
                    nodes[row_nodes[i + 1]].add_connection("左" if language == "cn" else "left", nodes[row_nodes[i]])

            if row < n:
                next_row_nodes = row_indices[row + 1]

                for i in range(len(row_nodes)):
                    nodes[row_nodes[i]].add_connection(
                        "左下" if language == "cn" else "bottom-left", nodes[next_row_nodes[i]]
                    )
                    nodes[next_row_nodes[i]].add_connection(
                        "右上" if language == "cn" else "top-right", nodes[row_nodes[i]]
                    )

                    nodes[row_nodes[i]].add_connection(
                        "右下" if language == "cn" else "bottom-right", nodes[next_row_nodes[i + 1]]
                    )
                    nodes[next_row_nodes[i + 1]].add_connection(
                        "左上" if language == "cn" else "top-left", nodes[row_nodes[i]]
                    )

        distribute_items_to_nodes(nodes, language)

        if language == "cn":
            description = "构建了一个N层的三角形网络，其中第1层有一个三角形，第2层有3个三角形，第3层有5个三角形，以此类推，第N层有2N+1个三角形，每个顶点上都有一个独特的物品。你将沿着这些瓷砖的边移动，并在每个顶点看到物体。"
        else:
            description = "There is an N-layer triangular network where the 1st layer has one triangle, the 2nd layer has 3 triangles, the 3rd layer has 5 triangles, and so on. The Nth layer has 2N+1 triangles, with each vertex having a unique item. You will move along the edges and see objects at each vertex."

        return description, nodes

    def find_path(self, nodes: List[Node], start_idx: int, end_idx: int) -> Optional[List[Tuple[int, Optional[str]]]]:
        if start_idx == end_idx:
            return [(start_idx, None)]

        node_to_idx = {id(node): idx for idx, node in enumerate(nodes)}

        queue = [start_idx]
        visited = {start_idx}

        predecessors = {}

        while queue:
            current_idx = queue.pop(0)
            current_node = nodes[current_idx]

            directions = list(current_node.connections.keys())
            random.shuffle(directions)

            for direction in directions:
                next_node = current_node.connections[direction]
                next_idx = node_to_idx[id(next_node)]

                if next_idx not in visited:
                    visited.add(next_idx)
                    queue.append(next_idx)

                    predecessors[next_idx] = (current_idx, direction)

                    if next_idx == end_idx:
                        path = []
                        current = next_idx

                        while current != start_idx:
                            prev, direction = predecessors[current]
                            path.append((prev, direction))
                            current = prev

                        path.reverse()
                        path.append((end_idx, None))
                        return path

        return None

    def generate_multi_node_path(
        self, nodes: List[Node], waypoint_indices: List[int], language: str = "cn"
    ) -> Tuple[List[Tuple[int, Optional[str]]], str]:
        if len(waypoint_indices) < 2:
            return None, "路径点数量不足，至少需要2个节点"

        complete_path = []
        path_description = ""

        for i in range(len(waypoint_indices) - 1):
            start_idx = waypoint_indices[i]
            end_idx = waypoint_indices[i + 1]

            segment_path = self.find_path(nodes, start_idx, end_idx)
            if not segment_path:
                return (
                    None,
                    "无法找到从节点{start_idx}到节点{end_idx}的路径"
                    if language == "cn"
                    else f"Cannot find a path from node {start_idx} to node {end_idx}",
                )

            if i == 0:
                complete_path.extend(segment_path)
            else:
                if complete_path:
                    next_direction = None

                    for j in range(len(segment_path) - 1):
                        if segment_path[j][0] == complete_path[-1][0]:
                            next_direction = segment_path[j][1]
                            break

                    if next_direction:
                        last_idx = complete_path[-1][0]
                        complete_path[-1] = (last_idx, next_direction)

                    complete_path.extend(segment_path[1:])

            if i == 0:
                if language == "cn":
                    path_description += f"看到了一个{nodes[start_idx].item}，"
                else:
                    path_description += f"You see a {nodes[start_idx].item}, "

            directions = []
            curr_dir = None
            dir_count = 0

            for j in range(len(segment_path) - 1):
                direction = segment_path[j][1]

                if direction != curr_dir:
                    if curr_dir and dir_count > 0:
                        if language == "cn":
                            dir_desc = f"然后向{curr_dir}方向{dir_count}步"
                        else:
                            dir_desc = f"then {dir_count} step(s) {curr_dir}"
                        directions.append(dir_desc)

                    curr_dir = direction
                    dir_count = 1
                else:
                    dir_count += 1

            if curr_dir and dir_count > 0:
                if language == "cn":
                    dir_desc = f"然后向{curr_dir}方向{dir_count}步"
                else:
                    dir_desc = f"then {dir_count} step(s) {curr_dir}"
                directions.append(dir_desc)

            path_description += "，".join(directions) if language == "cn" else ", ".join(directions)

            if language == "cn":
                path_description += f"，看到了一个{nodes[end_idx].item}。"
            else:
                path_description += f", and you see a {nodes[end_idx].item}. "

        return complete_path, path_description

    def generate_question(
        self, nodes: List[Node], start_idx: int, end_idx: int, seen_node: List[int], language: str = "cn"
    ) -> Tuple[str, str]:
        path = self.find_path(nodes, start_idx, end_idx)
        if not path:
            return "无法找到从起点到终点的路径", ""

        available_nodes = [i for i in range(len(nodes)) if i not in seen_node and i != start_idx and i != end_idx]

        num_waypoints = min(3, len(available_nodes))
        if num_waypoints > 0 and available_nodes:
            waypoints = random.sample(available_nodes, num_waypoints)

            waypoints = [start_idx] + waypoints + [end_idx]

            complete_path, _ = self.generate_multi_node_path(nodes, waypoints)

            if complete_path:
                path = complete_path

        question = "从当前位置开始，" if language == "cn" else "Starting from your current position, "

        directions = []
        curr_dir = None
        dir_count = 0

        for i in range(len(path) - 1):
            curr_idx, direction = path[i]

            if direction != curr_dir:
                if curr_dir and dir_count > 0:
                    if language == "cn":
                        dir_desc = f"往{curr_dir}{dir_count}步"
                    else:
                        dir_desc = f"go {dir_count} step(s) {curr_dir}"
                    directions.append(dir_desc)

                curr_dir = direction
                dir_count = 1
            else:
                dir_count += 1

        if curr_dir and dir_count > 0:
            if language == "cn":
                dir_desc = f"往{curr_dir}{dir_count}步"
            else:
                dir_desc = f"go {dir_count} step(s) {curr_dir}"
            directions.append(dir_desc)

        question += "，".join(directions) if language == "cn" else ", ".join(directions)

        if language == "cn":
            question += "，看到的物品是什么？"
        else:
            question += ", what item do you see?"

        answer = nodes[end_idx].item

        return question, answer

    def get_random_waypoints(self, node_count: int, num_waypoints: int = 3) -> List[int]:
        num_waypoints = min(num_waypoints, node_count)
        if num_waypoints < 2:
            num_waypoints = 2

        return random.sample(range(node_count), num_waypoints)

    def jump2unknown(self, nodes, seen_node, unknown_node_num, language: str = "cn"):
        node1 = random.sample(seen_node, 1)[0]
        other_indices = random.sample([i for i in range(len(nodes)) if i not in seen_node], unknown_node_num)
        other_indices.append(node1)
        random.shuffle(other_indices)
        paths, path_description = self.generate_multi_node_path(nodes, other_indices, language)
        seen_node = seen_node + other_indices
        return seen_node, path_description

    def generate_problem(self, shape="build_triangle_network", language="cn", num_waypoints=10, unknown_node_num=5):
        description, nodes = eval(f"self.{shape}(self.n,language)")
        path_description = f"{description}"

        if language == "cn":
            path_description += "你从其中某一个点开始，"
        else:
            path_description += "You start from one of the points, "

        waypoint_indices = self.get_random_waypoints(len(nodes), num_waypoints)
        paths, description = self.generate_multi_node_path(nodes, waypoint_indices, language)
        path_description += description

        if random.random() < self.jump_ratio:
            if language == "cn":
                path_description += "然后你跳到了一个任意的节点，"
            else:
                path_description += "Then you jump to a random node, "

            waypoint_indices, description = self.jump2unknown(nodes, waypoint_indices, unknown_node_num, language)
            path_description += description

            if random.random() < 0.5:
                if language == "cn":
                    path_description += "然后你回到了那个任意的节点，"
                else:
                    path_description += "Then you return to that random node, "

                waypoint_indices.append(waypoint_indices[-(unknown_node_num + 1)])

        question, answer = self.generate_question(
            nodes, waypoint_indices[-1], random.choice(waypoint_indices), waypoint_indices, language
        )
        return path_description, question, answer

    def generate(self, num_of_data=10, num_waypoints=10, unknown_node_num=5, language="mixed"):
        outputs = []
        for i in range(num_of_data):
            shape = random.choice(["build_triangle_network", "build_diamond_network", "build_square_network"])
            if language == "mixed":
                now_language = random.choice(["cn", "en"])
            else:
                now_language = language
            path_description, question, answer = self.generate_problem(
                shape, now_language, num_waypoints, unknown_node_num
            )
            if now_language == "cn":
                question = random.choice(prompts_zh).format(context=path_description, question=question)
            else:
                question = random.choice(prompts_en).format(context=path_description, question=question)

            outputs.append(
                Data(
                    question=question,
                    answer=answer,
                    difficulty=1,
                    metadata={
                        "shape": shape,
                        "language": now_language,
                        "num_waypoints": num_waypoints,
                        "unknown_node_num": unknown_node_num,
                        "n": self.n,
                    },
                )
            )
        return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成空间推理游戏数据")
    parser.add_argument("--jump_ratio", type=float, default=0.5, help="跳跃到未知节点的概率")
    parser.add_argument("--num_of_data", type=int, default=100, help="生成的题目数量")
    parser.add_argument("--num_waypoints", type=int, default=10, help="路径关键点数量")
    parser.add_argument("--unknown_node_num", type=int, default=5, help="未知节点数量")
    parser.add_argument("--language", type=str, default="mixed", help="语言")
    parser.add_argument("--n", type=int, default=5, help="网格大小参数")
    args = parser.parse_args()

    data_dir = pathlib.Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    game = SpaceReasoning(jump_ratio=args.jump_ratio, n=args.n)

    filename = f"data_n{args.n}_wp{args.num_waypoints}_un{args.unknown_node_num}_jr{args.jump_ratio}_{args.language}_n{args.num_of_data}.jsonl"
    output_file = data_dir / filename

    game_data_list = game.generate(
        num_of_data=args.num_of_data,
        num_waypoints=args.num_waypoints,
        unknown_node_num=args.unknown_node_num,
        language=args.language,
    )

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for game_data in game_data_list:
                f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")
    except Exception:
        pass
