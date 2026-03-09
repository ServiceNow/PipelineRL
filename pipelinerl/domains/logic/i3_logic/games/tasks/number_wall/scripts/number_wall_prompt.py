import random

prompt_candidates = {
    "Given the grid below, place walls (marked as 'A') to create islands where each island contains exactly one number, and the total number of cells in each island equals that number.": "en",
    "In this Number Wall puzzle, add walls (marked as 'A') to divide the grid into islands. Each island must contain exactly one number, and its size must equal that number.": "en",
    "Create walls (marked as 'A') in the grid to form islands. Each island should contain exactly one number, and the total cells in the island must equal that number.": "en",
    "Solve this Number Wall puzzle by placing walls (marked as 'A') to separate the grid into islands. Each island must contain exactly one number and have exactly that many cells.": "en",
    "Divide the grid into islands by placing walls (marked as 'A'). Each island must contain exactly one number, and the size of the island must match that number.": "en",
    "Place walls (marked as 'A') to separate the grid into islands. Each island must contain exactly one number, and the total number of cells in the island must equal that number.": "en",
    "In this Number Wall puzzle, add walls (marked as 'A') to create islands where each island has exactly one number, and the island's size equals that number.": "en",
    "Create a solution for this Number Wall puzzle by placing walls (marked as 'A') to form islands. Each island must contain exactly one number and have that many cells in total.": "en",
    "Solve this Number Wall puzzle by adding walls (marked as 'A') to divide the grid. Each resulting island must contain exactly one number and have that many cells.": "en",
    "Place walls (marked as 'A') to divide the grid into islands. Each island must contain exactly one number, and the number of cells in the island must equal that number.": "en",
    "在下面的网格中，放置墙壁（标记为'A'）来创建岛屿，每个岛屿恰好包含一个数字，且岛屿的总格子数等于该数字。": "zh",
    "在这个数字墙谜题中，添加墙壁（标记为'A'）来划分网格。每个岛屿必须恰好包含一个数字，且岛屿的大小必须等于该数字。": "zh",
    "在网格中创建墙壁（标记为'A'）形成岛屿。每个岛屿应该恰好包含一个数字，且岛屿的总格子数必须等于该数字。": "zh",
    "通过放置墙壁（标记为'A'）来解决这个数字墙谜题，将网格分隔成岛屿。每个岛屿必须包含恰好一个数字，且有与该数字相同数量的格子。": "zh",
    "通过放置墙壁（标记为'A'）将网格分割成岛屿。每个岛屿必须恰好包含一个数字，且岛屿的大小必须与该数字匹配。": "zh",
    "放置墙壁（标记为'A'）来分隔网格成为岛屿。每个岛屿必须恰好包含一个数字，且岛屿中的总格子数必须等于该数字。": "zh",
    "在这个数字墙谜题中，添加墙壁（标记为'A'）来创建岛屿，每个岛屿恰好有一个数字，且岛屿的大小等于该数字。": "zh",
    "通过放置墙壁（标记为'A'）来为这个数字墙谜题创建解决方案，形成岛屿。每个岛屿必须恰好包含一个数字，且总共有该数字数量的格子。": "zh",
    "通过添加墙壁（标记为'A'）来解决这个数字墙谜题，将网格划分开。每个形成的岛屿必须恰好包含一个数字，且有该数字数量的格子。": "zh",
    "放置墙壁（标记为'A'）将网格分割成岛屿。每个岛屿必须恰好包含一个数字，且岛屿中的格子数必须等于该数字。": "zh",
}


def format_grid(grid):
    n = len(grid)
    result = []
    top_border = "+----" * n + "+"
    result.append(top_border)
    for row in grid:
        row_str = "| "
        for cell in row:
            if isinstance(cell, int):
                row_str += f"{cell:<2} | "
            else:
                row_str += f"{cell}  | "
        result.append(row_str)
        result.append(top_border)
    return "\n".join(result)


def prompt_number_wall(grid: list[list]) -> str:
    prompt = random.choice(list(prompt_candidates.keys()))
    language = prompt_candidates[prompt]
    formatted_grid = format_grid(grid)
    if language == "en":
        prompt += "\n\nGrid:\n" + formatted_grid
        prompt += "\n\nRules:"
        prompt += "\n- Each island must contain exactly one number."
        prompt += "\n- The total number of cells in an island (including the number cell) must equal the value of that number."
        prompt += "\n- All cells within an island must be connected horizontally or vertically."
        prompt += "\n- Walls (marked as 'A') cannot form 2×2 or larger continuous rectangles."
        prompt += "\n- All islands must be separated by walls."
        prompt += "\n\nAt the end of your response, please output a ```python code block containing the solution grid as a 2D list."
    elif language == "zh":
        prompt += "\n\n网格：\n" + formatted_grid
        prompt += "\n\n规则："
        prompt += "\n- 每个岛屿必须恰好包含一个数字。"
        prompt += "\n- 岛屿中的总格子数（包括数字所在格子）必须等于该数字的值。"
        prompt += "\n- 岛屿内的所有格子必须通过水平或垂直方式相连。"
        prompt += "\n- 墙壁（标记为'A'）不能形成2×2或更大的连续矩形。"
        prompt += "\n- 所有岛屿必须被墙壁分隔开。"
        prompt += "\n\n在回答的最后，请输出一个```python代码块，其中包含作为二维列表的解决方案网格。"
    return prompt
