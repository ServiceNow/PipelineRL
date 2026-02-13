import random

chinese_prompt_candidates = [
    "给定一个大小为 {n} x {n} 的网格，网格被划分为若干个区域，每个区域用不同的字母标记。你需要在网格中放置星星，使得每行、每列以及每个分区内恰好有 {k} 颗星星，并且星星不能相邻（包括水平、垂直和对角线方向）。",
    "这是一个 {n} x {n} 的星星放置谜题。网格被分成几个用字母标记的区域。你的任务是放置星星，确保每行、每列和每个字母区域都恰好有 {k} 颗星星，并且任意两颗星星都不能相邻（水平、垂直或对角线方向）。",
    "在这个 {n} x {n} 的网格中，各个区域用不同字母标识。请放置星星，使每行、每列和每个字母区域恰好包含 {k} 颗星星，同时确保没有两颗星星彼此相邻（包括水平、垂直和对角线上的相邻）。",
    "这个 {n} x {n} 的星星布局游戏要求你在一个分区网格上放置星星。每个分区用一个字母标记。你需要确保每行、每列和每个分区恰好有 {k} 颗星星，且星星之间不能相邻（横向、纵向或对角线）。",
    "在一个 {n} x {n} 的网格上，每个单元格属于一个用字母标记的区域。你的任务是放置星星，使每行、每列和每个区域正好有 {k} 颗星星，同时任何两颗星星都不能在相邻的格子中（包括对角线相邻）。",
    "请在这个 {n} x {n} 的网格中放置星星。网格分为不同的区域，每个区域用一个字母表示。规则要求每行、每列以及每个字母区域都必须恰好包含 {k} 颗星星，并且星星之间不能相互接触（包括水平、垂直和对角线方向）。",
    "这个星星放置游戏在 {n} x {n} 的网格上进行，网格被划分为几个区域（用字母表示）。你的目标是放置星星，使每行、每列和每个区域都有 {k} 颗星星，同时确保没有任何星星是相邻的（包括对角线相邻）。",
    "在这个 {n} x {n} 的网格游戏中，网格被分成不同的区域（用字母标记）。你需要放置星星，使得每一行、每一列和每一个区域都恰好有 {k} 颗星星，并且所有星星都不能互相接触（包括斜向接触）。",
    "这个 {n} x {n} 的星星谜题要求你在网格中放置星星。网格被分为用字母标记的不同区域。每行、每列和每个字母区域必须恰好有 {k} 颗星星，同时任何两颗星星都不能放在相邻的格子里（包括对角线相邻的格子）。",
    "在这个分区的 {n} x {n} 网格中，每个分区用不同的字母标记。请放置星星，满足以下条件：每行恰好有 {k} 颗星星，每列恰好有 {k} 颗星星，每个字母区域恰好有 {k} 颗星星，并且任何两颗星星都不能相邻（包括对角线方向）。",
]
english_prompt_candidates = [
    "Given an {n} x {n} grid divided into regions marked with different letters, place stars so that each row, column, and region contains exactly {k} stars, and no stars are adjacent to each other (horizontally, vertically, or diagonally).",
    "In this {n} x {n} star placement puzzle, the grid is divided into letter-marked regions. Your task is to place stars ensuring each row, column, and letter region has exactly {k} stars, with no two stars adjacent (horizontally, vertically, or diagonally).",
    "For this {n} x {n} grid with different lettered regions, place stars so that each row, column, and letter region contains exactly {k} stars. Stars cannot be placed in adjacent cells, including diagonally adjacent cells.",
    "This {n} x {n} star battle puzzle requires you to place stars on a grid divided into regions (marked by letters). Every row, column, and region must contain exactly {k} stars, and no stars can touch each other, even diagonally.",
    "On this {n} x {n} grid with letter-marked regions, place stars ensuring that each row, column, and region has exactly {k} stars. No two stars may be adjacent to each other, including diagonally.",
    "In this {n} x {n} grid puzzle, the grid is divided into regions labeled with letters. Place stars so each row, column, and letter region contains exactly {k} stars, with no stars touching each other (including diagonally).",
    "For this star placement challenge on an {n} x {n} grid divided into lettered regions, place stars so that every row, column, and region has exactly {k} stars. No stars can be adjacent to each other (horizontally, vertically, or diagonally).",
    "This {n} x {n} puzzle requires placing stars on a grid with letter-marked regions. Each row, column, and region must contain exactly {k} stars, and no two stars can be adjacent (including diagonally adjacent).",
    "In this star battle puzzle with an {n} x {n} grid divided into lettered regions, place stars so that each row, column, and region contains exactly {k} stars. Stars cannot be adjacent to each other, even diagonally.",
    "On this {n} x {n} grid divided into letter-marked regions, your task is to place stars ensuring each row, column, and region has exactly {k} stars. No stars can be adjacent to each other (horizontally, vertically, or diagonally).",
]


def format_region_grid(region_grid):
    # n = len(region_grid)
    formatted = ""
    for row in region_grid:
        formatted += "    ["
        for j, cell in enumerate(row):
            if j > 0:
                formatted += ", "
            formatted += f'"{cell}"'
        formatted += "],\n"
    return formatted.rstrip(",\n")


def prompt_star_placement_puzzle(n, k, region_grid, is_chinese=True):
    if is_chinese:
        prompt_template = random.choice(chinese_prompt_candidates)
    else:
        prompt_template = random.choice(english_prompt_candidates)
    prompt = prompt_template.format(n=n, k=k)
    formatted_region_grid = format_region_grid(region_grid)
    if is_chinese:
        prompt += f"\n\n给定以下区域网格：\n\n[\n{formatted_region_grid}\n]\n\n"
        prompt += "按字母顺序输出星星的坐标，如果一个区域有多个星星坐标，先写行号较小的坐标，如果行号相同，则写列号较小的坐标，不同区域的坐标用换行符分隔。\n\n"
        prompt += "您必须使用Python字典格式输出答案，并用```python代码块包裹。注意：\n1. 代码块中只包含最终答案字典，不要包含其他代码、变量定义、注释或计算过程。\n2. 确保您的答案完全符合游戏规则：每行、每列和每个区域恰好有且只有一颗星星，并且没有星星相邻。\n3. 在提交答案前，请仔细检查您的答案是否满足这些条件。\n\n例如：\n\n```python\n{\n    'A': [(1, 1)],\n    'B': [(3, 2)],\n    'C': [(2, 3)]\n}\n```\n"
    else:
        prompt += f"\n\nGiven the following region grid:\n\n[\n{formatted_region_grid}\n]\n\n"
        prompt += "Output the coordinates of the stars in alphabetical order, if there are more than one stellar coordinates in a region, write the coordinates of the answer with the smallest number of rows first, if the number of rows is the same, then write the one with the smallest number of columns first, and the coordinates of the different regions are separated by a line breaker.\n\n"
        prompt += "You must output your answer in Python dictionary format and wrap it in a ```python code block. Note:\n1. The code block should ONLY contain the final answer dictionary, without any other code, variable definitions, comments or calculation process.\n2. Ensure your answer fully complies with the game rules: exactly one star in each row, column, and region, and no stars are adjacent.\n3. Before submitting your answer, carefully verify that it meets these conditions.\n\nLike this:\n\n```python\n{\n    'A': [(1, 1)],\n    'B': [(3, 2)],\n    'C': [(2, 3)]\n}\n```\n"
    return prompt


def generate_prompts(n, k, region_grid):
    prompts = []
    for _ in range(10):
        prompts.append(prompt_star_placement_puzzle(n, k, region_grid, is_chinese=True))
    for _ in range(10):
        prompts.append(prompt_star_placement_puzzle(n, k, region_grid, is_chinese=False))
    return prompts
