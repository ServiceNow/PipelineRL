import random

prompt_candidates = {
    "You are given a {n} x {n} grid representing a Minesweeper game. In this puzzle, you need to identify all cells that can be definitively determined as mines based on the currently revealed cells.\n1. Rules:\n  1. Each cell can be either a mine, a number (0-8), or an unrevealed cell (X).\n  2. Numbers indicate how many mines are in the 8 adjacent cells (horizontally, vertically, and diagonally).\n  3. Your task is to find all unrevealed cells that must contain mines based on the current state of the grid.\n2. Input:\n{puzzle}": "en",
    "This is a {n} x {n} Minesweeper grid. Your task is to identify all cells that must contain mines based on the currently revealed information.\n1. Rules:\n  1. Each cell is either a mine, a number (0-8), or unrevealed (X).\n  2. A number indicates the count of mines in the 8 surrounding cells.\n  3. Using only one-step deductive logic, determine which unrevealed cells must be mines.\n2. Input:\n{puzzle}": "en",
    "You're presented with a {n} x {n} Minesweeper puzzle grid. The goal is to identify all unrevealed cells that must contain mines based on the current revealed numbers.\n1. Rules:\n  1. Cells marked with numbers (0-8) indicate how many mines are in the 8 adjacent cells.\n  2. Cells marked with X are unrevealed.\n  3. Using logical deduction, identify all cells that must contain mines.\n2. Input:\n{puzzle}": "en",
    "Below is a {n} x {n} Minesweeper grid. Your objective is to find all unrevealed cells that must contain mines based on the currently revealed information.\n1. Rules:\n  1. Numbers (0-8) indicate how many mines are in the 8 adjacent cells.\n  2. Unrevealed cells are marked with X.\n  3. Using one-step logical deduction, determine which X cells must be mines.\n2. Input:\n{puzzle}": "en",
    "Here's a {n} x {n} Minesweeper puzzle. You need to identify all cells that must contain mines based on the current state of the grid.\n1. Rules:\n  1. Numbers show how many mines are in the 8 surrounding cells.\n  2. Unrevealed cells are marked with X.\n  3. Find all X cells that can be definitively identified as mines through logical deduction.\n2. Input:\n{puzzle}": "en",
    "I'm presenting you with a {n} x {n} Minesweeper grid. Your task is to identify all unrevealed cells that must contain mines based on the currently revealed numbers.\n1. Rules:\n  1. Each revealed number indicates how many mines are in the 8 adjacent cells.\n  2. Unrevealed cells are marked with X.\n  3. Using logical deduction, determine which X cells must be mines.\n2. Input:\n{puzzle}": "en",
    "Consider this {n} x {n} Minesweeper grid. You need to identify all unrevealed cells that must contain mines based on the current information.\n1. Rules:\n  1. Numbers (0-8) show how many mines are in the 8 adjacent cells.\n  2. Cells marked with X are unrevealed.\n  3. Using one-step logical reasoning, determine which X cells must be mines.\n2. Input:\n{puzzle}": "en",
    "You have a {n} x {n} Minesweeper grid. Your goal is to identify all unrevealed cells that must contain mines based on the currently revealed numbers.\n1. Rules:\n  1. Each number indicates how many mines are in the 8 surrounding cells.\n  2. Unrevealed cells are marked with X.\n  3. Find all X cells that can be definitively identified as mines through logical deduction.\n2. Input:\n{puzzle}": "en",
    "This {n} x {n} grid represents a Minesweeper game. Your task is to identify all unrevealed cells that must contain mines based on the current state.\n1. Rules:\n  1. Numbers (0-8) indicate how many mines are in the 8 adjacent cells.\n  2. Cells marked with X are unrevealed.\n  3. Using logical reasoning, determine which X cells must be mines.\n2. Input:\n{puzzle}": "en",
    "Examine this {n} x {n} Minesweeper grid. Your objective is to identify all unrevealed cells that must contain mines based on the currently revealed information.\n1. Rules:\n  1. Each number shows how many mines are in the 8 surrounding cells.\n  2. Unrevealed cells are marked with X.\n  3. Using one-step deductive logic, determine which X cells must be mines.\n2. Input:\n{puzzle}": "en",
    "这是一个 {n} x {n} 的扫雷游戏网格。在这个谜题中，你需要根据当前已揭示的单元格，确定所有可以明确判断为地雷的单元格。\n1. 规则：\n  1. 每个单元格可以是地雷、数字（0-8）或未揭示的单元格（X）。\n  2. 数字表示其周围8个相邻单元格（水平、垂直和对角线）中地雷的数量。\n  3. 你的任务是找出所有基于当前网格状态必定包含地雷的未揭示单元格。\n2. 输入：\n{puzzle}": "zh",
    "给你一个 {n} x {n} 的扫雷网格。你的任务是根据当前已揭示的信息，确定所有必定包含地雷的单元格。\n1. 规则：\n  1. 每个单元格可以是地雷、数字（0-8）或未揭示（X）。\n  2. 数字表示周围8个单元格中地雷的数量。\n  3. 使用一步推理逻辑，确定哪些未揭示的单元格必定是地雷。\n2. 输入：\n{puzzle}": "zh",
    "这是一个 {n} x {n} 的扫雷谜题网格。目标是根据当前已揭示的数字，确定所有必定包含地雷的未揭示单元格。\n1. 规则：\n  1. 标有数字（0-8）的单元格表示其周围8个相邻单元格中地雷的数量。\n  2. 标有X的单元格是未揭示的。\n  3. 使用逻辑推理，确定所有必定是地雷的单元格。\n2. 输入：\n{puzzle}": "zh",
    "下面是一个 {n} x {n} 的扫雷网格。你的目标是根据当前已揭示的信息，找出所有必定包含地雷的未揭示单元格。\n1. 规则：\n  1. 数字（0-8）表示周围8个相邻单元格中地雷的数量。\n  2. 未揭示的单元格标记为X。\n  3. 使用一步逻辑推理，确定哪些X单元格必定是地雷。\n2. 输入：\n{puzzle}": "zh",
    "这是一个 {n} x {n} 的扫雷谜题。你需要根据网格的当前状态，确定所有必定包含地雷的单元格。\n1. 规则：\n  1. 数字表示周围8个单元格中地雷的数量。\n  2. 未揭示的单元格标记为X。\n  3. 找出所有通过逻辑推理可以明确确定为地雷的X单元格。\n2. 输入：\n{puzzle}": "zh",
    "给你一个 {n} x {n} 的扫雷网格。你的任务是根据当前已揭示的数字，确定所有必定包含地雷的未揭示单元格。\n1. 规则：\n  1. 每个已揭示的数字表示其周围8个相邻单元格中地雷的数量。\n  2. 未揭示的单元格标记为X。\n  3. 使用逻辑推理，确定哪些X单元格必定是地雷。\n2. 输入：\n{puzzle}": "zh",
    "考虑这个 {n} x {n} 的扫雷网格。你需要根据当前信息，确定所有必定包含地雷的未揭示单元格。\n1. 规则：\n  1. 数字（0-8）表示周围8个相邻单元格中地雷的数量。\n  2. 标记为X的单元格是未揭示的。\n  3. 使用一步逻辑推理，确定哪些X单元格必定是地雷。\n2. 输入：\n{puzzle}": "zh",
    "你有一个 {n} x {n} 的扫雷网格。你的目标是根据当前已揭示的数字，确定所有必定包含地雷的未揭示单元格。\n1. 规则：\n  1. 每个数字表示周围8个单元格中地雷的数量。\n  2. 未揭示的单元格标记为X。\n  3. 找出所有通过逻辑推理可以明确确定为地雷的X单元格。\n2. 输入：\n{puzzle}": "zh",
    "这个 {n} x {n} 的网格代表一个扫雷游戏。你的任务是根据当前状态，确定所有必定包含地雷的未揭示单元格。\n1. 规则：\n  1. 数字（0-8）表示周围8个相邻单元格中地雷的数量。\n  2. 标记为X的单元格是未揭示的。\n  3. 使用逻辑推理，确定哪些X单元格必定是地雷。\n2. 输入：\n{puzzle}": "zh",
    "请分析这个 {n} x {n} 的扫雷网格。你的目标是根据当前已揭示的信息，确定所有必定包含地雷的未揭示单元格。\n1. 规则：\n  1. 每个数字表示周围8个单元格中地雷的数量。\n  2. 未揭示的单元格标记为X。\n  3. 使用一步推理逻辑，确定哪些X单元格必定是地雷。\n2. 输入：\n{puzzle}": "zh",
}


def prompt_minesweeper(grid):
    n = len(grid)
    grid_str = ""
    for row in grid:
        grid_str += "  ".join((str(cell) for cell in row)) + "\n"
    puzzle = f"{grid_str}"
    prompt = random.choice(list(prompt_candidates.keys()))
    language = prompt_candidates[prompt]
    prompt = prompt.format(n=n, puzzle=puzzle)
    if language == "en":
        prompt += "\n3. Task:\n  - Identify all unrevealed cells (X) that must contain mines based on the current information.\n  - Output the coordinates of all cells that must be mines as a list of (row, column) tuples.\n4. Example Output Format:\n[(0, 1), (2, 3), (3, 0)]"
    elif language == "zh":
        prompt += "\n3. 任务：\n  - 根据当前信息，确定所有必定包含地雷的未揭示单元格（X）。\n  - 以坐标列表的形式输出所有必定是地雷的单元格，格式为(行, 列)的元组列表。\n4. 示例输出格式：\n[(0, 1), (2, 3), (3, 0)]"
    return prompt
