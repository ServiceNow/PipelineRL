import random

prompt_candidates = {
    "You are given a {n}×{n} Numbrix puzzle. In this puzzle, you need to fill in the missing numbers from 1 to {n_squared} to create a continuous path where consecutive numbers are orthogonally adjacent (horizontally or vertically, not diagonally).\n1. Rules:\n  1. Each cell must contain a unique number from 1 to {n_squared}.\n  2. Consecutive numbers (like 1-2, 2-3, etc.) must be orthogonally adjacent (horizontally or vertically).\n  3. Some numbers are already placed as hints.\n2. Input:\n{puzzle}": "en",
    "This is a {n}×{n} Numbrix puzzle grid. Your task is to fill in all missing numbers from 1 to {n_squared} such that consecutive numbers are always adjacent horizontally or vertically (not diagonally).\n1. Rules:\n  1. Each cell must contain exactly one number from 1 to {n_squared}.\n  2. Each number from 1 to {n_squared} must appear exactly once in the grid.\n  3. Consecutive numbers must be orthogonally adjacent (up, down, left, or right).\n  4. Some numbers are already provided as clues.\n2. Input:\n{puzzle}": "en",
    "You're presented with a {n}×{n} Numbrix puzzle. The goal is to fill in all empty cells marked with 'X' with numbers from 1 to {n_squared} to form a continuous path where consecutive numbers are adjacent horizontally or vertically.\n1. Rules:\n  1. Each number from 1 to {n_squared} must appear exactly once in the grid.\n  2. Consecutive numbers (such as 3-4, 8-9, etc.) must be orthogonally adjacent.\n  3. Some numbers are already placed in the grid as hints.\n2. Input:\n{puzzle}": "en",
    "Below is a {n}×{n} Numbrix puzzle grid. Your objective is to place the numbers 1 through {n_squared} in the grid such that consecutive numbers are always adjacent horizontally or vertically (not diagonally).\n1. Rules:\n  1. Each cell must contain a unique integer from 1 to {n_squared}.\n  2. Consecutive integers must be orthogonally adjacent.\n  3. Some cells already contain numbers as hints.\n2. Input:\n{puzzle}": "en",
    "Here's a {n}×{n} Numbrix logic puzzle. You need to fill in all empty cells (marked with 'X') with numbers from 1 to {n_squared} to create a continuous path where consecutive numbers are adjacent.\n1. Rules:\n  1. Each number from 1 to {n_squared} must appear exactly once in the grid.\n  2. Consecutive numbers must be orthogonally adjacent (up, down, left, or right).\n  3. Some numbers are already placed as clues.\n2. Input:\n{puzzle}": "en",
    "I'm presenting you with a {n}×{n} Numbrix puzzle. Your task is to fill in the grid with numbers from 1 to {n_squared} so that consecutive numbers are always adjacent (horizontally or vertically).\n1. Rules:\n  1. Each cell must contain a unique number from 1 to {n_squared}.\n  2. Consecutive numbers must share an edge (not just a corner).\n  3. Some numbers are already placed in the grid as hints.\n2. Input:\n{puzzle}": "en",
    "Consider this {n}×{n} Numbrix puzzle grid. You need to place numbers from 1 to {n_squared} in the grid such that consecutive numbers are always orthogonally adjacent.\n1. Rules:\n  1. Each number from 1 to {n_squared} must appear exactly once.\n  2. Numbers that differ by 1 must be placed in cells that share an edge.\n  3. Some cells already contain numbers as clues.\n2. Input:\n{puzzle}": "en",
    "You have a {n}×{n} Numbrix puzzle grid. Your goal is to fill in all cells marked with 'X' with numbers from 1 to {n_squared} to create a continuous path of consecutive numbers.\n1. Rules:\n  1. Each number from 1 to {n_squared} must be used exactly once.\n  2. Consecutive numbers must be placed in cells that are horizontally or vertically adjacent.\n  3. Some numbers are already placed as hints.\n2. Input:\n{puzzle}": "en",
    "This {n}×{n} grid represents a Numbrix puzzle. Your task is to fill in all empty cells with numbers from 1 to {n_squared} to form a continuous path where consecutive numbers are adjacent.\n1. Rules:\n  1. Each cell must contain a unique number from 1 to {n_squared}.\n  2. Consecutive numbers must be placed in orthogonally adjacent cells.\n  3. Some numbers are already provided as clues.\n2. Input:\n{puzzle}": "en",
    "Examine this {n}×{n} Numbrix puzzle grid. Your objective is to place numbers from 1 to {n_squared} in the grid such that consecutive numbers are always adjacent horizontally or vertically.\n1. Rules:\n  1. Each number from 1 to {n_squared} must appear exactly once in the grid.\n  2. Numbers that differ by 1 must be placed in orthogonally adjacent cells.\n  3. Some cells already contain numbers as hints.\n2. Input:\n{puzzle}": "en",
    "Solve this {n}×{n} Numbrix puzzle by filling in all empty cells with numbers from 1 to {n_squared} to create a continuous path of consecutive numbers.\n1. Rules:\n  1. Each number from 1 to {n_squared} must appear exactly once in the grid.\n  2. Consecutive numbers must be placed in cells that share an edge.\n  3. Some numbers are already placed as hints.\n2. Input:\n{puzzle}": "en",
    "This is a Numbrix puzzle on a {n}×{n} grid. Your challenge is to fill in all empty cells with numbers from 1 to {n_squared} to create a continuous path.\n1. Rules:\n  1. Each cell must contain a unique number from 1 to {n_squared}.\n  2. Consecutive numbers must be placed in orthogonally adjacent cells.\n  3. Some numbers are already provided as clues.\n2. Input:\n{puzzle}": "en",
    "You're looking at a {n}×{n} Numbrix puzzle. The goal is to place numbers from 1 to {n_squared} in the grid to form a continuous path where consecutive numbers are adjacent.\n1. Rules:\n  1. Each number from 1 to {n_squared} must appear exactly once.\n  2. Consecutive numbers must be placed in cells that share an edge.\n  3. Some cells already contain numbers as hints.\n2. Input:\n{puzzle}": "en",
    "Complete this {n}×{n} Numbrix puzzle by filling in all empty cells with numbers from 1 to {n_squared} to create a continuous path.\n1. Rules:\n  1. Each number from 1 to {n_squared} must be used exactly once.\n  2. Consecutive numbers must be placed in orthogonally adjacent cells.\n  3. Some numbers are already placed as hints.\n2. Input:\n{puzzle}": "en",
    "This Numbrix puzzle is on a {n}×{n} grid. Your task is to fill in all empty cells with numbers from 1 to {n_squared} to form a continuous path of consecutive numbers.\n1. Rules:\n  1. Each cell must contain a unique number from 1 to {n_squared}.\n  2. Numbers that differ by 1 must be placed in cells that share an edge.\n  3. Some numbers are already provided as clues.\n2. Input:\n{puzzle}": "en",
    "这是一个 {n}×{n} 的 Numbrix 谜题。在这个谜题中，你需要填入从 1 到 {n_squared} 的缺失数字，创建一条连续路径，使得连续的数字在正交方向上相邻（水平或垂直，不能对角）。\n1. 规则：\n  1. 每个单元格必须包含从 1 到 {n_squared} 的唯一数字。\n  2. 连续的数字（如 1-2, 2-3 等）必须正交相邻（水平或垂直）。\n  3. 一些数字已经作为提示放置在网格中。\n2. 输入：\n{puzzle}": "zh",
    "给你一个 {n}×{n} 的 Numbrix 谜题网格。你的任务是填入所有从 1 到 {n_squared} 的缺失数字，使得连续的数字总是水平或垂直相邻（不能对角相邻）。\n1. 规则：\n  1. 每个单元格必须包含从 1 到 {n_squared} 的一个数字。\n  2. 从 1 到 {n_squared} 的每个数字在网格中必须恰好出现一次。\n  3. 连续的数字必须正交相邻（上、下、左、右）。\n  4. 一些数字已经作为线索提供。\n2. 输入：\n{puzzle}": "zh",
    "这是一个 {n}×{n} 的 Numbrix 逻辑谜题。目标是在所有标记为'X'的空单元格中填入从 1 到 {n_squared} 的数字，形成一条连续路径，使得连续的数字在水平或垂直方向上相邻。\n1. 规则：\n  1. 从 1 到 {n_squared} 的每个数字在网格中必须恰好出现一次。\n  2. 连续的数字（如 3-4, 8-9 等）必须正交相邻。\n  3. 一些数字已经作为提示放置在网格中。\n2. 输入：\n{puzzle}": "zh",
    "下面是一个 {n}×{n} 的 Numbrix 谜题网格。你的目标是在网格中放置从 1 到 {n_squared} 的数字，使得连续的数字总是水平或垂直相邻（不能对角相邻）。\n1. 规则：\n  1. 每个单元格必须包含从 1 到 {n_squared} 的唯一整数。\n  2. 连续的整数必须正交相邻。\n  3. 一些单元格已经包含作为提示的数字。\n2. 输入：\n{puzzle}": "zh",
    "这是一个 {n}×{n} 的 Numbrix 谜题。你需要在所有空单元格（标记为'X'）中填入从 1 到 {n_squared} 的数字，创建一条连续的路径，使得连续的数字相邻。\n1. 规则：\n  1. 从 1 到 {n_squared} 的每个数字在网格中必须恰好出现一次。\n  2. 连续的数字必须正交相邻（上、下、左、右）。\n  3. 一些数字已经作为线索放置。\n2. 输入：\n{puzzle}": "zh",
    "我给你一个 {n}×{n} 的 Numbrix 谜题。你的任务是在网格中填入从 1 到 {n_squared} 的数字，使得连续的数字总是相邻（水平或垂直）。\n1. 规则：\n  1. 每个单元格必须包含从 1 到 {n_squared} 的唯一数字。\n  2. 连续的数字必须共享一条边（不仅仅是一个角）。\n  3. 一些数字已经作为提示放置在网格中。\n2. 输入：\n{puzzle}": "zh",
    "考虑这个 {n}×{n} 的 Numbrix 谜题网格。你需要在网格中放置从 1 到 {n_squared} 的数字，使得连续的数字总是正交相邻。\n1. 规则：\n  1. 从 1 到 {n_squared} 的每个数字必须恰好出现一次。\n  2. 相差为 1 的数字必须放置在共享一条边的单元格中。\n  3. 一些单元格已经包含作为线索的数字。\n2. 输入：\n{puzzle}": "zh",
    "你有一个 {n}×{n} 的 Numbrix 谜题网格。你的目标是在所有标记为'X'的单元格中填入从 1 到 {n_squared} 的数字，创建一条连续数字的路径。\n1. 规则：\n  1. 从 1 到 {n_squared} 的每个数字必须恰好使用一次。\n  2. 连续的数字必须放置在水平或垂直相邻的单元格中。\n  3. 一些数字已经作为提示放置。\n2. 输入：\n{puzzle}": "zh",
    "这个 {n}×{n} 的网格代表一个 Numbrix 谜题。你的任务是在所有空单元格中填入从 1 到 {n_squared} 的数字，形成一条连续的路径，使得连续的数字相邻。\n1. 规则：\n  1. 每个单元格必须包含从 1 到 {n_squared} 的唯一数字。\n  2. 连续的数字必须放置在正交相邻的单元格中。\n  3. 一些数字已经作为线索提供。\n2. 输入：\n{puzzle}": "zh",
    "请分析这个 {n}×{n} 的 Numbrix 谜题网格。你的目标是在网格中放置从 1 到 {n_squared} 的数字，使得连续的数字总是水平或垂直相邻。\n1. 规则：\n  1. 从 1 到 {n_squared} 的每个数字在网格中必须恰好出现一次。\n  2. 相差为 1 的数字必须放置在正交相邻的单元格中。\n  3. 一些单元格已经包含作为线索的数字。\n2. 输入：\n{puzzle}": "zh",
    "解决这个 {n}×{n} 的 Numbrix 谜题，在所有空单元格中填入从 1 到 {n_squared} 的数字，创建一条连续数字的路径。\n1. 规则：\n  1. 从 1 到 {n_squared} 的每个数字在网格中必须恰好出现一次。\n  2. 连续的数字必须放置在共享一条边的单元格中。\n  3. 一些数字已经作为提示放置。\n2. 输入：\n{puzzle}": "zh",
    "这是一个在 {n}×{n} 网格上的 Numbrix 谜题。你的挑战是在所有空单元格中填入从 1 到 {n_squared} 的数字，创建一条连续的路径。\n1. 规则：\n  1. 每个单元格必须包含从 1 到 {n_squared} 的唯一数字。\n  2. 连续的数字必须放置在正交相邻的单元格中。\n  3. 一些数字已经作为线索提供。\n2. 输入：\n{puzzle}": "zh",
    "你正在看一个 {n}×{n} 的 Numbrix 谜题。目标是在网格中放置从 1 到 {n_squared} 的数字，形成一条连续的路径，使得连续的数字相邻。\n1. 规则：\n  1. 从 1 到 {n_squared} 的每个数字必须恰好出现一次。\n  2. 连续的数字必须放置在共享一条边的单元格中。\n  3. 一些单元格已经包含作为线索的数字。\n2. 输入：\n{puzzle}": "zh",
    "完成这个 {n}×{n} 的 Numbrix 谜题，在所有空单元格中填入从 1 到 {n_squared} 的数字，创建一条连续的路径。\n1. 规则：\n  1. 从 1 到 {n_squared} 的每个数字必须恰好使用一次。\n  2. 连续的数字必须放置在正交相邻的单元格中。\n  3. 一些数字已经作为提示放置。\n2. 输入：\n{puzzle}": "zh",
    "这个 Numbrix 谜题是在一个 {n}×{n} 的网格上。你的任务是在所有空单元格中填入从 1 到 {n_squared} 的数字，形成一条连续数字的路径。\n1. 规则：\n  1. 每个单元格必须包含从 1 到 {n_squared} 的唯一数字。\n  2. 相差为 1 的数字必须放置在共享一条边的单元格中。\n  3. 一些数字已经作为线索提供。\n2. 输入：\n{puzzle}": "zh",
}


def format_grid(grid):
    n = len(grid)
    max_width = max(len(str(n * n)), 1)
    h_line = "+" + "+".join(["-" * (max_width + 2)] * n) + "+"
    result = h_line + "\n"
    for row in grid:
        formatted_row = []
        for cell in row:
            if cell == "X":
                cell_str = " " * max_width + "X"
            else:
                cell_str = f"{cell:>{max_width + 1}} "
            formatted_row.append(cell_str)
        result += "|" + "|".join(formatted_row) + "|\n"
        result += h_line + "\n"
    return result


def prompt_numbrix(grid):
    n = len(grid)
    n_squared = n * n
    grid_str = format_grid(grid)
    puzzle = f"{grid_str}\n\nGrid size: {n}×{n}\nNumbers range: 1 to {n_squared}"
    prompt = random.choice(list(prompt_candidates.keys()))
    language = prompt_candidates[prompt]
    prompt = prompt.format(n=n, n_squared=n_squared, puzzle=puzzle)
    if language == "en":
        prompt += "\n3. Task:\n  - Fill in all empty cells marked with 'X' with appropriate numbers from 1 to {n_squared}.\n  - Ensure that consecutive numbers are always orthogonally adjacent.\n  - Output the completed grid as a 2D list (list of lists) in Python format.\n4. Example Output Format:\n[\n    [1, 2, 3, 4],\n    [8, 7, 6, 5],\n    [9, 10, 11, 12],\n    [16, 15, 14, 13]\n]".format(
            n_squared=n_squared
        )
    elif language == "zh":
        prompt += "\n3. 任务：\n  - 在所有标记为'X'的空单元格中填入从 1 到 {n_squared} 的适当数字。\n  - 确保连续的数字总是正交相邻。\n  - 以Python格式的二维列表(列表的列表)输出完成的网格。\n4. 示例输出格式：\n[\n    [1, 2, 3, 4],\n    [8, 7, 6, 5],\n    [9, 10, 11, 12],\n    [16, 15, 14, 13]\n]".format(
            n_squared=n_squared
        )
    return prompt
