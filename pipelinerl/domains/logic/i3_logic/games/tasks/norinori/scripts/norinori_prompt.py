import random

prompt_candidates = {
    "Solve this Norinori puzzle with a {n}×{n} grid. Place dominoes (1×2 or 2×1 rectangles) so that each region has exactly two cells covered.": "en",
    "In this Norinori puzzle, place dominoes on the {n}×{n} grid so that each region contains exactly two covered cells.": "en",
    "This is a {n}×{n} Norinori puzzle. Your task is to place dominoes (1×2 or 2×1 blocks) so that each region has exactly two cells covered.": "en",
    "Norinori puzzle: Place dominoes on the {n}×{n} grid ensuring each region has exactly two covered cells. Dominoes cannot share edges.": "en",
    "Can you solve this Norinori puzzle? Place dominoes on the {n}×{n} grid so each region has exactly two covered cells.": "en",
    "Solve the Norinori puzzle by placing dominoes on the {n}×{n} grid. Each region must have exactly two covered cells.": "en",
    "This Norinori puzzle requires you to place dominoes on a {n}×{n} grid so that each region has exactly two covered cells.": "en",
    "Place dominoes on this {n}×{n} Norinori grid. Each region must have exactly two covered cells, and dominoes cannot share edges.": "en",
    "In this {n}×{n} Norinori puzzle, place dominoes so that each region has exactly two covered cells. Dominoes cannot be adjacent.": "en",
    "Your challenge is to solve this Norinori puzzle by placing dominoes on the {n}×{n} grid so each region has exactly two covered cells.": "en",
    "解决这个 {n}×{n} 的 Norinori 谜题。放置多米诺（1×2 或 2×1 的矩形块），使每个区域恰好有两个格子被覆盖。": "zh",
    "在这个 Norinori 谜题中，在 {n}×{n} 的网格上放置多米诺，使每个区域恰好包含两个被覆盖的格子。": "zh",
    "这是一个 {n}×{n} 的 Norinori 谜题。你的任务是放置多米诺（1×2 或 2×1 的块），使每个区域恰好有两个格子被覆盖。": "zh",
    "Norinori 谜题：在 {n}×{n} 的网格上放置多米诺，确保每个区域恰好有两个格子被覆盖。多米诺不能共享边。": "zh",
    "你能解决这个 Norinori 谜题吗？在 {n}×{n} 的网格上放置多米诺，使每个区域恰好有两个格子被覆盖。": "zh",
    "通过在 {n}×{n} 的网格上放置多米诺来解决 Norinori 谜题。每个区域必须恰好有两个格子被覆盖。": "zh",
    "这个 Norinori 谜题要求你在 {n}×{n} 的网格上放置多米诺，使每个区域恰好有两个格子被覆盖。": "zh",
    "在这个 {n}×{n} 的 Norinori 网格上放置多米诺。每个区域必须恰好有两个格子被覆盖，多米诺不能共享边。": "zh",
    "在这个 {n}×{n} 的 Norinori 谜题中，放置多米诺使每个区域恰好有两个格子被覆盖。多米诺不能相邻。": "zh",
    "你的挑战是通过在 {n}×{n} 的网格上放置多米诺来解决这个 Norinori 谜题，使每个区域恰好有两个格子被覆盖。": "zh",
}


def prompt_norinori(region_grid):
    n = len(region_grid)
    grid_str = ""
    for row in region_grid:
        grid_str += "  ".join(row) + "\n"
    prompt_template = random.choice(list(prompt_candidates.keys()))
    language = prompt_candidates[prompt_template]
    prompt_text = prompt_template.format(n=n)
    if language == "en":
        rules = "\n# Norinori Puzzle\n\n## Rules:\n- Place dominoes (1×2 or 2×1 rectangles) on the grid.\n- Each region must have exactly two cells covered by dominoes.\n- Dominoes can cross region boundaries.\n- Dominoes cannot share edges (i.e., cannot be orthogonally adjacent), but they can touch diagonally.\n- Cells marked with 'X' are shadow cells that do not belong to any region, but must be part of a domino.\n\n## Grid:\n{}\n\nPlease solve this puzzle by finding all domino positions. List each domino as a pair of coordinates, for example [(1,2), (1,3)] represents a domino covering the cells at row 1, column 2 and row 1, column 3.\nNote: Coordinates are 1-indexed (the top-left cell is at position (1,1)).\n\nYour answer should be in the format: [[(r1,c1), (r1,c2)], [(r2,c1), (r2,c2)], ...] where each pair represents a domino.\n".format(
            grid_str
        )
    else:
        rules = "\n# Norinori 谜题\n\n## 规则：\n- 在网格中放置多米诺（1×2 或 2×1 的矩形块）。\n- 每个区域必须恰好有两个格子被多米诺覆盖。\n- 多米诺可以跨越区域边界。\n- 多米诺不能共享边（即不能正交相邻），但可以对角线相接。\n- 标记为 'X' 的格子是阴影格子，不属于任何区域，但是必须成为某个多米诺的一部分。\n\n## 网格：\n{}\n\n请解决这个谜题，找出所有多米诺的位置。将每个多米诺表示为一对坐标，例如 [(1,2), (1,3)] 表示一个覆盖第1行第2列和第1行第3列的多米诺。\n注意：坐标从1开始计数（左上角的格子位置是 (1,1)）。\n\n你的答案应该采用以下格式：[[(r1,c1), (r1,c2)], [(r2,c1), (r2,c2)], ...] 其中每对坐标表示一个多米诺。\n".format(
            grid_str
        )
    full_prompt = prompt_text + rules
    return full_prompt
