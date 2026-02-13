import random

chinese_prompt_candidates = [
    "这个摩天楼游戏在一个 {n}×{n} 的网格上进行，每个位置放置高度从 1 到 {n} 的摩天楼。同一行同一列中的高度不能重复。网格四周的数字表示从该方向看去，能看到几座摩天楼（较高的摩天楼会遮挡较矮的）。请填写每个格子的摩天楼高度。请将答案仅放在```python代码块中，不要包含任何解释。",
    "在这个 {n}×{n} 的摩天楼谜题中，每个格子需填入 1 到 {n} 的数字，代表摩天楼高度。每行每列的数字不能重复。游戏边缘的数字指示从该方向看进去能看到的摩天楼数量（高楼会挡住矮楼）。请解出所有格子的高度。请务必将答案放在```python代码块中，仅提供格式良好的Python列表。",
    "摩天楼谜题：一个 {n}×{n} 的网格，每格放置高度为 1 到 {n} 的摩天楼。每行每列中高度不能重复。网格边缘的数字表示从那个方向看进去能看到的摩天楼数量（因为高的建筑会遮挡矮的建筑）。请确定每个位置的摩天楼高度。必须将答案用```python代码块标注，不要解释过程。",
    "这是一个 {n}×{n} 的摩天楼逻辑谜题。填入 1 到 {n} 的数字作为摩天楼高度，同行同列不能有相同高度。周围的数字表示从该角度能看到的摩天楼数量（较高的摩天楼会挡住后面较矮的）。请解出每个格子应填的数字。将答案用```python代码块格式输出，不要包含任何解释。",
    "摩天楼游戏：在 {n}×{n} 网格中，每个格子放一座高度为 1 到 {n} 的摩天楼。每行每列的高度不能重复。边缘数字指示从该方向看进去时能看到的摩天楼数量（高楼会挡住矮楼）。请推理出每个位置的摩天楼高度。必须使用```python代码块格式提交答案，且只包含网格数据。",
    "解决这个 {n}×{n} 的摩天楼谜题：每格填入 1 到 {n} 的数字代表摩天楼高度，每行每列中不能有重复高度。外围数字表示从该方向看进去时能看到的摩天楼数量（高楼会遮挡矮楼）。请确定所有格子的高度。将最终答案使用```python格式包裹，便于自动提取，不要包含其他文本。",
    "这个 {n}×{n} 的摩天楼谜题中，您需要确定每个格子中摩天楼的高度（1 到 {n}）。同行同列中的高度不能重复。网格四周的数字告诉您从该位置看过去能看到多少座摩天楼（因为高楼会挡住矮楼）。请填写完整解答，并将答案放在```python代码块中，不要包含任何解释。",
    "在这个摩天楼谜题中，您需要在 {n}×{n} 网格的每个位置放置一座高为 1 到 {n} 的摩天楼。每行每列中的高度都必须唯一。边缘的数字表示从该方向看进去时能看到几座摩天楼（较高的摩天楼会遮挡较矮的摩天楼）。请解出每个格子的高度。答案必须用```python代码块格式提交，仅包含二维列表。",
    "摩天楼推理游戏：在 {n}×{n} 的网格内，每个格子填入 1 到 {n} 的数字，表示摩天楼高度。每行每列不能有重复数字。网格边上的数字表示从该处向内看能看到的摩天楼数量（因为高楼会挡住矮楼）。请确定每个格子的高度，并用```python代码块格式提交答案，不要包含任何分析过程。",
    "这是一个 {n}×{n} 的摩天楼游戏。在每个格子中填入 1 到 {n} 的数字，代表摩天楼高度。每行每列不能出现重复数字。外部的数字表示从该方向看进去时可见的摩天楼数量（高楼会遮挡矮楼）。请找出所有格子的正确高度，并使用```python代码块提交标准格式的答案，仅包含二维列表数据。",
]
english_prompt_candidates = [
    "In this Skyscraper puzzle, you have a {n}×{n} grid where each cell contains a skyscraper of height between 1 and {n}. Heights cannot repeat in the same row or column. The numbers around the grid indicate how many skyscrapers are visible when looking from that direction, as taller buildings block the view of shorter ones. Please determine the height of each skyscraper. Your answer must be submitted in a ```python code block format only, containing just the grid data.",
    "Solve this {n}×{n} Skyscraper puzzle where each cell must contain a skyscraper with height from 1 to {n}. No duplicate heights in any row or column. Edge numbers show how many skyscrapers are visible from that direction (taller buildings hide shorter ones behind them). Fill in the height for each cell. Submit your answer within a ```python code block, without explanations or analysis.",
    "This is a {n}×{n} Skyscraper puzzle. Place skyscrapers of heights 1 to {n} so that no row or column contains the same height twice. The clues around the grid indicate how many skyscrapers are visible when looking from that direction, as taller skyscrapers block the view of shorter ones. Find the correct height for each position. Format your answer inside a ```python code block only.",
    "You're solving a {n}×{n} Skyscraper logic puzzle. Each cell must contain a skyscraper of height 1 to {n}, with no repetition in any row or column. Numbers on the edges tell you how many skyscrapers are visible when looking from that direction (taller buildings hide shorter ones). Determine all cell values. Submit your answer inside a ```python code block only, with no additional text.",
    "In this {n}×{n} Skyscraper grid, place towers of heights 1 to {n} with no repeats in any row or column. The numbers outside the grid show how many skyscrapers you can see when looking from that direction (taller skyscrapers block the view of shorter ones). Find the height of each skyscraper. Put your answer inside a ```python code block, with no explanations or comments.",
    "Solve this {n}×{n} Skyscraper puzzle by placing buildings of heights 1 to {n} in each cell. Heights cannot repeat in any row or column. The numbers at the edges indicate how many skyscrapers are visible from that direction, as taller buildings block the view of shorter ones. Complete the grid with the correct heights. Format your answer inside a ```python code block only, containing just the 2D list.",
    "This {n}×{n} Skyscraper puzzle requires placing buildings of heights 1 to {n} in each cell, with no repetition in rows or columns. Edge numbers show how many buildings are visible from that direction (taller buildings block shorter ones behind them). Determine the height of each skyscraper. Your answer must be in a ```python code block format only, with no explanation.",
    "In this {n}×{n} Skyscraper puzzle, you need to place towers with heights from 1 to {n} in each cell. No height can appear twice in the same row or column. The numbers on the outside indicate how many skyscrapers can be seen when looking from that direction, as taller buildings hide shorter ones. Fill in the complete solution. Please provide your answer within a ```python code block only, containing just the grid data.",
    "You're tasked with solving a {n}×{n} Skyscraper puzzle. Each cell must contain a skyscraper of height 1 through {n}, with no repeats in any row or column. The clue numbers around the grid tell you how many skyscrapers are visible when looking from that direction (taller buildings block the view of shorter ones). Find all cell values. Format your answer as a properly formatted Python list inside a ```python code block only.",
    "For this {n}×{n} Skyscraper puzzle, place buildings of heights 1 to {n} so that each height appears exactly once in each row and column. The numbers around the grid show how many skyscrapers can be seen from that direction, as taller buildings hide shorter ones behind them. Determine the height of each skyscraper. Submit your answer in a ```python code block format without any explanations or additional text.",
]


def format_puzzle(n, grid, top, bottom, left, right):
    formatted = "      "
    for i in range(n):
        formatted += f"{top[i]}   "
    formatted += "\n"
    for i in range(n):
        formatted += f"{left[i]}     "
        for j in range(n):
            formatted += f"{grid[i][j]}   "
        formatted += f"{right[i]}\n"
    formatted += "      "
    for i in range(n):
        formatted += f"{bottom[i]}   "
    return formatted


def prompt_skyscraper_puzzle(n, grid, top, bottom, left, right, is_chinese=True):
    if is_chinese:
        prompt_template = random.choice(chinese_prompt_candidates)
    else:
        prompt_template = random.choice(english_prompt_candidates)
    prompt = prompt_template.format(n=n)
    formatted_puzzle = format_puzzle(n, grid, top, bottom, left, right)
    if is_chinese:
        prompt += f"\n\n1. 输入:\n\n{formatted_puzzle}\n\n2. 任务:\n   - 根据所有规则放置摩天楼。\n   - 输出完成的网格，使用Python的二维列表格式（列表的列表）。\n   - 每个子列表代表网格的一行。\n   - **极其重要**: 你的回答必须按以下格式提供：\n     1. 你可以先进行分析和推理\n     2. 然后在回答的最后，你必须添加一个```python代码块\n     3. 这个代码块中必须只包含一个二维列表（不要有任何注释或解释）\n     4. 不遵循此格式会导致回答被判为错误\n\n3. 输出格式示例:\n\n你的分析和推理...\n\n最终解答：\n\n```python\n[\n  [3, 2, 1, 4],\n  [1, 4, 3, 2],\n  [4, 1, 2, 3],\n  [2, 3, 4, 1]\n]\n```\n\n⚠️警告：代码块内只能有网格数据，不能包含任何文字、解释或注释！"
    else:
        prompt += f"\n\n1. Input:\n\n{formatted_puzzle}\n\n2. Task:\n   - Place skyscrapers in the grid following all rules.\n   - Output the completed grid as a 2D list (list of lists) in Python format.\n   - Each sub-list represents a single row of the grid.\n   - **CRITICAL**: Your answer must follow this format:\n     1. You may provide analysis and reasoning first\n     2. Then at the end of your answer, you must add a ```python code block\n     3. This code block must contain ONLY a 2D list (no comments or explanations)\n     4. Failure to follow this format will result in your answer being marked incorrect\n\n3. Example Output Format:\n\nYour analysis and reasoning...\n\nFinal answer:\n\n```python\n[\n  [3, 2, 1, 4],\n  [1, 4, 3, 2],\n  [4, 1, 2, 3],\n  [2, 3, 4, 1]\n]\n```\n\n⚠️WARNING: The code block must contain ONLY the grid data, with no text, explanations, or comments!"
    return prompt


def generate_prompts(n, grid, top, bottom, left, right):
    prompts = []
    for _ in range(10):
        prompts.append(prompt_skyscraper_puzzle(n, grid, top, bottom, left, right, is_chinese=True))
    for _ in range(10):
        prompts.append(prompt_skyscraper_puzzle(n, grid, top, bottom, left, right, is_chinese=False))
    return prompts
