import json
import random

chinese_prompt_candidates = [
    "迷宫由一个 {n} x {m} 的网格组成。\n其中 X 代表空白格子。数字代表射线箭头串的起点。\n请在空白格子中填入箭头，箭头可以向上（↑）、下（↓）、左（←）、右（→）或对角线方向（↖、↗、↘、↙）延伸。\n数字代表射线箭头串的起点，且从该数字出发的所有射线箭头串中箭头总数等于该数字。\n现在，有如下的网格，请按上述要求给出完整的网格：\n{maze_str}",
    "你面前有一个 {n} x {m} 的箭头迷宫。\n迷宫中，X 表示空白格子，数字表示射线箭头串的起点。\n你需要在空白格子中填入箭头（↑、↓、←、→、↖、↗、↘、↙），使得从每个数字出发的所有射线箭头串中的箭头总数等于该数字。\n箭头迷宫如下所示，请填写完整的解答：\n{maze_str}",
    "这是一个 {n} x {m} 的箭头迷宫游戏。\n在迷宫中，X 表示需要填入箭头的空白格子，数字表示射线箭头串的起点。\n箭头可以是上（↑）、下（↓）、左（←）、右（→）或对角线方向（↖、↗、↘、↙）。\n每个数字代表从该位置出发的所有射线箭头串中的箭头总数。\n请完成下面的迷宫：\n{maze_str}",
    "下面是一个 {n} x {m} 的箭头迷宫网格。\nX 代表空白格子，需要填入箭头；数字代表射线箭头串的起点。\n从一个数字出发，沿各个方向可以形成射线箭头串，所有串中的箭头总数必须等于该数字。\n箭头可以是：↑、↓、←、→、↖、↗、↘、↙。\n请填写完整的迷宫解答：\n{maze_str}",
    "请解决这个 {n} x {m} 的箭头迷宫。\n其中 X 是空白格子，数字是射线箭头串的起点。\n你需要用箭头（↑、↓、←、→、↖、↗、↘、↙）填充所有空白格子。\n每个数字指示从这里发出的所有射线箭头串中箭头的总数。\n迷宫网格如下：\n{maze_str}",
    "给你一个 {n} x {m} 的箭头迷宫问题。\n迷宫中，X 表示空白位置，数字表示射线箭头串的起点。\n你的任务是用箭头填满空白位置，箭头可以是八个方向之一：↑、↓、←、→、↖、↗、↘、↙。\n每个数字表示从该位置出发的所有射线箭头串中的箭头总数应等于该数字。\n迷宫如下：\n{maze_str}",
    "这是一个箭头迷宫，大小为 {n} x {m}。\n在迷宫中，X 代表空白格子，数字代表射线起点。\n你需要在每个空白格子中填入一个箭头符号（↑、↓、←、→、↖、↗、↘、↙）。\n每个数字表示从该位置出发，沿着各个方向的箭头射线总长度。\n请完成以下迷宫：\n{maze_str}",
    "解决这个 {n} x {m} 的箭头迷宫谜题。\n迷宫中，X 是需要填入箭头的空格，数字是箭头射线的起始点。\n箭头可以是八个方向中的任意一个：↑、↓、←、→、↖、↗、↘、↙。\n每个数字表示从该点出发的所有射线箭头总数。\n迷宫网格如下所示：\n{maze_str}",
    "你需要完成一个 {n} x {m} 的箭头迷宫。\n在迷宫中，X 表示空白格子，数字表示射线箭头串的起点。\n你的任务是在空白格子中填入箭头（↑、↓、←、→、↖、↗、↘、↙）。\n每个数字指示从该位置发出的所有射线箭头串的总长度。\n请填写完整的网格：\n{maze_str}",
    "这是一个 {n} x {m} 的箭头迷宫问题。\n迷宫中的 X 代表需要填入箭头的格子，数字代表射线起点。\n箭头可以是八个方向之一：上（↑）、下（↓）、左（←）、右（→）或对角线方向（↖、↗、↘、↙）。\n每个数字等于从该点出发的所有箭头射线的总长度。\n请解决下面的迷宫：\n{maze_str}",
]
english_prompt_candidates = [
    "The maze consists of a {n} x {m} grid.\nWhere X represents an empty cell. Numbers represent the starting point of ray arrow strings.\nPlease fill in the empty cells with arrows, which can extend in the directions up (↑), down (↓), left (←), right (→), or diagonally (↖, ↗, ↘, ↙).\nThe number represents the starting point of ray arrow strings, and the total number of arrows in all ray arrow strings extending from that number equals the number.\nNow, given the following grid, please provide the complete grid according to the requirements above:\n{maze_str}",
    "Before you is a {n} x {m} arrow maze.\nIn the maze, X represents an empty cell, and numbers represent the starting points of ray arrow strings.\nYou need to fill in the empty cells with arrows (↑, ↓, ←, →, ↖, ↗, ↘, ↙), ensuring that the total number of arrows in all ray arrow strings extending from each number equals that number.\nThe arrow maze is shown below, please provide the complete solution:\n{maze_str}",
    "This is a {n} x {m} arrow maze game.\nIn the maze, X represents empty cells that need to be filled with arrows, and numbers represent the starting points of ray arrow strings.\nArrows can be up (↑), down (↓), left (←), right (→), or diagonal (↖, ↗, ↘, ↙).\nEach number represents the total number of arrows in all ray arrow strings extending from that position.\nPlease complete the following maze:\n{maze_str}",
    "Below is a {n} x {m} arrow maze grid.\nX represents empty cells that need to be filled with arrows; numbers represent the starting points of ray arrow strings.\nFrom a number, ray arrow strings can form in various directions, and the total number of arrows in all strings must equal that number.\nArrows can be: ↑, ↓, ←, →, ↖, ↗, ↘, ↙.\nPlease provide the complete maze solution:\n{maze_str}",
    "Please solve this {n} x {m} arrow maze.\nWhere X is an empty cell, and numbers are the starting points of ray arrow strings.\nYou need to fill all empty cells with arrows (↑, ↓, ←, →, ↖, ↗, ↘, ↙).\nEach number indicates the total number of arrows in all ray arrow strings extending from here.\nThe maze grid is as follows:\n{maze_str}",
    "Here's a {n} x {m} arrow maze problem for you.\nIn the maze, X indicates empty positions, and numbers indicate the starting points of ray arrow strings.\nYour task is to fill the empty positions with arrows, which can be in one of eight directions: ↑, ↓, ←, →, ↖, ↗, ↘, ↙.\nEach number indicates that the total number of arrows in all ray arrow strings extending from that position should equal that number.\nThe maze is as follows:\n{maze_str}",
    "This is an arrow maze with dimensions {n} x {m}.\nIn the maze, X represents empty cells, and numbers represent ray starting points.\nYou need to fill each empty cell with an arrow symbol (↑, ↓, ←, →, ↖, ↗, ↘, ↙).\nEach number represents the total length of arrow rays extending from that position in various directions.\nPlease complete the following maze:\n{maze_str}",
    "Solve this {n} x {m} arrow maze puzzle.\nIn the maze, X are spaces that need to be filled with arrows, and numbers are the starting points of arrow rays.\nArrows can be in any of the eight directions: ↑, ↓, ←, →, ↖, ↗, ↘, ↙.\nEach number represents the total number of all ray arrows extending from that point.\nThe maze grid is shown below:\n{maze_str}",
    "You need to complete a {n} x {m} arrow maze.\nIn the maze, X represents empty cells, and numbers represent the starting points of ray arrow strings.\nYour task is to fill in the empty cells with arrows (↑, ↓, ←, →, ↖, ↗, ↘, ↙).\nEach number indicates the total length of all ray arrow strings extending from that position.\nPlease fill in the complete grid:\n{maze_str}",
    "This is a {n} x {m} arrow maze problem.\nThe X in the maze represents cells that need to be filled with arrows, and numbers represent ray starting points.\nArrows can be in one of eight directions: up (↑), down (↓), left (←), right (→), or diagonally (↖, ↗, ↘, ↙).\nEach number equals the total length of all arrow rays extending from that point.\nPlease solve the maze below:\n{maze_str}",
]


def format_maze(maze):
    return json.dumps(maze, ensure_ascii=False, indent=4)


def prompt_arrow_maze(maze, n, m, is_chinese=False):
    maze_str = format_maze(maze)
    if is_chinese:
        prompt = random.choice(chinese_prompt_candidates)
        prompt += '\n\n附加条件：完整的迷宫网格需要在回答的末尾的python代码块中给到，代码块内仅包含二维数组，示例如下：\n```python\n[\n  ["←","↓",...],\n  ["←","↑",...],\n  ...\n]\n```'
    else:
        prompt = random.choice(english_prompt_candidates)
        prompt += '\n\nAdditional requirement: The complete maze grid should be provided in a Python code block at the end of your answer. The code block should only contain a 2D array, for example:\n```python\n[\n  ["←","↓",...],\n  ["←","↑",...],\n  ...\n]\n```'
    prompt = prompt.format(n=n, m=m, maze_str=maze_str)
    return prompt
