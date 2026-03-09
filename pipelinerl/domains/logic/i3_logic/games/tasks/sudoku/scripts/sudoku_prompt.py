import random

chinese_prompt_candidates = [
    "这里有一个 9x9 的数独游戏，其中 X 表示数字被隐藏需要填写：\n{sudoku}\n请解答。",
    "下面是一个 9x9 的数独谜题，X 代表需要填入的数字：\n{sudoku}\n请解开这个数独谜题。",
    "请解决以下数独问题，X 表示需要填写的空格：\n{sudoku}\n请按照数独规则填入所有空格。",
    "这是一个标准的 9x9 数独，X 处需要填入1-9的数字：\n{sudoku}\n请完成此数独。",
    "看一下这个数独游戏，X 表示需要你填写的位置：\n{sudoku}\n请根据数独规则解决此问题。",
    "解决下面的数独题目，X 代表空白位置：\n{sudoku}\n请按照数独规则完成。",
    "这是一个 9x9 的数独板，X 代表未填写的位置：\n{sudoku}\n请填写所有的 X 位置使数独有效。",
    "试着解决这个数独，X 表示未知数字：\n{sudoku}\n请填入所有的 X。",
    "请解答这个数独谜题，X 是需要你填入的数字：\n{sudoku}\n请尝试解开此谜题。",
    "尝试解决这个 9x9 数独游戏，其中 X 是空位：\n{sudoku}\n请完成这个数独游戏。",
]
english_prompt_candidates = [
    "Here is a 9x9 Sudoku puzzle, where X represents a hidden number that needs to be filled in:\n{sudoku}\nPlease solve it.",
    "Below is a 9x9 Sudoku puzzle, where X represents empty cells that need to be filled:\n{sudoku}\nPlease solve this Sudoku puzzle.",
    "Solve the following Sudoku puzzle, where X indicates an empty space:\n{sudoku}\nPlease fill in all the empty spaces according to Sudoku rules.",
    "This is a standard 9x9 Sudoku, where X needs to be filled with digits 1-9:\n{sudoku}\nPlease complete this Sudoku.",
    "Take a look at this Sudoku game, where X marks the positions you need to fill:\n{sudoku}\nPlease solve this puzzle according to Sudoku rules.",
    "Solve the Sudoku puzzle below, where X represents blank positions:\n{sudoku}\nPlease complete it according to Sudoku rules.",
    "Here is a 9x9 Sudoku board, where X represents unfilled positions:\n{sudoku}\nPlease fill in all X positions to make a valid Sudoku.",
    "Try to solve this Sudoku, where X represents unknown digits:\n{sudoku}\nPlease fill in all the Xs.",
    "Please solve this Sudoku puzzle, where X represents the numbers you need to fill in:\n{sudoku}\nPlease try to solve this puzzle.",
    "Attempt to solve this 9x9 Sudoku game, where X represents empty cells:\n{sudoku}\nPlease complete this Sudoku game.",
]
chinese_rules = [
    "数独的规则很简单：每一行、每一列和每个3x3的小方格都必须包含1到9的数字，且不能重复。",
    "请按照数独规则（每行、每列、每个3x3宫格内数字1-9不重复）解答。",
    "数独规则：填入数字1-9，使得每行、每列和每个3x3子网格中的数字都不重复。",
    "解数独的规则是：在每一行、每一列和每个3x3的方格中，数字1-9各出现一次。",
    "数独游戏规则：每一行、每一列和每个3×3的小九宫格内，数字1-9都只能出现一次。",
]
english_rules = [
    "The rules of Sudoku are simple: each row, column, and 3x3 box must contain the numbers 1-9 without repetition.",
    "Please solve according to Sudoku rules (numbers 1-9 must appear exactly once in each row, column, and 3x3 box).",
    "Sudoku rules: Fill in digits 1-9 so that each digit appears exactly once in each row, column, and 3x3 sub-grid.",
    "The rules for solving Sudoku: in each row, column, and 3x3 box, the digits 1-9 must occur exactly once.",
    "Rules of the Sudoku game: Each row, column, and 3×3 box must contain the numbers 1-9 with no repetitions.",
]
chinese_output_format = [
    "请在最后使用Python的markdown代码块给出答案，以元组形式表示，例如：```python\n((1,2,3,4,5,6,7,8,9),(4,5,6,7,8,9,1,2,3),...)\n```"
]
english_output_format = [
    "Please provide your answer at the end using a Python markdown code block, represented as a tuple, for example: ```python\n((1,2,3,4,5,6,7,8,9),(4,5,6,7,8,9,1,2,3),...)\n```"
]


def format_sudoku_grid(sudoku_grid):
    formatted_rows = []
    for row in sudoku_grid:
        formatted_row = ""
        for cell in row:
            if cell == 0 or cell == "X" or cell == "x":
                formatted_row += "X"
            else:
                formatted_row += str(cell)
        formatted_rows.append(formatted_row)
    return "\n".join(formatted_rows)


def prompt_sudoku(sudoku_grid, is_chinese=False):
    formatted_sudoku = format_sudoku_grid(sudoku_grid)
    if is_chinese:
        prompt_template = random.choice(chinese_prompt_candidates)
        rules = random.choice(chinese_rules)
        output_format = random.choice(chinese_output_format)
    else:
        prompt_template = random.choice(english_prompt_candidates)
        rules = random.choice(english_rules)
        output_format = random.choice(english_output_format)
    prompt = prompt_template.format(sudoku=formatted_sudoku)
    prompt = f"{prompt}\n\n{rules}\n\n{output_format}"
    return prompt
