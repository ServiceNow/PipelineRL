import random

prompt_candidates = {
    "You are playing a Calcudoko puzzle on an N×N grid. The grid is divided into regions, each with a target number and an operator. Your task is to fill numbers 1 to N in each row and column, ensuring no number repeats in any row, column, or region. For example, (1,1)(2,1)(3,1):12+ means the sum of numbers in those cells must be 12.": "en",
    "Welcome to Calcudoko! You'll be solving a puzzle on an N×N grid. The grid contains several regions, each with its own mathematical operation and target number. Fill in numbers 1 to N in each row and column, making sure no number repeats in any row, column, or region. For instance, (1,1)(2,1)(3,1):12+ indicates that the sum of numbers in those positions should be 12.": "en",
    "Here's a Calcudoko puzzle for you. You have an N×N grid with various regions. Each region has a specific operation and target number. Place numbers 1 to N in each row and column, ensuring no number appears twice in any row, column, or region. For example, (1,1)(2,1)(3,1):12+ means the numbers in those cells must add up to 12.": "en",
    "这是一个 N×N 的计算数独游戏。网格被分成多个区域，每个区域都有目标数字和运算符。你需要在每行和每列中填入 1 到 N 的数字，确保在每行、每列和每个区域中数字都不重复。例如，(1,1)(2,1)(3,1):12+ 表示这些位置上的数字之和必须等于 12。": "zh",
    "欢迎来到计算数独！你需要在 N×N 的网格上解决这个谜题。网格被分成几个区域，每个区域都有自己的数学运算和目标数字。在每行和每列中填入 1 到 N 的数字，确保在每行、每列和每个区域中数字都不重复。比如，(1,1)(2,1)(3,1):12+ 表示这些格子中的数字之和应该等于 12。": "zh",
    "这是一个计算数独游戏，大小为 N×N。网格中有多个区域，每个区域都指定了运算方式和目标数字。请你在每行和每列中填入 1 到 N 的数字，注意在每行、每列和每个区域中数字都不能重复。例如，(1,1)(2,1)(3,1):12+ 表示这些位置上的数字相加必须等于 12。": "zh",
}


def format_regions(regions):
    regions_str = ""
    for region in regions:
        cells = region["cells"]
        target = region["target"]
        operator = region["operator"]
        regions_str += f"({','.join(cells)}):{target}{operator}\n"
    return regions_str


def prompt_calcudoko(grid_size: int, regions: list) -> str:
    prompt = random.choice(list(prompt_candidates.keys()))
    language = prompt_candidates[prompt]
    regions_str = format_regions(regions)
    puzzle_info = f"\nThe size of the grid is {grid_size}×{grid_size}.\n{regions_str}"
    if language == "en":
        format_instructions = "\nPlease provide each element in order from left to right, and from top to bottom, with each element separated by a space and each row separated by a comma. Ensure that your final answer is wrapped in double square brackets.\n\nFor example, if the answer is:\nA B C\nD E F\nG H I\n\nplease output [[A B C,D E F,G H I]]."
    else:
        format_instructions = "\n请按照从左到右、从上到下的顺序提供每个数字，数字之间用空格分隔，行之间用逗号分隔。请确保最终答案用双方括号括起来。\n\n例如，如果答案是：\nA B C\nD E F\nG H I\n\n请输出 [[A B C,D E F,G H I]]。"
    final_prompt = prompt + puzzle_info + format_instructions
    return final_prompt
