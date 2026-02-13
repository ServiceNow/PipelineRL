import random

import numpy as np

chinese_prompt_candidates = [
    "已知有一个{n}*{n}的二维矩阵，满足每行和每列的最后一个元素，等于该行和该列的其他元素的和。该二维矩阵为：\n{matrix}\n，其中部分元素用X代替。有一组候选数字{numbers}，可以填入矩阵中的X位置满足相应规则。请你填一下，每个数字只能使用一次，给出填充后的矩阵。",
    "现在有一个{n}*{n}的矩阵，矩阵中有些位置已经填了数字，有些位置用X表示。每行每列的最后一个数字表示该行或该列前面所有数字的和。矩阵为：\n{matrix}\n。你需要用{numbers}这些数字填充矩阵中的X，使得满足每行和每列的和等于最后一个数字，每个数字只能使用一次。",
    "给定一个{n}*{n}的数独式矩阵：\n{matrix}\n，其中X表示待填充的位置。矩阵的每行和每列的最后一个数字等于该行或该列其他数字的总和。现在给你一组数字{numbers}，请将这些数字填入X位置，使得矩阵满足要求。每个数字只能使用一次。",
    "有一个{n}*{n}的数字矩阵：\n{matrix}\n，其中部分位置标记为X需要填充。矩阵的特点是每行和每列的最后一个数字等于该行或列前面所有数字的和。请你用{numbers}中的数字填充X位置，满足条件。每个数字只能使用一次。",
    "请完成下面的Survo谜题。给定一个{n}*{n}矩阵：\n{matrix}\n，矩阵中的X需要填入数字。每行每列最后的数字代表该行或列其他数字之和。你有以下数字可以使用：{numbers}。每个数字只能使用一次，请给出填充后的完整矩阵。",
]
english_prompt_candidates = [
    "Given a {n}*{n} matrix where the last element of each row and column equals the sum of the other elements in that row or column. The matrix is:\n{matrix}\nwhere some elements are replaced with X. You have a set of numbers {numbers} that can be filled into the X positions to satisfy the rules. Please fill in the matrix. Each number can only be used once.",
    "You have a {n}*{n} matrix with some positions already filled with numbers and others marked with X. The matrix is:\n{matrix}\nThe last number in each row and column represents the sum of all other numbers in that row or column. You need to fill in the X positions using the numbers {numbers} to satisfy these conditions. Each number can only be used once.",
    "Complete the following Survo puzzle. In this {n}*{n} matrix:\n{matrix}\nthe cells marked with X need to be filled with numbers. The last number in each row and column equals the sum of all other numbers in that row or column. You can use the following numbers: {numbers}. Each number can only be used once.",
    "In this {n}*{n} Survo matrix puzzle:\n{matrix}\nthe X cells need to be filled with numbers from the set {numbers}. The last element in each row and column is the sum of all other elements in that row or column. Each number can only be used once. Provide the completed matrix.",
    "Solve this {n}*{n} matrix puzzle:\n{matrix}\nwhere X represents empty cells that need to be filled. The last number in each row and column equals the sum of all other numbers in that row or column. You have the numbers {numbers} to place in the empty cells. Each number can only be used once.",
]


def format_matrix(matrix, n):
    display_matrix = np.array(matrix, dtype=object).copy()
    for i in range(n):
        for j in range(n):
            if display_matrix[i, j] == 0:
                display_matrix[i, j] = "X"
    matrix_str = str(display_matrix.tolist())
    return matrix_str


def prompt_survo(matrix, candidate_numbers, n, is_chinese=True):
    matrix_str = format_matrix(matrix, n)
    numbers_str = str(candidate_numbers)
    if is_chinese:
        prompt = random.choice(chinese_prompt_candidates)
        prompt += "\n\n请在回答的末尾将完成后的完整矩阵放到python代码块中，注意代码块中应仅包含矩阵，无需其他内容，形如：\n```python\n[\n    [1,2,...],\n    [2,3,...],\n    ...\n]\n```"
    else:
        prompt = random.choice(english_prompt_candidates)
        prompt += "\n\nPlease provide the complete filled matrix at the end of your answer in a Python code block. The code block should contain ONLY the matrix and nothing else, like this:\n```python\n[\n    [1,2,...],\n    [2,3,...],\n    ...\n]\n```"
    prompt = prompt.format(n=n, matrix=matrix_str, numbers=numbers_str)
    return prompt
