import random

chinese_prompt_candidates = [
    "现在有一个运算表达式，{operation_symbols}，等号左边的数字缺失了，用?表示，每一个?，都代表着一个0-9之间的数字。请尝试还原该表达式。",
    "这里有个运算表达式 {operation_symbols}，等号左边的数字是缺失的，以?来呈现，每个?都意味着一个 0 到 9 之间的数字。请还原该表达式。",
    "有运算表达式 {operation_symbols}，等号左边数字缺失，用？表示，每一个？所对应的数字在 0 至 9 这个区间。请对该表达式进行还原。",
    "这儿有个运算式子 {operation_symbols}，等号左边的数字部分被问号替代，每个问号对应的数字在 0 至 9 之间。你要尝试把问号替换为正确数字，从而还原这个运算式子。",
    "当看到运算表达式 {operation_symbols} 时，会发现等号左边的数字被问号隐藏了，每个问号对应的数字都在 0 到 9 这个范围。请你通过分析，把问号还原为正确数字，展现出完整的运算表达式。",
    "现给出运算表达式 {operation_symbols}，等号左边的数字位置上是问号，且每个问号代表 0 - 9 中的一个数字。请你把问号替换成合适数字，完成表达式的还原。",
    "在一个运算式 {operation_symbols} 里，等号左侧有若干位置的数字被问号占位，每个问号所隐藏的数字取值范围在 0 到 9 之间。请你把这些隐藏数字找出来，还原完整的运算式。",
    "对于运算表达式 {operation_symbols}，等号左边有一些数字被问号取代，每个问号都蕴含着一个 0 到 9 的数字。现在，请你通过思考和计算，让问号变回原本的数字，还原整个表达式。",
    "有个运算式 {operation_symbols}，等号左边存在用问号表示的未知数字，这些问号所代表的数字都处于 0 - 9 的范围。请你把这些未知数字确定下来，还原这个运算式。",
    "给定运算表达式 {operation_symbols}，其中等号左边的数字被问号所遮蔽，而每个问号对应的数字都在 0 到 9 这个区间内。你需要做的是揭开问号的 “面纱”，还原出完整的运算表达式。",
]
english_prompt_candidates = [
    "Now there is an operation expression, {operation_symbols}, and the numbers on the left side of the equal sign are missing, represented by '?'. Each '?' represents a number between 0 and 9. Please try to restore the expression.",
    "Here is an operation expression {operation_symbols}, and the numbers on the left side of the equal sign are missing, presented as '?'. Each '?' means a number between 0 and 9. Please restore the expression.",
    "There is an operation expression {operation_symbols}, and the numbers on the left side of the equal sign are missing, represented by '?'. The number corresponding to each '?' is within the range of 0 to 9. Please restore the expression.",
    "Here is an operation expression {operation_symbols}, and the numbers on the left side of the equal sign are replaced by question marks. The number corresponding to each question mark is between 0 and 9. You need to try to replace the question marks with the correct numbers to restore this operation expression.",
    "When you see the operation expression {operation_symbols}, you will find that the numbers on the left side of the equal sign are hidden by question marks, and the number corresponding to each question mark is within the range of 0 to 9. Please analyze and restore the question marks to the correct numbers to show the complete operation expression.",
    "An operation expression {operation_symbols} is now given, and there are question marks in the positions of the numbers on the left side of the equal sign, and each question mark represents a number between 0 and 9. Please replace the question marks with appropriate numbers to complete the restoration of the expression.",
    "In an operation expression {operation_symbols}, there are several numbers on the left side of the equal sign that are occupied by question marks, and the number hidden by each question mark has a value range between 0 and 9. Please find out these hidden numbers and restore the complete operation expression.",
    "For the operation expression {operation_symbols}, some numbers on the left side of the equal sign are replaced by question marks, and each question mark contains a number between 0 and 9. Now, please think and calculate to change the question marks back to the original numbers and restore the entire expression.",
    "There is an operation expression {operation_symbols}, and there are unknown numbers represented by question marks on the left side of the equal sign. The numbers represented by these question marks are all within the range of 0 to 9. Please determine these unknown numbers and restore this operation expression.",
    "Given the operation expression {operation_symbols}, the numbers on the left side of the equal sign are obscured by question marks, and the number corresponding to each question mark is within the range of 0 to 9. What you need to do is to lift the 'veil' of the question marks and restore the complete operation expression.",
]


def prompt_mathPath(query_expr, is_chinese=True):
    if is_chinese:
        prompt = random.choice(chinese_prompt_candidates)
        prompt += "\n\n请在回答的末尾，将填充完整的运算表达式，放到 [[ 与 ]] 中间，无需其他内容，形如：\n[[2 + 4 * 3 - 4 = 10]]"
    else:
        prompt = random.choice(english_prompt_candidates)
        prompt += "\n\nPlease place the completed arithmetic expression at the end of the answer, enclosed between [[ and ]], with no additional content. For example:\n[[2 + 4 * 3 - 4 = 10]]"
    prompt = prompt.format(operation_symbols=query_expr)
    return prompt
