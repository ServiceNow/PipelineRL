import random

prompt_candidates = {
    "Given the numbers {numbers}, apply the arithmetic operations {operators} to get the result of {result}.": "en",
    "Using the numbers {numbers}, figure out how to combine them with the arithmetic operations {operators} to equal {result}.": "en",
    "With the numbers {numbers} at hand, use the arithmetic operations {operators} to yield {result}.": "en",
    "Try to make {result} by performing arithmetic operations {operators} on the numbers {numbers}.": "en",
    "From the numbers {numbers}, find a way to use the arithmetic operations {operators} to reach the value of {result}.": "en",
    "Given these numbers {numbers}, can you apply the arithmetic operations {operators} to calculate {result}?": "en",
    "Employ the arithmetic operations {operators} on the numbers {numbers} to obtain a sum of {result}.": "en",
    "Using only the numbers {numbers} and the arithmetic operations {operators}, construct an expression that equals {result}.": "en",
    "See if you can get {result} by performing combinations of the arithmetic operations {operators} on the numbers {numbers}.": "en",
    "Given the set of numbers {numbers}, find a combination of the arithmetic operations {operators} that results in {result}.": "en",
    "Use the numbers {numbers} and perform appropriate arithmetic operations {operators} to achieve the target value of {result}.": "en",
    "Try to calculate {result} by applying the operations {operators} to the numbers {numbers}.": "en",
    "From the numbers {numbers}, determine how to use the arithmetic operations {operators} to produce a value of {result}.": "en",
    "Given {numbers}, work out an arithmetic expression using {operators} that gives {result} as the answer.": "en",
    "With the numbers {numbers}, find a way to use the arithmetic operations {operators} to make the total {result}.": "en",
    "Can you use the numbers {numbers} and the arithmetic operations {operators} to get {result}?": "en",
    "Use the set of numbers {numbers} and the operations {operators} to come up with an equation that equals {result}.": "en",
    "Given the numbers {numbers}, apply the arithmetic operations {operators} in a way that the outcome is {result}.": "en",
    "Try to manipulate the numbers {numbers} using the arithmetic operations {operators} to arrive at {result}.": "en",
    "Using the numbers {numbers}, find a sequence of arithmetic operations {operators} that will result in the value of {result}.": "en",
    "给定数字 {numbers}，使用算术运算 {operators} 得到 {result}。": "zh",
    "用数字 {numbers}，想办法通过算术运算 {operators} 计算出 {result}。": "zh",
    "仅使用数字 {numbers} 和算术运算 {operators}，构造一个等于 {result} 的算式。": "zh",
    "试试用算术运算 {operators} 结合数字 {numbers} 计算出 {result}。": "zh",
    "从数字 {numbers} 中，找出一种使用算术运算 {operators} 得到 {result} 的方法。": "zh",
    "你能用数字 {numbers} 和算术运算 {operators} 算出 {result} 吗？": "zh",
    "通过对数字 {numbers} 进行算术运算 {operators}，看看是否能得到 {result}。": "zh",
    "试着运用算术运算 {operators} 组合数字 {numbers}，让结果等于 {result}。": "zh",
    "给定这些数字 {numbers}，能否找到合适的算术运算 {operators} 使其等于 {result}？": "zh",
    "你能使用数字 {numbers} 及算术运算 {operators}，得出 {result} 吗？": "zh",
    "用数字 {numbers}，通过合适的算术运算 {operators}，使其计算结果为 {result}。": "zh",
    "试试用 {operators} 对数字 {numbers} 进行运算，使最终结果为 {result}。": "zh",
    "设法利用算术运算 {operators} 处理数字 {numbers}，使其等于 {result}。": "zh",
    "运用数字 {numbers} 和算术运算 {operators}，找出一种方法得到 {result}。": "zh",
    "仅用数字 {numbers} 和算术运算 {operators}，构造一个等式，其结果为 {result}。": "zh",
    "你能找到一种用 {operators} 计算 {numbers} 的方法，使结果为 {result} 吗？": "zh",
    "试试看如何运用算术运算 {operators} 让 {numbers} 的结果等于 {result}。": "zh",
    "通过合理组合 {numbers} 和算术运算 {operators}，计算出 {result}。": "zh",
    "设法使用数字 {numbers} 和算术运算 {operators}，使最终答案是 {result}。": "zh",
    "你能用算术运算 {operators} 让 {numbers} 的运算结果等于 {result} 吗？": "zh",
}


def prompt_game_of_24(numbers: list[int], operators: list[str], result: int) -> str:
    prompt = random.choice(list(prompt_candidates.keys()))
    language = prompt_candidates[prompt]
    if language == "en":
        prompt += " At the end of your response, please output a ```python code block. The code block should contain only a single expression representing the answer, which can be directly evaluated using Python's eval() function."
    elif language == "zh":
        prompt += "在回答的最后，请输出一个 ```python 代码块。代码块中的仅包含一个代表答案的表达式，并且该表达式可以直接被 Python 中的 eval() 函数求值。"
    prompt = prompt.format(numbers=numbers, operators=operators, result=result)
    return prompt
