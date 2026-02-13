import random

chinese_prompt_candidates_single = [
    "已知：{word1} {operator} {word2} = {word3}，每个字母代表一个数字（{word1} 代表是 {len1} 位数、{word2} 代表是 {len2} 位数、{word3} 代表是 {len3} 位数），字母与字母之间代表的数字不重复。若要让等式成立，求得题中的数字等式。",
    "有一个等式：{word1} {operator} {word2} = {word3}，其中每个字母代表一个不同的数字（{word1} 是 {len1} 位数、{word2} 是 {len2} 位数、{word3} 是 {len3} 位数）。请找出每个字母代表的数字，使等式成立。",
    "给定一个字母等式：{word1} {operator} {word2} = {word3}，每个字母表示一个数字，不同字母表示不同数字。其中{word1}是{len1}位数，{word2}是{len2}位数，{word3}是{len3}位数。请计算出满足条件的数字等式。",
    "在密码算术题中：{word1} {operator} {word2} = {word3}，每个字母对应一个唯一的数字（{word1}代表{len1}位数、{word2}代表{len2}位数、{word3}代表{len3}位数）。请找出使等式成立的数字组合。",
    "字谜等式：{word1} {operator} {word2} = {word3}，每个字母都代表0-9之间的一个数字，不同字母代表不同数字。其中{word1}是{len1}位数，{word2}是{len2}位数，{word3}是{len3}位数。求解这个等式的数字形式。",
]
chinese_prompt_candidates_multi = [
    "已知字母等式：{equation}，每个字母代表一个数字，不同字母代表不同数字。请找出让等式成立的数字替换方案。",
    "解决密码算术谜题：{equation}，其中每个字母表示0-9之间的一个数字，不同字母代表不同数字。请计算出使等式成立的数字等式。",
    "在字母数学题中：{equation}，每个字母都代表一个唯一的数字。请找出每个字母对应的数字，使整个等式成立。",
    "密码等式：{equation}，每个字母替换为一个数字后使等式成立，相同字母对应相同数字，不同字母对应不同数字。请求出正确的数字等式。",
    "字谜数学等式：{equation}，每个字母表示一个唯一的数字。请解出这个数学谜题，找出使等式成立的数字替换方案。",
]
english_prompt_candidates_single = [
    "Given: {word1} {operator} {word2} = {word3}, where each letter represents a unique digit ({word1} is a {len1}-digit number, {word2} is a {len2}-digit number, and {word3} is a {len3}-digit number). Find the numeric equation that makes the equality valid.",
    "In the cryptarithm: {word1} {operator} {word2} = {word3}, each letter stands for a different digit ({word1} is {len1} digits, {word2} is {len2} digits, and {word3} is {len3} digits). Determine what each letter represents to make the equation true.",
    "Solve the following alphametic puzzle: {word1} {operator} {word2} = {word3}, where each letter represents a unique digit. Note that {word1} is a {len1}-digit number, {word2} is a {len2}-digit number, and {word3} is a {len3}-digit number.",
    "In this letter arithmetic problem: {word1} {operator} {word2} = {word3}, each letter represents a distinct digit. {word1} has {len1} digits, {word2} has {len2} digits, and {word3} has {len3} digits. Find the numerical equation.",
    "For the cryptarithmetic puzzle: {word1} {operator} {word2} = {word3}, assign a unique digit to each letter so that the equation is valid. Note that {word1} is {len1} digits long, {word2} is {len2} digits long, and {word3} is {len3} digits long.",
]
english_prompt_candidates_multi = [
    "Solve this cryptarithm: {equation}, where each letter represents a unique digit. Find the digit substitution that makes the equation true.",
    "In this alphametic puzzle: {equation}, each letter represents a distinct digit from 0-9. Determine the digits that make the equation valid.",
    "Decode the following letter arithmetic: {equation}, where each letter stands for a unique digit. Find the numeric values that satisfy the equation.",
    "For the cryptarithmetic puzzle: {equation}, assign a unique digit to each letter so that the equation is valid. What is the resulting numeric equation?",
    "In this verbal arithmetic problem: {equation}, substitute each letter with a unique digit to make the equation true. Find the correct digit assignment.",
]


def prompt_cryptarithm(words, operators, is_chinese=False):
    if len(operators) == 1:
        word1, word2, word3 = (words[0], words[1], words[2])
        len1, len2, len3 = (len(word1), len(word2), len(word3))
        op = operators[0]
        if is_chinese:
            prompt = random.choice(chinese_prompt_candidates_single)
            prompt += " 请在回答的最后一行使用以下格式：答案是 $YOUR_ANSWER。$YOUR_ANSWER 应该是替换为数字后的等式。"
        else:
            prompt = random.choice(english_prompt_candidates_single)
            prompt += " Please end your response in the last line with the following format: The answer is $YOUR_ANSWER. $YOUR_ANSWER should be the equation with letters replaced by digits."
        prompt = prompt.format(word1=word1, operator=op, word2=word2, word3=word3, len1=len1, len2=len2, len3=len3)
    else:
        equation = words[0]
        for i in range(len(operators)):
            equation += f" {operators[i]} {words[i + 1]}"
        equation += f" = {words[-1]}"
        words_description = ""
        for i, word in enumerate(words):
            words_description += f"{word}是{len(word)}位数、"
        words_description = words_description[:-1]
        if is_chinese:
            prompt = random.choice(chinese_prompt_candidates_multi)
            prompt = prompt.replace("{equation}", f"{equation}（其中{words_description}）")
            prompt += " 请在回答的最后一行使用以下格式：答案是 $YOUR_ANSWER。$YOUR_ANSWER 应该是替换为数字后的等式。"
        else:
            prompt = random.choice(english_prompt_candidates_multi)
            prompt = prompt.replace("{equation}", f"{equation} (where {words_description})")
            prompt += " Please end your response in the last line with the following format: The answer is $YOUR_ANSWER. $YOUR_ANSWER should be the equation with letters replaced by digits."
    return prompt
