import random

chinese_prompt_candidates = [
    "请补全下面序列的其余部分，确保括号正确闭合。支持的括号类型包括：圆括号(), 方括号[], 花括号{{}}, 尖括号<>。\n{{sequence}}\n\n附加要求：请直接输出补全完整后的序列，不要包含其他内容。\n\n示例输出格式：\n<()[]>",
    "下面是一个不完整的括号序列，请补充完整使其合法。序列中可以包含以下类型的括号：圆括号(), 方括号[], 花括号{{}}, 尖括号<>。\n{{sequence}}\n\n请在回答中直接输出完整的序列，不要有其他文字。格式如下：\n([{}])",
    "给定一个不完整的括号序列，请补充完整。可用的括号类型有：圆括号(), 方括号[], 花括号{{}}, 尖括号<>。\n{{sequence}}\n\n要求：补充后的序列必须合法，且补充的部分要最短。请直接输出完整序列，不要包含解释或其他内容。参考格式：\n{<>}[()]",
    "这是一个需要补全的括号序列。支持的括号包括：圆括号(), 方括号[], 花括号{{}}, 尖括号<>。\n{{sequence}}\n\n请补充完整，确保所有括号都正确闭合。直接输出完整序列，不要包含任何额外的说明文字。示例格式：\n([<>])",
    "请完成下面的括号序列，使其成为一个合法的括号序列。序列中的括号可以是以下类型：圆括号(), 方括号[], 花括号{{}}, 尖括号<>。\n{{sequence}}\n\n补充要求：使用最短的补充序列。直接输出完整结果，只需包含最终的序列。输出格式示例：\n{[()]}",
]
english_prompt_candidates = [
    "Complete the following sequence to ensure all brackets are properly closed. Supported bracket types include: parentheses (), square brackets [], curly braces {{}}, angle brackets <>.\n{{sequence}}\n\nPlease output the complete sequence directly. Example format:\n<()[]>",
    "Given an incomplete bracket sequence, please complete it. The sequence can contain these types of brackets: parentheses (), square brackets [], curly braces {{}}, angle brackets <>.\n{{sequence}}\n\nRequirements: The completed sequence must be valid, and the added part should be minimal. Output only the complete sequence. Like this:\n([{}])",
    "Here is a bracket sequence that needs to be completed. Available bracket types are: parentheses (), square brackets [], curly braces {{}}, angle brackets <>.\n{{sequence}}\n\nPlease complete it to make all brackets properly closed. Output only the complete sequence with no additional text. Reference format:\n{<>}[()]",
    "Complete the following sequence to make it a valid bracket sequence. The sequence may contain: parentheses (), square brackets [], curly braces {{}}, angle brackets <>.\n{{sequence}}\n\nRequirement: Use the minimal completion. Output only the final sequence. Example:\n([<>])",
    "Please complete the following sequence to ensure proper bracket closure. Supported brackets are: parentheses (), square brackets [], curly braces {{}}, angle brackets <>.\n{{sequence}}\n\nNote: The completion should be minimal. Output only the sequence itself. Output format:\n{[()]}",
]


def prompt_dyck_language(sequence: str, is_chinese: bool = False) -> str:
    if is_chinese:
        prompt = random.choice(chinese_prompt_candidates)
    else:
        prompt = random.choice(english_prompt_candidates)
    prompt = prompt.replace("{{sequence}}", sequence)
    return prompt
