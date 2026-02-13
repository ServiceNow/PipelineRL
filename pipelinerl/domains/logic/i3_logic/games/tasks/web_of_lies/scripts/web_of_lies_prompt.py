import random

chinese_prompt_templates = [
    '在这个问题中,每个人要么总是说真话,要么总是说谎话。\n线索：{statements}\n问题：{questions}\n请逐步思考,在最后一行以下列格式给出你的答案：答案是 $YOUR_ANSWER。$YOUR_ANSWER 应该是由几个加粗的词组成的列表,是或否（例如,"答案是 **是,否,是**。"）',
    '在这个谜题中,我们有{num_people}个人物,每个人只会说真话或只会说谎话。\n已知：{statements}\n请回答：{questions}\n请一步一步分析,在回答的最后一行使用以下格式：答案是 $YOUR_ANSWER。$YOUR_ANSWER 应该是一个由"是"或"否"组成的加粗列表（例如,"答案是 **是,否,是**。"）',
]
english_prompt_templates = [
    'In this question, assume each person either always tells the truth or always lies.\nClues: {statements}\nQuestions: {questions}\nThink step by step, end your response in the last line with the following format: The answer is $YOUR_ANSWER. $YOUR_ANSWER should be a list of words in bold, yes or no (for example, "The answer is **yes, no, yes**.").',
    'In this puzzle, we have {num_people} people, and each person either always tells the truth or always lies. \nGiven information: {statements}\nPlease answer: {questions}\nAnalyze step by step, and end your response in the last line using this format: The answer is $YOUR_ANSWER. $YOUR_ANSWER should be a list of bolded "yes" or "no" (for example, "The answer is **yes, no, yes**.").',
]


def format_statements(statements):
    return " ".join(statements)


def format_questions(questions, is_chinese=True):
    if is_chinese:
        formatted_questions = []
        for question in questions:
            formatted_questions.append(f"{question}说的是真话吗？")
        return "，".join(formatted_questions)
    else:
        formatted_questions = []
        for question in questions:
            formatted_questions.append(f"Does {question} tell the truth?")
        return " ".join(formatted_questions)


def prompt_web_of_lies(statements, questions, num_people, is_chinese=True):
    statements_str = format_statements(statements)
    questions_str = format_questions(questions, is_chinese)
    if is_chinese:
        prompt = random.choice(chinese_prompt_templates)
    else:
        prompt = random.choice(english_prompt_templates)
    prompt = prompt.format(statements=statements_str, questions=questions_str, num_people=num_people)
    return prompt
