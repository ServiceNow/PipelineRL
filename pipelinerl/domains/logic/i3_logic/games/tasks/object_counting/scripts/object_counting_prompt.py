CHINESE_PROMPTS = [
    "下面是一个关于物品计数的问题。请仔细阅读并解答。\n\n{context}\n\n{problem}\n\n请你仔细计算，给出准确答案。请将答案填写在\\boxed{{}}中，例如：\\boxed{{answer}}。",
    "这是一个需要统计物品数量的问题。请你先整理信息，然后计算答案。\n\n{context}\n\n{problem}\n\n请先分类整理所有信息，然后一步步计算得出结果。最后将答案填入\\boxed{{}}中，例如：\\boxed{{answer}}。",
    "想象你是一位数学老师，需要解答学生提出的物品计数问题。请展示你的思考过程。\n\n学生的问题：\n{context}\n\n{problem}\n\n作为老师，请你详细分析并给出正确答案。记得将最终答案填入\\boxed{{}}中，例如：\\boxed{{answer}}。",
    "以下是一个关于数量统计的问题，请确保你的计算万无一失。\n\n{context}\n\n{problem}\n\n请仔细阅读所有信息，避免遗漏任何细节，然后给出准确答案。请使用\\boxed{{}}格式填写答案，例如：\\boxed{{answer}}。",
    "下面是一个物品计数问题，请用清晰的思路解答。\n\n{context}\n\n{problem}\n\n请先理清所有信息，按类别整理，然后计算最终答案。最后请将答案填入\\boxed{{}}中，例如：\\boxed{{answer}}。",
    "我需要你帮我解决一个物品统计问题。请按照以下步骤思考：1)识别所有物品及数量 2)按类别分组 3)执行所需计算。\n\n{context}\n\n{problem}\n\n请按步骤思考并给出最终答案。记得使用\\boxed{{}}格式给出答案，例如：\\boxed{{answer}}。",
    "想象你正在帮朋友整理物品清单并回答他的问题。请仔细阅读清单内容。\n\n物品清单：\n{context}\n\n朋友的问题：\n{problem}\n\n请帮助你的朋友解答这个问题。最终答案请放在\\boxed{{}}中，例如：\\boxed{{answer}}。",
    "请对以下物品计数问题进行严谨分析和解答。\n\n问题描述：\n{context}\n\n具体要求：\n{problem}\n\n请进行细致分析，确保计算准确无误。答案必须使用\\boxed{{}}格式，例如：\\boxed{{answer}}。",
    "嘿，我有个关于数量的小问题需要你帮忙，可以看看下面的情况：\n\n{context}\n\n我想知道：{problem}\n\n能帮我计算一下正确答案吗？请把答案放在\\boxed{{}}里，比如这样：\\boxed{{answer}}。谢谢！",
    "这是一个需要仔细计算的物品统计问题。请证明你的数学能力！\n\n{context}\n\n你的挑战是：{problem}\n\n请展示你的计算过程，并给出准确答案。答案必须放在\\boxed{{}}中，如：\\boxed{{answer}}。",
]
ENGLISH_PROMPTS = [
    "Below is an object counting problem. Please read carefully and solve it.\n\n{context}\n\n{problem}\n\nPlease calculate carefully and provide the exact answer in a \\boxed{{}} format, for example: \\boxed{{answer}}.",
    "This is a problem about counting objects. Please organize the information and then calculate the answer.\n\n{context}\n\n{problem}\n\nFirst categorize all information, then calculate the result step by step. Put your final answer in \\boxed{{}} format, like this: \\boxed{{answer}}.",
    "Imagine you are a math teacher who needs to solve an object counting problem posed by a student. Show your thinking process.\n\nStudent's problem:\n{context}\n\n{problem}\n\nAs a teacher, please analyze in detail and provide the correct answer. Remember to put your final answer in \\boxed{{}} format, for example: \\boxed{{answer}}.",
    "Here's a quantity calculation problem. Please ensure your calculation is absolutely accurate.\n\n{context}\n\n{problem}\n\nRead all information carefully, don't miss any details, then provide the precise answer. Use the \\boxed{{}} format for your answer, like this: \\boxed{{answer}}.",
    "Below is an object counting problem. Please solve it with clear thinking.\n\n{context}\n\n{problem}\n\nFirst organize all information by category, then calculate the final answer. Put your answer in \\boxed{{}} format, for example: \\boxed{{answer}}.",
    "I need you to help me solve an object counting problem. Think through these steps: 1) Identify all items and quantities 2) Group by categories 3) Perform the required calculation.\n\n{context}\n\n{problem}\n\nPlease follow the steps and provide the final answer in \\boxed{{}} format, like this: \\boxed{{answer}}.",
    "Imagine you're helping a friend organize an inventory list and answering their question. Please read the list carefully.\n\nInventory list:\n{context}\n\nFriend's question:\n{problem}\n\nPlease help your friend answer this question. Put the final answer in \\boxed{{}} format, for example: \\boxed{{answer}}.",
    "Please provide a rigorous analysis and solution to the following object counting problem.\n\nProblem description:\n{context}\n\nSpecific requirement:\n{problem}\n\nPlease conduct a detailed analysis and ensure calculation accuracy. Your answer must be in \\boxed{{}} format, for example: \\boxed{{answer}}.",
    "Hey, I have a small counting problem I need help with. Can you look at the following situation:\n\n{context}\n\nI'd like to know: {problem}\n\nCan you help me calculate the correct answer? Please put your answer in \\boxed{{}} format, like this: \\boxed{{answer}}. Thanks!",
    "This is an object counting problem that requires careful calculation. Prove your mathematical ability!\n\n{context}\n\nYour challenge is: {problem}\n\nPlease show your calculation process and provide the exact answer. Your answer must be in \\boxed{{}} format, like this: \\boxed{{answer}}.",
]
