prompts_zh = [
    "下面是一个空间推理题：{context}回答以下问题:{question}将答案填入\\boxed{{}}中，如果答案为空则填入unknown。",
    "请分析以下树结构问题：{context}请回答：{question}请将最终答案放在\\boxed{{}}中，如果不知道答案，请写unknown。",
    "这是一道关于树结构的推理题：{context}{question}你需要把答案写在\\boxed{{}}中。如果无法确定答案，请写unknown。",
    "考虑以下树结构空间推理问题：{context}思考并回答：{question}答案需要放在\\boxed{{}}中表示，如果没有答案则写unknown。",
    "树结构推理问题：{context}根据以上信息，{question}将你的答案放在\\boxed{{}}符号内，如果找不到答案则填入unknown。",
    "以下是一个关于树结构的问题：{context}基于上述描述，请回答：{question}你的答案应该放在\\boxed{{}}内，如果无法确定则填入unknown。",
    "阅读下面关于树结构的描述：{context}请根据提供的信息回答问题：{question}答案需要用\\boxed{{}}括起来，无法确定时请填入unknown。",
    "这是一个树结构空间推理题目：{context}根据给定的树结构，{question}请将答案放在\\boxed{{}}中，若无法确定答案，请填入unknown。",
    "分析下面的树结构：{context}回答以下问题：{question}你的最终答案应该用\\boxed{{}}括起来，如果不确定，请填入unknown。",
    "观察以下树结构信息：{context}请你思考并回答：{question}将你的答案用\\boxed{{}}表示，若没有找到答案，请填写unknown。",
]
prompts_en = [
    "Below is a spatial reasoning problem: {context} Answer the following question: {question} Put your answer in \\boxed{{}}, or write 'unknown' if you can't determine the answer.",
    "Please analyze the following tree structure problem: {context} Please answer: {question} Place your final answer in \\boxed{{}}, or write 'unknown' if you don't know the answer.",
    "This is a reasoning problem about tree structure: {context} {question} You need to put your answer in \\boxed{{}}, or write 'unknown' if you're not sure.",
    "Consider the following tree structure spatial reasoning problem: {context} Think and answer: {question} Your answer should be in \\boxed{{}}, or write 'unknown' if there is no answer.",
    "Tree structure reasoning problem: {context} Based on the information above, {question} Place your answer inside \\boxed{{}}, or write 'unknown' if you can't find the answer.",
    "The following is a question about tree structure: {context} Based on the above description, please answer: {question} Your answer should be in \\boxed{{}}, or write 'unknown' if you are unsure.",
    "Read the description below about the tree structure: {context} Please answer the question based on the provided information: {question} The answer should be enclosed in \\boxed{{}}, or write 'unknown' if you cannot determine it.",
    "This is a tree structure spatial reasoning problem: {context} According to the given tree structure, {question} Please put the answer in \\boxed{{}}, or write 'unknown' if you cannot determine the answer.",
    "Analyze the tree structure below: {context} Answer the following question: {question} Your final answer should be enclosed in \\boxed{{}}, or write 'unknown' if you're not certain.",
    "Observe the following tree structure information: {context} Please think and answer: {question} Express your answer using \\boxed{{}}, or write 'unknown' if you haven't found the answer.",
]
