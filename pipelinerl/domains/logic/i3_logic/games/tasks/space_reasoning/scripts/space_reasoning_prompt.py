prompts_zh = [
    "请分析这个空间情境：{context}，并回答问题：{question}。答案需放入\\boxed{{}}中，确保与题目中的物品名称完全一致。如果没有物品则输出\\boxed{{unknown}}。",
    "根据所描述的场景：{context}，请解答：{question}。将答案填写在\\boxed{{}}内，注意必须与题目中提到的物品名称完全相符。如果没有物品则输出\\boxed{{unknown}}。",
    "分析下述空间关系：{context}。问题是：{question}。答案请用\\boxed{{}}标注，并确保答案与题目中的物品名称一字不差。如果没有物品则输出\\boxed{{unknown}}。",
    "给定以下空间信息：{context}，回答问题：{question}。答案必须放在\\boxed{{}}中，且与题目所列物品名称保持完全一致。如果没有物品则输出\\boxed{{unknown}}。",
    "仔细阅读这个情境：{context}，然后思考：{question}？请将最终答案放入\\boxed{{}}中，答案必须与题目中出现的物品名称完全相同。如果没有物品则输出\\boxed{{unknown}}。",
    "情境描述：{context}。请解决这个问题：{question}。将正确答案填入\\boxed{{}}，务必确保答案与题目中的物品名称完全一致。如果没有物品则输出\\boxed{{unknown}}。",
    "通过逻辑推理分析：{context}，找出问题{question}的答案。答案需使用\\boxed{{}}包围，并与题目中的物品名称完全一致。如果没有物品则输出\\boxed{{unknown}}。",
    "空间场景如下：{context}。需要解答的问题是：{question}。请用\\boxed{{}}框出你的答案，答案必须与题目中的物品名称完全匹配。如果没有物品则输出\\boxed{{unknown}}。",
    "读完这段描述：{context}，请回答以下问题：{question}。答案放在\\boxed{{}}中，注意答案必须与题目中提到的物品名称完全相同。如果没有物品则输出\\boxed{{unknown}}。",
    "基于以下场景信息：{context}，解答问题：{question}。回答时使用\\boxed{{}}格式，并确保答案与题目中提及的物品名称完全一致。如果没有物品则输出\\boxed{{unknown}}。",
]
prompts_en = [
    "Analyze this spatial scenario: {context}. Answer: {question}. Place your answer in \\boxed{{}} and ensure it exactly matches the item names in the problem. If there is no item, output \\boxed{{unknown}}.",
    "Based on the given spatial information: {context}, please solve: {question}. Your answer must be inside \\boxed{{}} and identical to the item names mentioned. If there is no item, output \\boxed{{unknown}}.",
    "Examine the following spatial arrangement: {context}. The question is: {question}. Put the answer in \\boxed{{}}, making sure it matches exactly with the item names in the problem. If there is no item, output \\boxed{{unknown}}.",
    "Consider these spatial relationships: {context}. Solve this question: {question}. Format your answer within \\boxed{{}}, using only the exact item names from the problem. If there is no item, output \\boxed{{unknown}}.",
    "Read this scenario carefully: {context}. What is the answer to: {question}? Enclose your final answer in \\boxed{{}}, using the exact item names as they appear in the problem. If there is no item, output \\boxed{{unknown}}.",
    "The spatial context is: {context}. Please determine: {question}. Your solution should be placed in \\boxed{{}} and must match the item names stated in the problem precisely. If there is no item, output \\boxed{{unknown}}.",
    "Using logical reasoning, analyze: {context} and solve: {question}. Present your answer in \\boxed{{}}, ensuring it contains the exact item names from the original problem. If there is no item, output \\boxed{{unknown}}.",
    "Spatial scenario: {context}. Question: {question}. Format the answer within \\boxed{{}}, using only the identical item names mentioned in the problem. If there is no item, output \\boxed{{unknown}}.",
    "After reading this description: {context}, respond to this question: {question}. The answer must be in \\boxed{{}} format and match exactly with the item names from the problem. If there is no item, output \\boxed{{unknown}}.",
    "From the information provided: {context}, determine the answer to: {question}. Write your answer in \\boxed{{}}, using precisely the same item names as given in the problem. If there is no item, output \\boxed{{unknown}}.",
]
