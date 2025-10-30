import re
import json
import jsondiff
from collections import Counter


def compute_score_reward(tool, label_tool):
    if not tool and not label_tool:
        return 1.0
    if not tool or not label_tool:
        return 0.0
    diff = jsondiff.diff(tool, label_tool)
    similarity_score = 1 - len(diff) / len(label_tool)
    return similarity_score

def compute_score_binary(tool, label_tool):    
    if not tool and not label_tool:
        return 1.0
    if not tool or not label_tool:
        return -1.0
    diff = jsondiff.diff(tool, label_tool)
    if len(diff) == 0:
        return 1.0
    else:
        return -1.0

def compute_score_penalty(tool, label_tool):
    # reward will either be 1.0 if everything is correct, or in [-1.0, -0.25]
    if not tool and not label_tool:
        return 1.0
    if not tool or not label_tool:
        return -1.0
    diff = jsondiff.diff(tool, label_tool)
    if len(diff) == 0:
        return 1.0
    # fixed_penalty can be changed and then it will change the range of variable_penalty as well
    fixed_penalty = -0.25
    # with fixed_penalty = -0.5, want variable_penalty to be within [-0.5, 0]
    # label = 5, diff = 0 -> 0
    # label = 5, diff = 5 -> -0.5
    # label = 5, diff = 2 -> 0.4 * -0.5 = -0.2
    # label = 5, diff = 1 -> 0.2 * -0.5 = -0.1
    # scale by the proportion of wrong answers
    # with fixed_penalty = -0.25, want variable penalty to be within [-0.75, 0]
    # 0 * -0.75 => 0
    # 1 * -0.75 => -0.75
    variable_penalty = len(diff) / len(label_tool) * -(1 + fixed_penalty)
    # final penalty will have range [-1.0, -0.25]
    return fixed_penalty + variable_penalty

def compare_tools(tools, label_tools):
    if not tools and not label_tools:
        return 1.0  # Perfect match
    if not tools or not label_tools:
        return -1.0  # Worst case if one is empty
    
    reward = 0.0
    num_tools_matched = 0
    for label_tool in label_tools:
        if label_tool not in tools:
            reward += -1.0
        else:
            num_tools_matched += 1
            reward += compute_score_reward(tools[label_tool], label_tools[label_tool])
    # additional_tool_count = 0
    # check if model generated tools that aren't in the label and slightly penalize
    # for tool in tools:
    #     if tool not in label_tools:
    #         reward += -0.5
    #         additional_tool_count += 1
    # return reward / (len(label_tools) + additional_tool_count)
    if len(tools) > num_tools_matched:
        additional_tool_count = len(tools) - num_tools_matched
        reward += -0.5 * additional_tool_count
    else:
        additional_tool_count = 0
    return reward / (len(label_tools) + additional_tool_count)

def agentic_fn_calling_reward_func(answer, label):
    try:
        tools_str = re.findall(r"<tool_calls>(.*)</tool_calls>", answer, re.DOTALL)[0].strip()
        tools = json.loads(tools_str)
        if type(tools) != list:
            raise Exception(f"tools is not a list: {type(label_tools)}")
            
        label_tools = label["tool_calls"]
        if label_tools is None:
            label_tools = []
    except:
        return -1.0, json.dumps({"agentic_fn_calling": -1.0})
    reformatted_tools = {}
    reformatted_label_tools = {}
    for tool in tools:
        try:
            reformatted_tools[tool["name"]] = tool["arguments"]
        except:
            continue
    for label_tool in label_tools:
        try:
            reformatted_label_tools[label_tool["function"]["name"]] = json.loads(label_tool["function"]["arguments"])
        except:
            continue
    print(reformatted_tools)
    reward = compare_tools(reformatted_tools, reformatted_label_tools)
    return reward, json.dumps({"agentic_fn_calling": reward})

if __name__ == "__main__":
#     answer = """
# <tool_calls>[{"name": "run_integration_test", "arguments": {"workflow_name": "report_generation", "integrated_systems": ["database"], "test_scope": "connectivity_check", "frequency": "weekly"}}, {"name": "run_functional_test", "arguments": {"workflow_name": "report_generation", "test_scope": ["data_format", "delivery"]}}]</tool_calls>
# """
#     answer = """
# <tool_calls>[{"name": "test", "arguments": {"page": 2, "a": 2, "page_size": 4}}, {"name": "test", "arguments": {"pag": 2, "b": 2, "page_width": 3}}]</tool_calls>
# """
    answer = """
<tool_calls>[{"name": "test", "arguments": {"page": 2, "a": 2, "page_size": 4}}]</tool_calls>
"""
    # label = {
    #     "tool_calls": [{"function": {"arguments": "{\"integrated_systems\":[\"database\"],\"workflow_name\":\"report_generation\",\"test_scope\":\"connectivity_check\",\"frequency\":\"weekly\"}", "name": "run_integration_test"}, "id": "bfb4f6dec", "type": "function"}]
    # }
    label = {
        "tool_calls": [{"function": {"arguments": "{\"page\":2,\"a\":2,\"page_size\":4}", "name": "test"}, "id": "bfb4f6dec", "type": "function"}]
    }
    print(agentic_fn_calling_reward_func(answer, label))