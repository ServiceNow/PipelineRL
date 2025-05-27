import re
import random
import ast
import operator


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except:
        return None


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None

    solution_str = solution_str.split("\n")[-1]

    matches = remove_boxed(last_boxed_only_string(solution_str))
    if matches:
        final_answer = matches.strip()
    else:
        final_answer = None
    return final_answer


def validate_format(solution_str):
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return False

    # return True

    has_think = "<think>" in solution_str
    has_end_think = "</think>" in solution_str
    if has_think and has_end_think:
        think_before_end_think = solution_str.index("<think>") < solution_str.index(
            "</think>"
        )
        has_string_in_between = (
            think_before_end_think
            and solution_str[
                solution_str.index("<think>") : solution_str.index("</think>")
            ].strip()
            != ""
        )

    end_in_box = True
    # last_box_idx = solution_str.rfind("\\boxed{")
    # if last_box_idx == -1:
    #     end_in_box = False
    # else:
    #     end_in_box = len(solution_str[last_box_idx:]) <= 100

    return (
        has_think
        and has_end_think
        and think_before_end_think
        and has_string_in_between
        and end_in_box
    )


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]

        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)

        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None
