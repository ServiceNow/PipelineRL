import requests
import logging
# from fastapi.responses import JSONResponse
import aiohttp
import asyncio

logging.basicConfig(
    level=logging.DEBUG,  # Or INFO, WARNING, etc.
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Logs to console
    ],
)

logger = logging.getLogger(__name__)


def extract_code(completion: str, language: str) -> str:
    """
    Extracts code from a model completion, handling code blocks if present.
    Code taken from NowVerl
    """
    solution = completion
    if f"```{language}" in completion:
        solution = completion.split("```python")[-1].split("```")[0]
    elif "```" in completion:
        # Handle cases like ```\ncode\n```
        parts = completion.split("```")
        if len(parts) >= 2:
            solution = parts[1]
            # Remove potential language specifier like 'python\n'
            if "\n" in solution:
                first_line, rest = solution.split("\n", 1)
                if first_line.strip().isalpha():  # Simple check for language name
                    solution = rest

    return solution


def wrap_code(code: str, fn_name: str, language: str) -> str:
    if language == "python":
        code = f"""
import traceback
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json

# === User's Original Code START ===
{code}
# === User's Original Code END ===

_SANDBOX_FN_NAME = "{fn_name}"

def _execute_user_function():
    # --- Input Parsing ---
    _raw_input_str = sys.stdin.read()
    _args = []
    if _raw_input_str.strip(): # If there's input
        try:
            _args = [json.loads(line) for line in _raw_input_str.split('\\n')]
        except json.JSONDecodeError as _je:
            sys.stderr.write(f"WrapperError: Invalid JSON input for '{{_SANDBOX_FN_NAME}}': {{_je}}\\nInput was: "
                              f"{{_raw_input_str[:200]}}\\n")
            return None, True # result, error_occurred

    # --- Function Location and Execution ---
    try:
        _target_callable = None
        # Try global scope first
        if _SANDBOX_FN_NAME in globals():
            _target_callable = globals()[_SANDBOX_FN_NAME]
        # Else, if 'Solution' class exists, try to get its method
        elif 'Solution' in globals():
            _Solution_class = globals()['Solution']
            # Attempt to instantiate and get method.
            # Errors (e.g., Solution not a class, instantiation fails, method missing)
            # will be caught by the broad except block below.
            _solution_instance = _Solution_class()
            _target_callable = getattr(_solution_instance, _SANDBOX_FN_NAME)

        if not _target_callable:
            sys.stderr.write(f"WrapperError: Function or method '{{_SANDBOX_FN_NAME}}' not found.\\n")
            return None, True # result, error_occurred

        _fn_result = _target_callable(*_args)
        return _fn_result, False # result, no_error
    except Exception: # Catches errors from Solution instantiation, getattr, or function call
        sys.stderr.write(f"Error during setup or execution of '{{_SANDBOX_FN_NAME}}':\\n{{traceback.format_exc()}}\\n")
        return None, True # result, error_occurred

if __name__ == '__main__':
    _result, _error_occurred = _execute_user_function()

    if not _error_occurred:
        # Serialize result to stdout
        if isinstance(_result, (dict, list, tuple)) or _result is None or isinstance(_result, bool):
            print(json.dumps(_result))
        elif isinstance(_result, (int, float, str)):
            print(str(_result)) # Ensure string conversion for print
        else:
            # For other types, default to string representation.
            print(str(_result))
    # Optional: To explicitly exit with an error code if the sandbox relies on it
    # else:
    #    sys.exit(1)
"""
    return code


class RemoteCodeExecutor:
    def __init__(self, server_url: str, timeout=10, memory_limit_mb=1024):
        self.server_url = server_url
        self.request_headers = headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb

    def _construct_remote_code_execution_contexts_std(
        self, generation: str, reward_context: dict, extra_info: dict
    ):
        """
        Construct the context for remote code execution for stdio-based verification.
        Verifies if all assert statements pass or not.
        Args:
            generation: The generated code.
            reward_context: The reward context containing assert statements.
            extra_info: Additional information such as programming language.
        Returns:
            request dictionary for remote execution.
        """
        language = extra_info["language"]
        timeout = extra_info.get("timeout", 10)
        memory_limit_mb = extra_info.get("memory_limit_mb", 1024)

        fn_name = reward_context.get("fn_name", "")
        test_inputs = reward_context["inputs"]
        test_expected_outputs = reward_context["outputs"]

        """Verfies if ALL the test cases pass or not"""
        solution = extract_code(generation, language)
        solution = wrap_code(solution, fn_name, language)
        payloads = []
        for idx, stdin in enumerate(test_inputs):
            payload = {
                "compile_timeout": timeout,
                "run_timeout": timeout,
                "code": solution,
                "stdin": stdin,
                "memory_limit_MB": memory_limit_mb,
                "language": language,
                "files": {},
                "fetch_files": [],
            }
            payloads.append(payload)

        return payloads, test_expected_outputs

    def _construct_remote_code_execution_contexts_assert(
        self, generation: str, reward_context: dict, extra_info: dict
    ):
        """
        Construct the context for remote code execution for assert-based verification.
        Verifies if all assert statements pass or not.
        Args:
            generation: The generated code.
            reward_context: The reward context containing assert statements.
            extra_info: Additional information such as programming language.
        Returns:
            request dictionary for remote execution.
        """
        language = extra_info["language"]
        timeout = extra_info.get("timeout", 10)
        memory_limit_mb = extra_info.get("memory_limit_mb", 1024)
        assert_stmts = reward_context["assert_case"]

        solution = extract_code(generation, language)
        payloads = []
        for idx, assert_stmt in enumerate(assert_stmts):
            payload = {
                "compile_timeout": timeout,
                "run_timeout": timeout,
                "code": solution + "\n" + assert_stmt,
                # "stdin": stdin,
                "memory_limit_MB": memory_limit_mb,
                "language": language,
                "files": {},
                "fetch_files": [],
            }
            payloads.append(payload)

        return payloads, None

    def _construct_remote_code_execution_contexts(
        self, generation: str, reward_context: dict, extra_info: dict
    ):
        match reward_context.get("call_type", ""):
            case "std":
                return self._construct_remote_code_execution_contexts_std(
                    generation, reward_context, extra_info
                )

            case "assert":
                return self._construct_remote_code_execution_contexts_assert(
                    generation, reward_context, extra_info
                )

            case _:
                raise ValueError(
                    f"Unknown Code execution call_type: {reward_context.get('call_type', '')}"
                )

    async def _forward_request(self, session, payload):
        async with session.post(
            self.server_url, headers=self.request_headers, json=payload
        ) as resp:
            try:
                result = await resp.json()
                return result
            except Exception as e:
                logger.error(f"Error executing code: {e}")
                return {"error": "Error executing code"}

    async def _forward_requests(self, session, requests: list[dict]) -> list[dict]:
        # Call forward_request for each request in the list asynchronously
        # TODO: Can this lead to token exhaustion?
        # Use the passed session instead of creating a new one
        tasks = [self._forward_request(session, payload) for payload in requests]
        results = await asyncio.gather(*tasks)
        return results

    def _verify_assert_executions(
        self, responses: list[dict], expected: list[str] | None
    ):
        unit_test_results = [
            "run_result" in data and (not data["run_result"].get("stderr", ""))
            for data in responses
        ]
        logger.info(unit_test_results)
        return "coding_correct" if all(unit_test_results) else "coding_incorrect"

    def _execution_is_correct(self, actual_output, expected_output) -> bool:
        return expected_output is None or str(actual_output).rstrip("\n") == str(
            expected_output
        ).rstrip("\n")

    def _verify_std_executions(self, responses: list[dict], expected: list[str] | None):
        unit_test_results = [
            "run_result" in data
            and self._execution_is_correct(data["run_result"].get("stdout", ""), exp)
            for data, exp in zip(responses, expected)
        ]

        logger.info(responses)
        return "coding_correct" if all(unit_test_results) else "coding_incorrect"

    async def execute(self, session, generation: str, reward_context: dict, extra_info: dict):
        payloads, expected = self._construct_remote_code_execution_contexts(
            generation, reward_context, extra_info
        )

        responses = await self._forward_requests(session, payloads)
        
        results = [resp for resp in responses]
        match reward_context.get("call_type", ""):
            case "std":
                return self._verify_std_executions(results, expected)

            case "assert":
                return self._verify_assert_executions(results, expected)

            case _:
                raise ValueError(
                    f"Unknown Code execution call_type: {reward_context.get('call_type', '')}"
                )

if __name__ == "__main__":
    # from datasets import load_dataset
    # import os

    # ds = load_dataset(
    #         "ServiceNow-AI/mixed-training-text-datasets", 
    #         token=os.environ.get("HUGGINGFACE_TOKEN", None),
    #         split="train",
    #         trust_remote_code=True)

    # dataset = ds.filter(lambda sample: sample['ability'] == 'code')[:5]

    code_executor = RemoteCodeExecutor(
        server_url='http://172.20.24.132:8080/run_code'
    )
    generation="""
def add_one(x):
    return x + 1

if __name__ == '__main__':
    import sys
    input_data = sys.stdin.read()
    x = int(input_data)
    print(add_one(x))
"""

    reward_context = {
        'call_type': 'std',
        'inputs': ['1', '5', '-3'],
        'outputs': ['2', '9', '-2'],
        'fn_name': 'add_one',
    }
    
    extra_info = {'language': 'python'}

    async def main():
        async with aiohttp.ClientSession() as session:
            result = await code_executor.execute(
                session,
                generation,
                reward_context,
                extra_info
            )
            print(result)

    asyncio.run(main())


# generation="""
# def add_one(x):
#     return x + 1
# """
# assert1 = 'assert add_one(1) == 2'
# assert2 = 'assert add_one(5) == 4'
# assert3 = 'assert add_one(-3) == -2'