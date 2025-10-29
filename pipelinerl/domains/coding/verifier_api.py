import time
import requests
import asyncio
from concurrent.futures import ProcessPoolExecutor
import aiohttp
import uvicorn
import logging
import signal
from contextlib import contextmanager
import json

from omegaconf import DictConfig
import math_verify  # Ensure math_verify is installed

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from functools import partial

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

def execution_is_correct(context) -> bool:
    actual_output = context["predicted_output"]
    expected_output = context["expected_output"]
    return expected_output is None or str(actual_output).rstrip("\n") == str(expected_output).rstrip("\n")

async def old_verify_answer_rpc(
    session: aiohttp.ClientSession,
    host: str,
    port: int,
    generation: str,
    language: str,
    test_inputs: list,
    test_expected_outputs: list,
    fn_name:str = "",
    timeout: int = 10,
    memory_limit_mb: int = 1024,
):
    """Verfies if ALL the test cases pass or not"""
    solution = extract_code(generation, language)
    solution = wrap_code(solution, fn_name, language)
    test_outputs = []

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
        # logger.info(f"Shiva-Payload Code: {solution}")
        # print(f"Shiva-Payload Code: {solution}")

        _endpoint = f"http://{host}:{port}/execute_code"
        # logger.info(f"Shiva-Calling URL: {_endpoint}")

        async with session.post(
            f"http://{host}:{port}/execute_code",
            json=payload,
        ) as response:
            if response.status == 200:
                try:
                    data = await response.json()
                    # TODO: Is this safe?
                    # logger.info(f"Shiva-Execution output - full: {data}")
                    # assert "stdout" in data
                    # logger.info(f"Shiva-Execution output: {data['run_result']['stdout']}")
                    test_outputs.append({
                        "predicted_output": data['run_result']["stdout"],
                        "expected_output": test_expected_outputs[idx]
                    })
                    continue
                except Exception as e:
                    raise e
            else:
                logger.error(f"Error verifying answer: {response.status}")
                logger.error(f"Response: {await response.text()}")
                raise ValueError("Error verifying answer")

    unit_test_results = [
        execution_is_correct(context) for context in test_outputs
    ]

    # TODO: Right now we have a binary reward... rewarded only if all test cases pass.
    if all(unit_test_results):
        logger.info("Shiva-All unit tests passed.")
    else:
        logger.info("Shiva-Some unit tests failed.")

    return "correct" if all(unit_test_results) else "incorrect"

    
async def verify_answer_std_format_rpc(
    session: aiohttp.ClientSession,
    host: str,
    port: int,
    generation: str,
    reward_context: dict,
    extra_info: dict = {},
    timeout: int = 10,
    memory_limit_mb: int = 1024,
):
    language = extra_info['language']
    fn_name = reward_context.get('fn_name', "")
    test_inputs = reward_context['inputs']
    test_expected_outputs = reward_context['outputs']

    """Verfies if ALL the test cases pass or not"""
    solution = extract_code(generation, language)
    solution = wrap_code(solution, fn_name, language)
    test_outputs = []

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
        # logger.info(f"Shiva-Payload Code: {solution}")
        # print(f"Shiva-Payload Code: {solution}")

        _endpoint = f"http://{host}:{port}/execute_code"
        # logger.info(f"Shiva-Calling URL: {_endpoint}")

        async with session.post(
            f"http://{host}:{port}/execute_code",
            json=payload,
        ) as response:
            if response.status == 200:
                try:
                    data = await response.json()
                    # TODO: Is this safe?
                    # logger.info(f"Shiva-Execution output - full: {data}")
                    # assert "stdout" in data
                    # logger.info(f"Shiva-Execution output: {data['run_result']['stdout']}")
                    test_outputs.append({
                        "predicted_output": data['run_result']["stdout"],
                        "expected_output": test_expected_outputs[idx]
                    })
                    continue
                except Exception as e:
                    raise e
            else:
                logger.error(f"Error verifying answer: {response.status}")
                logger.error(f"Response: {await response.text()}")
                raise ValueError("Error verifying answer")

    unit_test_results = [
        execution_is_correct(context) for context in test_outputs
    ]

    # TODO: Right now we have a binary reward... rewarded only if all test cases pass.
    if all(unit_test_results):
        logger.info("Shiva-All unit tests passed.")
    else:
        logger.info("Shiva-Some unit tests failed.")

    return "correct" if all(unit_test_results) else "incorrect"

async def verify_answer_assert_format_rpc(
    session: aiohttp.ClientSession,
    host: str,
    port: int,
    generation: str,
    reward_context: dict,
    extra_info: dict = {},
    timeout: int = 10,
    memory_limit_mb: int = 1024,
):
    language = extra_info['language']
    fn_name = reward_context.get('fn_name', "")
    assert_stmts = reward_context['assert_case']

    """Verfies if ALL the test cases pass or not"""
    solution = extract_code(generation, language)
    # solution = wrap_code(solution, fn_name, language)
    unit_test_results = []

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
        logger.info(f"Shiva-Payload Code: {solution}")
        print(f"Shiva-Payload Code: {solution}")

        _endpoint = f"http://{host}:{port}/execute_code"
        logger.info(f"Shiva-Calling URL: {_endpoint}")

        async with session.post(
            f"http://{host}:{port}/execute_code",
            json=payload,
        ) as response:
            if response.status == 200:
                try:
                    data = await response.json()
                    # TODO: Is this safe?
                    logger.info(f"Shiva-Execution output - full: {data}")
                    # assert "stdout" in data
                    logger.info(f"Shiva-Execution stderr: {data['run_result']['stderr']}")
                    unit_test_results.append(not data['run_result']["stderr"])
                    continue
                except Exception as e:
                    raise e
            else:
                logger.error(f"Error verifying answer: {response.status}")
                logger.error(f"Response: {await response.text()}")
                raise ValueError("Error verifying answer")

    # TODO: Right now we have a binary reward... rewarded only if all test cases pass.
    if all(unit_test_results):
        logger.info("Shiva-All unit tests passed.")
    else:
        logger.info("Shiva-Some unit tests failed.")

    return "correct" if all(unit_test_results) else "incorrect"



class CodeEnvironment:
    def launch(self, port: int):
        """
        Serve the verification API using FastAPI.
        """
        app = FastAPI()
        # Create a process pool with 64 workers
        # TODO: Thread the #workers here from the cfg.
        with ProcessPoolExecutor(max_workers=64) as process_pool:
            @app.post("/execute_code")
            async def verify(request: dict):
                # logger.info(f"Shiva-Received code execution request.")
                # TODO: How to thread this URL here from the config?
                url = "http://172.20.24.132:8080/run_code"
                headers = {"Content-Type": "application/json", "Accept": "application/json"}
                async with aiohttp.ClientSession() as session:
                    # logger.info(f"Shiva-Remote call request: {request}")
                    async with session.post(url, headers=headers, json=request) as resp:
                        try:
                            result = await resp.json()
                            # logger.info(f"Shiva-Remote call result: {result}")
                            return JSONResponse(content=result)
                        except Exception as e:
                            logger.error(f"Error executing code: {e}")
                            return JSONResponse(content={"error": "Error executing code"}, status_code=500)

            @app.get("/health")
            async def health():
                return JSONResponse(content={"status": "ok"})

            uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=60)


