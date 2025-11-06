import time
import requests
import asyncio
import re
from concurrent.futures import ProcessPoolExecutor
import aiohttp
import uvicorn
import logging
import signal
from contextlib import contextmanager

from omegaconf import DictConfig
import math_verify  # Ensure math_verify is installed

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from functools import partial

import pipelinerl.countdown_utils

logging.basicConfig(
    level=logging.DEBUG,  # Or INFO, WARNING, etc.
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Logs to console
    ],
)


logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    pass


class UnparsableException(Exception):
    pass


class NoAnswerException(Exception):
    pass


class EmptyBoxedException(Exception):
    pass


@contextmanager
def timeout(seconds=1):
    def timeout_handler(signum, frame):
        raise TimeoutException("Computation timed out")

    # Set the timeout handler
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield  # This is the key addition - context managers must yield
    finally:
        # Restore the original handler and disable the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


DELIMITER_STR = re.compile(r"\[END FINAL RESPONSE\]", flags=re.IGNORECASE)
ANSWER_PREFIX_RE = re.compile(
    r"^(final answer|answer|ans\.?|thus.*?is|therefore.*?is|so the answer is)[:=\-\s]*",
    re.IGNORECASE,
)


def _strip_answer_prefix(line: str) -> str:
    return ANSWER_PREFIX_RE.sub("", line).strip()


def _extract_fallback_expression(text: str) -> str | None:
    if not text:
        return None

    for raw_line in reversed(text.strip().splitlines()):
        cleaned = _strip_answer_prefix(raw_line.strip())
        cleaned = cleaned.rstrip(".;!")
        if not cleaned:
            continue
        if any(char.isdigit() for char in cleaned) or "\\" in cleaned:
            return cleaned
    return None


def strip_delimiter_strings(text: str) -> str:
    if not text:
        return text
    stripped = DELIMITER_STR.sub("", text)
    # Remove lines that became empty after sentinel stripping to avoid parsing noise
    cleaned_lines = [line for line in stripped.splitlines() if line.strip()]
    return "\n".join(cleaned_lines)


def extract_balanced_boxed_expression(text: str) -> str:
    """Return the smallest prefix containing a balanced ``\boxed{...}`` expression."""
    brace_depth = 0
    seen_open = False
    for idx, char in enumerate(text):
        if char == "{":
            brace_depth += 1
            seen_open = True
        elif char == "}" and seen_open:
            brace_depth -= 1
            if brace_depth == 0:
                return text[: idx + 1]
    return text


def verify_answer(prediction: str, gold: str, strict: bool = True, max_prediction_length: int = 1000) -> str:
    """
    Checks if a predicted answer matches a gold (correct) answer by making a request to the math_verify package.

    Args:
        prediction (str): The predicted answer to validate
        gold (str): The gold (correct) answer to compare against
        strict (bool): Whether to enforce strict comparison mode.
        - In strict mode: Variables matter and sets are not comparable with tuples
        - In non-strict mode: Variables are matched by position and sets can be compared with tuples
        url (str): URL of the validation service endpoint

    Returns:
        str: The status of the answer, which can be one of the following:
        - "correct": The prediction is correct
        - "wrong": The prediction is incorrect
        - "no_answer": The prediction is empty
        - "unparsable": The prediction cannot be parsed

    """
    if prediction.startswith("countdown"):
        return verify_countdown(prediction, gold)
    else:
        return verify_math(prediction, gold, strict=strict, max_prediction_length=max_prediction_length)


def verify_math(prediction: str, gold: str, strict: bool = True, max_prediction_length: int = 1000) -> str:
    try:
        # Input Sanitization / Validation
        if not isinstance(prediction, str) or not isinstance(gold, str):
            raise ValueError("Prediction and gold must be strings")

        prediction = strip_delimiter_strings(prediction)

        # Try extracting from \boxed{...} first
        boxed_start = prediction.rfind("\\boxed{")

        if boxed_start >= 0:
            boxed_prediction = prediction[boxed_start:]
            if "\\boxed{}" in boxed_prediction:
                raise EmptyBoxedException()
            if len(boxed_prediction) > max_prediction_length:
                raise UnparsableException()
            extracted_prediction = extract_balanced_boxed_expression(boxed_prediction)
        else:
            # Fallback: look for <answer>...</answer> tags
            answer_match = re.findall(r"<answer>(.*?)</answer>", prediction, re.DOTALL)
            if answer_match:
                extracted_prediction = strip_delimiter_strings(answer_match[-1].strip())  # last one
            else:
                fallback_expression = _extract_fallback_expression(prediction)
                if fallback_expression:
                    extracted_prediction = fallback_expression
                else:
                    raise NoAnswerException()

        if len(extracted_prediction) > max_prediction_length:
            raise UnparsableException()

        try:
            gold_parsed = math_verify.parse(gold)
            pred_parsed = math_verify.parse(extracted_prediction)
        except Exception as parse_exc:
            logger.debug("math_verify.parse failed", exc_info=parse_exc)
            raise UnparsableException() from parse_exc

        if not pred_parsed:
            raise UnparsableException("Prediction parsed to empty result.")

        try:
            with timeout(1):
                equivalent = math_verify.verify(gold_parsed, pred_parsed, strict=strict, timeout_seconds=1)
        except TimeoutException as timeout_exc:
            logger.debug("math_verify.verify timed out; treating as wrong", exc_info=timeout_exc)
            return "wrong"
        except (ValueError, TypeError, NotImplementedError) as verify_exc:
            logger.debug("math_verify.verify raised recoverable error; treating as wrong", exc_info=verify_exc)
            return "wrong"
        except Exception as verify_exc:
            logger.debug("math_verify.verify failed unexpectedly", exc_info=verify_exc)
            raise

        return "correct" if equivalent else "wrong"

    except Exception as e:
        match e:
            case NoAnswerException():
                answer_status = "no_answer"
            case (EmptyBoxedException() | UnparsableException()):
                answer_status = "unparsable"
            case _:
                logger.debug("Falling back to unparsable due to unexpected error", exc_info=e)
                answer_status = "unparsable"

    return answer_status



def verify_countdown(prediction: str, gold: str) -> str:
    target = eval(gold.split("-")[1])
    numbers = eval(gold.split("-")[2])

    equation = pipelinerl.countdown_utils.extract_solution(solution_str=prediction)

    if equation is None:
        return "no_answer"

    format_correct = pipelinerl.countdown_utils.validate_format(prediction)
    if not format_correct:
        return "unparsable"

    # Validate equation uses correct numbers
    if not pipelinerl.countdown_utils.validate_equation(equation, numbers):
        return "wrong"

    # Evaluate equation
    try:
        result = pipelinerl.countdown_utils.evaluate_equation(equation)
        if result is None:
            return "wrong"

        if abs(result - target) < 1e-5:  # Account for floating point precision
            return "correct"
        else:
            return "wrong"
    except Exception as _:
        return "wrong"


async def verify_answer_rpc(
    session: aiohttp.ClientSession,
    host: str,
    port: int,
    prediction: str,
    gold: str,
    strict: bool = True,
    max_prediction_length: int = 1000,
):
    """
    Verify the answer using the verifier API.
    """
    json = {
        "prediction": prediction,
        "gold": gold,
        "strict": strict,
        "max_prediction_length": max_prediction_length,
    }
    async with session.post(
        f"http://{host}:{port}/verify_answer",
        json=json,
    ) as response:
        if response.status == 200:
            data = await response.json()
            return data["answer_status"]
        else:
            logger.error(f"Error verifying answer: {response.status}")
            logger.error(f"Response: {await response.text()}")
            raise ValueError("Error verifying answer")


class MathEnvironment:

    def launch(self, port: int):
        """
        Serve the verification API using FastAPI.
        """
        app = FastAPI()
        # Create a process pool with 4 workers
        with ProcessPoolExecutor(max_workers=4) as process_pool:
            @app.post("/verify_answer")
            async def verify(request: dict):
                prediction = request["prediction"]
                gold = request["gold"]
                strict = request["strict"]
                max_prediction_length = request["max_prediction_length"]

                # Run verification in the process pool to avoid blocking the main thread
                loop = asyncio.get_event_loop()
                answer_status = await loop.run_in_executor(
                    process_pool, partial(verify_answer, prediction, gold, strict, max_prediction_length)
                )
                return JSONResponse(content={"answer_status": answer_status})

            @app.get("/health")
            async def health():
                return JSONResponse(content={"status": "ok"})

            uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=60)
