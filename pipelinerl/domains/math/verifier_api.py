import asyncio
import logging
import re
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from functools import partial

import aiohttp
import math_verify
import requests  # noqa: F401 - retained for parity with upstream
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

import pipelinerl.countdown_utils

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
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
def timeout(seconds: int = 1):
    def timeout_handler(signum, frame):
        raise TimeoutException("Computation timed out")

    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


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
        cleaned = _strip_answer_prefix(raw_line.strip()).rstrip(".;!")
        if cleaned and (any(ch.isdigit() for ch in cleaned) or "\\" in cleaned):
            return cleaned
    return None


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        if not s.startswith(left):
            raise UnparsableException()
        return s[len(left) :]

    left = "\\boxed{"
    if not s.startswith(left) or not s.endswith("}"):
        raise UnparsableException()
    return s[len(left) : -1]


def last_boxed_only_string(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if "\\boxed " in text:
        return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None

    right_brace_idx = None
    num_left_braces_open = 0
    i = idx
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        if text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return text[idx : right_brace_idx + 1]


def fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        return f"\\frac{{{a}}}{{{b}}}"
    except (AssertionError, ValueError):
        return string


def remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            a = split[0]
            new_substr = f"\\sqrt{{{a}}}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string: str) -> str:
    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)
    return string


def is_equiv(str1: str, str2: str) -> bool:
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def verify_math(prediction: str, gold: str, strict: bool = True, max_prediction_length: int = 1000) -> str:
    try:
        if not isinstance(prediction, str) or not isinstance(gold, str):
            raise ValueError("Prediction and gold must be strings")

        prediction = prediction.strip()
        if not prediction:
            raise NoAnswerException()

        extracted_prediction: str | None = None

        boxed_prediction = last_boxed_only_string(prediction)
        if boxed_prediction is not None:
            try:
                extracted_prediction = remove_boxed(boxed_prediction).strip()
            except UnparsableException as exc:
                logger.debug("Failed to remove boxed expression", exc_info=exc)
                extracted_prediction = None

        if not extracted_prediction:
            answer_match = re.findall(r"<answer>(.*?)</answer>", prediction, re.DOTALL)
            if answer_match:
                extracted_prediction = answer_match[-1].strip()
            else:
                fallback_expression = _extract_fallback_expression(prediction)
                if fallback_expression:
                    extracted_prediction = fallback_expression.strip()
                else:
                    raise NoAnswerException()

        if not extracted_prediction:
            raise EmptyBoxedException()

        if 0 < max_prediction_length < len(extracted_prediction):
            raise UnparsableException()

        if is_equiv(gold, extracted_prediction):
            return "correct"

        try:
            target_boxed = last_boxed_only_string(f"\\boxed{{{gold}}}") or f"\\boxed{{{gold}}}"
            pred_boxed = last_boxed_only_string(f"\\boxed{{{extracted_prediction}}}") or f"\\boxed{{{extracted_prediction}}}"
            gold_parsed = math_verify.parse(target_boxed)
            pred_parsed = math_verify.parse(pred_boxed)
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

    except Exception as error:
        match error:
            case NoAnswerException():
                answer_status = "no_answer"
            case (EmptyBoxedException() | UnparsableException()):
                answer_status = "unparsable"
            case _:
                logger.debug("Unexpected verifier error", exc_info=error)
                answer_status = "unparsable"
        return answer_status


def verify_countdown(prediction: str, gold: str) -> str:
    target = eval(gold.split("-")[1])
    numbers = eval(gold.split("-")[2])

    equation = pipelinerl.countdown_utils.extract_solution(solution_str=prediction)

    if equation is None:
        return "no_answer"

    if not pipelinerl.countdown_utils.validate_format(prediction):
        return "unparsable"

    if not pipelinerl.countdown_utils.validate_equation(equation, numbers):
        return "wrong"

    try:
        result = pipelinerl.countdown_utils.evaluate_equation(equation)
        if result is None:
            return "wrong"
        return "correct" if abs(result - target) < 1e-5 else "wrong"
    except Exception:
        return "wrong"


def verify_answer(prediction: str, gold: str, strict: bool = True, max_prediction_length: int = 1000) -> str:
    try:
        if prediction.startswith("countdown"):
            return verify_countdown(prediction, gold)
        return verify_math(prediction, gold, strict=strict, max_prediction_length=max_prediction_length)
    except NoAnswerException:
        return "no_answer"
    except UnparsableException:
        return "unparsable"
    except Exception as exc:
        logger.debug("verify_answer unexpected failure", exc_info=exc)
        return "unparsable"


async def verify_answer_rpc(
    session: aiohttp.ClientSession,
    host: str,
    port: int,
    prediction: str,
    gold: str,
    strict: bool = True,
    max_prediction_length: int = 1000,
):
    payload = {
        "prediction": prediction,
        "gold": gold,
        "strict": strict,
        "max_prediction_length": max_prediction_length,
    }
    async with session.post(f"http://{host}:{port}/verify_answer", json=payload) as response:
        if response.status == 200:
            data = await response.json()
            return data["answer_status"]
        logger.error("Error verifying answer: %s", response.status)
        logger.error("Response: %s", await response.text())
        raise ValueError("Error verifying answer")


class MathEnvironment:
    def launch(self, port: int):
        app = FastAPI()
        with ProcessPoolExecutor(max_workers=4) as process_pool:
            @app.post("/verify_answer")
            async def verify(request: dict):
                prediction = request["prediction"]
                gold = request["gold"]
                strict = request["strict"]
                max_prediction_length = request["max_prediction_length"]

                loop = asyncio.get_event_loop()
                answer_status = await loop.run_in_executor(
                    process_pool, partial(verify_answer, prediction, gold, strict, max_prediction_length)
                )
                return JSONResponse(content={"answer_status": answer_status})

            @app.get("/health")
            async def health():
                return JSONResponse(content={"status": "ok"})

            uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=60)
