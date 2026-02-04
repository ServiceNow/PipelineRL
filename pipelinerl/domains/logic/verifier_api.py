from __future__ import annotations

import asyncio
import logging
import signal
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from functools import partial

import aiohttp
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Lazy imports to avoid loading i3_logic until needed
_verifier_classes = None
_Data = None


class TimeoutException(Exception):
    pass


@contextmanager
def timeout(seconds=5):
    def timeout_handler(signum, frame):
        raise TimeoutException("Verification timed out")

    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


def _ensure_imports():
    global _verifier_classes, _Data
    if _verifier_classes is None:
        from i3_logic.task2verifier import verifier_classes
        from i3_logic.base import Data
        _verifier_classes = verifier_classes
        _Data = Data


def verify_answer(prediction: str, reward_context: dict) -> str:
    _ensure_imports()

    task_type = reward_context.get("task")
    game_data = reward_context.get("game_data")

    if not task_type or not game_data:
        return "unparsable"

    # Get the verifier class for this task
    verifier_cls = _verifier_classes.get(task_type)
    if verifier_cls is None:
        logger.warning("No verifier found for task type: %s", task_type)
        return "unparsable"

    if not prediction or not prediction.strip():
        return "no_answer"

    try:
        # Parse the game data
        if isinstance(game_data, str):
            data_obj = _Data.from_json_str(game_data)
        else:
            data_obj = _Data.from_json_dict(game_data)

        # Verify with timeout
        # Pass full prediction - each verifier has its own extract_answer method
        verifier = verifier_cls()
        with timeout(5):
            result = verifier.verify(data_obj, prediction)

        # Handle different return types (bool or float score)
        if isinstance(result, bool):
            return "correct" if result else "wrong"
        elif isinstance(result, (int, float)):
            return "correct" if result > 0 else "wrong"
        else:
            return "wrong"

    except TimeoutException:
        logger.warning("Verification timed out for task %s", task_type)
        return "unparsable"
    except Exception as e:
        logger.warning("Verification error for task %s: %s", task_type, str(e))
        return "unparsable"


async def verify_answer_rpc(
    session: aiohttp.ClientSession,
    host: str,
    port: int,
    prediction: str,
    reward_context: dict,
) -> str:
    json_payload = {
        "prediction": prediction,
        "reward_context": reward_context,
    }
    async with session.post(
        f"http://{host}:{port}/verify_answer",
        json=json_payload,
    ) as response:
        if response.status == 200:
            data = await response.json()
            return data["answer_status"]
        else:
            logger.error("Error verifying answer: %s", response.status)
            logger.error("Response: %s", await response.text())
            raise ValueError("Error verifying answer")


class LogicEnvironment:
    def launch(self, port: int):
        """Start the verification API using FastAPI."""
        app = FastAPI()

        with ProcessPoolExecutor(max_workers=4) as process_pool:

            @app.post("/verify_answer")
            async def verify(request: dict):
                prediction = request["prediction"]
                reward_context = request["reward_context"]

                loop = asyncio.get_event_loop()
                answer_status = await loop.run_in_executor(
                    process_pool,
                    partial(verify_answer, prediction, reward_context),
                )
                return JSONResponse(content={"answer_status": answer_status})

            @app.get("/health")
            async def health():
                return JSONResponse(content={"status": "ok"})

            uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=60)
