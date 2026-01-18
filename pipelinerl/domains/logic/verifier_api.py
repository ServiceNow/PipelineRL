"""Logic verification using PrimeIntellect i3-logic verifiers.

Supports both local (in-process) verification and RPC-based verification
via a FastAPI server, consistent with the math and coding domains.
"""

from __future__ import annotations

import asyncio
import logging
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any

import aiohttp
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Lazy imports to avoid loading i3_logic until needed
_verifier_classes = None
_Data = None


def _ensure_imports():
    """Lazily import i3_logic dependencies."""
    global _verifier_classes, _Data
    if _verifier_classes is None:
        from i3_logic.task2verifier import verifier_classes
        from i3_logic.base import Data
        _verifier_classes = verifier_classes
        _Data = Data


@dataclass
class VerificationResult:
    """Result of logic answer verification."""
    status: str  # "correct", "wrong", "no_answer", "error"
    task_type: str
    error_message: str | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "task_type": self.task_type,
            "error_message": self.error_message,
        }


def _extract_answer(text: str) -> str | None:
    """Extract answer from model output.

    Looks for answer in <answer>...</answer> tags or returns the full text
    if no tags are found.
    """
    # Try to extract from <answer> tags (as used by i3-logic)
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try \\boxed{} format (common in math-style answers)
    match = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Return the last non-empty line as a fallback
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if lines:
        return lines[-1]

    return None


def verify_logic_answer(
    prediction: str,
    reward_context: dict,
    timeout: float = 5.0,
) -> VerificationResult:
    """Verify a logic answer using i3-logic verifiers.

    Args:
        prediction: The model's response text.
        reward_context: Dict containing 'task' (task type) and 'game_data' (puzzle data).
        timeout: Timeout for verification (currently unused, verification is fast).

    Returns:
        VerificationResult with status and task type.
    """
    _ensure_imports()

    task_type = reward_context.get("task")
    game_data = reward_context.get("game_data")

    if not task_type:
        return VerificationResult(
            status="error",
            task_type="unknown",
            error_message="Missing 'task' in reward_context",
        )

    if not game_data:
        return VerificationResult(
            status="error",
            task_type=task_type,
            error_message="Missing 'game_data' in reward_context",
        )

    # Get the verifier class for this task
    verifier_cls = _verifier_classes.get(task_type)
    if verifier_cls is None:
        return VerificationResult(
            status="error",
            task_type=task_type,
            error_message=f"No verifier found for task type: {task_type}",
        )

    # Extract the answer from the prediction
    answer = _extract_answer(prediction)
    if not answer:
        return VerificationResult(
            status="no_answer",
            task_type=task_type,
        )

    try:
        # Parse the game data
        if isinstance(game_data, str):
            data_obj = _Data.from_json_str(game_data)
        else:
            data_obj = _Data.from_json_dict(game_data)

        # Instantiate verifier and verify
        verifier = verifier_cls()
        is_correct = verifier.verify(data_obj, answer)

        return VerificationResult(
            status="correct" if is_correct else "wrong",
            task_type=task_type,
        )

    except Exception as e:
        logger.warning("Verification error for task %s: %s", task_type, str(e))
        return VerificationResult(
            status="error",
            task_type=task_type,
            error_message=str(e),
        )


def get_available_verifiers() -> list[str]:
    """Return list of available logic task verifiers."""
    _ensure_imports()
    return sorted(_verifier_classes.keys())


async def verify_logic_answer_rpc(
    session: aiohttp.ClientSession,
    host: str,
    port: int,
    prediction: str,
    reward_context: dict[str, Any],
) -> VerificationResult:
    """Verify logic answer via RPC call to a LogicEnvironment server.

    Args:
        session: aiohttp session for making the request.
        host: Hostname of the logic environment server.
        port: Port of the logic environment server.
        prediction: The model's response text.
        reward_context: Dict containing 'task' and 'game_data'.

    Returns:
        VerificationResult with status and task type.
    """
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
            return VerificationResult(
                status=data["status"],
                task_type=data["task_type"],
                error_message=data.get("error_message"),
            )
        else:
            logger.error("Error verifying logic answer: %s", response.status)
            logger.error("Response: %s", await response.text())
            return VerificationResult(
                status="error",
                task_type=reward_context.get("task", "unknown"),
                error_message=f"RPC error: {response.status}",
            )


def _verify_sync(prediction: str, reward_context: dict) -> dict:
    """Synchronous wrapper for verification (used in process pool)."""
    result = verify_logic_answer(prediction, reward_context)
    return result.to_dict()


class LogicEnvironment:
    """FastAPI-based logic verification server.

    Launch this as a remote environment when you don't want to run
    verification in-process (e.g., for load balancing across multiple actors).
    """

    def launch(self, port: int, max_workers: int = 4):
        """Start the verification API using FastAPI.

        Args:
            port: Port to listen on.
            max_workers: Number of process pool workers for concurrent verification.
        """
        app = FastAPI()

        with ProcessPoolExecutor(max_workers=max_workers) as process_pool:

            @app.post("/verify_answer")
            async def verify(request: dict):
                prediction = request.get("prediction", "")
                reward_context = request.get("reward_context", {})

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    process_pool,
                    partial(_verify_sync, prediction, reward_context),
                )
                return JSONResponse(content=result)

            @app.get("/health")
            async def health():
                return JSONResponse(content={"status": "ok"})

            @app.get("/verifiers")
            async def list_verifiers():
                verifiers = get_available_verifiers()
                return JSONResponse(content={"verifiers": verifiers, "count": len(verifiers)})

            uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=60)
