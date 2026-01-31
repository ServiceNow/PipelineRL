"""IFEval instruction following verification.

Uses AllenAI's IFEvalG module which supports 67 constraint types.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import aiohttp
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Lazy import to avoid dependency issues
_IFEVAL_LOADED = False
_INSTRUCTION_DICT = None


def _lazy_import_ifeval():
    global _IFEVAL_LOADED, _INSTRUCTION_DICT

    if _IFEVAL_LOADED:
        return True

    try:
        from pipelinerl.domains.ifeval.ifevalg.instructions_registry import INSTRUCTION_DICT

        _INSTRUCTION_DICT = INSTRUCTION_DICT
        _IFEVAL_LOADED = True
        logger.info("IFEvalG loaded successfully (%d constraint types)", len(INSTRUCTION_DICT))
        return True
    except ImportError as e:
        logger.error("Failed to import IFEvalG: %s", e)
        return False


def _check_single_instruction(instruction_id: str, kwargs: dict, response: str) -> bool:
    if instruction_id not in _INSTRUCTION_DICT:
        logger.warning("Unknown instruction ID: %s", instruction_id)
        return False

    try:
        instruction_class = _INSTRUCTION_DICT[instruction_id]
        instruction = instruction_class(instruction_id)

        if kwargs:
            instruction.build_description(**kwargs)
        else:
            instruction.build_description()

        result = instruction.check_following(response)
        return bool(result) if result is not None else False

    except Exception as e:
        logger.debug("Error checking instruction %s: %s", instruction_id, e)
        return False


def verify_answer(prediction: str, reward_context: dict) -> dict:
    instruction_id_list = reward_context.get("instruction_id_list", [])
    kwargs_list = reward_context.get("kwargs", [])

    if not instruction_id_list:
        return {
            "answer_status": "correct",
            "followed_count": 0,
            "total_count": 0,
            "score": 1.0,
        }

    if not _lazy_import_ifeval():
        return {
            "answer_status": "unparsable",
            "followed_count": 0,
            "total_count": len(instruction_id_list),
            "score": 0.0,
        }

    if not prediction or not prediction.strip():
        return {
            "answer_status": "no_answer",
            "followed_count": 0,
            "total_count": len(instruction_id_list),
            "score": 0.0,
        }

    # Ensure kwargs_list matches instruction_id_list length
    if len(kwargs_list) < len(instruction_id_list):
        kwargs_list = list(kwargs_list) + [{}] * (len(instruction_id_list) - len(kwargs_list))

    # Normalize kwargs
    kwargs_list = [kw if kw is not None else {} for kw in kwargs_list]

    # Check each instruction
    instruction_results = []
    for instr_id, kwargs in zip(instruction_id_list, kwargs_list):
        try:
            result = _check_single_instruction(instr_id, kwargs, prediction)
            instruction_results.append(result)
        except Exception as e:
            logger.warning("IFEval verification error for %s: %s", instr_id, e)
            instruction_results.append(False)

    followed_count = sum(instruction_results)
    total_count = len(instruction_id_list)
    score = followed_count / total_count if total_count > 0 else 0.0

    # Determine answer status
    if all(instruction_results):
        answer_status = "correct"
    elif followed_count > 0:
        answer_status = "partial"  # Some but not all instructions followed
    else:
        answer_status = "wrong"

    return {
        "answer_status": answer_status,
        "followed_count": followed_count,
        "total_count": total_count,
        "score": score,
    }


async def verify_answer_rpc(
    session: aiohttp.ClientSession,
    host: str,
    port: int,
    prediction: str,
    reward_context: dict,
) -> dict:
    """Verify answer via RPC call to an IFEvalEnvironment server.

    Returns:
        Dict with: answer_status, followed_count, total_count, score
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
            return await response.json()
        else:
            logger.error("Error verifying answer: %s", response.status)
            logger.error("Response: %s", await response.text())
            raise ValueError("Error verifying answer")


class IFEvalEnvironment:
    """FastAPI-based IFEval verification server."""

    def launch(self, port: int):
        app = FastAPI()

        with ProcessPoolExecutor(max_workers=4) as process_pool:

            @app.post("/verify_answer")
            async def verify(request: dict):
                prediction = request["prediction"]
                reward_context = request["reward_context"]

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    process_pool,
                    partial(verify_answer, prediction, reward_context),
                )
                return JSONResponse(content=result)

            @app.get("/health")
            async def health():
                return JSONResponse(content={"status": "ok"})

            uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=60)
