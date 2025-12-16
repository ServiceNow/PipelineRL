import asyncio
from concurrent.futures import ProcessPoolExecutor
import aiohttp
import uvicorn
import logging
import json

from omegaconf import DictConfig
import re
from typing import List

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from functools import partial

from i3_logic.task2verifier import verifier_classes


logging.basicConfig(
    level=logging.DEBUG,  # Or INFO, WARNING, etc.
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Logs to console
    ],
)

logger = logging.getLogger(__name__)


# {'arc_agi',
#  'arrow_maze',
#  'boolean_expressions',
#  'buggy_tables',
#  'calcudoko',
#  'campsite',
#  'cipher',
#  'dyck_language',
#  'dyck_language_errors',
#  'futoshiki',
#  'goods_exchange',
#  'kukurasu',
#  'math_path',
#  'mathador',
#  'minesweeper',
#  'norinori',
#  'number_wall',
#  'numbrix',
#  'object_counting',
#  'object_properties',
#  'space_reasoning_tree',
#  'sudoku',
#  'survo',
#  'time_sequence',
#  'web_of_lies',
#  'word_sorting',
#  'word_sorting_mistake',
#  'wordscapes',
#  'zebra_puzzle'}

def get_verifier(extra_info):
    return verifier_classes.get(extra_info.get('task'))()

def logic_reward_func(answer, reward_context, extra_info):
    verifier = get_verifier(extra_info)
    logger.info(f"Shiva-logic_reward_func called with answer: {answer}, reward_context: {reward_context}, extra_info: {extra_info}")
    
    verification_passed = verifier.verify(data=reward_context,test_solution_str=answer)
    task_status = 'correct' if verification_passed else 'incorrect'

    logger.info(f"Shiva-logic_reward_func: reward_context: {reward_context}, task_status: {task_status}")
    return task_status        


async def verify_answer_rpc(
    session: aiohttp.ClientSession,
    host: str,
    port: int,
    generation: str,
    reward_context: dict,
    extra_info: dict = {},
):
    """
    Verify the answer using the verifier API.
    """
    json = {
        "generation": generation,
        "reward_context": reward_context,
        "extra_info": extra_info,
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



class LogicEnvironment:
    def launch(self, port: int):
        """
        Serve the verification API using FastAPI.
        """
        app = FastAPI()
        # Create a process pool with 4 workers
        with ProcessPoolExecutor(max_workers=4) as process_pool:
            @app.post("/verify_answer")
            async def verify(request: dict):
                generation = request["generation"]
                reward_context = request['reward_context']
                extra_info = request.get('extra_info', {})

                logger.info(f"Shiva-Received verification request with generation: {generation} and reward_context: {reward_context}")

                # Run verification in the process pool to avoid blocking the main thread
                loop = asyncio.get_event_loop()
                answer_status = await loop.run_in_executor(
                    process_pool, partial(logic_reward_func, generation, reward_context, extra_info)
                )
                return JSONResponse(content={"answer_status": answer_status})

            @app.get("/health")
            async def health():
                return JSONResponse(content={"status": "ok"})

            uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=60)
