import asyncio
from concurrent.futures import ProcessPoolExecutor
import aiohttp
import uvicorn
import logging
import json
import ast

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from functools import partial

from pipelinerl.domains.ifeval.utils import instructions_registry

logging.basicConfig(
    level=logging.DEBUG,  # Or INFO, WARNING, etc.
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Logs to console
    ],
)

logger = logging.getLogger(__name__)


def ifeval_reward_func(answer: str, reward_context: dict) -> str:
    """
    Verify ifeval constraints using the instructions_registry.
    """
    logger.info(f"ifeval_reward_func called with answer: {answer[:100]}... and reward_context: {reward_context}")
    
    instruction_dict = instructions_registry.INSTRUCTION_DICT
    
    # Parse the constraint dict from label
    label = reward_context
    constraint_dict = ast.literal_eval(label) if isinstance(label, str) else label
    if isinstance(constraint_dict, list):
        constraint_dict = constraint_dict[0]
    if isinstance(constraint_dict, str):
        constraint_dict = json.loads(constraint_dict)
    
    # Remove thinking section if present
    cleaned_answer = answer

    instruction_keys = constraint_dict["instruction_id"]
    args_list = constraint_dict["kwargs"]
    
    if len(answer) == 0 or len(cleaned_answer) == 0:
        logger.warning("Empty prediction received for ifeval_reward_func.")
        return "incorrect"
    
    # Check each instruction constraint
    rewards = []
    for instruction_key, args in zip(instruction_keys, args_list):
        if args is None:
            args = {}
        args = {k: v for k, v in args.items() if v is not None}
        
        instruction_cls = instruction_dict[instruction_key]
        instruction_instance = instruction_cls(instruction_key)
        instruction_instance.build_description(**args)
        
        if answer.strip() and instruction_instance.check_following(cleaned_answer):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    score = sum(rewards) / len(rewards)
    task_status = "correct" if score == 1.0 else "incorrect"
    
    logger.info(f"ifeval_reward_func: constraints checked: {len(rewards)}, score: {score}, task_status: {task_status}")
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
        "reward_context": reward_context
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



class IfEvalEnvironment:
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

                logger.info(f"Shiva-Received verification request with generation: {generation} and reward_context: {reward_context}")

                # Run verification in the process pool to avoid blocking the main thread
                loop = asyncio.get_event_loop()
                answer_status = await loop.run_in_executor(
                    process_pool, partial(ifeval_reward_func, generation, reward_context)
                )
                return JSONResponse(content={"answer_status": answer_status})

            @app.get("/health")
            async def health():
                return JSONResponse(content={"status": "ok"})

            uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=60)
