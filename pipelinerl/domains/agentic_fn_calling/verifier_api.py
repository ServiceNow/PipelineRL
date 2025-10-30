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

logging.basicConfig(
    level=logging.DEBUG,  # Or INFO, WARNING, etc.
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Logs to console
    ],
)

logger = logging.getLogger(__name__)

def agentic_fn_calling_reward_func(answer, label):
    # TODO: Simplify this function and make it readable. Right now, it's taken as-is from
    # /mnt/queue4/rishabh/NowRLHF/openrlhf/reward/agentic/agentic_deprecated.py
    label_tools = label["tool_calls"]
    if "irrelevant_tool_call" in str(label):
        try:
            tools_str = re.findall(r"<tool_calls>(.*)</tool_calls>", answer, re.DOTALL)
            tools = json.loads(tools_str[0].strip())
            return 'incorrect'
        except:
            return 'correct'
    try:
        tools_str = re.findall(r"<tool_calls>(.*)</tool_calls>", answer, re.DOTALL)
        tools = json.loads(tools_str[0].strip())
        if type(tools) != list:
            raise Exception(f"tools is not a list: {type(label_tools)}")

        label_tools = label["tool_calls"]
        if label_tools is None:
            label_tools = []
    except:
        return 'incorrect'
    if len(tools) == 0 or len(label_tools) == 0:
        if len(tools) == 0 and len(label_tools) == 0:
            return 'correct'
        else:
            return 'partially_correct'
    
    else:
        try:
            for item in label_tools:
                if "function" not in item:
                    continue
                if "arguments" not in item['function']:
                    continue
                args_val = item['function']['arguments']
                if isinstance(args_val, str):
                    item['function']['arguments'] = json.loads(args_val)
            counter1_full = Counter((item["name"], json.dumps(item["arguments"], sort_keys=True)) 
                                    for item in tools)
            counter2_full = Counter((item["function"]["name"], json.dumps(item["function"]["arguments"], sort_keys=True)) 
                                    for item in label_tools)
        except:
            return 'partially_correct'

        if counter1_full == counter2_full:
            return 'correct'
        
        return 'partially_correct'



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



class AgenticToolsEnvironment:
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
                    process_pool, partial(agentic_fn_calling_reward_func, generation, reward_context)
                )
                return JSONResponse(content={"answer_status": answer_status})

            @app.get("/health")
            async def health():
                return JSONResponse(content={"status": "ok"})

            uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=60)
