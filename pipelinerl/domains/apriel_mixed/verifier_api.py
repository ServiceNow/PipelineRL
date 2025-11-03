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

from pipelinerl.domains.math.verifier_api import verify_answer as math_verify_answer
from pipelinerl.domains.agentic_fn_calling.verifier_api import (
    agentic_fn_calling_reward_func as agentic_verify_answer,
)
from pipelinerl.domains.ifeval.verifier_api import (
    ifeval_reward_func as ifeval_verify_answer,
)

from pipelinerl.domains.coding.remote_code_executor import RemoteCodeExecutor

logging.basicConfig(
    level=logging.DEBUG,  # Or INFO, WARNING, etc.
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Logs to console
    ],
)

logger = logging.getLogger(__name__)


def math2apriel(math_status: str) -> str:
    """
    Needed because Math domain of PipelineRL has several sub-categories which Apriel does not need..
    """
    match math_status:
        case "correct":
            return "correct"
        case "wrong":
            return "incorrect"
        case _:
            return "unparsable"


def apriel_reward_func(domain, answer, label):
    if domain == "ifeval":
        return "ifeval_" + ifeval_verify_answer(answer, label)

    if domain == "math":
        return "math_" + math2apriel(math_verify_answer(answer, label))

    if domain == "agentic_fn_calling":
        return "agentic_fn_calling_" + agentic_verify_answer(answer, label)

    raise ValueError(f"Unknown domain: {domain}")


async def verify_answer_rpc(
    session: aiohttp.ClientSession,
    host: str,
    port: int,
    domain: str,
    generation: str,
    reward_context: dict,
    extra_info: dict = {},
):
    """
    Verify the answer using the verifier API.
    """
    extra_info = (
        {"language": extra_info.get("language", "python")} if domain == "coding" else {}
    )
    json = {
        "domain": domain,
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


class AprielToolsEnvironment:
    def launch(self, port: int):
        """
        Serve the verification API using FastAPI.
        """
        app = FastAPI()
        # Create a process pool with 4 workers
        with ProcessPoolExecutor(max_workers=4) as process_pool:

            @app.post("/verify_answer")
            async def verify(request: dict):
                domain = request["domain"]
                generation = request["generation"]
                reward_context = request["reward_context"]
                extra_info = request["extra_info"]

                logger.info(
                    f"Shiva-Received verification request with generation: {generation} and reward_context: {reward_context}"
                )

                # Run verification in the process pool to avoid blocking the main thread
                loop = asyncio.get_event_loop()
                if domain in ["math", "ifeval", "agentic_fn_calling"]:
                    answer_status = await loop.run_in_executor(
                        process_pool,
                        partial(apriel_reward_func, domain, generation, reward_context),
                    )
                    return JSONResponse(content={"answer_status": answer_status})
                elif domain == "coding":
                    remote_executor = RemoteCodeExecutor(
                        # TODO: Thread this here.
                        server_url="http://172.20.24.132:8080/run_code"
                    )
                    answer_status = await remote_executor.execute(
                        generation, reward_context, extra_info
                    )
                    return JSONResponse(content={"answer_status": answer_status})
                else:
                    raise ValueError(f"Unknown domain: {domain}")
                

            @app.get("/health")
            async def health():
                return JSONResponse(content={"status": "ok"})

            uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=60)
