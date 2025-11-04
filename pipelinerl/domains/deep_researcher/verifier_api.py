"""Verification API for DeepResearcher domain."""
import asyncio
from concurrent.futures import ProcessPoolExecutor
import aiohttp
import uvicorn
import logging
from functools import partial

from fastapi import FastAPI
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def verify_answer(generation: str, reward_context: dict) -> str:
    answer_type = reward_context.get("answer_type", "contains")
    ground_truth = reward_context.get("ground_truth", "")
    aliases = reward_context.get("aliases", [])
    
    generation_lower = generation.lower().strip()
    ground_truth_lower = ground_truth.lower().strip()
    
    if answer_type == "exact_match":
        if generation_lower == ground_truth_lower:
            return "correct"
        for alias in aliases:
            if generation_lower == alias.lower().strip():
                return "correct"
        return "incorrect"
    
    elif answer_type == "contains":
        if ground_truth_lower in generation_lower:
            return "correct"
        for alias in aliases:
            if alias.lower().strip() in generation_lower:
                return "correct"
        return "incorrect"
    
    elif answer_type == "fuzzy":
        ground_truth_terms = set(ground_truth_lower.split())
        generation_terms = set(generation_lower.split())
        
        overlap = ground_truth_terms & generation_terms
        if len(overlap) >= len(ground_truth_terms) * 0.5:
            return "correct"
        
        for alias in aliases:
            alias_terms = set(alias.lower().strip().split())
            overlap = alias_terms & generation_terms
            if len(overlap) >= len(alias_terms) * 0.5:
                return "correct"
        
        return "incorrect"
    
    else:
        logger.warning(f"Unknown answer_type: {answer_type}, defaulting to 'contains'")
        return "correct" if ground_truth_lower in generation_lower else "incorrect"


async def verify_answer_rpc(
    session: aiohttp.ClientSession,
    host: str,
    port: int,
    generation: str,
    reward_context: dict,
) -> str:
    json_data = {
        "generation": generation,
        "reward_context": reward_context
    }
    
    try:
        async with session.post(
            f"http://{host}:{port}/verify_answer",
            json=json_data,
            timeout=aiohttp.ClientTimeout(total=10)
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data["answer_status"]
            else:
                error_text = await response.text()
                logger.error(f"Verification failed with status {response.status}: {error_text}")
                return "incorrect"
    except asyncio.TimeoutError:
        logger.error("Verification request timed out")
        return "incorrect"
    except Exception as e:
        logger.error(f"Verification request failed: {e}")
        return "incorrect"


class DeepResearcherEnvironment:
    
    def launch(self, port: int):
        app = FastAPI(title="DeepResearcher Verification API")
        
        with ProcessPoolExecutor(max_workers=4) as process_pool:
            @app.post("/verify_answer")
            async def verify(request: dict):
                generation = request.get("generation", "")
                reward_context = request.get("reward_context", {})
                
                loop = asyncio.get_event_loop()
                answer_status = await loop.run_in_executor(
                    process_pool,
                    partial(verify_answer, generation, reward_context)
                )
                
                return JSONResponse(content={"answer_status": answer_status})
            
            @app.get("/health")
            async def health():

                return JSONResponse(content={"status": "ok"})
            
            logger.info(f"Starting DeepResearcher verification server on port {port}")
            uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=60)
