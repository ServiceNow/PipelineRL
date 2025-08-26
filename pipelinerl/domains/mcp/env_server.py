import os
from tapeagents.remote_environment import EnvironmentServer
from omegaconf import OmegaConf
from typing import List
from fastapi import HTTPException
from pydantic import BaseModel
import logging
import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from pipelinerl.domains.math.verifier_api import verify_answer

logger = logging.getLogger(__name__)


class EnvironmentServerWithVerifier(EnvironmentServer):
    """Environment server that includes the verify_answer endpoint."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.process_pool = ProcessPoolExecutor(max_workers=4)
    
    def create_app(self):
        app = super().create_app()
        
        class VerifyAnswerRequest(BaseModel):
            prediction: str
            gold: str
            strict: bool = True
            max_prediction_length: int = 1000
        
        @app.post("/verify_answer")
        async def verify_answer_endpoint(request: VerifyAnswerRequest):
            try:
                # Run verification in the process pool to avoid blocking the main thread
                loop = asyncio.get_event_loop()
                answer_status = await loop.run_in_executor(
                    self.process_pool, 
                    partial(
                        verify_answer, 
                        request.prediction, 
                        request.gold, 
                        request.strict, 
                        request.max_prediction_length
                    )
                )
                return {"answer_status": answer_status}
            except Exception as e:
                logger.exception(f"Error in verify_answer: {e}")
                raise HTTPException(status_code=500, detail=f"Error verifying answer: {str(e)}")
        
        return app
    
    def shutdown(self):
        super().shutdown()
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=True)


class MCPEnvironmentServer:

    def __init__(self,
        n_envs: int,
        host: str,
        mcp_target: str,
        mcp_config_path: str,
        mcp_tools_whitelist: List[str],
        exp_path: str,
        env_call_timeout: int = 60,
        mcp_read_timeout_seconds: int = 10,
    ):
        # Remote environment server configuration
        self.n_envs = n_envs
        self.host = host
        self.env_call_timeout = env_call_timeout
        # Individual web environment configuration
        self.mcp_target = mcp_target
        self.mcp_config_path = mcp_config_path
        self.mcp_tools_whitelist = mcp_tools_whitelist
        self.exp_path = exp_path
        self.mcp_read_timeout_seconds = mcp_read_timeout_seconds


    def launch(self, port: int):
        """
        Serve the environment in TapeAgent with verify_answer endpoint.
        """
        env_server = EnvironmentServerWithVerifier(
            n_envs=self.n_envs, 
            host=self.host, 
            port=port, 
            env_call_timeout=self.env_call_timeout
        )
        env_server.launch(OmegaConf.create({
            "_target_": self.mcp_target,
            "config_path": self.mcp_config_path,
            "tools_whitelist": self.mcp_tools_whitelist,
            "read_timeout_seconds": self.mcp_read_timeout_seconds,
        }))

