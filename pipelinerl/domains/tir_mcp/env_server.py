import os
from tapeagents.remote_environment import EnvironmentServer
from omegaconf import OmegaConf
from typing import List


from pipelinerl.domains.math import MathEnvironment

class MCPEnvironmentServer:

    def __init__(self,
        n_envs: int,
        n_envs_mcp: int,
        n_envs_math: int,
        host: str,
        mcp_target: str,
        mcp_config_path: str,
        mcp_tools_whitelist: List[str],
        math_target: str,
        exp_path: str,
        env_call_timeout: int = 60,
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


    def launch(self, port: int):
        """
        Serve the environment in TapeAgent.
        """
        if port != 7777:
            env_server = EnvironmentServer(n_envs=self.n_envs, host=self.host, port=port, env_call_timeout=self.env_call_timeout)
            env_server.launch(OmegaConf.create({
                "_target_": self.mcp_target,
                "config_path": self.mcp_config_path,
                "tools_whitelist": self.mcp_tools_whitelist,
            }))
        else:
            MathEnvironment().launch(port)

