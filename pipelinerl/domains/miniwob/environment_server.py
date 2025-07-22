import os
from tapeagents.remote_environment import EnvironmentServer
from omegaconf import OmegaConf


class WebEnvironmentServer:

    def __init__(self,
        miniwob_url: str,
        n_envs: int,
        host: str,
        web_env_target: str,
        exp_path: str,
        headless: bool = True,
        observation_format: str = "html",
        env_call_timeout: int = 60,
    ):
        os.environ["MINIWOB_URL"] = miniwob_url
        # Remote environment server configuration
        self.n_envs = n_envs
        self.host = host
        self.env_call_timeout = env_call_timeout
        # Individual web environment configuration
        self.web_env_target = web_env_target
        self.exp_path = exp_path
        self.headless = headless
        self.observation_format = observation_format


    def launch(self, port: int):
        """
        Serve the web environment in TapeAgent.
        """
        env_server = EnvironmentServer(n_envs=self.n_envs, host=self.host, port=port, env_call_timeout=self.env_call_timeout)
        env_server.launch(OmegaConf.create({
            "_target_": self.web_env_target,
            "exp_path": self.exp_path,
            "headless": self.headless,
            "observation_format": self.observation_format,
        }))

