from tapeagents.remote_environment import EnvironmentServer

class WebEnvironment:

    def launch(self, port: int):
        """
        Serve the verification API using FastAPI.
        """
        EnvironmentServer(cfg.environment_server).launch(cfg.environment)

