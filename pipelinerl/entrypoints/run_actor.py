import hydra
from omegaconf import DictConfig

from pipelinerl.actor import run_actor_loop
from pipelinerl.utils import better_crashing


@hydra.main(config_path="../../conf", config_name="rl_deepseek_async", version_base="1.3.2")
def hydra_entrypoint(cfg: DictConfig):
    with better_crashing("actor"):
        run_actor_loop(cfg)


if __name__ == "__main__":
    hydra_entrypoint()
