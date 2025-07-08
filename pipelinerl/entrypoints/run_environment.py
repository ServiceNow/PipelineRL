import hydra
from omegaconf import DictConfig

from pipelinerl.utils import better_crashing


@hydra.main(config_path="../../conf", config_name="base", version_base="1.3.2")
def hydra_entrypoint(cfg: DictConfig):
    with better_crashing("environment"):
        environment = hydra.utils.instantiate(cfg.environment)
        this_job, = [job for job in cfg.jobs if job["idx"] == cfg.me.job_idx]
        port = this_job["port"]
        environment.launch(port=port)


if __name__ == "__main__":
    hydra_entrypoint()
