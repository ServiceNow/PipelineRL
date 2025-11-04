import hydra
from omegaconf import DictConfig

from pipelinerl.utils import better_crashing, select_environment_config


@hydra.main(config_path="../../conf", config_name="base", version_base="1.3.2")
def hydra_entrypoint(cfg: DictConfig):
    with better_crashing("environment"):
        this_job, = [job for job in cfg.jobs if job["idx"] == cfg.me.job_idx]
        environment_cfg = select_environment_config(
            cfg,
            key=this_job.get("environment_key"),
            index=this_job.get("environment_index"),
        )
        if environment_cfg is None:
            raise ValueError("No environment configuration found for job")
        environment = hydra.utils.instantiate(environment_cfg)
        port = this_job["port"]
        environment.launch(port=port)


if __name__ == "__main__":
    hydra_entrypoint()
