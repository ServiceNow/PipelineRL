import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf

from pipelinerl.utils import better_crashing


_META_KEYS = ("key", "mode", "replicas_per_actor")


def _strip_environment_metadata(env_cfg):
    if isinstance(env_cfg, DictConfig):
        data = OmegaConf.to_container(env_cfg, resolve=False)
    elif isinstance(env_cfg, dict):
        data = OmegaConf.to_container(env_cfg, resolve=False)
    else:
        return env_cfg
    for meta_key in _META_KEYS:
        data.pop(meta_key, None)
    return OmegaConf.create(data)


def _select_environment_cfg(cfg: DictConfig, job: dict):
    env_cfgs = getattr(cfg, "environments", None)
    key = job.get("environment_key")
    index = job.get("environment_index")

    if env_cfgs:
        if isinstance(env_cfgs, ListConfig):
            if key is not None:
                for env_cfg in env_cfgs:
                    env_key = env_cfg.get("key") or env_cfg.get("name")
                    if env_key is not None and str(env_key) == str(key):
                        return _strip_environment_metadata(env_cfg)
            if index is not None and 0 <= index < len(env_cfgs):
                return _strip_environment_metadata(env_cfgs[index])
        elif isinstance(env_cfgs, DictConfig):
            if key is not None and key in env_cfgs:
                return _strip_environment_metadata(env_cfgs[key])
            if index is not None:
                for idx, (_, env_cfg) in enumerate(env_cfgs.items()):
                    if idx == index:
                        return _strip_environment_metadata(env_cfg)

    return getattr(cfg, "environment", None)


@hydra.main(config_path="../../conf", config_name="base", version_base="1.3.2")
def hydra_entrypoint(cfg: DictConfig):
    with better_crashing("environment"):
        this_job, = [job for job in cfg.jobs if job["idx"] == cfg.me.job_idx]
        environment_cfg = _select_environment_cfg(cfg, this_job)
        if environment_cfg is None:
            raise ValueError("No environment configuration found for job")
        environment = hydra.utils.instantiate(environment_cfg)
        port = this_job["port"]
        environment.launch(port=port)


if __name__ == "__main__":
    hydra_entrypoint()
