import os

import hydra
from omegaconf import DictConfig

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

from cube_pipelinerl.launch import run_actor_loop_ray
from pipelinerl.compat import patch_litellm_context_window_exception_for_pickle
from pipelinerl.utils import better_crashing

@hydra.main(config_path="../../conf", config_name="base", version_base="1.3.2")
def hydra_entrypoint(cfg: DictConfig):
    os.environ["RAY_DEBUG"] = str(cfg.get('ray_debug', 0))
    patch_litellm_context_window_exception_for_pickle()
    with better_crashing("actor_ray"):
        run_actor_loop_ray(cfg)


if __name__ == "__main__":
    hydra_entrypoint()
