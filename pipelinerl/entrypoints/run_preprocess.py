import os

import hydra
from pipelinerl.preprocess import run_preprocessing_loop
from pipelinerl.utils import better_crashing


@hydra.main(version_base=None, config_path="../../conf", config_name="finetune")
def preprocess_hydra_entry_point(cfg):
    with better_crashing("preprocess"):
        run_preprocessing_loop(cfg)


if __name__ == "__main__":
    preprocess_hydra_entry_point()
    # SharedMemoryManager / multiprocessing.Queue cleanup has occasionally
    # hung at interpreter shutdown after the main loop already finished,
    # pinning the launcher (which waits on this process before tearing down
    # vLLM/Ray services). The work is done at this point; force exit so the
    # launcher can proceed.
    os._exit(0)
