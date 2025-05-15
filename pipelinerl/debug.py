import hydra
from omegaconf import DictConfig

from pipelinerl.launch import main as launch_main


@hydra.main(config_path="../conf/", config_name="debug", version_base="1.3.2")
def main(cfg: DictConfig):
    launch_main(cfg)


if __name__ == "__main__":
    main()
