import hydra
from pipelinerl.run_preprocess import run_preprocessing_loop
from pipelinerl.utils import better_crashing


@hydra.main(version_base=None, config_path="../../conf", config_name="finetune")
def preprocess_hydra_entry_point(cfg):
    with better_crashing("preprocess"):
        run_preprocessing_loop(cfg)


if __name__ == "__main__":
    preprocess_hydra_entry_point()
