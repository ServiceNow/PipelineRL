import hydra
from pipelinerl.run_finetune import run_finetuning_loop
from pipelinerl.utils import better_crashing


@hydra.main(version_base=None, config_path="../../conf", config_name="finetune")
def finetune_with_config(cfg):
    with better_crashing("finetune"):
        run_finetuning_loop(cfg)


if __name__ == "__main__":
    finetune_with_config()
