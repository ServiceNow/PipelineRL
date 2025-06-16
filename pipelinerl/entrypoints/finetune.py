import sys
import hydra
from pipelinerl.run_finetune import run_finetuning_loop
from pipelinerl.utils import better_crashing


@hydra.main(version_base=None, config_path="../../conf", config_name="finetune")
def finetune_with_config(cfg):
    with better_crashing("finetune"):
        run_finetuning_loop(cfg)


if __name__ == "__main__":
    for i in range(len(sys.argv)):
        if sys.argv[i].startswith("--local_rank"):
            # Remove the redundant cmd argument from DeepSpeed
            sys.argv = sys.argv[:i] + sys.argv[i + 1:]
            break
    finetune_with_config()
