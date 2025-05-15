import hydra
from omegaconf import DictConfig

from pipelinerl.tapeagents_rollouts import generate_rollout


@hydra.main(config_path="../conf/", config_name="debug", version_base="1.3.2")
def main(cfg: DictConfig):
    llm = None
    problem = None
    session = None
    result = generate_rollout(cfg, llm, problem, session)
    print(result)


if __name__ == "__main__":
    main()
