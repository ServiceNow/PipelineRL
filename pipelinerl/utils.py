import contextlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
import traceback
from typing import Dict, Mapping, List, Any, Union
import numpy as np
from omegaconf import DictConfig
import psutil
import requests
from importlib.metadata import distributions
from transformers import PreTrainedTokenizer
from collections import defaultdict

from pipelinerl.world import Job
from tapeagents.llms import LLMOutput
from tapeagents.core import Prompt

import wandb
from wandb.sdk import wandb_run

logger = logging.getLogger(__name__)


def init_wandb(
    cfg: DictConfig,
    run_dir: Path,
    config_for_wandb: DictConfig | dict,
) -> wandb_run.Run:
    """Initialize W&B.

    config_for_wandb is the configuration that will be logged to W&B.

    """
    if config_for_wandb is None:
        config_for_wandb = cfg.dict()

    python_env = {}
    for dist in distributions():
        python_env[dist.metadata["Name"]] = dist.version
    config_for_wandb["python_env"] = python_env

    if cfg.wandb.wandb_resume == "always":
        resume = "allow"
    elif cfg.wandb.wandb_resume == "never":
        resume = "never"
    elif cfg.wandb.wandb_resume == "if_not_interactive":
        raise NotImplementedError()
    else:
        raise ValueError(f"Unknown value for wandb_resume: {cfg.finetune.wandb_resume}")

    wandb_name = str(run_dir)
    root = cfg.wandb.wandb_workspace_root
    if root:
        if not wandb_name.startswith(root + "/"):
            raise ValueError(f"run_dir {run_dir} does not start with root {root}")
        wandb_name = wandb_name[len(root) + 1 :]

    wandb_id = cfg.wandb.wandb_id
    if not wandb_id:
        wandb_id = wandb_name.replace("/", "_")

    if len(wandb_name) > 128:
        logger.warning(f"wandb_name: {wandb_name} is longer than 128 characters. Truncating to 128 characters.")

    logging.info(f"Initializing W&B with\nname: {wandb_name[:128]}\nid: {wandb_id}\nresume: {resume}")
    run = wandb.init(
        name=wandb_name[:128],  # wandb limits name to 128 characters
        entity=cfg.wandb.wandb_entity_name,
        project=cfg.wandb.wandb_project_name,
        group=cfg.wandb.wandb_group,
        dir=cfg.wandb.wandb_dir,
        config=config_for_wandb,  # type: ignore
        resume=resume,
        id=wandb_id,
        tags=cfg.wandb.tags,
    )
    if not isinstance(run, wandb_run.Run):
        raise ValueError("W&B init failed")
    return run


def generate_cuda_device_strings(total_gpus: int, gpus_per_model: int) -> List[str]:
    """
    Generate a list of CUDA device strings for assigning GPUs to models.

    Args:
    - total_gpus (int): The total number of GPUs available.
    - gpus_per_model (int): The number of GPUs required per model.

    Returns:
    - List[str]: A list of strings, each representing the CUDA devices for a model.
    """
    cuda_device_strings = []
    for start_gpu in range(0, total_gpus, gpus_per_model):
        end_gpu = start_gpu + gpus_per_model
        cuda_devices = ",".join(str(i) for i in range(start_gpu, end_gpu))
        cuda_device_strings.append(cuda_devices)
    return cuda_device_strings


def setup_logging(logging_dir: Path, stage: str):
    print(f"Setting up logging to {logging_dir}")

    logging_dir = Path(logging_dir)
    logging_dir.mkdir(parents=True, exist_ok=True)  # Create the output directory if it doesn't exist

    # Define log file paths
    info_log = logging_dir / "info.log"
    debug_log = logging_dir / "debug.log"
    error_log = logging_dir / "error.log"
    warning_log = logging_dir / "warning.log"

    # Clear any existing handlers
    logger = logging.getLogger()  # get root logger
    logger.handlers = []  # Clear existing handlers
    logger.setLevel(logging.DEBUG)  # Ensure all levels are captured at the root level

    # Create file handlers for each log level
    info_handler = logging.FileHandler(info_log)
    info_handler.setLevel(logging.INFO)

    debug_handler = logging.FileHandler(debug_log)
    debug_handler.setLevel(logging.DEBUG)

    error_handler = logging.FileHandler(error_log)
    error_handler.setLevel(logging.ERROR)

    warning_handler = logging.FileHandler(warning_log)
    warning_handler.setLevel(logging.WARNING)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.INFO)

    # Create formatters and set them to the handlers
    formatter = logging.Formatter(f"[{stage}]: %(asctime)s - %(name)s - %(levelname)s - %(message)s")

    info_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    warning_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(info_handler)
    logger.addHandler(debug_handler)
    logger.addHandler(error_handler)
    logger.addHandler(stdout_handler)
    logger.addHandler(warning_handler)


def load_state(state_path):
    if state_path.exists():
        with open(state_path, "r") as f:
            return json.load(f)
    else:
        return {"iteration": 0}


def save_state(state, state_path):
    with open(state_path, "w") as f:
        json.dump(state, f)


def clean_up(target_path: Path, state: Dict, state_path: str | Path) -> None:
    os.makedirs(target_path, exist_ok=True)

    def remove_dir(directory: Path):
        if directory.exists() and directory.is_dir():
            shutil.rmtree(directory)

    # Reset the state iteration steps
    state["iteration"] = 0
    save_state(state, state_path)

    logger.info("Cleaning up checkpoints and training state")
    # list of files to remove
    files = [
        target_path / "debug.log",
        target_path / "error.log",
        target_path / "info.log",
    ]

    for file in files:
        if file.exists():
            # erase the content but not the file
            with open(file, "w"):
                pass
            logger.info(f"{file} erased.")

    # List of directories to remove
    directories = [
        target_path / "llm_calls.sqlite",
        target_path / "dialogue_trace.log",
        target_path / "rollouts",
        target_path / "tapes",
        target_path / "conf",
        target_path / "finetune" / "current",
        target_path / "finetune" / "logs",
        target_path / "finetune" / "intermediate",
        target_path / "finetune" / "training_state",
    ]

    for directory in directories:
        remove_dir(directory)
        logger.info(f"{directory} removed.")


def always_or_never_success_stats(success_stats: Mapping[str, Mapping[str, list[int]]]) -> dict[str, float]:
    always_success = {}
    never_success = {}
    sometimes_success = {}
    for dataset in success_stats:
        for problem in success_stats[dataset]:
            always_success[problem] = all(success_stats[dataset][problem])
            never_success[problem] = not any(success_stats[dataset][problem])
            sometimes_success[problem] = not always_success[problem] and not never_success[problem]
    return {  # type: ignore
        "always_success": float(np.mean(list(always_success.values()))),
        "never_success": float(np.mean(list(never_success.values()))),
        "sometimes_success": float(np.mean(list(sometimes_success.values()))),
    }


def dict_to_list(d: Dict[Any, Any] | List[Any]) -> List[Any]:
    if isinstance(d, dict):
        return [item for v in d.values() for item in dict_to_list(v)]
    return d


def calculate_stats(stats: List | Dict[Any, Any]) -> Dict[str, float]:
    if isinstance(stats, dict):
        # stats is a dict of list
        stats = dict_to_list(stats)

    if not isinstance(stats, list):
        raise TypeError(f"Expected stats to be a list, got {type(stats)}")

    aggregated_stats = {
        "max": float(max(stats)),
        "min": float(min(stats)),
        "var": float(np.var(stats)),
        "mean": float(np.mean(stats)),
    }

    if aggregated_stats["var"] == 0:
        # pop max, min, and var
        aggregated_stats.pop("max")
        aggregated_stats.pop("min")
        aggregated_stats.pop("var")

    return aggregated_stats


def get_tokens_from_hf_tokenizer(tokenizer: PreTrainedTokenizer | None, prompt: Prompt, output: LLMOutput) -> list:
    if not tokenizer:
        return []
    prompt_token_ids = tokenizer.apply_chat_template(
        conversation=prompt.messages, tokenize=True, add_generation_prompt=True
    )
    text_token_ids = tokenizer.apply_chat_template(
        prompt.messages + [{"role": "assistant", "content": output.content}], tokenize=True
    )
    output_token_ids = text_token_ids[len(prompt_token_ids) :]
    output_tokens = [tokenizer.decode(output_token_id) for output_token_id in output_token_ids]
    return output_tokens


def wait_for_inference_servers(urls: list[str]):
    logger.info("Waiting for inference servers to be up")
    while True:
        all_servers_up = True
        still_not_up = None
        for url in urls:
            try:
                response = requests.get(f"{url}/health")
                if response.status_code != 200:
                    all_servers_up = False
                    still_not_up = url
                    break
            except requests.exceptions.ConnectionError:
                all_servers_up = False
                still_not_up = url
                break
        if all_servers_up:
            break
        logger.info(f"Still waiting for {still_not_up} ...")
        time.sleep(3.0)
    logger.info("All inference servers are up")


def wait_for_environments(cfg: DictConfig):
    """
    Wait for the verifier to be ready.
    """
    env_jobs = [Job(**job) for job in cfg.jobs if job.kind == "environment"]
    for job in env_jobs:
        while True:
            url = f"http://{job.hostname}:{job.port}/health"
            # use requests
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    break
            except:
                logger.info(f"Waiting for environment at {url} to be ready...")
                time.sleep(5.0)


@contextlib.contextmanager
def better_crashing(entrypoint_name: str):
    try:
        yield
    except Exception as e:
        # TODO: understand why the logging message can appear super late
        logger.error(f"Exception in {entrypoint_name}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # get process if of the current process
        process_id = os.getpid()
        terminate_with_children(process_id)
        logger.error(f"I should not even be here...")
        import sys

        sys.exit(1)


def terminate_with_children(process_id: int):
    """Terminate the process and all its children"""
    try:
        parent = psutil.Process(process_id)
        children = parent.children(recursive=True)

        # First attempt graceful termination of children
        for child in children:
            child.terminate()

        # Wait for children to terminate
        _, alive = psutil.wait_procs(children, timeout=5)

        if alive:
            logger.info(f"{len(alive)} children still alive, trying SIGKILL")
            for child in alive:
                child.kill()

        # Terminate parent process
        parent.terminate()
        parent.wait(timeout=3)

        # Force kill parent if still alive
        if parent.is_running():
            parent.kill()
            logger.info(f"Trying SIGKILL on parent process {process_id}")
            parent.wait()
            logger.info(f"Parent process {process_id} finished.")

    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        logger.error(f"Error stopping process {process_id}: {e}")
