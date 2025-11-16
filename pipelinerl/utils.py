import contextlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
import traceback
from typing import Dict, Mapping, List, Any

import numpy as np
import psutil
import requests
from importlib.metadata import distributions
from omegaconf import DictConfig, ListConfig, OmegaConf
from transformers import PreTrainedTokenizer

from pipelinerl.world import Job
from tapeagents.llms import LLMOutput
from tapeagents.core import Prompt

import wandb
from wandb.sdk import wandb_run

logger = logging.getLogger(__name__)

_ENV_METADATA_KEYS = {"key", "mode", "replicas_per_actor"}


def _strip_environment_metadata(env_cfg: DictConfig | dict | None):
    if env_cfg is None:
        return None
    if isinstance(env_cfg, DictConfig):
        data = OmegaConf.to_container(env_cfg, resolve=True)
    elif isinstance(env_cfg, dict):
        data = dict(env_cfg)
    else:
        return env_cfg
    for meta_key in _ENV_METADATA_KEYS:
        data.pop(meta_key, None)
    return OmegaConf.create(data)


def _env_cfg_type(env_cfg, field: str):
    if isinstance(env_cfg, DictConfig):
        return env_cfg.get(field, None)
    if isinstance(env_cfg, dict):
        return env_cfg.get(field)
    return None


def select_environment_config(cfg: DictConfig, *, key: str | None = None, index: int | None = None):
    env_cfgs = getattr(cfg, "environments", None)
    if env_cfgs:
        if isinstance(env_cfgs, (ListConfig, list)):
            if key is not None:
                for env_cfg in env_cfgs:
                    env_key = _env_cfg_type(env_cfg, "key") or _env_cfg_type(env_cfg, "name")
                    if env_key is not None and str(env_key) == str(key):
                        return _strip_environment_metadata(env_cfg)
            if index is not None and 0 <= index < len(env_cfgs):
                return _strip_environment_metadata(env_cfgs[index])
        elif isinstance(env_cfgs, (DictConfig, dict)):
            if key is not None and key in env_cfgs:
                return _strip_environment_metadata(env_cfgs[key])
            if index is not None:
                for idx, env_key in enumerate(env_cfgs):
                    if idx == index:
                        return _strip_environment_metadata(env_cfgs[env_key])

    return getattr(cfg, "environment", None)


def collect_environment_specs(cfg: DictConfig) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    env_cfgs = getattr(cfg, "environments", None)
    default_mode = str(getattr(cfg.world, "environment_mode", "remote"))
    default_replicas = getattr(cfg.world, "env_replicas_per_actor", 1)

    if isinstance(env_cfgs, (ListConfig, list)):
        iterable = list(env_cfgs)
        for idx, env_cfg in enumerate(iterable):
            if env_cfg is None:
                continue
            key = _env_cfg_type(env_cfg, "key") or _env_cfg_type(env_cfg, "name")
            mode = _env_cfg_type(env_cfg, "mode")
            replicas = _env_cfg_type(env_cfg, "replicas_per_actor")
            specs.append(
                {
                    "key": str(key) if key is not None else f"environment_{idx}",
                    "mode": str(mode) if mode is not None else default_mode,
                    "replicas_per_actor": replicas,
                    "index": idx,
                }
            )
    elif isinstance(env_cfgs, (DictConfig, dict)):
        items = env_cfgs.items()
        for idx, (key, env_cfg) in enumerate(items):
            if env_cfg is None:
                continue
            mode = _env_cfg_type(env_cfg, "mode")
            replicas = _env_cfg_type(env_cfg, "replicas_per_actor")
            specs.append(
                {
                    "key": str(key),
                    "mode": str(mode) if mode is not None else default_mode,
                    "replicas_per_actor": replicas,
                    "index": idx,
                }
            )
    else:
        single_env = getattr(cfg, "environment", None)
        if single_env:
            key = _env_cfg_type(single_env, "key") or _env_cfg_type(single_env, "name")
            specs.append(
                {
                    "key": str(key) if key is not None else "default",
                    "mode": default_mode,
                    "replicas_per_actor": default_replicas,
                    "index": 0,
                }
            )
    return specs


def resolve_environment_key(cfg: DictConfig, default: str | None = None) -> str | None:
    explicit = cfg.get("environment_key", None) if hasattr(cfg, "get") else None
    if explicit:
        return str(explicit)
    specs = collect_environment_specs(cfg)
    if len(specs) == 1:
        return specs[0]["key"]
    return default


def get_environment_jobs(cfg: DictConfig, key: str | None = None) -> list[Job]:
    jobs_cfg = getattr(cfg, "jobs", [])
    env_jobs = [Job(**job) for job in jobs_cfg if job["kind"] == "environment"]
    if key is None:
        return env_jobs
    filtered = [job for job in env_jobs if getattr(job, "environment_key", None) == key]
    return filtered or env_jobs

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

    if len(stats) == 0:
        return {}

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
    """Wait for remote environment servers to report healthy."""
    specs = collect_environment_specs(cfg)
    if not any(spec.get("mode") == "remote" for spec in specs):
        return

    env_jobs = get_environment_jobs(cfg)
    if not env_jobs:
        return
    for job in env_jobs:
        while True:
            url = f"http://{job.hostname}:{job.port}/health"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    logger.info(
                        "Environment %s ready at %s",
                        job.environment_key if job.environment_key is not None else job.replica_idx,
                        url,
                    )
                    break
            except requests.exceptions.RequestException:
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
        logger.error("I should not even be here...")
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
