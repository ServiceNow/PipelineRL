import json
import logging
import os
import time
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

import datasets
import transformers
from omegaconf import DictConfig

import wandb
from wandb.sdk import wandb_run

from .context import get_accelerator, logger


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

    if cfg.finetune.wandb_resume == "always":
        resume = "allow"
    elif cfg.finetune.wandb_resume == "never":
        resume = "never"
    elif cfg.finetune.wandb_resume == "if_not_interactive":
        raise NotImplementedError()
    else:
        raise ValueError(f"Unknown value for wandb_resume: {cfg.finetune.wandb_resume}")

    wandb_name = str(run_dir)
    root = cfg.finetune.wandb_workspace_root
    if root:
        if not wandb_name.startswith(root + "/"):
            raise ValueError(f"run_dir {run_dir} does not start with root {root}")
        wandb_name = wandb_name[len(root) + 1 :]

    wandb_id = cfg.finetune.wandb_id
    if not wandb_id:
        wandb_id = wandb_name.replace("/", "_")

    if len(wandb_name) > 128:
        logger.warning(
            f"wandb_name: {wandb_name} is longer than 128 characters. Truncating to 128 characters."
        )

    logging.info(
        f"Initializing W&B with\nname: {wandb_name[:128]}\nid: {wandb_id}\nresume: {resume}"
    )
    run = wandb.init(
        name=wandb_name[:128],  # wandb limits name to 128 characters
        entity=cfg.finetune.wandb_entity_name,
        project=cfg.finetune.wandb_project_name,
        group=cfg.finetune.wandb_group,
        dir=cfg.finetune.wandb_dir,
        config=config_for_wandb,  # type: ignore
        resume=resume,
        id=wandb_id,
        tags=cfg.finetune.tags,
    )
    if not isinstance(run, wandb_run.Run):
        raise ValueError("W&B init failed")
    return run


def setup_logging(cfg: DictConfig, output_dir: Path, run: wandb_run.Run | None = None):
    log_dir = output_dir / "log/"
    log_dir.mkdir(parents=True, exist_ok=True)
    debug_handler = logging.FileHandler(
        log_dir / f"info_{get_accelerator().process_index}.log"
    )
    debug_handler.setLevel(logging.INFO)
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[debug_handler, logging.StreamHandler()],
        force=True,  # forget previous handlers
    )
    if get_accelerator().is_main_process:  # we only want to setup logging once
        config_for_wandb = {
            str(k): str(v) for k, v in get_accelerator().state.__dict__.items()
        }
        config_for_wandb.update(flatten_dict_config(cfg))

        logger.setLevel(logging.INFO)
        if run is None and cfg.finetune.use_wandb:
            try:
                run = init_wandb(cfg, output_dir, config_for_wandb)
            except Exception as e:
                run = None
                logger.warning(f"Failed to initalize wandb: {e}")

        wandb_config = {}
        if run is not None:
            wandb_config = {
                "name": run.name[:128],  # wandb limits name to 128 characters
                "entity": run.entity,
                "project": run.project_name(),
                "id": run.id,
            }
        # Save wandb name, entity, project, and ID to JSON file in output_dir
        with open(os.path.join(output_dir, "wandb_info.json"), "w") as f:
            json.dump(wandb_config, f, indent=4)
    else:
        logger.setLevel(logging.ERROR)
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()


def log_metrics(logger: logging.Logger, completed_steps: int, metrics: dict[str, Any]):
    if not get_accelerator().is_main_process:
        return

    # Print metrics with 3 decimals
    metrics_pretty = {k: f"{v:.3f}" for k, v in metrics.items()}
    logger.info(f"Completed steps {completed_steps}: {metrics_pretty}")
    try:
        metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        wandb.log(metrics, step=completed_steps)
    except Exception as e:
        logger.error(f"Failed to log metrics to wandb with error: {e}")


def log_time(start_time: float, stats_dict: dict, msg: str):
    t = time.perf_counter()
    stats_dict[msg] = t - start_time
    return t


def flatten_dict_config(d: DictConfig | dict, separator=".") -> dict:
    result = {}
    for k, v in d.items():
        if isinstance(v, DictConfig) or isinstance(v, dict):
            for sub_k, sub_v in flatten_dict_config(v).items():
                result[str(k) + separator + str(sub_k)] = sub_v
        else:
            result[k] = v
    return result
