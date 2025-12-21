import json
import logging
import random
import re
from typing import Dict, List, Tuple, Literal
from enum import Enum
import os 

import datasets
import hydra
from datasets import load_dataset
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class IFEvalSource(str, Enum):
    IFEVAL_INST_LIST = "ifeval_inst_list"
    HF_MIXED_TRAINING = "ifeval"


def add_ids(dataset: list[dict]):
    for i, entry in enumerate(dataset):
        entry["id"] = i
    return dataset


def process_ifeval_inst_list(dataset: datasets.Dataset, dataset_name: str):
    """Process instruction list format."""
    for sample in dataset:
        yield {
            "dataset": dataset_name,
            "task": sample["messages"][0]['content'],
            'reward_context': sample['constraints']
        }


def process_ifeval_mixed_training(dataset: datasets.Dataset, dataset_name: str):
    """Process HF mixed-training-text-datasets format."""
    for sample in dataset:
        reward_data = json.loads(sample['reward_model']['ground_truth'])
        yield {
            "dataset": dataset_name,
            "task": sample["prompt"][0]['content'],
            'reward_context': reward_data
        }


def load_ifeval_dataset(source: IFEvalSource, seed: int | None = None):
    """Load IFEval data from the specified source."""
    if source == IFEvalSource.IFEVAL_INST_LIST:
        dataset = load_dataset(
            "ServiceNow-AI/instruction_following_rl", 
            "v1",
            split="train",
            token=os.environ.get("HF_TOKEN")
        )
        processor = process_ifeval_inst_list
    elif source == IFEvalSource.HF_MIXED_TRAINING:
        dataset = load_dataset(
            "ServiceNow-AI/mixed-training-text-datasets", 
            '80k-if-math-coding-fncalling-stem',
            split="train",
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN")
        )
        dataset = dataset.filter(lambda sample: sample['ability'] == 'ifeval')
        processor = process_ifeval_mixed_training
    else:
        raise ValueError(f"Unknown IFEval source: {source}")
    
    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    
    return dataset, processor


def load_datasets(dataset_names: List[str] | str | None, seed: int | None = None) -> List[Dict]:
    logger.info(f"Entered load_datasets with dataset_names: {dataset_names} and seed: {seed}")
    if dataset_names is None:
        return []

    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    all_datasets = []

    ifeval_sources = [name for name in dataset_names if name in {e.value for e in IFEvalSource}]
    
    for source_name in ifeval_sources:
        source = IFEvalSource(source_name)
        dataset, processor = load_ifeval_dataset(source, seed)
        logger.info(f"Loaded dataset of size {len(dataset)} from {source.value}")
        
        samples = [s for s in processor(dataset, source_name) if s is not None][:100]
        logger.info(f"Filtered to {len(samples)} samples")
        all_datasets += add_ids(samples)

    if len(all_datasets) == 0:
        logger.warning("No datasets loaded. Please check the dataset names provided.")

    return all_datasets