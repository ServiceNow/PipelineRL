import json
import logging
import random
import re
from typing import Dict, List, Tuple
import os 

from i3_logic.task2verifier import verifier_classes

import datasets
import hydra
from datasets import load_dataset
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def add_ids(dataset: list[dict]):
    for i, entry in enumerate(dataset):
        entry["id"] = i
    return dataset

def process_logic(dataset, dataset_name):
    for sample in dataset:
        reward_data = json.loads(sample['reward_model']['ground_truth'])
        yield {
            "dataset": dataset_name,
            # Single turn only.
            "task": sample["prompt"][0]['content'],
            'reward_context': reward_data,
            'extra_info': json.loads(sample['extra_info'])
        }

def load_datasets(dataset_names: List[str] | str | None, seed: int | None = None) -> List[Tuple[str, Dict]]:
    if dataset_names is None:
        return []

    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    datasets = []
    if "logic" in dataset_names:
        dataset = load_dataset(
            "ServiceNow-AI/mixed-training-text-datasets", 
            "prime-intellect-logic",
            token=os.environ.get("HF_TOKEN"))['train']
        
        dataset = dataset.filter(lambda sample: sample['ability'] == 'logic' and json.loads(sample['extra_info']).get('task', '') in verifier_classes)
        logger.info(f"Loaded dataset of size {len(dataset)}")
        dataset = dataset.shuffle(seed=seed) if seed is not None else dataset
        # TODO: Marker remove the limit of 5 samples.
        samples = [s for s in process_logic(dataset, "logic") if s is not None]
        logger.info(f"Filtered dataset of size {len(dataset)}")
        logger.info(f"Loading ServiceNow-AI/mixed-training-text-datasets train dataset - logic samples: {len(samples)} samples")
        datasets += add_ids(samples)

    if len(datasets) == 0:
        logger.warning("No datasets loaded. Please check the dataset names provided.")

    return datasets
