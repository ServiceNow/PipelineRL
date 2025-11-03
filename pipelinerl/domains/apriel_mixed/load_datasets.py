import json
import logging
import random
import re
from typing import Dict, List, Tuple
import os 

import datasets
import hydra
from datasets import load_dataset
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def add_ids(dataset: list[dict]):
    for i, entry in enumerate(dataset):
        entry["id"] = i
    return dataset

def remove_null_fields(turn: dict) -> dict:
    return {k: v for k, v in turn.items() if v is not None}

def preprocess_prompt(prompt: list[dict]) -> list[dict]:
    # Remove null fields from each turn in the prompt
    return [remove_null_fields(turn) for turn in prompt]


def process_apriel_mixed(dataset, dataset_name):
    for sample in dataset:
        try:
            reward_data = json.loads(sample['reward_model']['ground_truth'])
        except:
            reward_data = sample['reward_model']['ground_truth']
        
        try:
            extra_info = json.loads(sample.get('extra_info', '{}'))
        except:
            extra_info = {}
        yield {
            "dataset": dataset_name,
            "prompt": preprocess_prompt(sample["prompt"]),
            'reward_context': reward_data,
            'domain': sample.get('ability', 'Unknown'),
            'extra_info': extra_info
        }

def load_datasets(dataset_names: List[str] | str | None, seed: int | None = None) -> List[Tuple[str, Dict]]:
    logger.info(f"Shiva-Entered load_datasets with dataset_names: {dataset_names} and seed: {seed}")
    if dataset_names is None:
        return []

    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    datasets = []

    if "apriel_mixed" in dataset_names:
        dataset = load_dataset(
            "ServiceNow-AI/mixed-training-text-datasets", 
            split="train",
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"))

        dataset = dataset.filter(lambda sample: sample['ability'] != 'code')
        logger.info(f"Shiva-Loaded dataset of size {len(dataset)}")
        dataset = dataset.shuffle(seed=seed) if seed is not None else dataset

        # TODO: Marker remove the limit of 5 samples.
        samples = [s for s in process_apriel_mixed(dataset, "apriel_mixed") if s is not None][:5]
        logger.info(f"Shiva-Filtered dataset of size {len(dataset)}")
        logger.info(f"Loading ServiceNow-AI/mixed-training-text-datasets train dataset - apriel_mixed samples: {len(samples)} samples")
        datasets += add_ids(samples)

    if len(datasets) == 0:
        logger.warning("No datasets loaded. Please check the dataset names provided.")

    return datasets
