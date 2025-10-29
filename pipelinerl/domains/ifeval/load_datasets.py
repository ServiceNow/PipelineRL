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

# dict_keys(['func_name', 'N', 'quantifier', 'end_phrase', 'keyword_list', 'word', 'forbidden_words', 'letter', 'i', 'first_word', 'postscript_marker', 'options', 'section_splitter', 'original_prompt'])
def process_ifeval(dataset, dataset_name):
    for sample in dataset:
        reward_data = json.loads(sample['reward_model']['ground_truth'])
        yield {
            "dataset": dataset_name,
            "task": sample["prompt"][0]['content'],
            'reward_context': reward_data
        }

def load_datasets(dataset_names: List[str] | str | None, seed: int | None = None) -> List[Tuple[str, Dict]]:
    logger.info(f"Shiva-Entered load_datasets with dataset_names: {dataset_names} and seed: {seed}")
    if dataset_names is None:
        return []

    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    datasets = []

    if "ifeval" in dataset_names:
        dataset = load_dataset(
            "ServiceNow-AI/mixed-training-text-datasets", 
            split="train",
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"))
        
        dataset = dataset.filter(lambda sample: sample['ability'] == 'ifeval')
        logger.info(f"Shiva-Loaded dataset of size {len(dataset)}")
        dataset = dataset.shuffle(seed=seed) if seed is not None else dataset
        # TODO: Marker remove the limit of 5 samples.
        samples = [s for s in process_ifeval(dataset, "ifeval") if s is not None][:5]
        logger.info(f"Shiva-Filtered dataset of size {len(dataset)}")
        logger.info(f"Loading ServiceNow-AI/mixed-training-text-datasets train dataset - ifeval samples: {len(samples)} samples")
        datasets += add_ids(samples)

    if len(datasets) == 0:
        logger.warning("No datasets loaded. Please check the dataset names provided.")

    return datasets
