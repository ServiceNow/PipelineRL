import json
import logging
import random
import re
from typing import Dict, List, Tuple

import datasets
import hydra
from datasets import load_dataset
from omegaconf import DictConfig
import os

logger = logging.getLogger(__name__)

def add_ids(dataset: list[dict]):
    for i, entry in enumerate(dataset):
        entry["id"] = i
    return dataset


def process_coding_dataset(dataset, dataset_name):
    for sample in dataset:
        reward_data = json.loads(sample['reward_model']['ground_truth'])
        extra_info = sample.get('extra_info', {})
        if type(extra_info) == str:
            extra_info = json.loads(extra_info)
        yield {
            "dataset": dataset_name,
            "task": sample["prompt"][0]['content'],
            'reward_context': reward_data,
            'extra_info': extra_info
        }

def process_coding_dummy():
    task1 = """Implement a Python function `add_one` that takes an integer x and returns x + 1. Then implement the main block to read an integer from standard input, call add_one, and print the result."""

    task2 = """Implement a Python function `add_two` that takes an integer x and returns x + 2. Then implement the main block to read an integer from standard input, call add_two, and print the result."""

    task3 = """Implement a Python function `square_num` that takes an integer x and returns it's square x^2. Then implement the main block to read an integer from standard input, call square_num, and print the result."""

    dummy_sample1 = {
        "dataset": "dummy",
        "task": task1,
        "language": "python",
        "test_inputs": ["1", "5", "-3"],
        "test_expected_outputs": ["2", "6", "-2"],
        "fn_name": "add_one",
    }
    dummy_sample2 = {
        "dataset": "dummy",
        "task": task2,
        "language": "python",
        "test_inputs": ["1", "5", "-3"],
        "test_expected_outputs": ["3", "7", "-1"],
        "fn_name": "add_two",
    }
    dummy_sample3 = {
        "dataset": "square_num",
        "task": task3,
        "language": "python",
        "test_inputs": ["1", "5", "-3"],
        "test_expected_outputs": ["1", "25", "9"],
        "fn_name": "square_num",
    }

    dataset = [dummy_sample1, dummy_sample2, dummy_sample3]

    for sample in dataset:
        yield sample


def load_datasets(dataset_names: List[str] | str | None, seed: int | None = None) -> List[Tuple[str, Dict]]:
    if dataset_names is None:
        return []

    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    datasets = []

    if 'mixed-training-text-datasets' in dataset_names:
        dataset = load_dataset(
            "ServiceNow-AI/mixed-training-text-datasets", 
            token=os.environ.get("HUGGINGFACE_TOKEN", None),
            split="train",
            trust_remote_code=True)

        dataset = dataset.filter(lambda sample: sample['ability'] == 'code')
        dataset = dataset.shuffle(seed=seed) if seed is not None else dataset
        samples = [s for s in process_coding_dataset(dataset, "mixed-training-text-datasets-code") if s is not None and s['reward_context']['call_type'] == 'assert'][:5]
        logger.info(f"Loading ServiceNow-AI/mixed-training-text-datasets train dataset - code samples: {len(samples)} samples")
        sample_types = [s['reward_context']['call_type'] for s in samples]
        logger.info(f"Samples types: {sample_types}")
        datasets += add_ids(samples)

    if "dummy" in dataset_names:
        samples = [s for s in process_coding_dummy() if s is not None]
        logger.info(f"Loading dummy dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if len(datasets) == 0:
        logger.warning("No datasets loaded. Please check the dataset names provided.")

    return datasets