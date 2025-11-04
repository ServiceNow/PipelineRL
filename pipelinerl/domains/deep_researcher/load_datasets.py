"""Load research questions dataset for DeepResearcher domain."""
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def add_ids(dataset: list[dict]):
    for i, entry in enumerate(dataset):
        entry["id"] = i
    return dataset


def load_datasets(dataset_names: List[str] | str | None, seed: int | None = None) -> List[Dict]:
    if dataset_names is None:
        return []
    
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    datasets = []
    
    for dataset_name in dataset_names:
        if "deep_researcher" in dataset_name:
            samples = _load_mock_dataset(dataset_name)
            logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
            datasets += add_ids(samples)
    
    if len(datasets) == 0:
        logger.warning("No datasets loaded. Check dataset names.")
    
    return datasets


def _load_mock_dataset(dataset_name: str) -> List[Dict]:
    mock_questions = [
        {
            "question": "What is the capital of France?",
            "reward_context": {
                "answer_type": "contains",
                "ground_truth": "Paris",
                "aliases": ["paris"]
            },
            "dataset": dataset_name,
            "category": "geography"
        },
        {
            "question": "What is the population of Tokyo?",
            "reward_context": {
                "answer_type": "contains",
                "ground_truth": "14 million",
                "aliases": ["14000000", "fourteen million", "37 million"]
            },
            "dataset": dataset_name,
            "category": "geography"
        },
        {
            "question": "How tall is the Eiffel Tower?",
            "reward_context": {
                "answer_type": "contains",
                "ground_truth": "330 meters",
                "aliases": ["330m", "1083 feet", "1083 ft"]
            },
            "dataset": dataset_name,
            "category": "facts"
        },
        {
            "question": "What programming language was created by Guido van Rossum?",
            "reward_context": {
                "answer_type": "contains",
                "ground_truth": "Python",
                "aliases": ["python"]
            },
            "dataset": dataset_name,
            "category": "technology"
        },
        {
            "question": "What is the capital of Japan?",
            "reward_context": {
                "answer_type": "contains",
                "ground_truth": "Tokyo",
                "aliases": ["tokyo"]
            },
            "dataset": dataset_name,
            "category": "geography"
        },
        {
            "question": "What is machine learning?",
            "reward_context": {
                "answer_type": "fuzzy",
                "ground_truth": "subset of artificial intelligence that enables systems to learn from data",
                "aliases": ["AI technique", "learning from data"]
            },
            "dataset": dataset_name,
            "category": "technology"
        },
        {
            "question": "What is the population of France?",
            "reward_context": {
                "answer_type": "contains",
                "ground_truth": "67 million",
                "aliases": ["67000000", "sixty-seven million"]
            },
            "dataset": dataset_name,
            "category": "geography"
        },
        {
            "question": "When was the Eiffel Tower built?",
            "reward_context": {
                "answer_type": "contains",
                "ground_truth": "1889",
                "aliases": ["eighteen eighty-nine"]
            },
            "dataset": dataset_name,
            "category": "history"
        },
    ]
    
    if "train" in dataset_name:
        return mock_questions[:6]
    elif "test" in dataset_name:
        return mock_questions[6:]
    else:
        return mock_questions
