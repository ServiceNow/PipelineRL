import logging
from typing import List
import datasets
from datasets import load_dataset

logger = logging.getLogger(__name__)


def process_chartqa(dataset, dataset_name: str):
    """Process ChartQA dataset into standardized format."""
    for item in dataset:
        # ChartQA uses 'query' instead of 'question' and 'label' instead of 'answer'
        if "image" not in item or "query" not in item or "label" not in item:
            continue
            
        try:
            answer_str = str(item["label"][0]) if isinstance(item["label"], list) else str(item["label"])
            yield {
                "dataset": dataset_name,
                "image": item["image"],  # PIL Image object
                "question": item["query"],  # Use 'query' field
                "answer": answer_str,  # Use first label if list
                "human_or_machine": item.get("human_or_machine", 0),
            }
        except Exception as e:
            logger.error(f"Error processing item: {e}")
            continue


def add_ids(dataset: list[dict]):
    """Add sequential IDs to dataset items."""
    for i, entry in enumerate(dataset):
        entry["id"] = i
    return dataset


def load_problems(dataset_names: List[str] | str | None) -> List[dict]:
    """Load ChartQA datasets and return list of problems."""
    if dataset_names is None:
        return []

    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    datasets_list = []
    
    if "chartqa_train" in dataset_names:
        dataset = load_dataset("HuggingFaceM4/ChartQA", split="train", trust_remote_code=True)
        samples = list(process_chartqa(dataset, "chartqa_train"))
        datasets_list += add_ids(samples)
    
    if "chartqa_test" in dataset_names:
        dataset = load_dataset("HuggingFaceM4/ChartQA", split="test", trust_remote_code=True)
        samples = list(process_chartqa(dataset, "chartqa_test"))
        datasets_list += add_ids(samples)
    
    if "chartqa_val" in dataset_names:
        dataset = load_dataset("HuggingFaceM4/ChartQA", split="val", trust_remote_code=True)
        samples = list(process_chartqa(dataset, "chartqa_val"))
        datasets_list += add_ids(samples)

    if len(datasets_list) == 0:
        raise ValueError("No ChartQA datasets loaded")

    return datasets_list