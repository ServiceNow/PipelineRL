import logging
import re
from typing import Dict, Any, List, Union
from datasets import load_dataset

logger = logging.getLogger(__name__)


def _load_gsm8k_dataset(split: str) -> List[Dict[str, Any]]:
    dataset = load_dataset("openai/gsm8k", "main", split=split, trust_remote_code=True)
    samples = []
    for item in dataset:
        problem = extract_result_value(item)
        problem.update({
            "task": item["question"],
            "dataset": f"gsm8k_{split}",
        })
        samples.append(problem)
    return samples


def _load_math_dataset(split: str) -> List[Dict[str, Any]]:
    from pipelinerl.domains.math.load_datasets import load_datasets as math_load_datasets
    return math_load_datasets([f"math_{split}"])


def _load_aime_dataset(year: int) -> List[Dict[str, Any]]:
    aime_dataset = load_dataset("AI-MO/aimo-validation-aime", split="train", trust_remote_code=True)
    aime_dataset = aime_dataset.filter(lambda x: str(year) in x["url"])
    
    samples = []
    for item in aime_dataset:
        problem = {
            "task": item["problem"],
            "answer": f"\\boxed{{{item['answer']}}}",
            "dataset": f"aime_{year}",
            "level": "",
            "type": "aime",
        }
        samples.append(problem)
    
    logger.info(f"Loaded AIME {year}: {len(samples)} samples")
    return add_ids(samples)


def _load_amc_dataset(year: int) -> List[Dict[str, Any]]:
    amc_dataset = load_dataset("AI-MO/aimo-validation-amc", split="train", trust_remote_code=True)
    amc_dataset = amc_dataset.filter(lambda x: str(year) in x["url"])
    
    samples = []
    for item in amc_dataset:
        problem = {
            "task": item["problem"],
            "answer": f"\\boxed{{{item['answer']}}}",
            "dataset": f"amc_{year}",
            "level": "",
            "type": "amc",
        }
        samples.append(problem)
    
    logger.info(f"Loaded AMC {year}: {len(samples)} samples")
    return add_ids(samples)


def add_ids(dataset):
    """Add sequential IDs to dataset items."""
    for i, entry in enumerate(dataset):
        entry["id"] = i
    return dataset


def load_datasets(dataset_names: List[str], **kwargs) -> List[Dict[str, Any]]:
    """Load datasets for TIR domain."""
    all_problems = []
    
    # Dataset loading map for cleaner logic
    dataset_loaders = {
        "gsm8k_train": lambda: _load_gsm8k_dataset("train"),
        "gsm8k_test": lambda: _load_gsm8k_dataset("test"),
        "math_train": lambda: _load_math_dataset("train"),
        "math_test": lambda: _load_math_dataset("test"),
        "aime_2024": lambda: _load_aime_dataset(2024),
        "aime_2023": lambda: _load_aime_dataset(2023),
        "aime_2022": lambda: _load_aime_dataset(2022),
        "amc_2023": lambda: _load_amc_dataset(2023),
        "amc_2022": lambda: _load_amc_dataset(2022),
    }
    
    for name in dataset_names:
        if name in dataset_loaders:
            samples = dataset_loaders[name]()
            logger.info(f"Loaded {name}: {len(samples)} samples")
            
            # GSM8K dataset needs IDs
            if name.startswith("gsm8k"):
                samples = add_ids(samples)
            
            all_problems.extend(samples)
            
        else:
            logger.warning(f"Unknown dataset: {name}")
    
    logger.info(f"Loaded {len(all_problems)} problems from {len(dataset_names)} datasets")
    return all_problems


def extract_result_value(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Extract numerical result from dataset sample."""
    sample = dict(sample)
    
    if "answer" in sample:
        # GSM8K format
        answer_text = sample["answer"]
        sample["answer"] = answer_text
        match = re.search(r"####\s*([+-]?\d*\.?\d+)", answer_text)
        if match:
            value = match.group(1)
            sample["value"] = float(value) if '.' in value else int(value)
        else:
            sample["value"] = None
            
    elif "solution" in sample:
        # MATH format
        solution = sample["solution"]
        sample["answer"] = solution
        
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", solution)
        if boxed_match:
            boxed_content = boxed_match.group(1).strip()
            sample["value"] = _parse_mathematical_value(boxed_content)
        else:
            sample["value"] = _extract_answer_from_text(solution)
    # Add missing fields
    sample.setdefault("level", "")
    sample.setdefault("type", "")
    
    return sample


def _parse_mathematical_value(content: str) -> Union[float, int, None]:
    """Parse mathematical expressions."""
    try:
        import sympy as sp
        content = content.replace("\\pi", "*pi").replace("π", "*pi")
        result = sp.sympify(content)
        return float(result.evalf())
    except Exception as e:
        try:
            return float(content) if '.' in content else int(content)
        except ValueError:
            number_match = re.search(r"([+-]?\d*\.?\d+)", content)
            if number_match:
                value = number_match.group(1)
                return float(value) if '.' in value else int(value)
            return None


def _extract_answer_from_text(text: str) -> Union[float, int, None]:
    """Extract numerical answer from text."""
    patterns = [
        r"(?:answer|result)\s+is\s+([+-]?\d*\.?\d+)",
        r"([+-]?\d*\.?\d+)$"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            return float(value) if '.' in value else int(value)
    
    return None 