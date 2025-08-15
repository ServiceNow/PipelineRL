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
    """Load MATH dataset directly"""
    from datasets import load_dataset
    from pipelinerl.domains.math.load_datasets import add_ids
    
    dataset = load_dataset("hendrycks/competition_math", "main", split=split, trust_remote_code=True)
    samples = [s for s in _process_math_for_tir(dataset, f"math_{split}") if s is not None]
    logger.info(f"Loading math {split} dataset for TIR: {len(samples)} samples")
    return add_ids(samples)


def _process_math_for_tir(dataset, dataset_name):
    """Process MATH dataset for TIR domain with proper boxed answer extraction."""
    for item in dataset:
        if "correctness_math_verify" in item:
            if not any(item["correctness_math_verify"]):
                yield None
                continue
        if "problem" in item:
            question = item["problem"]
        elif "question" in item:
            question = item["question"]
        else:
            yield None
            continue
        if "subject" in item and "type" not in item:
            item["type"] = item["subject"]
        
        if "answer" in item:
            answer = "\\boxed{" + item["answer"] + "}"
        elif "solution" in item:
            solution = item["solution"]
            answer = _extract_boxed_answer(solution)
        else:
            yield None
            continue
            
        sample = {
            "dataset": dataset_name,
            "level": item.get("level", ""),
            "type": item.get("type", ""),
            "task": question,
            "answer": answer,
        }
        yield sample


def _extract_boxed_answer(solution: str) -> str:
    """Extract the boxed answer from a solution, fallback to full solution if not found."""
    boxed_start = solution.rfind("\\boxed{")
    if boxed_start >= 0:
        brace_count = 0
        i = boxed_start + 7
        while i < len(solution):
            if solution[i] == '{':
                brace_count += 1
            elif solution[i] == '}':
                if brace_count == 0:
                    boxed_content = solution[boxed_start + 7:i]
                    return f"\\boxed{{{boxed_content}}}"
                else:
                    brace_count -= 1
            i += 1
    
    return solution


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


def _load_openreasoner_dataset(dataset_name: str) -> List[Dict[str, Any]]:
    """Load OpenReasoner datasets following the math domain pattern."""
    try:
        data_file_urls = {
            "open_reasoner_zero_57k": "https://raw.githubusercontent.com/Open-Reasoner-Zero/Open-Reasoner-Zero/refs/heads/main/data/orz_math_57k_collected.json",
            "open_reasoner_zero_extended_72k": "https://raw.githubusercontent.com/Open-Reasoner-Zero/Open-Reasoner-Zero/refs/heads/main/data/orz_math_72k_collection_extended.json",
            "open_reasoner_zero_hard_13k": "https://raw.githubusercontent.com/Open-Reasoner-Zero/Open-Reasoner-Zero/refs/heads/main/data/orz_math_13k_collection_hard.json",
        }
        
        if dataset_name not in data_file_urls:
            logger.error(f"Unknown OpenReasoner dataset: {dataset_name}")
            return []
        
        # Load the dataset from the JSON file
        dataset = load_dataset(
            "json",
            data_files=data_file_urls[dataset_name],
            split="train",
            trust_remote_code=True,
        )
        
        samples = []
        for item in dataset:
            # Format: item["0"]["value"] = task, item["1"]["ground_truth"]["value"] = answer
            try:
                task = item["0"]["value"]
                answer_value = item["1"]["ground_truth"]["value"]
                
                # Ensure answer is in boxed format
                if not answer_value.startswith("\\boxed"):
                    answer = f"\\boxed{{{answer_value}}}"
                else:
                    answer = answer_value
                
                problem = {
                    "task": task,
                    "answer": answer,
                    "dataset": dataset_name,
                    "level": "",
                    "type": "reasoning",
                }
                samples.append(problem)
                
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping malformed item in {dataset_name}: {e}")
                continue
        
        logger.info(f"Loaded {dataset_name}: {len(samples)} samples")
        return samples
        
    except Exception as e:
        logger.error(f"Failed to load {dataset_name}: {e}")
        return []


def load_datasets(dataset_names: List[str], **kwargs) -> List[Dict[str, Any]]:
    """Load datasets for TIR domain."""
    all_problems = []
    
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
        "open_reasoner_zero_57k": lambda: _load_openreasoner_dataset("open_reasoner_zero_57k"),
        "open_reasoner_zero_extended_72k": lambda: _load_openreasoner_dataset("open_reasoner_zero_extended_72k"),
        "open_reasoner_zero_hard_13k": lambda: _load_openreasoner_dataset("open_reasoner_zero_hard_13k"),
    }
    
    logger.info(f"Attempting to load datasets: {dataset_names}")
    
    for name in dataset_names:
        if name in dataset_loaders:
            try:
                samples = dataset_loaders[name]()
                logger.info(f"Loaded {name}: {len(samples)} samples")
                
                if not samples:
                    logger.warning(f"Dataset {name} returned 0 samples!")
                
                if name.startswith("gsm8k"):
                    samples = add_ids(samples)
                
                all_problems.extend(samples)
            except Exception as e:
                logger.error(f"Failed to load dataset {name}: {e}")
                continue
            
        else:
            logger.warning(f"Unknown dataset: {name}")
    
    logger.info(f"Total problems loaded: {len(all_problems)}")
    
    if not all_problems:
        raise ValueError(f"No problems loaded from any datasets: {dataset_names}. Check dataset names and network connectivity.")
    
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