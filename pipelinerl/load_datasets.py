import logging
import re
from typing import Dict, List, Tuple

import json
import datasets
import hydra
import random
from datasets import load_dataset
from omegaconf import DictConfig
from tqdm import tqdm

"""
math_verify expects the following LaTeX format for the gold answer (with $ or \\boxed).
For example, this will parse correctly:
\\boxed{\\begin{pmatrix} -\\frac{1}{3} \\ \\frac{2}{3} \\ \\frac{5}{3} \\end{pmatrix}}$
and this will not parse:
\\begin{pmatrix} -\\frac{1}{3} \\ \\frac{2}{3} \\ \\frac{5}{3} \\end{pmatrix}
"""

logger = logging.getLogger(__name__)


def process_eurus(dataset):
    for item in dataset:
        if item["ability"] != "math":
            # discard the coding problems for now
            yield None
        answer = "\\boxed{" + str(item["reward_model"]["ground_truth"]) + "}"
        task = item["prompt"][1]["content"]
        task = task.replace(
            "\n\nPresent the answer in LaTex format: \\boxed{Your answer}", ""
        )
        yield {
            "dataset": item["data_source"],
            "task": task,
            "answer": answer,
        }


def process_math(dataset, dataset_name):
    for item in dataset:
        if "correctness_math_verify" in item:
            if not any(item["correctness_math_verify"]):
                # correctness cannot be verified with math_verify
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
            answer = item["solution"]
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


def process_gsm8k(dataset, dataset_name):
    for item in dataset:
        sample = {
            "dataset": dataset_name,
            "task": item["question"],
            "answer": item["answer"].split("####")[1],
        }
        yield sample

def process_limo(dataset):
    for item in dataset:
        task = item["question"]
        answer = "\\boxed{" + str(item["answer"]) + "}"
        yield {
            "dataset": "limo",
            "task": task,
            "answer": answer,
        }

def process_aime_and_amc(dataset, dataset_name):
    for item in dataset:
        task = item["problem"]
        answer = "\\boxed{" + str(item["answer"]) + "}"
        yield {
            "dataset": dataset_name,
            "task": task,
            "answer": answer,
        }

def process_open_reasoner(dataset, dataset_name):
    for item in dataset:
        # Note: Open Reasoner tasks sometimes have preamble, e.g.
        # - Example 31 (2004 College Entrance Examination Hunan Paper) 
        # - 8. 
        # - 4. (7 points)
        # We are currently ignoring the preamble
        task = item['0']['value']
        answer =  "\\boxed{" + item['1']['ground_truth']['value'] + "}"
        yield {
            "dataset": dataset_name,
            "task": task,
            "answer": answer
        }

def process_gpqa(dataset, dataset_name):
    for item in dataset:
        yield {
            "dataset": dataset_name,
            "task": item["problem"],
            "answer": item["solution"],
        }

def process_countdown(dataset):
    counter = 0
    for item in dataset:
        problem = item['prompt'][0]['content']
        problem = problem.split('<|im_start|>user')[-1]
        problem = problem.split('<|im_start|>assistant')[0]
        problem = problem.split('<|im_end|>')[0]
        problem = problem.strip()
        answer = '-'.join(['countdown', str(item['target']),str(item['nums'])]) 
        yield {
            "dataset": "countdown",
            "task": problem,
            "answer": answer,
            "id": counter   
        }
        counter += 1

def load_math(split):
    # FIXME?
    data = []
    for config in [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]:
        dataset = load_dataset(
            "EleutherAI/hendrycks_math", config, split=split, trust_remote_code=True
        )
        for sample in dataset:
            data.append(sample)
    return datasets.Dataset.from_list(data)


def _load_aime_dataset(year: int, upsample_factor: int = 0) -> list[dict]:
    aime_dataset = load_dataset(
        "AI-MO/aimo-validation-aime", split="train", trust_remote_code=True
    )
    aime_dataset = aime_dataset.filter(lambda x: str(year) in x["url"])
    
    dataset_name = f"aime_{year}" + ("" if upsample_factor > 0 else "_original")
    samples = [
        s for s in process_aime_and_amc(aime_dataset, dataset_name) if s is not None
    ]

    original_size = len(samples)
    if upsample_factor > 0:
        samples *= upsample_factor

    logger.info(f"Loading aime {year} dataset: {len(samples)} samples" + (f" (upsampled from {original_size})" if upsample_factor > 0 else ""))
    return add_ids(samples)


def _load_amc_dataset(year: int, upsample_factor: int = 0) -> list[dict]:
    amc_dataset = load_dataset(
        "AI-MO/aimo-validation-amc", split="train", trust_remote_code=True
    )
    amc_dataset = amc_dataset.filter(lambda x: str(year) in x["url"])

    dataset_name = f"amc_{year}" + ("" if upsample_factor > 0 else "_original")
    samples = [
        s for s in process_aime_and_amc(amc_dataset, dataset_name) if s is not None
    ]

    original_size = len(samples)
    if upsample_factor > 0:
        samples *= upsample_factor

    logger.info(f"Loading amc {year} dataset: {len(samples)} samples" + (f" (upsampled from {original_size})" if upsample_factor > 0 else ""))
    return add_ids(samples)


def add_ids(dataset: list[dict]):
    for i, entry in enumerate(dataset):
        entry["id"] = i
    return dataset


def load_datasets(dataset_names: List[str] | str | None) -> List[Tuple[str, Dict]]:
    if dataset_names is None:
        return []

    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    datasets = []
    if "eurus_train" in dataset_names:
        dataset = load_dataset(
            "PRIME-RL/Eurus-2-RL-Data", split="train", trust_remote_code=True
        )
        samples = [s for s in process_eurus(dataset) if s is not None]
        logger.info(f"Loading eurus train dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    # great for debugging since its much smaller than eurus train
    if "eurus_validation" in dataset_names:
        dataset = load_dataset(
            "PRIME-RL/Eurus-2-RL-Data", split="validation", trust_remote_code=True
        )
        samples = [s for s in process_eurus(dataset) if s is not None]
        logger.info(f"Loading eurus validation dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "math_train" in dataset_names:
        # math_dataset = load_math("train")
        dataset = load_dataset(
            "hendrycks/competition_math", split="train", trust_remote_code=True
        )
        samples = [s for s in process_math(dataset, "math_train") if s is not None]
        logger.info(f"Loading math train dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "math_simplerl_train" in dataset_names:
        # SimpleRL MATH dataset
        #   level 3-5 math problems from both train and test sets of the original MATH dataset (excluding problems from MATH-500)
        # math_dataset = load_math("train")
        dataset = load_dataset(
            "json",
            data_files="https://raw.githubusercontent.com/hkust-nlp/simpleRL-reason/refs/heads/v0/train/data/math_level3to5_data_processed_with_qwen_prompt.json",
            split="train",
            trust_remote_code=True,
        )
        samples = [
            s for s in process_math(dataset, "math_simplerl_train") if s is not None
        ]
        logger.info(f"Loading math simplerl train dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "simplerl_math_subset_1000" in dataset_names:
        # SimpleRL MATH dataset subset
        #   level 3-5 math problems from both train and test sets of the original MATH dataset (excluding problems from MATH-500)
        # math_dataset = load_math("train")
        dataset = load_dataset(
            "json",
            data_files="https://raw.githubusercontent.com/hkust-nlp/simpleRL-reason/refs/heads/v0/train/data/math_level3to5_data_processed_with_qwen_prompt.json",
            split="train",
            trust_remote_code=True,
        )
        samples = [
            s for s in process_math(dataset, "math_simplerl_subset") if s is not None
        ]
        random.seed(42)
        random.shuffle(samples)
        samples = samples[:1000]
        logger.info(f"Loading math simplerl subset test dataset: {len(samples)} samples")
        datasets += add_ids(samples)
    
    if "deepscaler_preview" in dataset_names:
        dataset = load_dataset(
            "agentica-org/DeepScaleR-Preview-Dataset", split="train", trust_remote_code=True
        )
        samples = [s for s in process_math(dataset, "deepscaler") if s is not None]
        logger.info(f"Loading deepscaler preview train dataset: {len(samples)} samples")
        datasets += add_ids(samples)


    if "math_test" in dataset_names:
        # math_dataset = load_math("test")
        dataset = load_dataset(
            "hendrycks/competition_math", split="test", trust_remote_code=True
        )
        samples = [s for s in process_math(dataset, "math_test") if s is not None]
        logger.info(f"Loading math test dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "omni_math_500" in dataset_names:
        dataset = load_dataset(
            "reliable-agents/Omni-MATH-500", split="test", trust_remote_code=True
        )
        samples = [s for s in process_math(dataset, "omni_math_500") if s is not None]
        logger.info(f"Loading omni math 500 dataset: {len(samples)} samples")
        datasets += add_ids(samples)


    if "math_500" in dataset_names:
        dataset = load_dataset(
            "HuggingFaceH4/MATH-500", split="test", trust_remote_code=True
        )
        samples = [s for s in process_math(dataset, "math_500") if s is not None]
        logger.info(f"Loading math 500 dataset: {len(samples)} samples")
        datasets += add_ids(samples)
    
    if "open_r1_math_220k" in dataset_names:
        dataset = load_dataset(
            "open-r1/OpenR1-Math-220k", split="default", trust_remote_code=True
        )
        samples = [s for s in process_math(dataset, "open_r1_math_220k") if s is not None]
        logger.info(f"Loading open r1 math 220k dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "gpqa_main" in dataset_names:
        dataset = load_dataset(
            "hendrydong/gpqa_main", split="test", trust_remote_code=True
        )
        samples = [s for s in process_gpqa(dataset, "gpqa_main") if s is not None]
        logger.info(f"Loading gpqa main dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "gpqa_diamond" in dataset_names:
        dataset = load_dataset(
            "hendrydong/gpqa_diamond", split="test", trust_remote_code=True
        )
        samples = [s for s in process_gpqa(dataset, "gpqa_diamond") if s is not None]
        logger.info(f"Loading gpqa diamond dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "gpqa_diamond" in dataset_names:
        pass

    if "gsm8k_train" in dataset_names:
        dataset = load_dataset(
            "openai/gsm8k", "main", split="train", trust_remote_code=True
        )
        samples = [s for s in process_gsm8k(dataset, "gsm8k_train") if s is not None]
        logger.info(f"Loading gsm8k train dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "gsm8k_test" in dataset_names:
        dataset = load_dataset(
            "openai/gsm8k", "main", split="test", trust_remote_code=True
        )
        samples = [s for s in process_gsm8k(dataset, "gsm8k_test") if s is not None]
        logger.info(f"Loading gsm8k test dataset: {len(samples)} samples")
        datasets += add_ids(samples)
    
    if "limo" in dataset_names:
        dataset = load_dataset(
            "GAIR/LIMO", split="train", trust_remote_code=True
        )
        samples = [s for s in process_limo(dataset) if s is not None]
        logger.info(f"Loading limo dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "aime_2022" in dataset_names:
        datasets += _load_aime_dataset(2022, upsample_factor=16)

    if "aime_2022_original" in dataset_names:
        datasets += _load_aime_dataset(2022)

    if "aime_2023" in dataset_names:
        datasets += _load_aime_dataset(2023, upsample_factor=16)

    if "aime_2023_original" in dataset_names:
        datasets += _load_aime_dataset(2023)

    if "aime_2024" in dataset_names:
        datasets += _load_aime_dataset(2024, upsample_factor=16)

    if "aime_2024_original" in dataset_names:
        datasets += _load_aime_dataset(2024)

    if "amc_2022" in dataset_names:
        # TODO: AMC 2022 is 43 problems, is that to be expected?
        datasets += _load_amc_dataset(2022, upsample_factor=16)

    if "amc_2022_original" in dataset_names:
        datasets += _load_amc_dataset(2022)

    if "amc_2023" in dataset_names:
        datasets += _load_amc_dataset(2023, upsample_factor=16)

    if "amc_2023_original" in dataset_names:
        datasets += _load_amc_dataset(2023)
    
    if "sometimes_success_data" in dataset_names:
        PATH = "/mnt/llmd/data/sometimes_success_data/data.jsonl"
        with open(PATH, "r") as f:
            samples = [json.loads(line) for line in f]
        logger.info(f"Loading easy data dataset: {len(samples)} samples")
        datasets += add_ids(samples)
    
    if "open_reasoner_zero_57k" in dataset_names:
        dataset = load_dataset(
            "json",
            data_files="https://raw.githubusercontent.com/Open-Reasoner-Zero/Open-Reasoner-Zero/refs/heads/main/data/orz_math_57k_collected.json",
            split="train",
            trust_remote_code=True,
        )
        samples = [s for s in process_open_reasoner(dataset, "open_reasoner_zero_57k") if s is not None]
        logger.info(f"Loading Open Reasoner Zero dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "open_reasoner_zero_extended_72k" in dataset_names:
        dataset = load_dataset(
            "json",
            data_files="https://raw.githubusercontent.com/Open-Reasoner-Zero/Open-Reasoner-Zero/refs/heads/main/data/orz_math_72k_collection_extended.json",
            split="train",
            trust_remote_code=True,
        )
        samples = [s for s in process_open_reasoner(dataset, "open_reasoner_zero_extended_72k") if s is not None]
        logger.info(f"Loading Open Reasoner Zero extended dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "open_reasoner_zero_hard_13k" in dataset_names:
        dataset = load_dataset(
            "json",
            data_files="https://raw.githubusercontent.com/Open-Reasoner-Zero/Open-Reasoner-Zero/refs/heads/main/data/orz_math_13k_collection_hard.json",
            split="train",
            trust_remote_code=True,
        )
        samples = [s for s in process_open_reasoner(dataset, "open_reasoner_zero_hard_13k") if s is not None]
        logger.info(f"Loading Open Reasoner Zero hard dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    for dataset_name in dataset_names:
        test_matched = re.match(r"multiplication_(\d+)_by_(\d+)_(\d+)_test", dataset_name)
        train_matched = re.match(r"multiplication(_upto)?_(\d+)_by_(\d+)_(\d+)_train", dataset_name)
        if test_matched:
            num_digits_1 = int(test_matched.group(1))
            num_digits_2 = int(test_matched.group(2))
            num_samples = int(test_matched.group(3))
            dataset = load_dataset("json", data_files=f"/mnt/llmd/data/ehsan_kamalloo/multiplication/multiplication_{num_digits_1}_by_{num_digits_2}_{num_samples}_test.jsonl", split="train")
            samples = [s for s in process_math(dataset, f"multiplication_{num_digits_1}_by_{num_digits_2}_{num_samples}_test") if s is not None]
            logger.info(f"Loading multiplication {num_digits_1}_by_{num_digits_2} dataset: {len(samples)} samples")
            datasets += add_ids(samples)
        elif train_matched:
            upto_prefix = train_matched.group(1) or ""
            num_digits_1 = int(train_matched.group(2))
            num_digits_2 = int(train_matched.group(3))
            num_samples = int(train_matched.group(4))
            dataset = load_dataset("json", data_files=f"/mnt/llmd/data/ehsan_kamalloo/multiplication/multiplication{upto_prefix}_{num_digits_1}_by_{num_digits_2}_{num_samples}_train.jsonl", split="train")
            samples = [s for s in process_math(dataset, f"multiplication{upto_prefix}_{num_digits_1}_by_{num_digits_2}_{num_samples}_train") if s is not None]
            logger.info(f"Loading multiplication {upto_prefix}_{num_digits_1}_by_{num_digits_2} dataset: {len(samples)} samples")
            datasets += add_ids(samples)

    if 'countdown' in dataset_names:
        dataset = load_dataset(
            'parquet', data_files='/mnt/llmd/data/xiaoyin/train.parquet', trust_remote_code=True, split='train'
        )
        samples = [s for s in process_countdown(dataset) if s is not None]
        logger.info(f"Loading countdown dataset: {len(samples)} samples")
        datasets += samples

    if len(datasets) == 0:
        raise ValueError("No datasets loaded")


    return datasets


@hydra.main(
    config_path="../conf/",
    config_name="base",
    version_base="1.3.2",
)
def main(cfg: DictConfig):
    train_samples = load_datasets(cfg.train_dataset_names)
    test_samples = load_datasets(cfg.test_dataset_names)



if __name__ == "__main__":
    main()
