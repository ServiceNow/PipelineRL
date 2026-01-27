import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import datasets
import hydra
from datasets import load_dataset
from omegaconf import DictConfig

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
        task = task.replace("\n\nPresent the answer in LaTex format: \\boxed{Your answer}", "")
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
            answer = item["answer"]
            # Only box if not already boxed
            if not answer.startswith("\\boxed{"):
                answer = "\\boxed{" + answer + "}"
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
        task = item["0"]["value"]
        answer = "\\boxed{" + item["1"]["ground_truth"]["value"] + "}"
        yield {"dataset": dataset_name, "task": task, "answer": answer}


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
        problem = item["prompt"][0]["content"]
        problem = problem.split("<|im_start|>user")[-1]
        problem = problem.split("<|im_start|>assistant")[0]
        problem = problem.split("<|im_end|>")[0]
        problem = problem.strip()
        answer = "-".join(["countdown", str(item["target"]), str(item["nums"])])
        yield {"dataset": "countdown", "task": problem, "answer": answer, "id": counter}
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
        dataset = load_dataset("EleutherAI/hendrycks_math", config, split=split, trust_remote_code=True)
        for sample in dataset:
            data.append(sample)
    return datasets.Dataset.from_list(data)


def _load_aime_dataset(year: int, upsample_factor: int = 0) -> list[dict]:
    if year == 2025:
        aime_dataset = load_dataset("MathArena/aime_2025", split="train", trust_remote_code=True)
    else:
        aime_dataset = load_dataset("AI-MO/aimo-validation-aime", split="train", trust_remote_code=True)
        aime_dataset = aime_dataset.filter(lambda x: str(year) in x["url"])

    dataset_name = f"aime_{year}" + ("" if upsample_factor > 0 else "_original")
    samples = [s for s in process_aime_and_amc(aime_dataset, dataset_name) if s is not None]

    original_size = len(samples)
    if upsample_factor > 0:
        samples *= upsample_factor

    logger.info(
        f"Loading aime {year} dataset: {len(samples)} samples"
        + (f" (upsampled from {original_size})" if upsample_factor > 0 else "")
    )
    return add_ids(samples)


def _load_amc_dataset(year: int, upsample_factor: int = 0) -> list[dict]:
    amc_dataset = load_dataset("AI-MO/aimo-validation-amc", split="train", trust_remote_code=True)
    amc_dataset = amc_dataset.filter(lambda x: str(year) in x["url"])

    dataset_name = f"amc_{year}" + ("" if upsample_factor > 0 else "_original")
    samples = [s for s in process_aime_and_amc(amc_dataset, dataset_name) if s is not None]

    original_size = len(samples)
    if upsample_factor > 0:
        samples *= upsample_factor

    logger.info(
        f"Loading amc {year} dataset: {len(samples)} samples"
        + (f" (upsampled from {original_size})" if upsample_factor > 0 else "")
    )
    return add_ids(samples)


def add_ids(dataset: list[dict]):
    for i, entry in enumerate(dataset):
        entry["id"] = i
    return dataset


def _resolve_custom_path(relative_paths: str | Sequence[str]) -> Path:
    """
    Resolve a path for locally generated datasets.

    Hydra jobs may change the working directory, so we check both the current
    directory and the repository root.
    """
    if isinstance(relative_paths, str):
        relative_paths = [relative_paths]

    resolved = Path(__file__).resolve()
    base_candidates = [Path.cwd()]
    if len(resolved.parents) >= 5:
        base_candidates.append(resolved.parents[4])

    candidates: List[Path] = []
    for rel in relative_paths:
        rel_path = Path(rel)
        candidates.append(rel_path)
        for base in base_candidates:
            if base == Path.cwd():
                continue
            candidates.append(base / rel_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Custom dataset not found. Tried: {[str(path) for path in candidates]}"
    )


def _load_custom_dataset(dataset_name: str) -> list[dict]:
    """
    Load a locally generated dataset by name.

    The loader searches under `datasets/custom/` and `datasets/custom_runs/` for either
    `<dataset_name>` or `<dataset_name>.jsonl`.
    """
    candidate_names: List[str] = []
    if dataset_name.endswith(".jsonl"):
        candidate_names.append(dataset_name)
    else:
        candidate_names.extend([dataset_name, f"{dataset_name}.jsonl"])

    search_paths: List[str] = []
    for name in candidate_names:
        search_paths.extend(
            [
                f"datasets/custom/{name}",
                f"datasets/custom_runs/{name}",
                name,
            ]
        )

    dataset_path = _resolve_custom_path(search_paths)
    with dataset_path.open("r", encoding="utf-8") as handle:
        samples = [json.loads(line) for line in handle if line.strip()]

    dataset_label = dataset_name[:-6] if dataset_name.endswith(".jsonl") else dataset_name

    for idx, sample in enumerate(samples):
        sample.setdefault("source_dataset", sample.get("dataset", dataset_label))
        sample.setdefault("source_id", sample.get("id"))
        sample["dataset"] = dataset_label
        sample["id"] = idx

    logger.info(f"Loading custom dataset {dataset_name}: {len(samples)} samples from {dataset_path}")
    return samples


def _is_hf_dataset_path(name: str) -> bool:
    """
    Check if a name looks like a HuggingFace dataset path (format: "org/dataset-name").

    Returns False for local file paths (multiple slashes, file extensions like .jsonl).
    """
    if "/" not in name:
        return False
    # Local paths typically have multiple slashes or file extensions
    if name.count("/") > 1:
        return False
    if name.endswith(".jsonl") or name.endswith(".json"):
        return False
    # Check both parts are non-empty
    parts = name.split("/")
    return len(parts) == 2 and all(parts)


# HuggingFace datasets that have custom loaders in load_datasets()
_PREDEFINED_HF_DATASETS = {
    "PRIME-RL/Eurus-2-RL-Data",
    "agentica-org/DeepScaleR-Preview-Dataset",
    "reliable-agents/Omni-MATH-500",
    "HuggingFaceH4/MATH-500",
    "open-r1/OpenR1-Math-220k",
    "hendrydong/gpqa_main",
    "hendrydong/gpqa_diamond",
    "GAIR/LIMO",
}


def _load_hf_dataset(dataset_name: str) -> list[dict]:
    """
    Load a HuggingFace dataset by its path (format: "org/dataset-name").

    Uses process_math to extract samples from the dataset.
    """
    dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)
    samples = [s for s in process_math(dataset, dataset_name) if s is not None]
    logger.info(f"Loading {dataset_name} dataset: {len(samples)} samples")
    return add_ids(samples)


def load_datasets(dataset_names: List[str] | str | None, seed: int | None = None) -> List[Tuple[str, Dict]]:
    if dataset_names is None:
        return []

    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    # Preserve order while de-duplicating
    dataset_names = list(dict.fromkeys(dataset_names))
    datasets = []
    remaining = set(dataset_names)
    if "eurus_train" in dataset_names:
        dataset = load_dataset("PRIME-RL/Eurus-2-RL-Data", split="train", trust_remote_code=True)
        samples = [s for s in process_eurus(dataset) if s is not None]
        logger.info(f"Loading eurus train dataset: {len(samples)} samples")
        datasets += add_ids(samples)
        remaining.discard("eurus_train")

    # great for debugging since its much smaller than eurus train
    if "eurus_validation" in dataset_names:
        dataset = load_dataset("PRIME-RL/Eurus-2-RL-Data", split="validation", trust_remote_code=True)
        samples = [s for s in process_eurus(dataset) if s is not None]
        logger.info(f"Loading eurus validation dataset: {len(samples)} samples")
        datasets += add_ids(samples)
        remaining.discard("eurus_validation")

    if "math_train" in dataset_names:
        # math_dataset = load_math("train")
        dataset = load_dataset("hendrycks/competition_math", split="train", trust_remote_code=True)
        samples = [s for s in process_math(dataset, "math_train") if s is not None]
        logger.info(f"Loading math train dataset: {len(samples)} samples")
        datasets += add_ids(samples)
        remaining.discard("math_train")

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
        samples = [s for s in process_math(dataset, "math_simplerl_train") if s is not None]
        logger.info(f"Loading math simplerl train dataset: {len(samples)} samples")
        datasets += add_ids(samples)
        remaining.discard("math_simplerl_train")

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
        samples = [s for s in process_math(dataset, "math_simplerl_subset") if s is not None]
        if seed is not None:
            random.seed(seed)
        random.shuffle(samples)
        samples = samples[:1000]
        logger.info(f"Loading math simplerl subset test dataset: {len(samples)} samples")
        datasets += add_ids(samples)
        remaining.discard("simplerl_math_subset_1000")

    if "deepscaler_preview" in dataset_names:
        dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train", trust_remote_code=True)
        samples = [s for s in process_math(dataset, "deepscaler") if s is not None]
        logger.info(f"Loading deepscaler preview train dataset: {len(samples)} samples")
        datasets += add_ids(samples)
        remaining.discard("deepscaler_preview")

    if "math_test" in dataset_names:
        # math_dataset = load_math("test")
        dataset = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)
        samples = [s for s in process_math(dataset, "math_test") if s is not None]
        logger.info(f"Loading math test dataset: {len(samples)} samples")
        datasets += add_ids(samples)
        remaining.discard("math_test")

    if "omni_math_500" in dataset_names:
        dataset = load_dataset("reliable-agents/Omni-MATH-500", split="test", trust_remote_code=True)
        samples = [s for s in process_math(dataset, "omni_math_500") if s is not None]
        logger.info(f"Loading omni math 500 dataset: {len(samples)} samples")
        datasets += add_ids(samples)
        remaining.discard("omni_math_500")

    if "math_500" in dataset_names:
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test", trust_remote_code=True)
        samples = [s for s in process_math(dataset, "math_500") if s is not None]
        logger.info(f"Loading math 500 dataset: {len(samples)} samples")
        datasets += add_ids(samples)
        remaining.discard("math_500")

    if "open_r1_math_220k" in dataset_names:
        dataset = load_dataset("open-r1/OpenR1-Math-220k", split="default", trust_remote_code=True)
        samples = [s for s in process_math(dataset, "open_r1_math_220k") if s is not None]
        logger.info(f"Loading open r1 math 220k dataset: {len(samples)} samples")
        datasets += add_ids(samples)
        remaining.discard("open_r1_math_220k")

    if "gpqa_main" in dataset_names:
        dataset = load_dataset("hendrydong/gpqa_main", split="test", trust_remote_code=True)
        samples = [s for s in process_gpqa(dataset, "gpqa_main") if s is not None]
        logger.info(f"Loading gpqa main dataset: {len(samples)} samples")
        datasets += add_ids(samples)
        remaining.discard("gpqa_main")

    if "gpqa_diamond" in dataset_names:
        dataset = load_dataset("hendrydong/gpqa_diamond", split="test", trust_remote_code=True)
        samples = [s for s in process_gpqa(dataset, "gpqa_diamond") if s is not None]
        logger.info(f"Loading gpqa diamond dataset: {len(samples)} samples")
        datasets += add_ids(samples)
        remaining.discard("gpqa_diamond")

    if "gpqa_diamond" in dataset_names:
        pass

    if "gsm8k_train" in dataset_names:
        dataset = load_dataset("openai/gsm8k", "main", split="train", trust_remote_code=True)
        samples = [s for s in process_gsm8k(dataset, "gsm8k_train") if s is not None]
        logger.info(f"Loading gsm8k train dataset: {len(samples)} samples")
        datasets += add_ids(samples)
        remaining.discard("gsm8k_train")

    if "gsm8k_test" in dataset_names:
        dataset = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
        samples = [s for s in process_gsm8k(dataset, "gsm8k_test") if s is not None]
        logger.info(f"Loading gsm8k test dataset: {len(samples)} samples")
        datasets += add_ids(samples)
        remaining.discard("gsm8k_test")

    if "limo" in dataset_names:
        dataset = load_dataset("GAIR/LIMO", split="train", trust_remote_code=True)
        samples = [s for s in process_limo(dataset) if s is not None]
        logger.info(f"Loading limo dataset: {len(samples)} samples")
        datasets += add_ids(samples)
        remaining.discard("limo")

    if "aime_2022" in dataset_names:
        datasets += _load_aime_dataset(2022, upsample_factor=16)
        remaining.discard("aime_2022")

    if "aime_2022_original" in dataset_names:
        datasets += _load_aime_dataset(2022)
        remaining.discard("aime_2022_original")

    if "aime_2023" in dataset_names:
        datasets += _load_aime_dataset(2023, upsample_factor=16)
        remaining.discard("aime_2023")

    if "aime_2023_original" in dataset_names:
        datasets += _load_aime_dataset(2023)
        remaining.discard("aime_2023_original")

    if "aime_2024" in dataset_names:
        datasets += _load_aime_dataset(2024, upsample_factor=16)
        remaining.discard("aime_2024")

    if "aime_2024_original" in dataset_names:
        datasets += _load_aime_dataset(2024)
        remaining.discard("aime_2024_original")

    if "aime_2025" in dataset_names:
        datasets += _load_aime_dataset(2025, upsample_factor=16)
        remaining.discard("aime_2025")

    if "aime_2025_original" in dataset_names:
        datasets += _load_aime_dataset(2025)
        remaining.discard("aime_2025_original")

    if "amc_2022" in dataset_names:
        # TODO: AMC 2022 is 43 problems, is that to be expected?
        datasets += _load_amc_dataset(2022, upsample_factor=16)
        remaining.discard("amc_2022")

    if "amc_2022_original" in dataset_names:
        datasets += _load_amc_dataset(2022)
        remaining.discard("amc_2022_original")

    if "amc_2023" in dataset_names:
        datasets += _load_amc_dataset(2023, upsample_factor=16)
        remaining.discard("amc_2023")

    if "amc_2023_original" in dataset_names:
        datasets += _load_amc_dataset(2023)
        remaining.discard("amc_2023_original")

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
        remaining.discard("open_reasoner_zero_57k")

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
        remaining.discard("open_reasoner_zero_extended_72k")

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
        remaining.discard("open_reasoner_zero_hard_13k")

    # Load any HuggingFace dataset (format: "org/dataset-name") not already handled above
    for dataset_name in list(remaining):
        if _is_hf_dataset_path(dataset_name) and dataset_name not in _PREDEFINED_HF_DATASETS:
            datasets += _load_hf_dataset(dataset_name)
            remaining.discard(dataset_name)

    # resolve any remaining names as local custom datasets.
    unresolved: List[str] = []
    for dataset_name in list(remaining):
        try:
            datasets += _load_custom_dataset(dataset_name)
            remaining.discard(dataset_name)
        except FileNotFoundError:
            unresolved.append(dataset_name)

    if unresolved:
        raise ValueError(f"Unknown dataset(s): {unresolved}")

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
