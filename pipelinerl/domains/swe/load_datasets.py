# Supported datasets
# ──────────────────────────────────────────────────────────────────────────────
# Ready to use (have gold_file_contents pre-extracted):
#   SWE-bench/SWE-smith-py        local preprocessed disk dataset or Hub ID
#   SWE-bench/SWE-smith-java      "
#   SWE-bench/SWE-smith-rs        "
#   SWE-bench/SWE-smith-go        "
#
# Require preprocessing first (clone repos, extract file contents at base_commit):
#   princeton-nlp/SWE-bench
#   princeton-nlp/SWE-bench_Lite
#   princeton-nlp/SWE-bench_Verified
#   SWE-bench/SWE-Pro (if/when released publicly)
#
#   Run: python -m pipelinerl.domains.swe.swe_preprocessor --config-name=swe/preprocess
# ──────────────────────────────────────────────────────────────────────────────

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

from datasets import load_dataset, load_from_disk

logger = logging.getLogger(__name__)


def _parse_file_contents(raw: Any) -> Dict[str, str]:
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    return {}


def _load_single_dataset(path: str) -> List[Dict]:
    """Load a dataset from a local disk path or a HuggingFace Hub ID.

    Local path:   /path/to/ds_train
    Hub ID:       SWE-bench/SWE-smith-py          (all splits concatenated)
    Hub ID+split: SWE-bench/SWE-smith-py:train
    """
    if os.path.exists(path):
        logger.info("Loading from disk: %s", path)
        dataset = load_from_disk(path)
    else:
        # Hub ID, optionally with ":split" suffix
        if ":" in path:
            hub_id, split = path.rsplit(":", 1)
        else:
            hub_id, split = path, None

        logger.info("Loading from HuggingFace Hub: %s (split=%s)", hub_id, split or "all")
        loaded = load_dataset(hub_id, split=split)

        if split is None:
            # DatasetDict — concatenate all splits
            from datasets import concatenate_datasets
            dataset = concatenate_datasets(list(loaded.values()))
        else:
            dataset = loaded

    logger.info("Loaded %d rows from %s", len(dataset), path)

    samples = []
    for row in dataset:
        item = dict(row)
        try:
            file_contents = _parse_file_contents(item.get("gold_file_contents", "{}"))
            if not file_contents:
                continue
            samples.append({
                "id": item.get("id", "") or item.get("instance_id", "") or item.get("issue_id", ""),
                "dataset": item.get("dataset", "") or path,
                "repo": item.get("repo", ""),
                "base_commit": item.get("base_commit", ""),
                "problem_statement": item.get("problem_statement", ""),
                "patch": item.get("patch", ""),
                "file_contents": file_contents,
            })
        except Exception as e:
            logger.warning("Skipping malformed item: %s", e)

    return samples


def load_local_swe_dataset(
    dataset_paths: List[str],
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Load one or more SWE-style datasets from disk and return a combined, shuffled list.

    Args:
        dataset_paths: Passed via cfg.train_dataset_names / cfg.test_dataset_names.
                       Each entry is a filesystem path to a HuggingFace disk dataset.
                       Add multiple paths to mix datasets (e.g. swe-smith + swe-bench).
        seed: Random seed for shuffling (inherit from cfg.seed via dataset_loader_params).
        max_samples: Optional cap on the total number of returned samples.
    """
    if not dataset_paths:
        logger.error("No dataset paths provided")
        return []

    all_samples: List[Dict] = []
    for path in dataset_paths:
        try:
            all_samples.extend(_load_single_dataset(path))
        except Exception as e:
            logger.error("Failed to load dataset from %s: %s", path, e, exc_info=True)

    random.Random(seed).shuffle(all_samples)
    logger.info("Shuffled %d samples (seed=%d)", len(all_samples), seed)

    if max_samples and len(all_samples) > max_samples:
        all_samples = all_samples[:max_samples]
        logger.info("Trimmed to max_samples=%d", max_samples)

    logger.info("Returning %d samples total", len(all_samples))
    return all_samples
