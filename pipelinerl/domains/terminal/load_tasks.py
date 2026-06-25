"""Load TMax terminal tasks from the published parquet into problem dicts.

``allenai/TMax-15K`` ships every task self-contained in a single parquet, so we
need no fixtures zip for text/bash tasks. Each row carries the Apptainer
``container_def`` (build recipe), the ``test_initial_state`` / ``test_final_state``
pytest verifiers, and the task ``description``. See ``TMAX_ENV_RECIPE.md``.

The actor calls ``load_problems(dataset_names, **dataset_loader_params)`` for both
the train and test splits, so ``dataset_names`` is the first positional argument
and selects the split (a name containing "test" -> the held-out split).
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_REPO = "allenai/TMax-15K"
DEFAULT_FILE = "data/train-00000-of-00001.parquet"


def load_problems(
    dataset_names: Optional[Sequence[str] | str] = None,
    repo_id: str = DEFAULT_REPO,
    parquet_file: str = DEFAULT_FILE,
    train_ratio: float = 0.95,
    tmax_domains: Optional[List[str]] = None,
    task_complexity_keep: Optional[List[str]] = None,
    limit: Optional[int] = None,
    seed: int = 0,
    subset: Optional[str] = None,
    **_ignored: Any,
) -> List[dict[str, Any]]:
    """Return TMax tasks as PipelineRL problem dicts.

    Args:
        dataset_names: list (or single name) for the split; a name containing
            "test" loads the held-out split, otherwise the train split.
        tmax_domains: optional subset of the 9 TMax domains (security, ...).
        task_complexity_keep: curriculum filter. The dataset's ``task_complexity``
            field is a long string starting with one of ``short`` / ``moderate`` /
            ``complex`` / ``intricate``; keep only rows whose leading bucket word is
            in this list. Used to remove trivial ``short`` tasks (all-pass ->
            zero-advantage filtered) and, in the cut regime, ``intricate`` tasks
            (all-fail within the turn cap). None = keep all buckets.
        limit: cap the number of returned problems (after the split).
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    if isinstance(dataset_names, str):
        names = [dataset_names]
    else:
        names = list(dataset_names or [])
    is_test = any("test" in n for n in names)
    split = "test" if is_test else "train"

    path = hf_hub_download(repo_id=repo_id, filename=parquet_file, repo_type="dataset")
    df = pd.read_parquet(path)
    logger.info("loaded %d TMax tasks from %s", len(df), repo_id)

    if tmax_domains:
        df = df[df["domain"].isin(list(tmax_domains))]

    if task_complexity_keep:
        keep = {c.strip().lower() for c in task_complexity_keep}
        before = len(df)
        bucket = df["task_complexity"].fillna("").str.split().str[0].str.lower()
        df = df[bucket.isin(keep)]
        logger.info("curriculum: kept %d/%d tasks with complexity bucket in %s", len(df), before, sorted(keep))

    # Deterministic shuffle then split, so train/test are stable across runs.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cut = int(len(df) * train_ratio)
    df = df.iloc[cut:] if split == "test" else df.iloc[:cut]

    if limit is not None:
        df = df.iloc[:limit]

    problems = []
    for i, row in df.reset_index(drop=True).iterrows():
        problems.append(
            {
                "id": int(i),
                "domain": "terminal",
                "dataset": f"tmax-15k@{row['domain']}",
                "tmax_domain": row["domain"],
                "task_id": row["task_id"],
                "task": row["description"],
                "container_def": row["container_def"],
                "test_initial_state": row["test_initial_state"],
                "test_final_state": row["test_final_state"],
                "task_complexity": row.get("task_complexity", ""),
                "command_complexity": row.get("command_complexity", ""),
            }
        )
    logger.info("returning %d problems (split=%s)", len(problems), split)
    return problems
