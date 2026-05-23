import base64

from terminalbench_cube.benchmark import TerminalBenchBenchmark


def load_tasks(
    dataset_names: list[str],
    train_split: float = 0.6,
    difficulty_filter: str | None = None,
    max_train_examples: int | None = None,  # cap number of train tasks (intended for debugging)
    max_test_examples: int | None = None,   # cap number of test tasks (intended for debugging)
) -> list[dict]:
    benchmark = TerminalBenchBenchmark(
        shuffle=True,
        shuffle_seed=42,
        difficulty_filter=difficulty_filter,
    )
    benchmark.install()
    benchmark._setup()

    all_ids = list(TerminalBenchBenchmark.task_metadata.keys())
    n_train = int(len(all_ids) * train_split)
    train_ids = all_ids[:n_train]
    test_ids = all_ids[n_train:]

    if max_train_examples is not None:
        train_ids = train_ids[:max_train_examples]
    if max_test_examples is not None:
        test_ids = test_ids[:max_test_examples]

    problems = []
    for name in dataset_names:
        if name == "train":
            problems.extend([_make_problem(tid, "terminalbench.train") for tid in train_ids])
        elif name == "test":
            problems.extend([_make_problem(tid, "terminalbench.test") for tid in test_ids])
    return problems


def _make_problem(task_id: str, dataset: str) -> dict:
    meta = TerminalBenchBenchmark.task_metadata[task_id]
    extra = meta.extra_info
    return {
        "task_id": task_id,
        "instruction": extra["instruction"],
        "archive_b64": base64.b64encode(extra["archive"]).decode("ascii"),
        "dataset": dataset,
        "difficulty": extra.get("difficulty", "unknown"),
        "max_test_timeout_sec": extra.get("max_test_timeout_sec", 900),
        "container_config": meta.container_config.model_dump(),
    }
