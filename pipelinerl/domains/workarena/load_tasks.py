"""
WorkArena L1 task definitions and loading utilities.

Splits:
- train/test: Main curated train/test splits for final experiments
- debug_train/debug_test: Tiny subsets for quick testing
"""

# ============================================================
# CURATED TASK SPLITS
# ============================================================

# Main curated train/test splits for final experiments
TRAIN = [
    "workarena.servicenow.all-menu",
    "workarena.servicenow.create-hardware-asset",
    "workarena.servicenow.create-incident",
    "workarena.servicenow.filter-asset-list",
    "workarena.servicenow.filter-change-request-list",
    "workarena.servicenow.filter-hardware-list",
    "workarena.servicenow.filter-service-catalog-item-list",
    "workarena.servicenow.filter-user-list",
    "workarena.servicenow.knowledge-base-search",
    "workarena.servicenow.order-apple-mac-book-pro15",
    "workarena.servicenow.order-development-laptop-p-c",
    "workarena.servicenow.order-ipad-mini",
    "workarena.servicenow.order-ipad-pro",
    "workarena.servicenow.order-sales-laptop",
    "workarena.servicenow.sort-change-request-list",
    "workarena.servicenow.sort-hardware-list",
    "workarena.servicenow.sort-incident-list",
    "workarena.servicenow.sort-service-catalog-item-list",
    "workarena.servicenow.sort-user-list",
    "workarena.servicenow.single-chart-value-retrieval",
    "workarena.servicenow.create-change-request",
    "workarena.servicenow.order-loaner-laptop",
    "workarena.servicenow.order-standard-laptop",
    "workarena.servicenow.create-problem",
]

TEST = [
    "workarena.servicenow.create-user",
    "workarena.servicenow.filter-incident-list",
    "workarena.servicenow.sort-asset-list",
    "workarena.servicenow.impersonation",
    "workarena.servicenow.order-apple-watch",
    "workarena.servicenow.order-developer-laptop",
    "workarena.servicenow.single-chart-min-max-retrieval",
]

# Debug tasks (tiny subset for quick testing)
DEBUG_TRAIN = [
    "workarena.servicenow.filter-incident-list",
    "workarena.servicenow.order-ipad-pro",
]

DEBUG_TEST = [
    "workarena.servicenow.create-hardware-asset",
    "workarena.servicenow.filter-service-catalog-item-list",
]

# All available seeds for WorkArena evaluation
WORKARENA_ALL_SEEDS = list(range(0, 1_000))  # [0, 1, 2, ..., 999]
WORKARENA_HELD_OUT_SEEDS = list(range(10))  # [0, 1, 2, ..., 9] - for evaluation
WORKARENA_TRAIN_SEEDS = list(set(range(0, 1_000)) - set(WORKARENA_HELD_OUT_SEEDS))  # [10, 11, ..., 999]


def make_task_list_fixed_seeds(
    task_list: list[str],
    sampling_seeds: list[int],
    total_episodes: int,
) -> list[dict]:
    """
    Create a list of task dicts for a given list of tasks and sampling seeds.
    There will be a total of total_episodes tasks created.

    This cycles through seeds round-robin style across all tasks:
    - seed 0 -> all tasks, seed 1 -> all tasks, etc.
    - Stops when total_episodes is reached
    - If total_episodes > len(tasks) * len(seeds), cycles back to seed 0

    Args:
        task_list: List of task names
        sampling_seeds: List of seeds to sample from (e.g., [0,1,2,...,999])
        total_episodes: Total number of episodes to create

    Returns:
        List of task dicts with 'dataset', 'task', and 'seed' keys
    """
    tasks = []
    episode_count = 0
    seed_id = 0

    while episode_count < total_episodes:
        seed = sampling_seeds[seed_id]
        for task in task_list:
            tasks.append({
                "dataset": task,
                "task": task,
                "seed": int(seed),
            })
            episode_count += 1
            if episode_count >= total_episodes:
                return tasks

        seed_id += 1
        if seed_id >= len(sampling_seeds):
            seed_id = 0

    return tasks


def load_tasks(
    dataset_names: list[str],
    train_seeds: list[int] = None,
    test_seeds: list[int] = None,
    total_eval_episodes: int = None,
):
    """Load WorkArena tasks by split name.
    
    Supported splits:
    - train, test: main curated train/test splits for final experiments
    - debug_train, debug_test: tiny subsets for quick testing
    
    Args:
        dataset_names: List of split names to load
        train_seeds: Seeds for train/debug_train splits 
                     (default: [10, 11, ..., 999] - 990 seeds for training)
        test_seeds: Seeds for test/debug_test splits 
                    (default: all 1000 seeds [0-999] for evaluation)
        total_eval_episodes: If provided, limits the total number of episodes for
                             test/eval splits by cycling through seeds. If None,
                             uses all combinations of tasks x seeds.
    """
    if train_seeds is None:
        train_seeds = WORKARENA_TRAIN_SEEDS  # [10, 11, ..., 999]
    if test_seeds is None:
        test_seeds = WORKARENA_ALL_SEEDS  # Use all seeds [0-999] for evaluation
    
    split_map = {
        "train": (TRAIN, train_seeds, False),
        "test": (TEST, test_seeds, True),
        "debug_train": (DEBUG_TRAIN, train_seeds, False),
        "debug_test": (DEBUG_TEST, test_seeds, True),
    }
    
    tasks = []
    for name in dataset_names:
        if name not in split_map:
            raise ValueError(f"Unknown split '{name}'. Valid: {list(split_map.keys())}")
        
        task_list, split_seeds, is_eval_split = split_map[name]
        
        # For eval splits with total_eval_episodes, use fixed seed cycling
        if is_eval_split and total_eval_episodes is not None:
            tasks.extend(make_task_list_fixed_seeds(
                task_list=task_list,
                sampling_seeds=split_seeds,
                total_episodes=total_eval_episodes,
            ))
        else:
            # Default behavior: all tasks x all seeds
            tasks.extend([
                {"dataset": task, "task": task, "seed": seed}
                for task in task_list
                for seed in split_seeds
            ])
    
    return tasks
