import random
from browsergym.workarena import ALL_WORKARENA_TASKS, ATOMIC_TASKS

def load_tasks(dataset_names: list[str], seeds: list[int] = [0, 1, 2, 3, 4]):
    all_shuffled_tasks = list(ALL_WORKARENA_TASKS)
    atomic_shuffled_tasks = list(ATOMIC_TASKS)
    random.seed(42)
    random.shuffle(all_shuffled_tasks)
    random.shuffle(atomic_shuffled_tasks)

    tasks = []
    for name in dataset_names:
        if name == "all":
            tasks.extend([
                {"dataset": task, "task": task, "seed": seed} for task in all_shuffled_tasks for seed in seeds
            ])
        elif name == "atomic":
            tasks.extend([
                {"dataset": task, "task": task, "seed": seed} for task in atomic_shuffled_tasks for seed in seeds
            ])
        else:
            raise ValueError(f"Invalid dataset name: {name}")
    return tasks