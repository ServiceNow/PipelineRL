import random
from browsergym.core.task import AbstractBrowserTask
from browsergym.workarena import ALL_WORKARENA_TASKS, workarena_tasks_all, workarena_tasks_l1, workarena_tasks_atomic

ALL_TASKS_DICT = {task.get_task_id(): task for task in ALL_WORKARENA_TASKS}


def load_tasks(dataset_names: list[str], seeds: list[int] = [0, 1, 2, 3, 4]):
    all_shuffled_task_ids = list(workarena_tasks_all)
    atomic_shuffled_task_ids = list(workarena_tasks_atomic)
    l1_shuffled_task_ids = list(workarena_tasks_l1)
    random.seed(42)
    random.shuffle(all_shuffled_task_ids)
    random.shuffle(atomic_shuffled_task_ids)
    random.shuffle(l1_shuffled_task_ids)
    tasks = []
    for name in dataset_names:
        if name == "all":
            tasks.extend(
                [
                    {"dataset": "workarena.all", "task": task_id, "seed": seed}
                    for task_id in all_shuffled_task_ids
                    for seed in seeds
                ]
            )
        elif name == "atomic":
            tasks.extend(
                [
                    {"dataset": "workarena.atomic", "task": task_id, "seed": seed}
                    for task_id in atomic_shuffled_task_ids
                    for seed in seeds
                ]
            )
        elif name == "l1":
            tasks.extend(
                [
                    {"dataset": "workarena.l1", "task": task_id, "seed": seed}
                    for task_id in l1_shuffled_task_ids
                    for seed in seeds
                ]
            )
        else:
            raise ValueError(f"Invalid dataset name: {name}")
    return tasks


def get_task_by_id(task_id: str) -> AbstractBrowserTask:
    if task_id in ALL_TASKS_DICT:
        return ALL_TASKS_DICT[task_id]
    else:
        raise ValueError(f"Task {task_id} not found")
