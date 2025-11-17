import random
from browsergym.miniwob import ALL_MINIWOB_TASKS

DEBUG_SPLIT = [
    "miniwob.buy-ticket",
    "miniwob.bisect-angle",
    "miniwob.choose-list",
    "miniwob.click-checkboxes-large",
    "miniwob.click-checkboxes-soft",
]
EASY_SPLIT = [
    "miniwob.click-color",
    "miniwob.click-test-2",
    "miniwob.click-test-transfer",
    "miniwob.enter-password",
    "miniwob.focus-text-2",
    "miniwob.identify-shape",
    "miniwob.navigate-tree",
    "miniwob.phone-book",
    "miniwob.read-table",
    "miniwob.use-autocomplete",
    "miniwob.use-autocomplete",
    "miniwob.buy-ticket",
    "miniwob.click-checkboxes-soft",
    "miniwob.click-collapsible-2",
    "miniwob.click-collapsible-2-nodelay",
    "miniwob.click-collapsible-nodelay",
    "miniwob.click-dialog-2",
    "miniwob.click-tab-2",
    "miniwob.click-tab-2-medium",
    "miniwob.form-sequence-3",
    "miniwob.hot-cold",
    "miniwob.multi-orderings",
    "miniwob.tic-tac-toe",
    "miniwob.use-autocomplete-nodelay"
]
TRAIN_SPLIT = None
TEST_SPLIT = None


def load_tasks(dataset_names: list[str], train_split: float = 0.6, seeds: list[int] = [0, 1, 2, 3, 4]):
    # set global variables if needed
    global TRAIN_SPLIT, TEST_SPLIT
    if TRAIN_SPLIT is None or TEST_SPLIT is None:
        # Make a copy of tasks to avoid modifying the original
        all_tasks = list(ALL_MINIWOB_TASKS)
        # Use fixed seed for consistent shuffling
        rng = random.Random(1406)
        rng.shuffle(all_tasks)

        n_train_tasks = int(len(ALL_MINIWOB_TASKS) * train_split)
        TRAIN_SPLIT = [t.get_task_id() for t in ALL_MINIWOB_TASKS[:n_train_tasks]]
        TEST_SPLIT = [t.get_task_id() for t in ALL_MINIWOB_TASKS[n_train_tasks:]]

    tasks = []
    for name in dataset_names:
        if name == "debug":
            tasks.extend([
                {"dataset": "miniwob.debug", "task": task, "seed": 0} for task in DEBUG_SPLIT
            ])
        elif name == "easy":
            tasks.extend([
                {"dataset": "miniwob.easy", "task": task, "seed": 0} for task in EASY_SPLIT
            ])
        elif name == "train":
            tasks.extend([
                {"dataset": "miniwob.train", "task": task, "seed": seed}
                for task in TRAIN_SPLIT for seed in seeds
            ])
        elif name == "test":
            tasks.extend([
                {"dataset": "miniwob.test", "task": task, "seed": seed}
                for task in TEST_SPLIT for seed in seeds
            ])
    return tasks

