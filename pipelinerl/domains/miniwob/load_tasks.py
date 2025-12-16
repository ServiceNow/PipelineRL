import random
from browsergym.miniwob import ALL_MINIWOB_TASKS

# Main curated train/test splits for final experiments
TRAIN = [
    "miniwob.ascending-numbers",
    "miniwob.bisect-angle",
    "miniwob.book-flight",
    "miniwob.choose-date",
    "miniwob.choose-date-easy",
    "miniwob.choose-date-medium",
    "miniwob.choose-date-nodelay",
    "miniwob.choose-list",
    "miniwob.circle-center",
    "miniwob.click-button-sequence",
    "miniwob.click-checkboxes-soft",
    "miniwob.click-checkboxes-transfer",
    "miniwob.click-collapsible-2",
    "miniwob.click-collapsible-2-nodelay",
    "miniwob.click-collapsible-nodelay",
    "miniwob.click-color",
    "miniwob.click-dialog",
    "miniwob.click-dialog-2",
    "miniwob.click-link",
    "miniwob.click-menu",
    "miniwob.click-menu-2",
    "miniwob.click-scroll-list",
    "miniwob.click-shape",
    "miniwob.click-tab",
    "miniwob.click-tab-2",
    "miniwob.click-tab-2-hard",
    "miniwob.click-tab-2-medium",
    "miniwob.click-test",
    "miniwob.click-test-2",
    "miniwob.click-test-transfer",
    "miniwob.click-widget",
    "miniwob.copy-paste",
    "miniwob.copy-paste-2",
    "miniwob.count-shape",
    "miniwob.count-sides",
    "miniwob.daily-calendar",
    "miniwob.drag-box",
    "miniwob.drag-circle",
    "miniwob.drag-cube",
    "miniwob.drag-items",
    "miniwob.drag-items-grid",
    "miniwob.drag-shapes",
    "miniwob.drag-shapes-2",
    "miniwob.drag-sort-numbers",
    "miniwob.draw-circle",
    "miniwob.draw-line",
    "miniwob.email-inbox",
    "miniwob.email-inbox-delete",
    "miniwob.email-inbox-forward",
    "miniwob.email-inbox-forward-nl",
    "miniwob.email-inbox-forward-nl-turk",
    "miniwob.email-inbox-important",
    "miniwob.email-inbox-noscroll",
    "miniwob.email-inbox-reply",
    "miniwob.email-inbox-star-reply",
    "miniwob.enter-date",
    "miniwob.enter-text",
    "miniwob.enter-text-dynamic",
    "miniwob.enter-time",
    "miniwob.find-greatest",
    "miniwob.find-word",
    "miniwob.focus-text-2",
    "miniwob.form-sequence",
    "miniwob.form-sequence-2",
    "miniwob.generate-number",
    "miniwob.grid-coordinate",
    "miniwob.guess-number",
    "miniwob.highlight-text",
    "miniwob.hot-cold",
    "miniwob.identify-shape",
    "miniwob.login-user",
    "miniwob.login-user-popup",
    "miniwob.multi-layouts",
    "miniwob.multi-orderings",
    "miniwob.navigate-tree",
    "miniwob.odd-or-even",
    "miniwob.order-food",
    "miniwob.phone-book",
    "miniwob.read-table",
    "miniwob.read-table-2",
    "miniwob.resize-textarea",
    "miniwob.right-angle",
    "miniwob.scroll-text",
    "miniwob.scroll-text-2",
    "miniwob.search-engine",
    "miniwob.sign-agreement",
    "miniwob.simple-algebra",
    "miniwob.social-media",
    "miniwob.social-media-all",
    "miniwob.social-media-some",
    "miniwob.text-editor",
    "miniwob.text-transform",
    "miniwob.tic-tac-toe",
    "miniwob.use-autocomplete",
    "miniwob.use-autocomplete-nodelay",
    "miniwob.use-colorwheel",
    "miniwob.use-colorwheel-2",
    "miniwob.use-spinner",
    "miniwob.visual-addition",
]

TEST = [
    "miniwob.buy-ticket",
    "miniwob.click-button",
    "miniwob.click-option",
    "miniwob.click-pie-nodelay",
    "miniwob.drag-single-shape",
    "miniwob.email-inbox-nl-turk",
    "miniwob.enter-text-2",
    "miniwob.find-midpoint",
    "miniwob.focus-text",
    "miniwob.simple-arithmetic",
    "miniwob.stock-market",
    "miniwob.use-slider-2",
    "miniwob.click-checkboxes",
    "miniwob.click-checkboxes-large",
    "miniwob.click-collapsible",
    "miniwob.click-pie",
    "miniwob.click-shades",
    "miniwob.click-tab-2-easy",
    "miniwob.enter-password",
    "miniwob.form-sequence-3",
    "miniwob.highlight-text-2",
    "miniwob.unicode-test",
    "miniwob.use-slider",
]

# Easy tasks (smaller curated subset)
EASY_TRAIN = [
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
    "miniwob.focus-text",
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
    "miniwob.use-autocomplete-nodelay",
]
EASY_TEST = [
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
    "miniwob.use-autocomplete-nodelay",
]

# Debug tasks (tiny subset for quick testing)
DEBUG_TRAIN = [
    "miniwob.click-dialog",
    "miniwob.click-checkboxes",
]
DEBUG_TEST = [
    "miniwob.click-tab",
    "miniwob.click-menu",
]


# All available seeds for MiniWoB evaluation
MINIWOB_ALL_SEEDS = list(range(0, 10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


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
        sampling_seeds: List of seeds to sample from (e.g., [0,1,2,3,4,5,6,7,8,9])
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
    """Load MiniWoB tasks by split name.
    
    Supported splits:
    - train, test: main curated train/test splits for final experiments
    - easy_train, easy_test: smaller easy task subsets
    - debug_train, debug_test: tiny subsets for quick testing
    
    Args:
        dataset_names: List of split names to load
        train_seeds: Seeds for train/easy_train/debug_train splits 
                     (default: [3,4,5,6,7,8,9] - matches finetuning/benchmarks.py)
        test_seeds: Seeds for test/easy_test/debug_test splits 
                    (default: [0,1,2] - held out seeds for evaluation)
        total_eval_episodes: If provided, limits the total number of episodes for
                             test/eval splits by cycling through seeds. If None,
                             uses all combinations of tasks x seeds.
    """
    # MiniWoB seed configuration (from finetuning/core/benchmarks.py):
    # - All seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # - Train seeds: [3, 4, 5, 6, 7, 8, 9] (7 seeds for training)
    # - Held-out/test seeds: [0, 1, 2] (3 seeds for evaluation)
    if train_seeds is None:
        train_seeds = [3, 4, 5, 6, 7, 8, 9]
    if test_seeds is None:
        test_seeds = MINIWOB_ALL_SEEDS  # Use all seeds [0-9] for evaluation
    
    split_map = {
        "train": (TRAIN, train_seeds, False),
        "test": (TEST, test_seeds, True),
        "easy_train": (EASY_TRAIN, train_seeds, False),
        "easy_test": (EASY_TEST, test_seeds, True),
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

