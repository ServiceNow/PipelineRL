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
UIC_TRAIN_SPLIT = [
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
UIC_TEST_SPLIT = [
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
        elif name == "uic_train":
            tasks.extend([
                {"dataset": "miniwob.uic_train", "task": task, "seed": seed}
                for task in UIC_TRAIN_SPLIT for seed in range(3,10)  # seeds 0-2 are used for held out goals in Mass setup
            ])
        elif name == "uic_train_heldout_goals":
            tasks.extend([
                {"dataset": "miniwob.uic_train_heldout_goals", "task": task, "seed": seed}
                for task in UIC_TRAIN_SPLIT for seed in range(3)  # seeds 0-2 are used for held out goals in Mass setup
            ])
        elif name == "uic_test":
            tasks.extend([
                {"dataset": "miniwob.uic_test", "task": task, "seed": seed}
                for task in UIC_TEST_SPLIT for seed in range(10)
            ])
    return tasks

