"""Hint-bank + MiniWoB benchmark helpers for hint-conditioned RL (variant A).

Used by the ``conf/cube_miniwob_hint_*.yaml`` recipes via Hydra ``_target_``:

- :func:`make_miniwob_benchmark` builds a MiniWoB ``BenchmarkConfig`` restricted to a
  task list, with ``port=0`` (each Ray worker's benchmark auto-assigns a free port —
  requires cube-harness branch ``oo/jefhinter-workarena``).
- :func:`load_hint_map` loads the per-task hint map for one hint condition
  ``{good, none, distractor}`` from a bank JSON produced by
  ``cube-harness/scripts/build_hint_bank.py`` (mined by the frozen JefHinter miner).

The GRPO group never mixes hint conditions: each condition is its own cube entry
(distinct ``cube_id``), and a group = ``attempts`` rollouts of one (cube, task).

Research thread: ``plans_rl/research/hint_conditioned_rl/THREAD.md``.
"""

import json
from pathlib import Path

from cube_browser_tool.bgym_tool import BgymToolConfig
from miniwob_cube.benchmark import MiniWobBenchmarkConfig

# Frozen T_train / T_eval split of the 28-task MINIWOB_HARD set (2026-06-11).
# T_eval is held out from ALL training arms; the post-hoc lift eval (frozen JefHinter
# harness) mines its hints fresh on T_eval. Spans all four task categories.
MINIWOB_T_EVAL = [
    "click-checkboxes-soft",
    "navigate-tree",
    "enter-time",
    "search-engine",
    "count-shape",
    "guess-number",
    "copy-paste",
    "social-media",
]
MINIWOB_T_TRAIN = [
    "click-checkboxes-large",
    "click-checkboxes-transfer",
    "click-collapsible-2",
    "grid-coordinate",
    "use-autocomplete",
    "use-spinner",
    "use-slider",
    "choose-date-easy",
    "enter-date",
    "login-user",
    "simple-algebra",
    "identify-shape",
    "find-word",
    "tic-tac-toe",
    "enter-text-dynamic",
    "click-tab-2-hard",
    "email-inbox",
    "multi-layouts",
    "click-pie",
    "read-table",
]


_SPLITS = {"train": MINIWOB_T_TRAIN, "eval": MINIWOB_T_EVAL}


def make_miniwob_benchmark(
    split: str = "train",
    port: int = 0,
    use_html: bool = True,
    use_axtree: bool = False,
    use_screenshot: bool = False,
) -> MiniWobBenchmarkConfig:
    """MiniWoB benchmark config over the frozen ``train``/``eval`` task split; text-only obs.

    ``port=0`` auto-assigns a free port per benchmark instance so parallel Ray
    workers don't collide on :8000.
    """
    cfg = MiniWobBenchmarkConfig(
        tool_config=BgymToolConfig(use_html=use_html, use_axtree=use_axtree, use_screenshot=use_screenshot),
        port=port,
    )
    return cfg.subset_from_list(_SPLITS[split])


def load_hint_map(bank_path: str, condition: str, split: str = "train") -> dict[str, str]:
    """Per-task hint map for one hint condition, restricted to a split.

    ``none`` returns {} (the agent gets no hint). Tasks absent from the bank
    (never failed during mining) simply have no entry — Genny falls back to no hint.
    """
    if condition == "none":
        return {}
    if condition not in ("good", "distractor"):
        raise ValueError(f"unknown hint condition {condition!r} (expected good | none | distractor)")
    bank = json.loads(Path(bank_path).read_text())
    tasks = set(_SPLITS[split])
    return {t: h for t, h in bank[condition].items() if t in tasks}
