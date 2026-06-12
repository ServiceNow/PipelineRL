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
# SPLIT v2 (2026-06-12, learner = Qwen2.5-3B-Instruct): selected from the 3B's 49 mid-band
# tasks (0.10 <= p <= 0.90 at 8 reps; full 125-task screen, W&B group `task-screen` + r8-grid),
# stratified by task family with difficulty interleaving (mean baseline p: train .482 / eval .492).
# v1 (the 20/8 split for the 7B) is superseded — see the research THREAD Exp 0.7.
MINIWOB_T_EVAL = [
    "buy-ticket",
    "click-checkboxes",
    "click-dialog-2",
    "click-option",
    "click-tab-2-easy",
    "count-sides",
    "email-inbox-delete",
    "email-inbox-important",
    "enter-time",
    "find-word",
    "navigate-tree",
    "scroll-text",
    "simple-algebra",
    "use-autocomplete-nodelay",
    "use-colorwheel",
]
MINIWOB_T_TRAIN = [
    "bisect-angle",
    "choose-list",
    "click-button-sequence",
    "click-checkboxes-large",
    "click-checkboxes-soft",
    "click-collapsible",
    "click-collapsible-nodelay",
    "click-link",
    "click-scroll-list",
    "click-tab",
    "click-tab-2-medium",
    "click-widget",
    "copy-paste",
    "draw-line",
    "email-inbox",
    "enter-password",
    "enter-text",
    "enter-text-2",
    "enter-text-dynamic",
    "focus-text",
    "focus-text-2",
    "form-sequence-2",
    "generate-number",
    "identify-shape",
    "login-user",
    "multi-orderings",
    "phone-book",
    "read-table",
    "read-table-2",
    "simple-arithmetic",
    "text-transform",
    "tic-tac-toe",
    "use-autocomplete",
    "use-colorwheel-2",
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
