"""Potential function and local-reward computation.

This is potential-based reward shaping (Ng, Harada, Russell 1999) using
purely generic state and action-quality signals — no task instruction is
parsed.

    Phi(s, h) =
        w_terminal              * 1[terminal_success(s)]
      + w_form_completion       * form_completion_fraction(s)
      + w_affordance_breadth    * affordance_engagement_breadth(h)
      - w_error                 * failed_tool_count(h)
      - w_stuck                 * stuck_count(h)
      - w_bad_target            * bad_target_count(h)

    local_reward = gamma * Phi(next_state, next_h) - Phi(prev_state, prev_h)

`h` is an episode-level counter bundle (`HistoryCounters` in `shaper.py`)
that monotonically grows over the episode. `Phi` reads it; the shaper
mutates it between steps.

The action itself never enters `Phi`. Actions only influence the reward
through their effect on the next state (DOM diffs) and on the cumulative
counters (failed/stuck/bad-target).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from miniwob_cube.shaping.dom import State
from miniwob_cube.shaping.signals import (
    affordance_engagement_breadth,
    form_completion_fraction,
)

if TYPE_CHECKING:
    from miniwob_cube.shaping.shaper import HistoryCounters


@dataclass
class RewardConfig:
    """Config for the components of Phi. Defaults are conservative — every
    component contributes a small bounded amount, so no single signal can
    dominate.

    Set `enable_step_verifier_rewards=False` to disable shaping entirely
    without touching any other config (the task class consults this flag
    before invoking the shaper).
    """

    enable_step_verifier_rewards: bool = False
    # When True, the task appends a short qualitative feedback block
    # (derived from the same shaper signals) to the observation handed
    # back to the LLM. The text never includes numeric debits — only the
    # qualitative event ("your last action had no effect") — to avoid
    # turning shaping weights into a reward-hacking target.
    inject_step_feedback: bool = False
    terminal: float = 0.7
    form_completion: float = 0.15
    affordance_breadth: float = 0.15
    error: float = 0.1
    stuck: float = 0.05
    bad_target: float = 0.1
    gamma: float = 1.0


@dataclass
class LocalRewardInfo:
    """Debug bundle for one shaped step."""

    reward: float
    prev_phi: float
    next_phi: float
    terminal_score_before: float
    terminal_score_after: float
    form_completion_before: float
    form_completion_after: float
    affordance_breadth_before: float
    affordance_breadth_after: float
    failed_tool_before: int
    failed_tool_after: int
    stuck_before: int
    stuck_after: int
    bad_target_before: int
    bad_target_after: int
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "prev_phi": self.prev_phi,
            "next_phi": self.next_phi,
            "terminal_score_before": self.terminal_score_before,
            "terminal_score_after": self.terminal_score_after,
            "form_completion_before": self.form_completion_before,
            "form_completion_after": self.form_completion_after,
            "affordance_breadth_before": self.affordance_breadth_before,
            "affordance_breadth_after": self.affordance_breadth_after,
            "failed_tool_before": self.failed_tool_before,
            "failed_tool_after": self.failed_tool_after,
            "stuck_before": self.stuck_before,
            "stuck_after": self.stuck_after,
            "bad_target_before": self.bad_target_before,
            "bad_target_after": self.bad_target_after,
            "reasons": list(self.reasons),
        }


def _terminal_score(state: State) -> float:
    return 1.0 if state.terminal_success else 0.0


def format_step_feedback(info: "LocalRewardInfo") -> str | None:
    """Render a `LocalRewardInfo` into LLM-facing qualitative feedback.

    Only events that *changed this step* are mentioned. Numeric weights are
    intentionally omitted — the LLM should see what happened, not what it
    costs in the loss (otherwise it can game the shaping weights).
    """
    lines: list[str] = []
    if info.failed_tool_after > info.failed_tool_before:
        lines.append(
            f"Your last tool call failed "
            f"(failed calls so far: {info.failed_tool_after})."
        )
    if info.bad_target_after > info.bad_target_before:
        lines.append(
            "Your last action targeted an element that does not exist, is disabled, "
            "or is not interactive "
            f"(bad targets so far: {info.bad_target_after})."
        )
    if info.stuck_after > info.stuck_before:
        lines.append(
            "Your last action did not change the page "
            f"(steps with no effect so far: {info.stuck_after})."
        )
    if info.form_completion_after > info.form_completion_before:
        lines.append(
            f"Form completion: {info.form_completion_before:.0%} -> "
            f"{info.form_completion_after:.0%}."
        )
    if info.affordance_breadth_after > info.affordance_breadth_before:
        lines.append(
            f"Interactive elements touched: {info.affordance_breadth_before:.0%} -> "
            f"{info.affordance_breadth_after:.0%}."
        )
    if info.terminal_score_after > info.terminal_score_before:
        lines.append("Task completed successfully.")
    if not lines:
        return None
    return "[Step feedback]\n" + "\n".join(f"- {line}" for line in lines)


def phi(state: State, counters: "HistoryCounters", config: RewardConfig) -> float:
    """Potential Phi(state, counters).

    Pure function of state + cumulative counters + static weights — never
    the tool name or action arguments.
    """
    ts = _terminal_score(state)
    fc = form_completion_fraction(state)
    ab = affordance_engagement_breadth(counters.touched_bids, counters.interactive_universe)
    return (
        config.terminal * ts
        + config.form_completion * fc
        + config.affordance_breadth * ab
        - config.error * float(counters.failed_tool_count)
        - config.stuck * float(counters.stuck_count)
        - config.bad_target * float(counters.bad_target_count)
    )
