"""Potential-based local reward shaping for MiniWob++ HTML tasks.

The shaper is deliberately *instruction-blind*: it never parses the task
text into predicates or answer keys. All progress signals are generic
state-invariant (Tier 1) or action-quality (Tier 2) measurements that
generalize across every MiniWob task family.

Pipeline:

    DOM snapshot + last action + tool result + env success
              -> generic progress signals
              -> Phi(state, counters)
              -> local_reward = gamma * Phi(next) - Phi(prev)

Public API:

    shaper = EpisodeShaper(initial_html, weights=RewardWeights(), gamma=1.0)
    info = shaper.step(
        next_html=...,
        action=ActionView.from_cube_action(last_action),
        tool_result=ToolResult(failed=False),
        terminal_success=False,
    )
    info.reward
    info.to_dict()
"""

from miniwob_cube.shaping.dom import (
    ActionView,
    Element,
    State,
    ToolResult,
    build_state,
    parse_dom,
)
from miniwob_cube.shaping.potential import (
    LocalRewardInfo,
    RewardConfig,
    format_step_feedback,
    phi,
)
from miniwob_cube.shaping.shaper import EpisodeShaper, HistoryCounters
from miniwob_cube.shaping.signals import (
    affordance_engagement_breadth,
    count_interactive,
    detect_bad_target,
    detect_stuck,
    form_completion_fraction,
    update_touched,
)

__all__ = [
    "ActionView",
    "Element",
    "EpisodeShaper",
    "HistoryCounters",
    "LocalRewardInfo",
    "RewardConfig",
    "State",
    "ToolResult",
    "affordance_engagement_breadth",
    "build_state",
    "count_interactive",
    "detect_bad_target",
    "detect_stuck",
    "form_completion_fraction",
    "format_step_feedback",
    "parse_dom",
    "phi",
    "update_touched",
]
