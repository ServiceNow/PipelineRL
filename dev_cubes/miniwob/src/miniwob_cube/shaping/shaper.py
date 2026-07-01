"""Episode-level shaper.

Owns the cumulative counters that turn per-step events (failed tool calls,
DOM-unchanged ticks, bad targets, touched affordances) into the
state-and-history bundle `Phi` reads.

Typical use:

    shaper = EpisodeShaper(initial_html, config=RewardConfig())
    # ... per step:
    info = shaper.step(
        next_html=curr_html,
        action=ActionView.from_cube_action(last_action),
        tool_result=ToolResult(failed=is_failure),
        terminal_success=bool(env_success),
    )
    info.reward                # local reward for this step
    info.to_dict()              # full debug bundle
"""

from __future__ import annotations

from dataclasses import dataclass, field

from miniwob_cube.shaping.dom import (
    ActionView,
    State,
    ToolResult,
    build_state,
)
from miniwob_cube.shaping.potential import (
    LocalRewardInfo,
    RewardConfig,
    phi,
)
from miniwob_cube.shaping.signals import (
    affordance_engagement_breadth,
    count_interactive,
    detect_bad_target,
    detect_stuck,
    form_completion_fraction,
    update_touched,
)


@dataclass
class HistoryCounters:
    """Episode-cumulative counters consumed by Phi."""

    failed_tool_count: int = 0
    stuck_count: int = 0
    bad_target_count: int = 0
    touched_bids: set[str] = field(default_factory=set)
    # Denominator for affordance breadth — set once from the initial DOM.
    interactive_universe: int = 0

    def snapshot(self) -> "HistoryCounters":
        return HistoryCounters(
            failed_tool_count=self.failed_tool_count,
            stuck_count=self.stuck_count,
            bad_target_count=self.bad_target_count,
            touched_bids=set(self.touched_bids),
            interactive_universe=self.interactive_universe,
        )


class EpisodeShaper:
    """Holds prev-state + cumulative counters, emits one LocalRewardInfo per step."""

    def __init__(
        self,
        initial_html: str | None,
        config: RewardConfig | None = None,
    ) -> None:
        self.config = config or RewardConfig()
        self.gamma = self.config.gamma
        self.prev_state: State = build_state(initial_html, terminal_success=False)
        self.counters: HistoryCounters = HistoryCounters(
            interactive_universe=count_interactive(self.prev_state.elements),
        )

    def step(
        self,
        *,
        next_html: str | None,
        action: ActionView | None,
        tool_result: ToolResult,
        terminal_success: bool,
    ) -> LocalRewardInfo:
        next_state = build_state(next_html, terminal_success=bool(terminal_success))
        act = action or ActionView()

        # Snapshot counters BEFORE mutation — Phi(prev) uses the pre-step
        # counter values; Phi(next) uses post-step values.
        prev_counters = self.counters.snapshot()

        # Update cumulative counters from this transition.
        if tool_result.failed:
            self.counters.failed_tool_count += 1
        if detect_bad_target(self.prev_state, act):
            self.counters.bad_target_count += 1
        if detect_stuck(self.prev_state, next_state, act, tool_result.failed):
            self.counters.stuck_count += 1
        update_touched(self.prev_state, next_state, self.counters.touched_bids)

        prev_phi = phi(self.prev_state, prev_counters, self.config)
        next_phi = phi(next_state, self.counters, self.config)
        reward = self.gamma * next_phi - prev_phi

        info = LocalRewardInfo(
            reward=float(reward),
            prev_phi=float(prev_phi),
            next_phi=float(next_phi),
            terminal_score_before=1.0 if self.prev_state.terminal_success else 0.0,
            terminal_score_after=1.0 if next_state.terminal_success else 0.0,
            form_completion_before=form_completion_fraction(self.prev_state),
            form_completion_after=form_completion_fraction(next_state),
            affordance_breadth_before=affordance_engagement_breadth(
                prev_counters.touched_bids, prev_counters.interactive_universe
            ),
            affordance_breadth_after=affordance_engagement_breadth(
                self.counters.touched_bids, self.counters.interactive_universe
            ),
            failed_tool_before=prev_counters.failed_tool_count,
            failed_tool_after=self.counters.failed_tool_count,
            stuck_before=prev_counters.stuck_count,
            stuck_after=self.counters.stuck_count,
            bad_target_before=prev_counters.bad_target_count,
            bad_target_after=self.counters.bad_target_count,
            reasons=_reasons(prev_counters, self.counters, self.prev_state, next_state),
        )

        self.prev_state = next_state
        return info


def _reasons(
    prev_c: HistoryCounters,
    next_c: HistoryCounters,
    prev_s: State,
    next_s: State,
) -> list[str]:
    out: list[str] = []
    if next_s.terminal_success and not prev_s.terminal_success:
        out.append("terminal success achieved")
    if next_c.failed_tool_count > prev_c.failed_tool_count:
        out.append(
            f"failed tool calls: {prev_c.failed_tool_count} -> {next_c.failed_tool_count}"
        )
    if next_c.stuck_count > prev_c.stuck_count:
        out.append(f"stuck step (DOM unchanged after action): {prev_c.stuck_count} -> {next_c.stuck_count}")
    if next_c.bad_target_count > prev_c.bad_target_count:
        out.append(
            f"bad-target action (missing/disabled/non-interactive bid): "
            f"{prev_c.bad_target_count} -> {next_c.bad_target_count}"
        )
    new_touched = len(next_c.touched_bids) - len(prev_c.touched_bids)
    if new_touched > 0:
        out.append(f"affordance breadth grew by {new_touched}")
    fc_before = form_completion_fraction(prev_s)
    fc_after = form_completion_fraction(next_s)
    if fc_after > fc_before:
        out.append(f"form completion: {fc_before:.2f} -> {fc_after:.2f}")
    elif fc_after < fc_before:
        out.append(f"form completion regressed: {fc_before:.2f} -> {fc_after:.2f}")
    return out
