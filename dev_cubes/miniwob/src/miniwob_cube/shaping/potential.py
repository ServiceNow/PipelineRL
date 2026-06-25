"""Potential function and local-reward computation.

This implements potential-based reward shaping (Ng, Harada, Russell 1999):

    Phi(s) =
        w_constraints * constraint_score(s)
      + w_terminal   * terminal_score(s)
      - w_forbidden  * violation_score(s)
      - w_error      * failed_tool_count(s)
      - w_noop       * noop_count(s)

    local_reward = gamma * Phi(next_state) - Phi(prev_state)

Because the reward is a pure difference of a state-only potential, the
optimal policy under the shaped reward equals the optimal policy under the
unshaped (terminal-only) reward — no biased solutions. The action itself
never enters Phi; it only influences the reward through its effect on the
next state (and on the failed-tool/noop counters).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from miniwob_cube.shaping.dom import State, ToolResult
from miniwob_cube.shaping.goal_spec import Constraint, GoalSpec


@dataclass
class RewardWeights:
    """Weights for the components of Phi. Defaults match the spec."""

    constraints: float = 0.3
    terminal: float = 0.7
    forbidden: float = 0.3
    error: float = 0.1
    noop: float = 0.05


@dataclass
class LocalRewardInfo:
    """Debug bundle for one shaped step."""

    reward: float
    prev_phi: float
    next_phi: float
    constraint_score_before: float
    constraint_score_after: float
    terminal_score_before: float
    terminal_score_after: float
    violation_score_before: float
    violation_score_after: float
    failed_tool_before: int
    failed_tool_after: int
    goal_spec: GoalSpec
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "prev_phi": self.prev_phi,
            "next_phi": self.next_phi,
            "constraint_score_before": self.constraint_score_before,
            "constraint_score_after": self.constraint_score_after,
            "terminal_score_before": self.terminal_score_before,
            "terminal_score_after": self.terminal_score_after,
            "violation_score_before": self.violation_score_before,
            "violation_score_after": self.violation_score_after,
            "failed_tool_before": self.failed_tool_before,
            "failed_tool_after": self.failed_tool_after,
            "goal_spec": self.goal_spec.describe(),
            "reasons": list(self.reasons),
        }


# ---------------------------------------------------------------------------
# Component scores
# ---------------------------------------------------------------------------


def _fraction_satisfied(constraints: list[Constraint], state: State) -> float:
    if not constraints:
        return 0.0
    sat = sum(1 for c in constraints if c.is_satisfied(state))
    return sat / float(len(constraints))


def _fraction_violated(forbidden: list[Constraint], state: State) -> float:
    if not forbidden:
        return 0.0
    viol = sum(1 for c in forbidden if c.is_satisfied(state))
    return viol / float(len(forbidden))


def _terminal_score(goal: GoalSpec, state: State) -> float:
    if not goal.terminal:
        return 1.0 if state.terminal_success else 0.0
    return 1.0 if goal.terminal.is_reached(state) else 0.0


def phi(state: State, goal: GoalSpec, weights: RewardWeights) -> float:
    """Potential function Phi(state).

    NOTE: This is potential-based shaping — Phi must depend only on the
    state (and the static goal/weights), never on the tool name.
    """
    cs = _fraction_satisfied(goal.constraints, state)
    ts = _terminal_score(goal, state)
    vs = _fraction_violated(goal.forbidden, state)
    return (
        weights.constraints * cs
        + weights.terminal * ts
        - weights.forbidden * vs
        - weights.error * float(state.failed_tool_count)
        - weights.noop * float(state.noop_count)
    )


# ---------------------------------------------------------------------------
# Local reward
# ---------------------------------------------------------------------------


def compute_local_reward(
    *,
    goal: GoalSpec,
    prev_state: State,
    next_state: State,
    tool_result: ToolResult | None = None,
    weights: RewardWeights | None = None,
    gamma: float = 1.0,
) -> LocalRewardInfo:
    """Compute potential-difference local reward and a debug bundle.

    If `tool_result.failed` is set and `next_state.failed_tool_count` was
    not pre-incremented by the caller, we add one here so the failure
    actually shows up in the next-state potential. Same for `is_noop`.

    Mirroring this in the caller is fine too; this helper just makes it
    safe to forget.
    """
    w = weights or RewardWeights()

    if tool_result is not None:
        if tool_result.failed and next_state.failed_tool_count == prev_state.failed_tool_count:
            next_state.failed_tool_count = prev_state.failed_tool_count + 1
        if tool_result.is_noop and next_state.noop_count == prev_state.noop_count:
            next_state.noop_count = prev_state.noop_count + 1

    cs_b = _fraction_satisfied(goal.constraints, prev_state)
    cs_a = _fraction_satisfied(goal.constraints, next_state)
    ts_b = _terminal_score(goal, prev_state)
    ts_a = _terminal_score(goal, next_state)
    vs_b = _fraction_violated(goal.forbidden, prev_state)
    vs_a = _fraction_violated(goal.forbidden, next_state)

    prev_phi = phi(prev_state, goal, w)
    next_phi = phi(next_state, goal, w)
    reward = gamma * next_phi - prev_phi

    reasons: list[str] = []
    if cs_a > cs_b:
        reasons.append(f"constraint progress: {cs_b:.3f} -> {cs_a:.3f}")
    elif cs_a < cs_b:
        reasons.append(f"constraint regression: {cs_b:.3f} -> {cs_a:.3f}")
    if ts_a > ts_b:
        reasons.append("terminal success achieved")
    if vs_a > vs_b:
        reasons.append(f"forbidden violation: {vs_b:.3f} -> {vs_a:.3f}")
    elif vs_a < vs_b:
        reasons.append(f"forbidden recovered: {vs_b:.3f} -> {vs_a:.3f}")
    if next_state.failed_tool_count > prev_state.failed_tool_count:
        reasons.append(
            f"failed tool calls: {prev_state.failed_tool_count} -> {next_state.failed_tool_count}"
        )
    if next_state.noop_count > prev_state.noop_count:
        reasons.append(f"noop: {prev_state.noop_count} -> {next_state.noop_count}")
    if goal.confidence < 1.0:
        reasons.append(f"low-confidence goal (confidence={goal.confidence:.2f})")

    return LocalRewardInfo(
        reward=float(reward),
        prev_phi=float(prev_phi),
        next_phi=float(next_phi),
        constraint_score_before=float(cs_b),
        constraint_score_after=float(cs_a),
        terminal_score_before=float(ts_b),
        terminal_score_after=float(ts_a),
        violation_score_before=float(vs_b),
        violation_score_after=float(vs_a),
        failed_tool_before=int(prev_state.failed_tool_count),
        failed_tool_after=int(next_state.failed_tool_count),
        goal_spec=goal,
        reasons=reasons,
    )
