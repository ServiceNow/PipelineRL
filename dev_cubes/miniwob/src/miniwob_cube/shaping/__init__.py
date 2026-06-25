"""Potential-based local reward shaping for MiniWob++ HTML tasks.

This is deterministic shaping built from the visible DOM only — no LLM
judge, no hidden-state inspection. The pipeline is:

    instruction + initial DOM
              -> GoalSpec DSL
              -> deterministic predicate evaluators
              -> Phi(state)
              -> local_reward = gamma * Phi(next) - Phi(prev)

Because the local reward is a pure potential difference, GRPO/PPO with this
shaping is invariant to the choice of Phi at convergence (Ng et al., 1999),
and we never reward an action merely because a predicate was already
satisfied before the action.

Public API:

    goal = build_goal_spec(task_text, initial_html)
    prev_state = build_state(prev_html, env_metadata)
    next_state = build_state(next_html, env_metadata)
    info = compute_local_reward(goal, prev_state, next_state, tool_result, ...)
"""

from miniwob_cube.shaping.dom import (
    Element,
    State,
    ToolResult,
    build_state,
    parse_dom,
)
from miniwob_cube.shaping.goal_spec import (
    AllCheckboxesUnchecked,
    AnyCheckboxChecked,
    CheckboxChecked,
    CheckboxUnchecked,
    ClickButton,
    ClickElement,
    Constraint,
    DropdownEquals,
    GoalSpec,
    InputEquals,
    RadioSelected,
    TerminalGoal,
    TerminalSuccess,
    WrongCheckboxChecked,
)
from miniwob_cube.shaping.parser import build_goal_spec
from miniwob_cube.shaping.potential import (
    LocalRewardInfo,
    RewardWeights,
    compute_local_reward,
    phi,
)

__all__ = [
    "AllCheckboxesUnchecked",
    "AnyCheckboxChecked",
    "CheckboxChecked",
    "CheckboxUnchecked",
    "ClickButton",
    "ClickElement",
    "Constraint",
    "DropdownEquals",
    "Element",
    "GoalSpec",
    "InputEquals",
    "LocalRewardInfo",
    "RadioSelected",
    "RewardWeights",
    "State",
    "TerminalGoal",
    "TerminalSuccess",
    "ToolResult",
    "WrongCheckboxChecked",
    "build_goal_spec",
    "build_state",
    "compute_local_reward",
    "parse_dom",
    "phi",
]
