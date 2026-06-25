"""Unit tests for the potential-based local reward shaper.

Run with:
    uv run --extra cube pytest dev_cubes/miniwob/tests/test_shaping.py -v
"""

from __future__ import annotations

import pytest

from miniwob_cube.shaping import (
    AnyCheckboxChecked,
    AllCheckboxesUnchecked,
    CheckboxChecked,
    ClickButton,
    GoalSpec,
    RewardWeights,
    ToolResult,
    WrongCheckboxChecked,
    build_goal_spec,
    build_state,
    compute_local_reward,
)


# ---- Fixtures ----

INITIAL_HTML = """\
<head bid="1">
 <title bid="2">Click Checkboxes Task</title>
</head>
<div bid="13" id="wrap">
 <div bid="14" id="query"></div>
 <div bid="15" id="area">
  <div bid="16" id="boxes">
   <label bid="34">
    <input bid="35" id="ch0" type="checkbox" value="on"/>
    GjVJ8fQ
   </label>
   <label bid="37">
    <input bid="38" id="ch1" type="checkbox" value="on"/>
    8MoLcKO
   </label>
  </div>
  <button bid="18" class="secondary-action" id="subbtn" value="">Submit</button>
 </div>
</div>
"""


def _html_with_checked(checked_bids: set[str]) -> str:
    """Build a clone of INITIAL_HTML where listed checkbox bids are checked."""
    html = INITIAL_HTML
    for bid in checked_bids:
        html = html.replace(
            f'<input bid="{bid}" id="',
            f'<input checked bid="{bid}" id="',
            1,
        )
    return html


# ---- Goal spec parsing ----


def test_parse_select_nothing():
    g = build_goal_spec("Select nothing and click Submit.", INITIAL_HTML)
    assert g.confidence == 1.0
    assert len(g.constraints) == 1
    assert isinstance(g.constraints[0], AllCheckboxesUnchecked)
    assert isinstance(g.terminal, ClickButton)
    assert g.terminal.text.lower() == "submit"
    assert any(isinstance(c, AnyCheckboxChecked) for c in g.forbidden)


def test_parse_select_apple_and_banana():
    html = INITIAL_HTML.replace("GjVJ8fQ", "apple").replace("8MoLcKO", "banana")
    g = build_goal_spec("Select apple and banana and click Submit.", html)
    labels = sorted(c.label for c in g.constraints if isinstance(c, CheckboxChecked))
    assert labels == ["apple", "banana"]
    assert any(isinstance(c, WrongCheckboxChecked) for c in g.forbidden)


def test_parse_click_target():
    g = build_goal_spec("Click apple.", INITIAL_HTML)
    assert g.confidence > 0
    assert g.terminal is not None


def test_parse_unknown_fallback():
    # Instruction with no matching rule -> safe fallback.
    g = build_goal_spec("Do the thing that pleases the spirit.", INITIAL_HTML)
    assert g.confidence == 0.0
    assert g.constraints == []


# ---- Worked examples from the spec ----


def test_failed_clear_on_checkbox_negative_reward():
    """`Select nothing and click Submit` with failed clear on a checkbox.

    DOM unchanged, but failed_tool_count goes 0 -> 1. Expected reward = -0.1.
    """
    goal = build_goal_spec("Select nothing and click Submit.", INITIAL_HTML)
    prev = build_state(INITIAL_HTML, env_metadata={"failed_tool_count": 0})
    nxt = build_state(INITIAL_HTML, env_metadata={"failed_tool_count": 0})

    info = compute_local_reward(
        goal=goal,
        prev_state=prev,
        next_state=nxt,
        tool_result=ToolResult(failed=True, error_message="cannot be filled"),
        weights=RewardWeights(),
        gamma=1.0,
    )

    assert info.prev_phi == pytest.approx(0.3, abs=1e-6)
    assert info.next_phi == pytest.approx(0.2, abs=1e-6)
    assert info.reward == pytest.approx(-0.1, abs=1e-6)
    # Predicate was already satisfied before the action — no positive credit.
    assert info.constraint_score_before == 1.0
    assert info.constraint_score_after == 1.0


def test_successful_submit_terminal_positive_reward():
    """`Select nothing and click Submit` -> click Submit; env reports success.

    constraint_score stays at 1.0, terminal_score 0 -> 1, no failure.
    Expected reward = +0.7.
    """
    goal = build_goal_spec("Select nothing and click Submit.", INITIAL_HTML)
    prev = build_state(INITIAL_HTML, env_metadata={"failed_tool_count": 0})
    nxt = build_state(INITIAL_HTML, env_metadata={"failed_tool_count": 0, "terminal_success": True})

    info = compute_local_reward(
        goal=goal,
        prev_state=prev,
        next_state=nxt,
        tool_result=ToolResult(),
        weights=RewardWeights(),
        gamma=1.0,
    )

    assert info.prev_phi == pytest.approx(0.3, abs=1e-6)
    assert info.next_phi == pytest.approx(1.0, abs=1e-6)
    assert info.reward == pytest.approx(0.7, abs=1e-6)


def test_select_nothing_but_checked_a_box_is_negative():
    """A forbidden constraint becomes true (a checkbox got checked)."""
    goal = build_goal_spec("Select nothing and click Submit.", INITIAL_HTML)
    prev = build_state(INITIAL_HTML)
    nxt = build_state(_html_with_checked({"35"}))

    info = compute_local_reward(
        goal=goal,
        prev_state=prev,
        next_state=nxt,
        tool_result=ToolResult(),
        weights=RewardWeights(),
    )

    # Constraint regressed AND forbidden violated.
    assert info.constraint_score_before == 1.0
    assert info.constraint_score_after == 0.0
    assert info.violation_score_before == 0.0
    assert info.violation_score_after == 1.0
    assert info.reward < 0


def test_select_x_checking_x_is_positive():
    html = INITIAL_HTML.replace("GjVJ8fQ", "apple").replace("8MoLcKO", "banana")
    goal = build_goal_spec("Select apple and click Submit.", html)
    prev = build_state(html)
    nxt = build_state(_html_with_checked({"35"}).replace("GjVJ8fQ", "apple").replace("8MoLcKO", "banana"))

    info = compute_local_reward(
        goal=goal, prev_state=prev, next_state=nxt, tool_result=ToolResult()
    )
    assert info.reward > 0
    assert info.constraint_score_after > info.constraint_score_before


def test_select_x_checking_wrong_box_is_worse_than_checking_x():
    html = INITIAL_HTML.replace("GjVJ8fQ", "apple").replace("8MoLcKO", "banana")
    goal = build_goal_spec("Select apple and click Submit.", html)
    prev = build_state(html)
    # Wrong: checking banana when apple was requested.
    wrong = build_state(
        _html_with_checked({"38"}).replace("GjVJ8fQ", "apple").replace("8MoLcKO", "banana")
    )
    # Right: checking apple.
    right = build_state(
        _html_with_checked({"35"}).replace("GjVJ8fQ", "apple").replace("8MoLcKO", "banana")
    )
    r_wrong = compute_local_reward(goal=goal, prev_state=prev, next_state=wrong).reward
    r_right = compute_local_reward(goal=goal, prev_state=prev, next_state=right).reward
    assert r_wrong < r_right
    # Wrong action triggers a forbidden constraint.
    assert r_wrong <= 0 or r_wrong < r_right


def test_noop_action_is_zero_or_small_negative():
    goal = build_goal_spec("Select nothing and click Submit.", INITIAL_HTML)
    prev = build_state(INITIAL_HTML)
    nxt = build_state(INITIAL_HTML)
    # No failure, no noop counted by the env.
    info = compute_local_reward(
        goal=goal, prev_state=prev, next_state=nxt, tool_result=ToolResult()
    )
    assert info.reward == pytest.approx(0.0, abs=1e-9)
    # With the env reporting a noop:
    info2 = compute_local_reward(
        goal=goal,
        prev_state=prev,
        next_state=build_state(INITIAL_HTML),
        tool_result=ToolResult(is_noop=True),
    )
    assert info2.reward <= 0.0


def test_unparsed_task_falls_back_to_terminal_only():
    goal = build_goal_spec("Do an unknown task.", INITIAL_HTML)
    assert goal.confidence == 0.0
    prev = build_state(INITIAL_HTML)
    nxt = build_state(INITIAL_HTML, env_metadata={"terminal_success": True})
    info = compute_local_reward(goal=goal, prev_state=prev, next_state=nxt)
    # Only the terminal-success potential contributes (w_terminal = 0.7).
    assert info.prev_phi == pytest.approx(0.0, abs=1e-9)
    assert info.next_phi == pytest.approx(0.7, abs=1e-9)
    assert info.reward == pytest.approx(0.7, abs=1e-9)
