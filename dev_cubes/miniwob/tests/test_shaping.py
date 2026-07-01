"""Unit tests for the generic potential-based local reward shaper.

The shaper is *instruction-blind* — every progress signal is computed from
DOM state and last-action quality. There is no GoalSpec, no parser, no
constraints/forbidden — those have been deleted.

Run with:
    uv run --extra cube python -m pytest dev_cubes/miniwob/tests/test_shaping.py -v
"""

from __future__ import annotations

import pytest

from miniwob_cube.shaping import (
    ActionView,
    EpisodeShaper,
    RewardWeights,
    ToolResult,
    affordance_engagement_breadth,
    build_state,
    count_interactive,
    detect_bad_target,
    detect_stuck,
    form_completion_fraction,
    format_step_feedback,
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

FORM_HTML = """\
<form bid="1">
 <input bid="2" id="user" type="text" value=""/>
 <input bid="3" id="pass" type="password" value=""/>
 <button bid="4">Login</button>
</form>
"""


def _html_with_checked(checked_bids: set[str], html: str = INITIAL_HTML) -> str:
    for bid in checked_bids:
        html = html.replace(f'<input bid="{bid}" id="', f'<input checked bid="{bid}" id="', 1)
    return html


def _html_with_value(bid: str, value: str, html: str = FORM_HTML) -> str:
    # Replace the existing `value=""` for the targeted bid (single occurrence).
    needle_prefix = f'<input bid="{bid}" '
    start = html.index(needle_prefix)
    end = html.index("/>", start)
    chunk = html[start:end]
    new_chunk = chunk.replace('value=""', f'value="{value}"', 1)
    return html[:start] + new_chunk + html[end:]


def _default_weights() -> RewardWeights:
    return RewardWeights()


# ---- Tier 1 signals ----


def test_form_completion_fraction_no_inputs():
    state = build_state(INITIAL_HTML)
    assert form_completion_fraction(state) == 0.0


def test_form_completion_fraction_partial():
    state = build_state(_html_with_value("2", "alice"))
    # 1 of 2 inputs filled.
    assert form_completion_fraction(state) == pytest.approx(0.5)


def test_form_completion_fraction_full():
    html = _html_with_value("2", "alice")
    html = _html_with_value("3", "secret", html=html)
    state = build_state(html)
    assert form_completion_fraction(state) == 1.0


def test_affordance_engagement_breadth_caps_at_one():
    assert affordance_engagement_breadth({"a", "b", "c"}, interactive_universe=2) == 1.0


def test_affordance_engagement_breadth_zero_universe():
    assert affordance_engagement_breadth({"a"}, interactive_universe=0) == 0.0


def test_count_interactive_on_initial_html():
    elements = build_state(INITIAL_HTML).elements
    # 2 checkboxes + 1 button = 3 interactive elements.
    assert count_interactive(elements) == 3


# ---- Tier 2 signals ----


def test_detect_bad_target_missing_bid():
    state = build_state(INITIAL_HTML)
    act = ActionView(name="click", bid="999")  # no such bid in DOM
    assert detect_bad_target(state, act) is True


def test_detect_bad_target_disabled_element():
    html = INITIAL_HTML.replace('<button bid="18"', '<button disabled bid="18"', 1)
    state = build_state(html)
    act = ActionView(name="click", bid="18")
    assert detect_bad_target(state, act) is True


def test_detect_bad_target_clicking_non_interactive():
    state = build_state(INITIAL_HTML)
    act = ActionView(name="click", bid="14")  # the empty <div id="query">
    assert detect_bad_target(state, act) is True


def test_detect_bad_target_valid_click_is_fine():
    state = build_state(INITIAL_HTML)
    act = ActionView(name="click", bid="18")  # the Submit button
    assert detect_bad_target(state, act) is False


def test_detect_stuck_for_dom_changing_action_with_no_change():
    prev = build_state(INITIAL_HTML)
    next_ = build_state(INITIAL_HTML)
    act = ActionView(name="click", bid="18")
    assert detect_stuck(prev, next_, act, failed=False) is True


def test_detect_stuck_excludes_non_dom_changing_actions():
    prev = build_state(INITIAL_HTML)
    next_ = build_state(INITIAL_HTML)
    for name in ["focus", "hover", "scroll", "press"]:
        act = ActionView(name=name, bid="18")
        assert detect_stuck(prev, next_, act, failed=False) is False, name


def test_detect_stuck_excludes_failed_actions():
    prev = build_state(INITIAL_HTML)
    next_ = build_state(INITIAL_HTML)
    act = ActionView(name="click", bid="18")
    # A failed call is debited via failed_tool_count, not via stuck_count.
    assert detect_stuck(prev, next_, act, failed=True) is False


def test_detect_stuck_does_not_trigger_when_dom_changed():
    prev = build_state(INITIAL_HTML)
    next_ = build_state(_html_with_checked({"35"}))
    act = ActionView(name="click", bid="35")
    assert detect_stuck(prev, next_, act, failed=False) is False


# ---- EpisodeShaper end-to-end ----


def test_failed_tool_call_produces_negative_reward():
    """`clear` on a checkbox: Playwright rejects, DOM unchanged.

    Φ_prev = 0 (nothing has happened yet)
    Φ_next = -w_error · 1 = -0.1
    reward = -0.1
    """
    shaper = EpisodeShaper(initial_html=INITIAL_HTML, weights=_default_weights())
    info = shaper.step(
        next_html=INITIAL_HTML,
        action=ActionView(name="clear", bid="35"),
        tool_result=ToolResult(failed=True, error_message="cannot be filled"),
        terminal_success=False,
    )
    assert info.prev_phi == pytest.approx(0.0, abs=1e-9)
    assert info.next_phi == pytest.approx(-0.1, abs=1e-9)
    assert info.reward == pytest.approx(-0.1, abs=1e-9)
    assert info.failed_tool_after == 1
    # No bad-target debit: bid 35 is a valid checkbox; the failure is on
    # the choice of tool, not the choice of target.
    assert info.bad_target_after == 0


def test_clicking_disabled_or_missing_bid_is_negative():
    shaper = EpisodeShaper(initial_html=INITIAL_HTML, weights=_default_weights())
    info = shaper.step(
        next_html=INITIAL_HTML,
        action=ActionView(name="click", bid="999"),
        tool_result=ToolResult(failed=False),
        terminal_success=False,
    )
    # Bad-target debit (w_bad_target=0.1) + stuck debit (DOM unchanged,
    # w_stuck=0.05). Total -0.15.
    assert info.bad_target_after == 1
    assert info.stuck_after == 1
    assert info.reward == pytest.approx(-0.15, abs=1e-9)


def test_successful_submit_yields_terminal_reward():
    """Click Submit, env reports success.

    Φ_prev = 0; Φ_next = w_terminal · 1 + (no penalties) = 0.7.
    Bid 18 is also "touched" because the env transitions to terminal,
    but in this simple case the DOM signature for bid 18 doesn't change —
    the affordance-breadth signal won't fire unless the DOM mutates.
    """
    shaper = EpisodeShaper(initial_html=INITIAL_HTML, weights=_default_weights())
    info = shaper.step(
        next_html=INITIAL_HTML,
        action=ActionView(name="click", bid="18"),
        tool_result=ToolResult(failed=False),
        terminal_success=True,
    )
    # DOM didn't actually change here (this is a synthetic test), so the
    # stuck signal does fire — but the terminal reward dwarfs it.
    # Expected: terminal +0.7, stuck -0.05 -> 0.65.
    assert info.terminal_score_after == 1.0
    assert info.reward == pytest.approx(0.65, abs=1e-9)


def test_successful_action_with_dom_change_is_positive():
    """Click a checkbox; DOM updates. No stuck, no bad target.

    Φ_next gains: w_affordance_breadth · 1/3 ≈ 0.05
    """
    shaper = EpisodeShaper(initial_html=INITIAL_HTML, weights=_default_weights())
    info = shaper.step(
        next_html=_html_with_checked({"35"}),
        action=ActionView(name="click", bid="35"),
        tool_result=ToolResult(failed=False),
        terminal_success=False,
    )
    assert info.affordance_breadth_after > info.affordance_breadth_before
    assert info.reward > 0
    # No mistake debits.
    assert info.stuck_after == 0
    assert info.bad_target_after == 0


def test_form_completion_signal_grows_when_input_filled():
    weights = _default_weights()
    shaper = EpisodeShaper(initial_html=FORM_HTML, weights=weights)
    info = shaper.step(
        next_html=_html_with_value("2", "alice"),
        action=ActionView(name="fill", bid="2"),
        tool_result=ToolResult(failed=False),
        terminal_success=False,
    )
    # form_completion went 0 -> 0.5; affordance breadth went 0 -> 1/3.
    assert info.form_completion_after == pytest.approx(0.5)
    assert info.reward > 0


def test_pure_noop_focus_action_is_zero_or_small():
    """`focus` doesn't change DOM. We exempt it from the stuck signal."""
    shaper = EpisodeShaper(initial_html=FORM_HTML, weights=_default_weights())
    info = shaper.step(
        next_html=FORM_HTML,
        action=ActionView(name="focus", bid="2"),
        tool_result=ToolResult(failed=False),
        terminal_success=False,
    )
    assert info.stuck_after == 0
    assert info.bad_target_after == 0
    assert info.reward == pytest.approx(0.0, abs=1e-9)


def test_disabled_shaper_via_phi_components_zeroed():
    """All weights = 0 -> all rewards = 0 regardless of state.

    This is the smoke test for "disable shaping entirely from config."
    """
    zero = RewardWeights(
        enable_step_verifier_rewards=False,
        terminal=0.0,
        form_completion=0.0,
        affordance_breadth=0.0,
        error=0.0,
        stuck=0.0,
        bad_target=0.0,
        gamma=1.0,
    )
    shaper = EpisodeShaper(initial_html=INITIAL_HTML, weights=zero)
    info = shaper.step(
        next_html=INITIAL_HTML,
        action=ActionView(name="clear", bid="35"),
        tool_result=ToolResult(failed=True),
        terminal_success=False,
    )
    assert info.reward == 0.0


def test_format_step_feedback_describes_failure():
    shaper = EpisodeShaper(initial_html=INITIAL_HTML, weights=_default_weights())
    info = shaper.step(
        next_html=INITIAL_HTML,
        action=ActionView(name="clear", bid="35"),
        tool_result=ToolResult(failed=True),
        terminal_success=False,
    )
    text = format_step_feedback(info)
    assert text is not None
    assert "[Step feedback]" in text
    assert "failed" in text.lower()
    # No numeric reward debits should leak into the text.
    assert "-0.1" not in text
    assert "weight" not in text.lower()


def test_format_step_feedback_describes_progress():
    shaper = EpisodeShaper(initial_html=INITIAL_HTML, weights=_default_weights())
    info = shaper.step(
        next_html=_html_with_checked({"35"}),
        action=ActionView(name="click", bid="35"),
        tool_result=ToolResult(failed=False),
        terminal_success=False,
    )
    text = format_step_feedback(info)
    assert text is not None
    assert "Interactive elements touched" in text or "interactive elements" in text.lower()


def test_format_step_feedback_none_when_nothing_changed():
    shaper = EpisodeShaper(initial_html=INITIAL_HTML, weights=_default_weights())
    info = shaper.step(
        next_html=INITIAL_HTML,
        action=ActionView(name="focus", bid="35"),
        tool_result=ToolResult(failed=False),
        terminal_success=False,
    )
    # focus on a non-changing element triggers no debit and no progress —
    # feedback should be None (nothing to say).
    assert format_step_feedback(info) is None


def test_counters_monotone_across_multiple_steps():
    shaper = EpisodeShaper(initial_html=INITIAL_HTML, weights=_default_weights())
    info1 = shaper.step(
        next_html=INITIAL_HTML,
        action=ActionView(name="clear", bid="35"),
        tool_result=ToolResult(failed=True),
        terminal_success=False,
    )
    info2 = shaper.step(
        next_html=INITIAL_HTML,
        action=ActionView(name="click", bid="999"),
        tool_result=ToolResult(failed=False),
        terminal_success=False,
    )
    # Counters must only increase.
    assert info2.failed_tool_after >= info1.failed_tool_after
    assert info2.bad_target_after >= info1.bad_target_after
    assert info2.stuck_after >= info1.stuck_after
