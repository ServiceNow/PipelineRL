"""Generic per-step progress signals — no instruction parsing.

Two tiers:

  Tier 1 (state invariants)        — score state quality from DOM alone.
  Tier 2 (action quality)          — score whether the last action was a
                                     plausible operation on the current DOM.

None of these signals read the task instruction. They are all deterministic
functions of (state, optional last action, optional history counters), so
they generalize across every MiniWob task family without encoding
task-specific answer keys.
"""

from __future__ import annotations

from miniwob_cube.shaping.dom import ActionView, Element, State

# Actions that legitimately do NOT change the pruned DOM. Excluded from the
# "stuck" signal so the agent isn't penalized for necessary preparatory
# actions (focus before typing, hover to reveal, scroll, key navigation).
_NON_DOM_CHANGING_ACTIONS = frozenset(
    {"focus", "hover", "scroll", "press", "key_press", "wait", "noop", "go_back", "go_forward"}
)


# ---------------------------------------------------------------------------
# Tier 1 — state invariants
# ---------------------------------------------------------------------------


def form_completion_fraction(state: State) -> float:
    """Fraction of visible text inputs that have a non-empty value.

    Returns 0.0 when no text inputs exist (signal disabled on non-form tasks).
    """
    inputs = state.text_inputs()
    if not inputs:
        return 0.0
    filled = sum(1 for i in inputs if (i.value or "").strip())
    return filled / float(len(inputs))


def affordance_engagement_breadth(touched_bids: set[str], interactive_universe: int) -> float:
    """Fraction of the initial-state interactive elements whose state has
    been touched (checkbox toggled, input filled, selection changed).

    Caps at 1.0 so spamming clicks on the same element doesn't add credit.
    """
    if interactive_universe <= 0:
        return 0.0
    return min(1.0, len(touched_bids) / float(interactive_universe))


# ---------------------------------------------------------------------------
# Tier 2 — action quality
# ---------------------------------------------------------------------------


def detect_bad_target(prev_state: State, action: ActionView) -> bool:
    """True if the action targets a bid that is missing, disabled, or
    non-interactive in `prev_state`.

    Pure mistake-detection: doesn't know what the task is, only whether the
    target the agent picked is even a thing it could operate on.
    """
    if action is None or not action.bid:
        return False
    target = prev_state.by_bid().get(action.bid)
    if target is None:
        return True
    if target.disabled:
        return True
    # For click-like actions, demand interactivity. Tools that read state
    # (focus, hover) are exempt — they can target any element.
    if action.name in {"click", "dblclick", "select_option", "check", "uncheck"}:
        if not target.is_interactive():
            return True
    return False


def detect_stuck(
    prev_state: State,
    next_state: State,
    action: ActionView,
    failed: bool,
) -> bool:
    """True if the agent ran a should-change action but DOM didn't move.

    Excludes:
      - failed tool calls (already debited via failed_tool_count)
      - actions that legitimately leave the DOM unchanged (focus, hover, …)
      - empty / missing actions
    """
    if failed:
        return False
    if action is None or not action.name:
        return False
    if action.name in _NON_DOM_CHANGING_ACTIONS:
        return False
    return _dom_state_unchanged(prev_state, next_state)


def _dom_state_unchanged(prev: State, next_: State) -> bool:
    """DOM is "unchanged" if every (bid, state_signature) pair matches."""
    prev_sigs = {e.bid: e.state_signature() for e in prev.elements if e.bid}
    next_sigs = {e.bid: e.state_signature() for e in next_.elements if e.bid}
    return prev_sigs == next_sigs


# ---------------------------------------------------------------------------
# Touched-element diffing — used to maintain `touched_bids` for breadth.
# ---------------------------------------------------------------------------


def update_touched(prev_state: State, next_state: State, touched_bids: set[str]) -> None:
    """Mutate `touched_bids` in place with bids whose state changed."""
    prev_by_bid = {e.bid: e for e in prev_state.elements if e.bid}
    for e in next_state.elements:
        if not e.bid:
            continue
        prev_e = prev_by_bid.get(e.bid)
        if prev_e is None:
            continue
        if e.state_signature() != prev_e.state_signature():
            touched_bids.add(e.bid)


def count_interactive(elements: list[Element]) -> int:
    return sum(1 for e in elements if e.is_interactive() and e.visible)
