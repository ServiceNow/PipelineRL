"""Rule-based deterministic parser for MiniWob++ instructions.

We only emit a dense `GoalSpec` when we are confident. When we are not,
we return `safe_fallback()` so the only shaping signal is the terminal
reward (i.e., we do not invent uncertain dense rewards).

Supported families (high-precision):
  - checkbox tasks ("Select X and Y and click Submit", "Select nothing ...")
  - pure click tasks ("Click X", "Click the Submit button")
  - text input tasks ("Type X", "Enter X into the text field")
  - dropdown/select tasks ("Choose X", "Select X from the dropdown")

Any other instruction shape falls through to the safe fallback.
"""

from __future__ import annotations

import re
from typing import Iterable

from miniwob_cube.shaping.dom import parse_dom
from miniwob_cube.shaping.goal_spec import (
    AllCheckboxesUnchecked,
    AnyCheckboxChecked,
    CheckboxChecked,
    ClickButton,
    ClickElement,
    DropdownEquals,
    GoalSpec,
    InputEquals,
    TerminalSuccess,
    WrongCheckboxChecked,
    safe_fallback,
)


_NOTHING_RE = re.compile(r"\bselect\s+nothing\b", re.IGNORECASE)
_SELECT_AND_SUBMIT_RE = re.compile(
    r"\bselect\s+(?P<items>.+?)\s+and\s+click\s+(?P<button>[A-Za-z0-9_\- ]+)\b\.?$",
    re.IGNORECASE,
)
_SELECT_ONLY_RE = re.compile(r"\bselect\s+(?P<items>.+?)\b\.?$", re.IGNORECASE)
_CLICK_RE = re.compile(
    r"^\s*click(?:\s+the)?\s+(?P<target>[A-Za-z0-9_\-\. ]+?)(?:\s+button|\s+link)?\s*\.?\s*$",
    re.IGNORECASE,
)
_TYPE_RE = re.compile(
    r"^\s*(?:type|enter|fill(?:\s+in)?|input|write)\s+"
    r"(?:in\s+|into\s+)?"
    r"(?P<value>.+?)"
    r"(?:\s+(?:in|into|to)(?:\s+the)?\s+(?:text\s+)?(?:field|box|input|textbox))?\s*\.?\s*$",
    re.IGNORECASE,
)
_DROPDOWN_RE = re.compile(
    r"^\s*(?:choose|select|pick)\s+(?P<value>[A-Za-z0-9_\-\. ]+?)"
    r"(?:\s+from(?:\s+the)?\s+(?:dropdown|list|menu|select))\s*\.?\s*$",
    re.IGNORECASE,
)


def _split_items(items: str) -> list[str]:
    """Split "apple, banana and cherry" / "apple and banana" into tokens."""
    raw = re.split(r"\s*,\s*|\s+and\s+", items, flags=re.IGNORECASE)
    return [s.strip(" .") for s in raw if s.strip(" .")]


def _has_checkboxes(html: str | None) -> bool:
    if not html:
        return False
    for e in parse_dom(html):
        if e.is_checkbox():
            return True
    return False


def _has_select(html: str | None) -> bool:
    if not html:
        return False
    for e in parse_dom(html):
        if e.is_select():
            return True
    return False


def _has_text_input(html: str | None) -> bool:
    if not html:
        return False
    for e in parse_dom(html):
        if e.is_text_input():
            return True
    return False


def build_goal_spec(task_text: str, initial_html: str | None) -> GoalSpec:
    """Turn an instruction + initial DOM into a `GoalSpec`.

    Returns `safe_fallback()` when no rule matches confidently.
    """
    text = (task_text or "").strip()
    if not text:
        return safe_fallback()

    # ---- Checkbox tasks ----
    if _has_checkboxes(initial_html):
        m = _SELECT_AND_SUBMIT_RE.search(text)
        if m:
            items_raw = m.group("items").strip()
            button = m.group("button").strip()
            return _build_checkbox_select_goal(items_raw, button)
        if _NOTHING_RE.search(text) and re.search(r"\bclick\b", text, re.IGNORECASE):
            # "Select nothing and click X" — submit button best-guessed from text
            m2 = re.search(r"\bclick\s+(?P<button>[A-Za-z0-9_\- ]+)\b\.?", text, re.IGNORECASE)
            button = (m2.group("button").strip() if m2 else "Submit")
            return _build_checkbox_select_goal("nothing", button)
        # "Select X" with no submit clause — bare checkbox selection.
        m3 = _SELECT_ONLY_RE.search(text)
        if m3 and "click" not in text.lower():
            items_raw = m3.group("items").strip()
            return _build_checkbox_select_goal(items_raw, button=None)

    # ---- Dropdown ----
    if _has_select(initial_html):
        m = _DROPDOWN_RE.match(text)
        if m:
            value = m.group("value").strip()
            return GoalSpec(
                constraints=[DropdownEquals(target=None, value=value)],
                terminal=TerminalSuccess(),
                forbidden=[],
                confidence=0.9,
            )

    # ---- Pure click ----
    m = _CLICK_RE.match(text)
    if m:
        target = m.group("target").strip()
        # If the target word is literally "submit" or "ok", treat as button.
        if target.lower() in {"submit", "ok", "cancel"}:
            return GoalSpec(
                constraints=[],
                terminal=ClickButton(text=target),
                forbidden=[],
                confidence=0.9,
            )
        return GoalSpec(
            constraints=[],
            terminal=ClickElement(text_or_label=target),
            forbidden=[],
            confidence=0.9,
        )

    # ---- Text input ----
    if _has_text_input(initial_html):
        m = _TYPE_RE.match(text)
        if m:
            value = m.group("value").strip().strip("\"'")
            terminal = None
            if re.search(r"\bsubmit\b", text, re.IGNORECASE):
                terminal = ClickButton(text="Submit")
            return GoalSpec(
                constraints=[InputEquals(target=None, value=value)],
                terminal=terminal,
                forbidden=[],
                confidence=0.8,
            )

    return safe_fallback()


def _build_checkbox_select_goal(items_raw: str, button: str | None) -> GoalSpec:
    """Construct a GoalSpec for "Select <items> and click <button>"."""
    nothing = items_raw.strip().lower() == "nothing" or items_raw.strip() == ""
    terminal = ClickButton(text=button or "Submit") if button else None
    if nothing:
        return GoalSpec(
            constraints=[AllCheckboxesUnchecked()],
            terminal=terminal,
            forbidden=[AnyCheckboxChecked()],
            confidence=1.0,
        )
    items = _split_items(items_raw)
    constraints = [CheckboxChecked(label=item) for item in items]
    forbidden = [WrongCheckboxChecked(allowed_labels=tuple(items))]
    return GoalSpec(
        constraints=constraints,
        terminal=terminal,
        forbidden=forbidden,
        confidence=0.95,
    )


def _items_list(items: Iterable[str]) -> list[str]:
    return [s for s in items if s]
