"""GoalSpec DSL — constraint and terminal-goal classes.

Constraints score the *state* (not the action). Each `is_satisfied(state)`
returns a bool indicating whether the predicate currently holds. The
potential function aggregates `is_satisfied` across `constraints` (positive
contribution) and `forbidden` (penalty contribution) — see `potential.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from miniwob_cube.shaping.dom import Element, State


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _label_matches(elem_label: str, target: str) -> bool:
    a, b = _norm(elem_label), _norm(target)
    if not a or not b:
        return False
    if a == b:
        return True
    # Allow substring match in either direction — MiniWob labels are short
    # tokens like "apple" while DOM labels can include surrounding whitespace.
    return b in a or a in b


# ===========================================================================
# Constraints
# ===========================================================================


class Constraint:
    """Base class. Subclasses implement `is_satisfied(state) -> bool`."""

    def is_satisfied(self, state: State) -> bool:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> str:
        return self.__class__.__name__


@dataclass(frozen=True)
class AllCheckboxesUnchecked(Constraint):
    scope: str = "all_visible_checkboxes"

    def is_satisfied(self, state: State) -> bool:
        boxes = state.checkboxes()
        if not boxes:
            return True
        return all(not b.checked for b in boxes)

    def describe(self) -> str:
        return f"AllCheckboxesUnchecked(scope={self.scope!r})"


@dataclass(frozen=True)
class AnyCheckboxChecked(Constraint):
    scope: str = "all_visible_checkboxes"

    def is_satisfied(self, state: State) -> bool:
        return any(b.checked for b in state.checkboxes())

    def describe(self) -> str:
        return f"AnyCheckboxChecked(scope={self.scope!r})"


@dataclass(frozen=True)
class CheckboxChecked(Constraint):
    label: str

    def is_satisfied(self, state: State) -> bool:
        for b in state.checkboxes():
            if _label_matches(b.best_label(), self.label) and b.checked:
                return True
        return False

    def describe(self) -> str:
        return f"CheckboxChecked(label={self.label!r})"


@dataclass(frozen=True)
class CheckboxUnchecked(Constraint):
    label: str

    def is_satisfied(self, state: State) -> bool:
        for b in state.checkboxes():
            if _label_matches(b.best_label(), self.label):
                return not b.checked
        # Element not present — treat as satisfied (cannot be wrong-checked).
        return True

    def describe(self) -> str:
        return f"CheckboxUnchecked(label={self.label!r})"


@dataclass(frozen=True)
class WrongCheckboxChecked(Constraint):
    """Violated when any checkbox NOT in `allowed_labels` is currently checked."""

    allowed_labels: tuple[str, ...] = ()

    def is_satisfied(self, state: State) -> bool:
        for b in state.checkboxes():
            if not b.checked:
                continue
            if not _label_in(b.best_label(), self.allowed_labels):
                return True
        return False

    def describe(self) -> str:
        return f"WrongCheckboxChecked(allowed={list(self.allowed_labels)!r})"


def _label_in(label: str, allowed: Iterable[str]) -> bool:
    for a in allowed:
        if _label_matches(label, a):
            return True
    return False


@dataclass(frozen=True)
class RadioSelected(Constraint):
    label: str

    def is_satisfied(self, state: State) -> bool:
        for r in state.radios():
            if _label_matches(r.best_label(), self.label) and r.checked:
                return True
        return False

    def describe(self) -> str:
        return f"RadioSelected(label={self.label!r})"


@dataclass(frozen=True)
class InputEquals(Constraint):
    """`target` is None to mean "any visible text input"."""

    target: str | None
    value: str

    def is_satisfied(self, state: State) -> bool:
        wanted = _norm(self.value)
        if not wanted:
            return False
        for inp in state.text_inputs():
            if self.target and not _label_matches(_input_handle(inp), self.target):
                continue
            actual = _norm(inp.value)
            if actual == wanted:
                return True
            # Allow surrounding text — useful when value is a substring.
            if wanted in actual:
                return True
        return False

    def describe(self) -> str:
        return f"InputEquals(target={self.target!r}, value={self.value!r})"


def _input_handle(elem: Element) -> str:
    return elem.label or elem.aria_label or elem.placeholder or elem.id or elem.name or ""


@dataclass(frozen=True)
class DropdownEquals(Constraint):
    target: str | None
    value: str

    def is_satisfied(self, state: State) -> bool:
        wanted = _norm(self.value)
        if not wanted:
            return False
        for sel in state.selects():
            if self.target and not _label_matches(_input_handle(sel), self.target):
                continue
            chosen = None
            for opt in sel.options:
                if opt.selected:
                    chosen = opt
                    break
            if chosen is None:
                # Fall back to the `value` attribute on the select.
                cur = _norm(sel.value)
                if cur and cur == wanted:
                    return True
                continue
            txt = _norm(chosen.text) or _norm(chosen.value)
            if txt == wanted or wanted in txt:
                return True
        return False

    def describe(self) -> str:
        return f"DropdownEquals(target={self.target!r}, value={self.value!r})"


@dataclass(frozen=True)
class InvalidToolCount(Constraint):
    """Satisfied iff the cumulative failed-tool count is zero.

    Mostly useful as a forbidden constraint — but exposed here so callers can
    introspect it. The actual failed-tool penalty is computed directly in
    `phi()` via the dedicated `w_error` weight.
    """

    max_failed: int = 0

    def is_satisfied(self, state: State) -> bool:
        return state.failed_tool_count <= self.max_failed


# ===========================================================================
# Terminal goals
# ===========================================================================


class TerminalGoal:
    """Marker base class for terminal goals. Evaluation is *not* by
    inspecting tool names — it is by inspecting `state.terminal_success`,
    which the environment sets from the actual JS reward signal."""

    def is_reached(self, state: State) -> bool:  # pragma: no cover
        return state.terminal_success

    def describe(self) -> str:
        return self.__class__.__name__


@dataclass(frozen=True)
class TerminalSuccess(TerminalGoal):
    def is_reached(self, state: State) -> bool:
        return state.terminal_success


@dataclass(frozen=True)
class ClickButton(TerminalGoal):
    text: str = "Submit"

    def is_reached(self, state: State) -> bool:
        return state.terminal_success

    def describe(self) -> str:
        return f"ClickButton(text={self.text!r})"


@dataclass(frozen=True)
class ClickElement(TerminalGoal):
    text_or_label: str

    def is_reached(self, state: State) -> bool:
        return state.terminal_success

    def describe(self) -> str:
        return f"ClickElement(text_or_label={self.text_or_label!r})"


# ===========================================================================
# GoalSpec
# ===========================================================================


@dataclass
class GoalSpec:
    constraints: list[Constraint] = field(default_factory=list)
    terminal: TerminalGoal | None = None
    forbidden: list[Constraint] = field(default_factory=list)
    confidence: float = 1.0

    def describe(self) -> dict:
        return {
            "constraints": [c.describe() for c in self.constraints],
            "terminal": self.terminal.describe() if self.terminal else None,
            "forbidden": [c.describe() for c in self.forbidden],
            "confidence": self.confidence,
        }


# Safe fallback when we cannot confidently parse the instruction. Only
# terminal-driven shaping remains — no invented dense predicates.
def safe_fallback() -> GoalSpec:
    return GoalSpec(
        constraints=[],
        terminal=TerminalSuccess(),
        forbidden=[],
        confidence=0.0,
    )
