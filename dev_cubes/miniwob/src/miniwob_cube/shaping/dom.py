"""DOM abstraction for MiniWob++ pruned HTML.

Parses pruned HTML into a flat list of `Element`s tagged with browsergym
`bid`s. The shaper only needs visible affordances (checkboxes, radios, text
inputs, dropdowns, buttons) and their basic state — `checked`, `selected`,
`value`, `disabled` — to compute generic progress signals.

Label inference for checkboxes/radios walks the surrounding `<label>`
element because MiniWob's pruned HTML rarely emits `<label for="...">`
linking.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

logger = logging.getLogger(__name__)


_TEXT_INPUT_TYPES = frozenset(
    {"", "text", "search", "email", "url", "password", "tel", "number", "date", "time"}
)


@dataclass
class Element:
    bid: str | None = None
    tag: str = ""
    type: str | None = None
    id: str | None = None
    text: str = ""
    label: str | None = None
    value: str | None = None
    checked: bool = False
    selected: bool = False
    disabled: bool = False
    visible: bool = True
    name: str | None = None
    aria_label: str | None = None
    title: str | None = None
    placeholder: str | None = None
    options: list["Element"] = field(default_factory=list)

    def is_checkbox(self) -> bool:
        return self.tag == "input" and (self.type or "").lower() == "checkbox"

    def is_radio(self) -> bool:
        return self.tag == "input" and (self.type or "").lower() == "radio"

    def is_text_input(self) -> bool:
        if self.tag == "textarea":
            return True
        if self.tag != "input":
            return False
        return (self.type or "").lower() in _TEXT_INPUT_TYPES

    def is_select(self) -> bool:
        return self.tag == "select"

    def is_button(self) -> bool:
        if self.tag == "button":
            return True
        if self.tag == "input" and (self.type or "").lower() in {"submit", "button", "reset"}:
            return True
        return False

    def is_interactive(self) -> bool:
        return (
            self.is_checkbox()
            or self.is_radio()
            or self.is_text_input()
            or self.is_select()
            or self.is_button()
            or self.tag in {"a", "option"}
        )

    def state_signature(self) -> tuple:
        """Stable tuple summarizing the element's interactive state."""
        return (self.checked, self.selected, (self.value or ""))


@dataclass
class State:
    """Snapshot of one observation. Holds DOM elements only.

    Episode-level counters live on `HistoryCounters` in `shaper.py`; Φ takes
    both so it remains a pure function of (state, counters, weights).
    """

    elements: list[Element] = field(default_factory=list)
    terminal_success: bool = False

    def checkboxes(self) -> list[Element]:
        return [e for e in self.elements if e.is_checkbox() and e.visible]

    def radios(self) -> list[Element]:
        return [e for e in self.elements if e.is_radio() and e.visible]

    def text_inputs(self) -> list[Element]:
        return [e for e in self.elements if e.is_text_input() and e.visible]

    def selects(self) -> list[Element]:
        return [e for e in self.elements if e.is_select() and e.visible]

    def buttons(self) -> list[Element]:
        return [e for e in self.elements if e.is_button() and e.visible]

    def interactive(self) -> list[Element]:
        return [e for e in self.elements if e.is_interactive() and e.visible]

    def by_bid(self) -> dict[str, Element]:
        return {e.bid: e for e in self.elements if e.bid}


@dataclass
class ToolResult:
    """Tool-call outcome for one step.

    `failed=True` means the tool reported an error (e.g. Playwright rejected
    the action). The shaper uses it as the ground-truth signal for
    `failed_tool_count`.
    """

    failed: bool = False
    error_message: str | None = None


@dataclass
class ActionView:
    """Lightweight view of the agent's last action — keeps the shaper
    testable without depending on cube.core.Action."""

    name: str = ""
    bid: str | None = None

    @classmethod
    def from_cube_action(cls, action) -> "ActionView":
        if action is None:
            return cls()
        name = getattr(action, "name", "") or ""
        args = getattr(action, "arguments", None) or {}
        return cls(name=str(name), bid=args.get("bid") if isinstance(args, dict) else None)


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


class _DomParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: list[Element] = []
        self._stack: list[tuple[Element, dict[str, str], list[str]]] = []
        self._by_id: dict[str, Element] = {}
        self._label_for: list[tuple[Element, str]] = []

    def _make(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> tuple[Element, dict[str, str]]:
        attrs = {k.lower(): (v or "") for k, v in attrs_list}
        type_attr = attrs.get("type", "")
        elem = Element(
            bid=attrs.get("bid") or attrs.get("data-bid") or None,
            tag=tag.lower(),
            type=type_attr.lower() if type_attr else None,
            id=attrs.get("id") or None,
            value=attrs.get("value") if "value" in attrs else None,
            checked=("checked" in attrs),
            selected=("selected" in attrs),
            disabled=("disabled" in attrs),
            visible=True,
            name=attrs.get("name") or None,
            aria_label=attrs.get("aria-label") or None,
            title=attrs.get("title") or None,
            placeholder=attrs.get("placeholder") or None,
        )
        if elem.id:
            self._by_id[elem.id] = elem
        return elem, attrs

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        elem, raw = self._make(tag, attrs)
        self.elements.append(elem)
        self._stack.append((elem, raw, []))
        if elem.tag == "label" and "for" in raw and raw["for"]:
            self._label_for.append((elem, raw["for"]))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        elem, raw = self._make(tag, attrs)
        self.elements.append(elem)
        if elem.tag == "label" and "for" in raw and raw["for"]:
            self._label_for.append((elem, raw["for"]))

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if not self._stack:
            return
        for i in range(len(self._stack) - 1, -1, -1):
            elem, _raw, parts = self._stack[i]
            if elem.tag == tag:
                elem.text = _norm(" ".join(parts))
                self._stack = self._stack[:i] + self._stack[i + 1 :]
                return
        elem, _raw, parts = self._stack.pop()
        elem.text = _norm(" ".join(parts))

    def handle_data(self, data: str) -> None:
        if not data or not data.strip():
            return
        for _elem, _raw, parts in self._stack:
            parts.append(data)


def parse_dom(html: str | None) -> list[Element]:
    if not html:
        return []
    p = _DomParser()
    try:
        p.feed(html)
    except Exception as e:  # pragma: no cover — pruned HTML is generally well-formed
        logger.debug("html parse failed: %s", e)
    for elem, _raw, parts in p._stack:
        if not elem.text:
            elem.text = _norm(" ".join(parts))

    # Build select.options.
    current_select: Element | None = None
    for elem in p.elements:
        if elem.tag == "select":
            current_select = elem
        elif elem.tag == "option" and current_select is not None:
            current_select.options.append(elem)

    _resolve_labels(p.elements, p._by_id, p._label_for)
    return p.elements


def _resolve_labels(
    elements: list[Element],
    by_id: dict[str, Element],
    label_for: list[tuple[Element, str]],
) -> None:
    for label_elem, target_id in label_for:
        target = by_id.get(target_id)
        if target is not None and (target.is_checkbox() or target.is_radio()):
            text = _norm(label_elem.text)
            if text and not target.label:
                target.label = text

    n = len(elements)
    for i, lbl in enumerate(elements):
        if lbl.tag != "label" or not lbl.text:
            continue
        for j in range(i + 1, min(i + 6, n)):
            cand = elements[j]
            if (cand.is_checkbox() or cand.is_radio()) and not cand.label:
                cand.label = lbl.text
                break


def build_state(html: str | None, terminal_success: bool = False) -> State:
    """Build a `State` from pruned HTML and an env-supplied success flag.

    Terminal success comes from `evaluate()` (i.e. the MiniWob JS reward
    signal), not from any DOM heuristic. Episode-level counters
    (failed_tool_count, stuck_count, …) live on `HistoryCounters` outside
    the state.
    """
    return State(elements=parse_dom(html), terminal_success=bool(terminal_success))
