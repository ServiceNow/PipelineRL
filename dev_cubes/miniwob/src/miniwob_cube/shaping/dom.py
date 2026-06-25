"""DOM abstraction for MiniWob++ pruned HTML.

Parses pruned HTML into a flat list of `Element`s tagged with browsergym
`bid`s. We deliberately keep this small: the only consumer is the deterministic
predicate evaluator in `potential.py`, which only needs visible affordances
(checkboxes, radios, text inputs, dropdowns, buttons, clickables) and their
labels.

Label inference for checkboxes/radios walks the surrounding `<label>` element
because MiniWob's pruned HTML rarely emits `<label for="...">` linking.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

logger = logging.getLogger(__name__)


_CHECKBOX_TYPES = frozenset({"checkbox", "radio"})
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
    # Helpful for grouping radios.
    name: str | None = None
    aria_label: str | None = None
    title: str | None = None
    placeholder: str | None = None
    # Children options for select elements.
    options: list["Element"] = field(default_factory=list)

    # ---- type predicates ----
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

    def best_label(self) -> str:
        for candidate in (self.label, self.text, self.aria_label, self.title, self.value, self.placeholder):
            if candidate:
                s = _norm(candidate)
                if s:
                    return s
        return ""


@dataclass
class ToolResult:
    """Lightweight summary of what happened at one step.

    The shaping logic never sees the LLM's tool call directly — it only needs
    to know whether the call failed, so it can debit failed_tool_count.
    """

    failed: bool = False
    error_message: str | None = None
    is_noop: bool = False  # DOM unchanged with no error.


@dataclass
class State:
    """Snapshot of one observation moment.

    `terminal_success` comes from the environment (the MiniWob JS reward),
    not from any DOM heuristic. `failed_tool_count` and `noop_count` are
    cumulative counts since episode start (they monotonically grow), so a
    failed step debits Phi at next_state but not at prev_state, producing a
    small negative local reward.
    """

    elements: list[Element] = field(default_factory=list)
    terminal_success: bool = False
    failed_tool_count: int = 0
    noop_count: int = 0

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


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


class _DomParser(HTMLParser):
    """Two-pass-friendly DOM scraper that retains parent stack so checkbox
    labels can be inferred from the enclosing `<label>` element."""

    def __init__(self) -> None:
        super().__init__()
        self.elements: list[Element] = []
        # Stack entries: (Element, attrs_dict, accumulated text parts).
        self._stack: list[tuple[Element, dict[str, str], list[str]]] = []
        # Map id -> element for <label for="..."> resolution.
        self._by_id: dict[str, Element] = {}
        # Element -> "for" target id, resolved at end.
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
                # Pop closed entry; preserve any deeper open tags as tolerable.
                self._stack = self._stack[:i] + self._stack[i + 1 :]
                return
        # Unmatched close — drop top entry.
        elem, _raw, parts = self._stack.pop()
        elem.text = _norm(" ".join(parts))

    def handle_data(self, data: str) -> None:
        if not data or not data.strip():
            return
        for _elem, _raw, parts in self._stack:
            parts.append(data)


def parse_dom(html: str | None) -> list[Element]:
    """Parse pruned HTML into a flat list of `Element`s.

    Labels for checkboxes/radios are inferred from any enclosing `<label>`
    element (the common MiniWob shape) and, as a fallback, from explicit
    `<label for="id">` references.
    """
    if not html:
        return []
    p = _DomParser()
    try:
        p.feed(html)
    except Exception as e:  # pragma: no cover — pruned_html is generally well-formed
        logger.debug("html parse failed: %s", e)
    # Finalize any stray open elements.
    for elem, _raw, parts in p._stack:
        if not elem.text:
            elem.text = _norm(" ".join(parts))

    # ---- Build select.options ----
    current_select: Element | None = None
    for elem in p.elements:
        if elem.tag == "select":
            current_select = elem
        elif elem.tag == "option" and current_select is not None:
            current_select.options.append(elem)

    # ---- Wire checkbox/radio labels ----
    _resolve_labels(p.elements, p._by_id, p._label_for)
    return p.elements


def _resolve_labels(
    elements: list[Element],
    by_id: dict[str, Element],
    label_for: list[tuple[Element, str]],
) -> None:
    """Infer labels for checkboxes/radios using surrounding <label> text."""
    # Pass 1: explicit `<label for="...">` linkage.
    for label_elem, target_id in label_for:
        target = by_id.get(target_id)
        if target is not None and (target.is_checkbox() or target.is_radio()):
            text = _norm(label_elem.text)
            if text and not target.label:
                target.label = text

    # Pass 2: implicit — a checkbox/radio nested inside <label>...text...</label>.
    # We use document order: for every <label>, find the checkbox/radio that
    # appears between its start and (effectively) its end.
    # MiniWob nests like: `<label> <input type="checkbox"/> apple </label>`.
    # The implementation here is index-based on `elements`.
    n = len(elements)
    for i, lbl in enumerate(elements):
        if lbl.tag != "label" or not lbl.text:
            continue
        # Heuristic: associate the nearest following checkbox/radio
        # whose label is still empty. This is a flat-list scan, but since
        # pruned HTML rarely interleaves siblings before the input, it works.
        for j in range(i + 1, min(i + 6, n)):
            cand = elements[j]
            if (cand.is_checkbox() or cand.is_radio()) and not cand.label:
                cand.label = lbl.text
                break


# ---------------------------------------------------------------------------
# State builder
# ---------------------------------------------------------------------------


def build_state(
    html: str | None,
    env_metadata: dict[str, Any] | None = None,
) -> State:
    """Build a `State` from pruned HTML and optional environment metadata.

    `env_metadata` may include:
        - `terminal_success`: bool
        - `failed_tool_count`: int
        - `noop_count`: int
    """
    md = env_metadata or {}
    return State(
        elements=parse_dom(html),
        terminal_success=bool(md.get("terminal_success", False)),
        failed_tool_count=int(md.get("failed_tool_count", 0) or 0),
        noop_count=int(md.get("noop_count", 0) or 0),
    )
