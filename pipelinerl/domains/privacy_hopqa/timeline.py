"""Interactive timeline artifact generation for privacy_hopqa rollouts."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


_STAGE_COLORS = {
    "hop_plan": "#2f6fed",
    "search": "#14866d",
    "doc_choose": "#9467bd",
    "doc_read": "#e17c05",
    "hop_resolve": "#d62728",
    "report": "#4c78a8",
    "error": "#7f1d1d",
}


def _format_seconds(value: float) -> str:
    if value < 0.1:
        return f"{value:.3f}"
    if value < 10.0:
        return f"{value:.2f}"
    return f"{value:.1f}"


def _lane_order(events: list[dict[str, Any]]) -> list[str]:
    first_seen: dict[str, float] = {}
    for event in events:
        lane = str(event.get("lane") or "other")
        first_seen.setdefault(lane, float(event.get("start_s") or 0.0))
    return [lane for lane, _ in sorted(first_seen.items(), key=lambda item: item[1])]


def _prompt_name_for_event(event: dict[str, Any]) -> str | None:
    stage = str(event.get("stage") or "")
    meta = event.get("meta") or {}
    hop_number = meta.get("hop_number")
    iteration = meta.get("iteration")
    if hop_number in (None, ""):
        return None
    if stage == "hop_plan" and iteration not in (None, ""):
        return f"hop_plan_iter{int(iteration)}_hop{int(hop_number)}.txt"
    if stage == "doc_choose" and iteration not in (None, ""):
        return f"doc_choose_iter{int(iteration)}_hop{int(hop_number)}.txt"
    if stage == "hop_resolve" and iteration not in (None, ""):
        return f"hop_resolve_iter{int(iteration)}_hop{int(hop_number)}.txt"
    if stage == "doc_read" and iteration not in (None, ""):
        doc_id = str(meta.get("doc_id") or "").replace("/", "_")
        if doc_id:
            return f"doc_read_iter{int(iteration)}_hop{int(hop_number)}_{doc_id}.txt"
    return None


def _truncate(text: str, limit: int = 280) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _build_event_details(
    summary: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    trace: dict[str, Any] | None = None,
    report_text: str | None = None,
    prompt_dir: Path | None = None,
) -> list[dict[str, Any]]:
    trace = trace or {}
    hop_states = list(trace.get("hop_states") or [])
    actions = list(trace.get("actions") or [])
    errors = list(trace.get("error_records") or [])

    actions_by_id = {str(action.get("id") or ""): action for action in actions if action.get("id")}
    hop_by_number = {int(hop.get("hop_number") or 0): hop for hop in hop_states if hop.get("hop_number") is not None}
    reader_results_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    search_history_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for hop in hop_states:
        hop_number = int(hop.get("hop_number") or 0)
        for result in hop.get("reader_results") or []:
            reader_results_by_key[(hop_number, str(result.get("doc_id") or ""))] = result
        for entry in hop.get("search_history") or []:
            search_history_by_key[(hop_number, str(entry.get("action_id") or ""))] = entry

    prompt_cache: dict[str, str] = {}

    def load_prompt(prompt_name: str | None) -> str | None:
        if not prompt_name or prompt_dir is None:
            return None
        if prompt_name in prompt_cache:
            return prompt_cache[prompt_name]
        path = prompt_dir / prompt_name
        if not path.exists():
            prompt_cache[prompt_name] = ""
            return None
        prompt_cache[prompt_name] = path.read_text(encoding="utf-8", errors="ignore")
        return prompt_cache[prompt_name]

    details: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        stage = str(event.get("stage") or "")
        lane = str(event.get("lane") or "")
        meta = dict(event.get("meta") or {})
        start_s = float(event.get("start_s") or 0.0)
        end_s = float(event.get("end_s") or start_s)
        duration_s = max(0.0, end_s - start_s)
        hop_number = int(meta.get("hop_number") or 0) if meta.get("hop_number") not in (None, "") else None
        iteration = int(meta.get("iteration") or 0) if meta.get("iteration") not in (None, "") else None
        hop_state = hop_by_number.get(hop_number or -1)
        prompt_name = _prompt_name_for_event(event)
        prompt_text = load_prompt(prompt_name)
        detail: dict[str, Any] = {
            "index": index,
            "stage": stage,
            "lane": lane,
            "label": str(event.get("label") or ""),
            "start_s": round(start_s, 6),
            "end_s": round(end_s, 6),
            "duration_s": round(duration_s, 6),
            "hop_number": hop_number,
            "iteration": iteration,
            "meta": meta,
            "question": str(hop_state.get("question") or "") if hop_state else "",
        }
        if prompt_name:
            detail["prompt_name"] = prompt_name
        if prompt_text:
            detail["prompt_text"] = prompt_text

        if stage == "search":
            action_id = str(meta.get("action_id") or "")
            if not action_id and lane.startswith("search:"):
                action_id = lane.split(":", 1)[1]
            action = actions_by_id.get(action_id)
            search_entry = search_history_by_key.get((hop_number or 0, action_id))
            detail["search_action"] = {
                "action_id": action_id,
                "query": str(meta.get("query") or ""),
                "result_count": int(meta.get("result_count") or 0),
                "top_doc_ids": list(meta.get("top_doc_ids") or []),
                "action_record": action,
                "search_history_entry": search_entry,
            }
            if action and action.get("actual_output"):
                detail["result_preview"] = {
                    "tool": action["actual_output"].get("tool"),
                    "results_count": action["actual_output"].get("results_count"),
                    "results": action["actual_output"].get("results", [])[:5],
                }
        elif stage == "doc_read":
            doc_id = str(meta.get("doc_id") or "")
            reader_result = reader_results_by_key.get((hop_number or 0, doc_id))
            detail["reader_result"] = reader_result
        elif stage == "doc_choose":
            detail["chooser_summary"] = {
                "candidate_doc_ids": list(hop_state.get("candidate_doc_ids") or []) if hop_state else [],
                "selected_doc_ids": list(meta.get("selected_doc_ids") or hop_state.get("selected_doc_ids") or []) if hop_state else [],
                "candidate_count": meta.get("candidate_count"),
            }
        elif stage == "hop_plan":
            detail["planned_actions"] = list(meta.get("planned_actions") or [])
            if hop_state:
                detail["recent_search_history"] = [
                    entry
                    for entry in (hop_state.get("search_history") or [])
                    if iteration is None or int(entry.get("iteration") or -1) == iteration
                ]
        elif stage == "hop_resolve":
            detail["resolution"] = {
                "status": str(hop_state.get("status") or "") if hop_state else "",
                "answer": str(hop_state.get("answer") or "") if hop_state else "",
                "justification": str(hop_state.get("justification") or "") if hop_state else "",
                "confidence": hop_state.get("confidence") if hop_state else None,
                "reason": str(hop_state.get("resolution_reason") or "") if hop_state else "",
                "reader_results": list(hop_state.get("reader_results") or []) if hop_state else [],
            }
        elif stage == "report":
            detail["report"] = {
                "resolved_hops": int(meta.get("resolved_hops") or 0),
                "report_path": str(meta.get("report_path") or trace.get("report_path") or ""),
                "final_report": report_text or "",
                "hop_answers": [
                    {
                        "hop_number": hop.get("hop_number"),
                        "question": hop.get("question"),
                        "answer": hop.get("answer"),
                        "justification": hop.get("justification"),
                    }
                    for hop in hop_states
                ],
            }
        elif stage == "error":
            detail["error"] = meta
            matching_error = next(
                (
                    err
                    for err in errors
                    if str(err.get("stage") or "") == str(meta.get("stage") or "")
                    and err.get("hop_number") == meta.get("hop_number")
                    and err.get("iteration") == meta.get("iteration")
                    and str(err.get("message") or "") == str(meta.get("message") or "")
                ),
                None,
            )
            if matching_error:
                detail["error_record"] = matching_error

        details.append(detail)
    return details


def render_timeline_html(
    summary: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    trace: dict[str, Any] | None = None,
    report_text: str | None = None,
    prompt_dir: Path | None = None,
) -> str:
    total_duration = max((float(event.get("end_s") or 0.0) for event in events), default=0.001)
    lanes = _lane_order(events)
    details = _build_event_details(summary, events, trace=trace, report_text=report_text, prompt_dir=prompt_dir)
    detail_lookup = {detail["index"]: detail for detail in details}

    lane_blocks: list[str] = []
    for lane in lanes:
        bars: list[str] = []
        lane_events = [(idx, evt) for idx, evt in enumerate(events) if str(evt.get("lane") or "other") == lane]
        for idx, event in lane_events:
            start_s = float(event.get("start_s") or 0.0)
            end_s = float(event.get("end_s") or start_s)
            duration_s = max(end_s - start_s, 0.01)
            left = 100.0 * start_s / total_duration
            width = max(0.8, 100.0 * duration_s / total_duration)
            stage = str(event.get("stage") or "other")
            color = _STAGE_COLORS.get(stage, "#6b7280")
            label = html.escape(str(event.get("label") or stage))
            title = html.escape(
                json.dumps(
                    {
                        "stage": stage,
                        "label": event.get("label"),
                        "start_s": round(start_s, 3),
                        "duration_s": round(duration_s, 3),
                        "meta": event.get("meta", {}),
                    },
                    ensure_ascii=False,
                )
            )
            bars.append(
                f"<button class='bar' data-event-index='{idx}' style='left:{left:.3f}%;width:{width:.3f}%;background:{color};' title='{title}'>{label}</button>"
            )
        lane_blocks.append(
            "<div class='lane-row'>"
            f"<div class='lane-label'>{html.escape(lane)}</div>"
            f"<div class='lane-track'>{''.join(bars)}</div>"
            "</div>"
        )

    rows = []
    for idx, event in enumerate(sorted(enumerate(events), key=lambda item: (float(item[1].get("start_s") or 0.0), str(item[1].get("lane") or "")))):
        event_index, payload = event
        rows.append(
            f"<tr class='event-row' data-event-index='{event_index}'>"
            f"<td>{html.escape(str(payload.get('lane') or ''))}</td>"
            f"<td>{html.escape(str(payload.get('stage') or ''))}</td>"
            f"<td>{html.escape(str(payload.get('label') or ''))}</td>"
            f"<td>{_format_seconds(float(payload.get('start_s') or 0.0))}</td>"
            f"<td>{_format_seconds(max(float(payload.get('end_s') or 0.0) - float(payload.get('start_s') or 0.0), 0.0))}</td>"
            f"<td>{html.escape(_truncate(json.dumps(payload.get('meta', {}), ensure_ascii=False), 160))}</td>"
            "<td><button class='inspect-btn' type='button'>Inspect</button></td>"
            "</tr>"
        )

    questions = []
    trace = trace or {}
    for hop in trace.get("hop_states") or []:
        answer = str(hop.get("answer") or "UNRESOLVED")
        justification = str(hop.get("justification") or "")
        questions.append(
            "<div class='question-card'>"
            f"<div class='question-hop'>Hop {int(hop.get('hop_number') or 0)}</div>"
            f"<div class='question-text'>{html.escape(str(hop.get('question') or ''))}</div>"
            f"<div class='question-answer'>{html.escape(_truncate(answer, 180))}</div>"
            f"<div class='question-note'>{html.escape(_truncate(justification, 220))}</div>"
            "</div>"
        )

    report_preview = ""
    if report_text:
        report_preview = (
            "<details class='report-preview'><summary>Final report preview</summary>"
            f"<pre>{html.escape(report_text)}</pre>"
            "</details>"
        )

    timeline_data = {
        "summary": summary,
        "events": details,
        "questions": [
            {
                "hop_number": hop.get("hop_number"),
                "question": hop.get("question"),
                "answer": hop.get("answer"),
                "justification": hop.get("justification"),
                "status": hop.get("status"),
            }
            for hop in (trace.get("hop_states") or [])
        ],
        "report_text": report_text or "",
    }
    timeline_json = json.dumps(timeline_data, ensure_ascii=False).replace("</", "<\\/")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>privacy_hopqa timeline</title>
  <style>
    :root {{
      --bg: #f5f4ef;
      --panel: #ffffff;
      --ink: #1e2430;
      --muted: #5f6b7a;
      --line: #d7dbe2;
      --accent: #1f5da8;
      --accent-soft: #e7f0fb;
      --shadow: 0 12px 36px rgba(30, 36, 48, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #f8f7f2 0%, #eef3f8 100%);
      color: var(--ink);
    }}
    button {{
      font: inherit;
    }}
    .page {{
      max-width: 1680px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero {{
      background: radial-gradient(circle at top left, #ffffff 0%, #eef4fb 45%, #f8f7f2 100%);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 28px 30px;
      margin-bottom: 22px;
      box-shadow: var(--shadow);
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 30px;
    }}
    .hero p {{
      margin: 6px 0;
      color: var(--muted);
      max-width: 1100px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 22px 0 0;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px 18px;
      box-shadow: 0 8px 24px rgba(30, 36, 48, 0.04);
    }}
    .card .label {{
      font-size: 12px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .card .value {{
      font-size: 28px;
      font-weight: 700;
    }}
    .section {{
      margin: 28px 0;
    }}
    .section h2 {{
      margin: 0 0 12px;
      font-size: 22px;
    }}
    .section-note {{
      margin: 0 0 14px;
      color: var(--muted);
    }}
    .question-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
    }}
    .question-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px 16px;
      box-shadow: 0 8px 24px rgba(30, 36, 48, 0.04);
    }}
    .question-hop {{
      font-size: 12px;
      text-transform: uppercase;
      color: var(--accent);
      letter-spacing: 0.06em;
      margin-bottom: 6px;
    }}
    .question-text {{
      font-weight: 600;
      margin-bottom: 10px;
    }}
    .question-answer {{
      font-size: 13px;
      margin-bottom: 6px;
    }}
    .question-note {{
      font-size: 12px;
      color: var(--muted);
    }}
    .lane-row {{
      display: grid;
      grid-template-columns: 260px 1fr;
      gap: 12px;
      align-items: center;
      margin-bottom: 10px;
    }}
    .lane-label {{
      font-size: 13px;
      color: #334155;
      overflow-wrap: anywhere;
    }}
    .lane-track {{
      position: relative;
      height: 28px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .bar {{
      position: absolute;
      top: 3px;
      bottom: 3px;
      border-radius: 6px;
      color: white;
      font-size: 11px;
      line-height: 22px;
      padding: 0 6px;
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
      border: none;
      cursor: pointer;
      text-align: left;
    }}
    .bar:hover,
    .event-row:hover {{
      filter: brightness(1.03);
    }}
    .inspect-btn {{
      background: var(--accent-soft);
      color: var(--accent);
      border: 1px solid #cfe0f6;
      border-radius: 999px;
      padding: 6px 10px;
      cursor: pointer;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 10px 28px rgba(30, 36, 48, 0.05);
    }}
    th, td {{
      border-bottom: 1px solid #e2e8f0;
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
      font-size: 13px;
    }}
    th {{
      background: #f1f5f9;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      background: #f7f9fc;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      max-height: 420px;
      overflow: auto;
    }}
    details {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      margin-top: 12px;
      box-shadow: 0 10px 28px rgba(30, 36, 48, 0.05);
    }}
    summary {{
      cursor: pointer;
      font-weight: 600;
    }}
    .modal-backdrop {{
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.52);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 20px;
      z-index: 1000;
    }}
    .modal-backdrop.open {{
      display: flex;
    }}
    .modal {{
      width: min(1100px, 100%);
      max-height: min(90vh, 900px);
      overflow: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 24px 80px rgba(15, 23, 42, 0.28);
    }}
    .modal-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      padding: 18px 20px 12px;
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      background: var(--panel);
      z-index: 2;
    }}
    .modal-title {{
      margin: 0;
      font-size: 22px;
    }}
    .modal-subtitle {{
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .modal-close {{
      background: transparent;
      border: none;
      font-size: 28px;
      line-height: 1;
      cursor: pointer;
      color: #475569;
    }}
    .modal-body {{
      padding: 18px 20px 22px;
    }}
    .detail-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .detail-card {{
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
    }}
    .detail-card .label {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 6px;
    }}
    .detail-card .value {{
      font-size: 14px;
      font-weight: 600;
      overflow-wrap: anywhere;
    }}
    .modal-section {{
      margin-bottom: 16px;
    }}
    .modal-section h3 {{
      margin: 0 0 8px;
      font-size: 16px;
    }}
    .json-note {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>privacy_hopqa rollout timeline</h1>
      <p>This trace viewer shows the full hop-QA loop for a single rollout. Click any timeline bar or event row to inspect the current hop question, search outputs, reader decisions, resolver state, and final report context.</p>
      <div class="summary">
        <div class="card"><div class="label">Task</div><div class="value">{html.escape(str(summary.get("task_id") or ""))}</div></div>
        <div class="card"><div class="label">Chain</div><div class="value">{html.escape(str(summary.get("chain_id") or ""))}</div></div>
        <div class="card"><div class="label">Iterations</div><div class="value">{int(summary.get("iterations_used") or 0)}</div></div>
        <div class="card"><div class="label">Resolved Hops</div><div class="value">{int(summary.get("resolved_hops") or 0)}/{int(summary.get("total_hops") or 0)}</div></div>
        <div class="card"><div class="label">Searches</div><div class="value">{int(summary.get("searches_total") or 0)}</div></div>
        <div class="card"><div class="label">Docs Read</div><div class="value">{int(summary.get("docs_read") or 0)}</div></div>
        <div class="card"><div class="label">Parse Errors</div><div class="value">{int(summary.get("parse_errors_total") or 0)}</div></div>
        <div class="card"><div class="label">Context Overflows</div><div class="value">{int(summary.get("context_overflow_errors") or 0)}</div></div>
        <div class="card"><div class="label">Duration (s)</div><div class="value">{_format_seconds(float(summary.get("duration_s") or 0.0))}</div></div>
      </div>
    </section>

    <section class="section">
      <h2>Questions</h2>
      <p class="section-note">This is the full hop chain the agent was trying to answer. The cards also show the final answer state for each hop.</p>
      <div class="question-grid">{''.join(questions)}</div>
      {report_preview}
    </section>

    <section class="section">
      <h2>Timeline</h2>
      <p class="section-note">Bars are clickable. Parallel lanes show where retrieval or document-reading overlapped in wall-clock time.</p>
      <div>{''.join(lane_blocks)}</div>
    </section>

    <section class="section">
      <h2>Events</h2>
      <p class="section-note">The event table is also clickable and opens the same modal as the bars above.</p>
      <table>
        <thead>
          <tr>
            <th>Lane</th>
            <th>Stage</th>
            <th>Label</th>
            <th>Start (s)</th>
            <th>Duration (s)</th>
            <th>Metadata</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </section>
  </div>

  <div class="modal-backdrop" id="timeline-modal-backdrop" aria-hidden="true">
    <div class="modal" role="dialog" aria-modal="true" aria-labelledby="timeline-modal-title">
      <div class="modal-header">
        <div>
          <h2 class="modal-title" id="timeline-modal-title">Event details</h2>
          <p class="modal-subtitle" id="timeline-modal-subtitle"></p>
        </div>
        <button class="modal-close" type="button" id="timeline-modal-close" aria-label="Close">&times;</button>
      </div>
      <div class="modal-body" id="timeline-modal-body"></div>
    </div>
  </div>

  <script id="timeline-data" type="application/json">{timeline_json}</script>
  <script>
    const timelineData = JSON.parse(document.getElementById("timeline-data").textContent);
    const modalBackdrop = document.getElementById("timeline-modal-backdrop");
    const modalTitle = document.getElementById("timeline-modal-title");
    const modalSubtitle = document.getElementById("timeline-modal-subtitle");
    const modalBody = document.getElementById("timeline-modal-body");
    const closeButton = document.getElementById("timeline-modal-close");

    function escapeHtml(value) {{
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }}

    function formatSeconds(value) {{
      const numeric = Number(value || 0);
      if (numeric < 0.1) return numeric.toFixed(3);
      if (numeric < 10) return numeric.toFixed(2);
      return numeric.toFixed(1);
    }}

    function renderKvCards(detail) {{
      const cards = [
        ["Stage", detail.stage],
        ["Lane", detail.lane],
        ["Start", `${{formatSeconds(detail.start_s)}}s`],
        ["Duration", `${{formatSeconds(detail.duration_s)}}s`],
      ];
      if (detail.hop_number !== null && detail.hop_number !== undefined) {{
        cards.push(["Hop", String(detail.hop_number)]);
      }}
      if (detail.iteration !== null && detail.iteration !== undefined) {{
        cards.push(["Iteration", String(detail.iteration)]);
      }}
      return `<div class="detail-grid">${{cards.map(([label, value]) => `
        <div class="detail-card">
          <div class="label">${{escapeHtml(label)}}</div>
          <div class="value">${{escapeHtml(value)}}</div>
        </div>`).join("")}}</div>`;
    }}

    function renderJsonSection(title, value, note = "") {{
      if (value === undefined || value === null || value === "" || (Array.isArray(value) && !value.length)) {{
        return "";
      }}
      return `
        <section class="modal-section">
          <h3>${{escapeHtml(title)}}</h3>
          ${{note ? `<div class="json-note">${{escapeHtml(note)}}</div>` : ""}}
          <pre>${{escapeHtml(typeof value === "string" ? value : JSON.stringify(value, null, 2))}}</pre>
        </section>`;
    }}

    function openEvent(index) {{
      const detail = timelineData.events[index];
      if (!detail) return;
      modalTitle.textContent = detail.label || detail.stage || "Event details";
      modalSubtitle.textContent = detail.question
        ? `Hop ${{detail.hop_number ?? "?"}}: ${{detail.question}}`
        : `${{detail.stage}} on ${{detail.lane}}`;

      const sections = [];
      sections.push(renderKvCards(detail));
      if (detail.question) {{
        sections.push(`<section class="modal-section"><h3>Current hop question</h3><pre>${{escapeHtml(detail.question)}}</pre></section>`);
      }}
      if (detail.prompt_text) {{
        sections.push(renderJsonSection("Prompt input", detail.prompt_text, detail.prompt_name || ""));
      }}
      if (detail.planned_actions) {{
        sections.push(renderJsonSection("Planned retrieval actions", detail.planned_actions));
      }}
      if (detail.search_action) {{
        sections.push(renderJsonSection("Search action", detail.search_action));
      }}
      if (detail.result_preview) {{
        sections.push(renderJsonSection("Search results preview", detail.result_preview));
      }}
      if (detail.chooser_summary) {{
        sections.push(renderJsonSection("Chooser summary", detail.chooser_summary));
      }}
      if (detail.reader_result) {{
        sections.push(renderJsonSection("Reader result", detail.reader_result));
      }}
      if (detail.resolution) {{
        sections.push(renderJsonSection("Resolver output", detail.resolution));
      }}
      if (detail.report) {{
        sections.push(renderJsonSection("Final report", detail.report.final_report || ""));
        sections.push(renderJsonSection("Per-hop answers", detail.report.hop_answers || []));
      }}
      if (detail.error) {{
        sections.push(renderJsonSection("Error details", detail.error));
      }}
      sections.push(renderJsonSection("Raw event metadata", detail.meta));
      modalBody.innerHTML = sections.join("");
      modalBackdrop.classList.add("open");
      modalBackdrop.setAttribute("aria-hidden", "false");
    }}

    function closeModal() {{
      modalBackdrop.classList.remove("open");
      modalBackdrop.setAttribute("aria-hidden", "true");
      modalBody.innerHTML = "";
    }}

    document.querySelectorAll("[data-event-index]").forEach((node) => {{
      node.addEventListener("click", () => openEvent(Number(node.getAttribute("data-event-index"))));
    }});

    closeButton.addEventListener("click", closeModal);
    modalBackdrop.addEventListener("click", (event) => {{
      if (event.target === modalBackdrop) closeModal();
    }});
    document.addEventListener("keydown", (event) => {{
      if (event.key === "Escape") closeModal();
    }});
  </script>
</body>
</html>
"""


def write_timeline_artifacts(
    output_dir: Path,
    *,
    summary: dict[str, Any],
    events: list[dict[str, Any]],
    trace: dict[str, Any] | None = None,
    report_text: str | None = None,
    report_path: str | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "timeline_events.json"
    summary_path = output_dir / "timeline_summary.json"
    html_path = output_dir / "timeline.html"

    events_path.write_text(json.dumps(events, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    prompt_dir = output_dir / "prompts"
    html_path.write_text(
        render_timeline_html(
            summary,
            events,
            trace=trace,
            report_text=report_text,
            prompt_dir=prompt_dir if prompt_dir.exists() else None,
        ),
        encoding="utf-8",
    )

    return {
        "events_path": str(events_path),
        "summary_path": str(summary_path),
        "html_path": str(html_path),
        "report_path": report_path or "",
    }
