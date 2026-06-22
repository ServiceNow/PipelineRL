"""Anchor-match rate analysis for GiGPO.

Reads PipelineRL actor jsonl (one group per line, list of training_text dicts)
and asks: how often does a step's anchor observation recur across sibling
rollouts within the same group? That recurrence rate determines whether
GiGPO's step-level advantage carries any signal — singletons collapse to 0.

Anchor extraction here mirrors `pipelinerl/domains/cube/result_builder.py`:
the prompt portion of `text` is a canonical JSON `{"messages":[...],"tools":[...]}`,
and the anchor is SHA1 of the *content* of the last user message (the
observation the LLM was asked to act on). We reconstruct it from the saved
`text` because the rollouts in `results/` predate the anchor_obs plumbing.

Run:
    uv run python -m analysis.anchor_match \\
        results/cube_qwen3_1.7b/streams/actor/0/0/0.jsonl
    uv run python -m analysis.anchor_match path/to/file.jsonl --by-step --json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


def extract_anchor(text_dict: dict[str, Any]) -> str:
    """Reconstruct the anchor_obs hash from a stored training_text record.

    Matches `_compute_anchor_obs` in pipelinerl/domains/cube/result_builder.py:
    SHA1 over the canonical-JSON content of the last user message in the prompt.
    """
    text = text_dict.get("text") or ""
    n_predicted = int(text_dict.get("n_predicted") or 0)
    if not text:
        return ""
    prompt_text = text[:-n_predicted] if n_predicted else text
    try:
        prompt = json.loads(prompt_text)
    except json.JSONDecodeError:
        return ""
    messages = prompt.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if content is None:
                return ""
            blob = json.dumps(content, sort_keys=True, default=str, separators=(",", ":"))
            return hashlib.sha1(blob.encode("utf-8")).hexdigest()
    return ""


def iter_groups(path: Path) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Yield (group_id, list of training_text dicts) from a PipelineRL actor jsonl."""
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records = json.loads(line)
            if not records:
                continue
            group_id = records[0].get("group_id") or "<unknown>"
            yield group_id, records


@dataclass
class GroupStats:
    group_id: str
    n_rollouts: int
    n_steps: int
    n_unique_anchors: int  # within this group
    n_steps_in_shared_anchor: int  # steps whose anchor recurs (cluster_size >= 2)
    cluster_size_hist: dict[int, int]  # cluster_size -> count of clusters of that size
    max_cluster_size: int

    @property
    def share_rate(self) -> float:
        return self.n_steps_in_shared_anchor / self.n_steps if self.n_steps else 0.0


def _step_index(record: dict[str, Any]) -> int:
    return int(record.get("metadata", {}).get("step_index") or 0)


def _filter_by_min_step(records: list[dict[str, Any]], min_step: int) -> list[dict[str, Any]]:
    return records if min_step <= 0 else [r for r in records if _step_index(r) >= min_step]


def analyze_group(records: list[dict[str, Any]], min_step: int = 0) -> GroupStats:
    """Compute anchor-cluster stats for one group of sibling rollouts.

    `min_step` drops records with step_index < min_step before clustering. Use
    min_step=1 to exclude the structurally-guaranteed step-0 cluster and see the
    signal GiGPO can actually exploit.
    """
    group_id = records[0].get("group_id") or "<unknown>"
    filtered = _filter_by_min_step(records, min_step)
    rollout_indices = {
        r.get("metadata", {}).get("rollout_index") for r in filtered
    }
    anchors: list[str] = [extract_anchor(r) for r in filtered]
    cluster_counts = collections.Counter(anchors)
    cluster_size_hist: dict[int, int] = collections.Counter(cluster_counts.values())
    n_steps_in_shared = sum(1 for a in anchors if cluster_counts[a] >= 2)
    return GroupStats(
        group_id=group_id,
        n_rollouts=len(rollout_indices),
        n_steps=len(filtered),
        n_unique_anchors=len(cluster_counts),
        n_steps_in_shared_anchor=n_steps_in_shared,
        cluster_size_hist=dict(cluster_size_hist),
        max_cluster_size=max(cluster_counts.values(), default=0),
    )


def per_step_share_rate(
    records_by_group: Iterable[tuple[str, list[dict[str, Any]]]],
    min_step: int = 0,
) -> dict[int, dict[str, float]]:
    """Break the anchor-share rate down by step_index across the whole file."""
    bucket: dict[int, list[int]] = collections.defaultdict(list)  # step_idx -> [cluster_size per record]
    for _, records in records_by_group:
        filtered = _filter_by_min_step(records, min_step)
        anchors_in_group: dict[str, int] = collections.Counter(extract_anchor(r) for r in filtered)
        for r in filtered:
            bucket[_step_index(r)].append(anchors_in_group[extract_anchor(r)])
    out: dict[int, dict[str, float]] = {}
    for si, sizes in sorted(bucket.items()):
        out[si] = {
            "n": len(sizes),
            "share_rate": sum(1 for s in sizes if s >= 2) / len(sizes),
            "mean_cluster_size": statistics.mean(sizes),
            "max_cluster_size": max(sizes),
        }
    return out


def summarize(path: Path, min_step: int = 0) -> dict[str, Any]:
    groups = list(iter_groups(path))
    stats = [analyze_group(records, min_step=min_step) for _, records in groups]
    total_steps = sum(s.n_steps for s in stats)
    total_shared = sum(s.n_steps_in_shared_anchor for s in stats)
    overall_hist: dict[int, int] = collections.Counter()
    for s in stats:
        for size, count in s.cluster_size_hist.items():
            overall_hist[size] += count
    n_groups_with_signal = sum(1 for s in stats if s.n_steps_in_shared_anchor > 0 and s.n_steps > 0)
    return {
        "path": str(path),
        "min_step": min_step,
        "n_groups": len(stats),
        "n_groups_with_nontrivial_signal": n_groups_with_signal,
        "n_steps": total_steps,
        "overall_share_rate": total_shared / total_steps if total_steps else 0.0,
        "mean_per_group_share_rate": (
            statistics.mean(s.share_rate for s in stats) if stats else 0.0
        ),
        "cluster_size_histogram": dict(sorted(overall_hist.items())),
        "groups": [
            {
                "group_id": s.group_id,
                "n_rollouts": s.n_rollouts,
                "n_steps": s.n_steps,
                "n_unique_anchors": s.n_unique_anchors,
                "share_rate": round(s.share_rate, 4),
                "max_cluster_size": s.max_cluster_size,
            }
            for s in stats
        ],
    }


def _format_hist(hist: dict[int, int]) -> str:
    if not hist:
        return "<empty>"
    lines = []
    total = sum(hist.values())
    for size in sorted(hist):
        count = hist[size]
        bar = "#" * min(40, int(40 * count / total))
        lines.append(f"  size={size:>3d}: {count:>5d}  {bar}")
    return "\n".join(lines)


def _print_per_group_table(groups: list[dict[str, Any]]) -> None:
    """One row per group, sorted by share rate descending."""
    rows = sorted(groups, key=lambda g: g["share_rate"], reverse=True)
    print("per-group anchor sharing:")
    print(
        f"  {'group_id':<32s} {'rollouts':>8s} {'records':>8s} "
        f"{'distinct':>8s} {'max':>6s} {'share':>6s}"
    )
    for g in rows:
        print(
            f"  {g['group_id']:<32s} {g['n_rollouts']:>8d} {g['n_steps']:>8d} "
            f"{g['n_unique_anchors']:>8d} {g['max_cluster_size']:>6d} "
            f"{g['share_rate']:>6.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="actor jsonl file (one group per line)")
    parser.add_argument(
        "--min-step",
        type=int,
        default=0,
        help=(
            "drop records with step_index < N before clustering. Use --min-step 1 to "
            "factor out the structurally-guaranteed step-0 cluster and see GiGPO's actual signal."
        ),
    )
    parser.add_argument(
        "--by-step",
        action="store_true",
        help="break share rate down by step_index",
    )
    parser.add_argument(
        "--per-group",
        action="store_true",
        help="print one line per group (sorted by share rate desc)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the full summary as JSON",
    )
    parser.add_argument(
        "--top-groups",
        type=int,
        default=5,
        help="when not using --per-group, print this many top/bottom groups (default 5)",
    )
    args = parser.parse_args()

    summary = summarize(args.path, min_step=args.min_step)

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print(f"file: {summary['path']}")
    if args.min_step:
        print(f"filter: step_index >= {args.min_step}")
    print(
        f"groups: {summary['n_groups']}   "
        f"with non-trivial signal: {summary['n_groups_with_nontrivial_signal']}/{summary['n_groups']}   "
        f"total records: {summary['n_steps']}"
    )
    print(f"overall anchor share rate (cluster_size>=2):  {summary['overall_share_rate']:.3f}")
    print(f"mean per-group share rate:                    {summary['mean_per_group_share_rate']:.3f}")
    print()
    print("cluster size histogram (across all groups):")
    print(_format_hist(summary["cluster_size_histogram"]))
    print()

    if args.per_group:
        _print_per_group_table(summary["groups"])
    else:
        groups_sorted = sorted(summary["groups"], key=lambda g: g["share_rate"], reverse=True)
        k = min(args.top_groups, len(groups_sorted))
        print(f"top {k} groups by share rate:")
        for g in groups_sorted[:k]:
            print(
                f"  {g['group_id']:<32s} rollouts={g['n_rollouts']:>2d} "
                f"records={g['n_steps']:>3d} distinct={g['n_unique_anchors']:>3d} "
                f"max={g['max_cluster_size']:>2d} share={g['share_rate']:.3f}"
            )
        print(f"bottom {k} groups by share rate:")
        for g in groups_sorted[-k:]:
            print(
                f"  {g['group_id']:<32s} rollouts={g['n_rollouts']:>2d} "
                f"records={g['n_steps']:>3d} distinct={g['n_unique_anchors']:>3d} "
                f"max={g['max_cluster_size']:>2d} share={g['share_rate']:.3f}"
            )

    if args.by_step:
        print()
        print("per step_index breakdown:")
        by_step = per_step_share_rate(iter_groups(args.path), min_step=args.min_step)
        for si, row in by_step.items():
            print(
                f"  step={si:>2d}  n={int(row['n']):>4d}  "
                f"share_rate={row['share_rate']:.3f}  "
                f"mean_cluster={row['mean_cluster_size']:.2f}  "
                f"max_cluster={int(row['max_cluster_size']):>2d}"
            )


if __name__ == "__main__":
    main()
