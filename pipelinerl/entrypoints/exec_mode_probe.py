"""Exec-mode parity probe: pty vs subprocess on real tasks (workstream B4).

Replays recorded agent command sequences from a training run's streams through
TerminalSession under both exec modes and compares outcomes. Run it INSIDE an
eai CPU job with the fleet image/mounts (see exec_mode_probe_job.yaml); the
toolkit box is not the target environment.

Protocol per task:
  oracle leg   fresh session, no agent commands: test_initial_state must pass
               (init_ok) and test_final_state must not be solved yet.
  replay leg   for each sampled recorded rollout: fresh session, replay the
               recorded commands in order, then run the final verifier.
Parity = agreement between modes on per-command success bits and on the final
verifier verdict, plus per-command latency. The recorded run's verdict is a
soft reference only: fresh replays legitimately diverge from the original run
(network installs, timestamps), but the two modes see identical inputs.

Example:
  python -m pipelinerl.entrypoints.exec_mode_probe \\
    --exp-dir /mnt/llmd/results/exps/rafa/terminal/terminal_qwen35_9b_gspo_50 \\
    --tasks 8 --rollouts-per-task 2 --out exec_mode_probe_results.json
"""

import argparse
import hashlib
import json
import logging
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from pipelinerl.domains.terminal.environment import TerminalSession
from pipelinerl.domains.terminal.load_tasks import load_problems

logger = logging.getLogger(__name__)

# Strict wrapper: require the full tool-call envelope so tool OUTPUT that merely
# quotes a Qwen tool-call block is never parsed as a command to replay.
_COMMAND_RE = re.compile(
    r"<tool_call>\s*<function=bash>\s*<parameter=command>\n(.*?)\n</parameter>\s*</function>\s*</tool_call>",
    re.DOTALL,
)


@dataclass
class RecordedRollout:
    group_id: str
    rollout_index: int
    task_id: str
    commands: list[str]
    model_version: int
    reward: float
    verifier_pass: Optional[bool]
    pass_fraction: Optional[float]
    dropped: bool


@dataclass
class CommandResult:
    success: bool
    abort_kind: Optional[str]
    output_chars: int
    output_sha256: str
    output_tail: str
    seconds: float


@dataclass
class LegResult:
    started: bool
    init_ok: Optional[bool] = None
    commands: list[CommandResult] = field(default_factory=list)
    final_passed: Optional[bool] = None
    passed_tests: Optional[int] = None
    total_tests: Optional[int] = None
    final_abort_kind: Optional[str] = None
    wall_seconds: float = 0.0


def load_audit_index(exp_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted((exp_dir / "streams" / "rollout_audit").rglob("*.jsonl")):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                index[(rec["group_id"], rec["rollout_index"])] = rec
    return index


def load_recorded_rollouts(exp_dir: Path, min_commands: int) -> list[RecordedRollout]:
    audit = load_audit_index(exp_dir)
    longest_text: dict[tuple[str, int], tuple[int, str, int]] = {}
    for path in sorted((exp_dir / "streams" / "actor").rglob("*.jsonl")):
        with open(path) as f:
            for line in f:
                for text in json.loads(line):
                    md = text.get("metadata") or {}
                    key = (text["group_id"], md.get("rollout_index"))
                    if key[1] is None:
                        continue
                    body = text.get("text") or ""
                    prev = longest_text.get(key)
                    if prev is None or len(body) > prev[0]:
                        longest_text[key] = (len(body), body, int(md.get("model_version", -1)))

    rollouts = []
    for key, (_, body, model_version) in longest_text.items():
        audit_rec = audit.get(key)
        if audit_rec is None:
            continue
        commands = [c for c in _COMMAND_RE.findall(body)]
        if len(commands) < min_commands:
            continue
        rollouts.append(
            RecordedRollout(
                group_id=key[0],
                rollout_index=key[1],
                task_id=audit_rec["task_id"],
                commands=commands,
                model_version=model_version,
                reward=audit_rec.get("reward", float("nan")),
                verifier_pass=audit_rec.get("verifier_pass"),
                pass_fraction=audit_rec.get("pass_fraction"),
                dropped=bool(audit_rec.get("dropped")),
            )
        )
    return rollouts


def make_session(args: argparse.Namespace, exec_mode: str, cache_dir: Path) -> TerminalSession:
    return TerminalSession(
        bases_dir=Path(args.bases_dir),
        proot_bin=args.proot_bin,
        nameserver=os.environ.get("TERMINAL_NAMESERVER", "10.150.0.10"),
        max_observation_chars=4000,
        cache_dir=str(cache_dir),
        command_timeout=args.command_timeout,
        verifier_timeout=args.verifier_timeout,
        check_initial_state=True,
        # Match conf/terminal.yaml production values: same disk cap, and rootfs
        # retention so consecutive legs of the same task reuse the built tree
        # instead of evict/rebuild looping.
        max_session_disk_bytes=966367641,
        rootfs_retention_seconds=600.0,
        contamination_check=False,
        session_delta_isolation=False,
        exec_mode=exec_mode,
    )


def run_leg(args: argparse.Namespace, task: dict, exec_mode: str, cache_dir: Path, commands: list[str]) -> LegResult:
    session = make_session(args, exec_mode, cache_dir)
    t0 = time.time()
    result = LegResult(started=False)
    try:
        flags = session.start(task)
        result.started = bool(flags.get("started"))
        result.init_ok = bool(flags.get("init_ok")) if result.started else None
        if not result.started or not result.init_ok:
            # Production drops not_runnable tasks (started/init failures) without
            # stepping; mirror that so parity never replays into a broken session.
            return result
        for command in commands:
            c0 = time.time()
            step = session.exec(command)
            output = step.get("output") or ""
            result.commands.append(
                CommandResult(
                    success=bool(step["success"]),
                    abort_kind=step.get("abort_kind"),
                    output_chars=len(output),
                    output_sha256=hashlib.sha256(output.encode()).hexdigest()[:16],
                    output_tail=output[-160:],
                    seconds=round(time.time() - c0, 3),
                )
            )
            if step.get("abort_kind"):
                break
        finish = session.finish()
        result.final_passed = bool(finish["passed"])
        result.passed_tests = finish.get("passed_tests")
        result.total_tests = finish.get("total_tests")
        result.final_abort_kind = finish.get("abort_kind")
        return result
    finally:
        result.wall_seconds = round(time.time() - t0, 1)
        session.close(contamination_sample=False)


def command_agreement(a: LegResult, b: LegResult) -> Optional[float]:
    """Agreement on (success, abort_kind) over the FULL max length: a leg that
    stopped early disagrees on every command the other leg still ran."""
    n = max(len(a.commands), len(b.commands))
    if n == 0:
        return None
    same = 0
    for i in range(n):
        if i >= len(a.commands) or i >= len(b.commands):
            continue
        ca, cb = a.commands[i], b.commands[i]
        if (ca.success, ca.abort_kind) == (cb.success, cb.abort_kind):
            same += 1
    return same / n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, required=True, help="training run output dir with streams/")
    parser.add_argument("--tasks", type=int, default=8, help="distinct tasks to probe")
    parser.add_argument("--rollouts-per-task", type=int, default=2)
    parser.add_argument("--min-commands", type=int, default=3, help="skip rollouts with fewer recorded commands")
    parser.add_argument("--modes", default="pty,subprocess")
    parser.add_argument("--command-timeout", type=float, default=180.0)
    parser.add_argument("--verifier-timeout", type=float, default=180.0)
    parser.add_argument("--bases-dir", default=os.environ.get("TERMINAL_BASES_DIR", "/mnt/llmd/data/terminal_bases"))
    parser.add_argument("--proot-bin", default=os.environ.get("PROOT_BIN", "/mnt/llmd/data/terminal_bin/proot"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/exec_mode_probe_cache"))
    parser.add_argument("--out", type=Path, default=Path("exec_mode_probe_results.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="stop after extraction; print the plan")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    rollouts = load_recorded_rollouts(args.exp_dir, args.min_commands)
    by_task: dict[str, list[RecordedRollout]] = defaultdict(list)
    for r in rollouts:
        by_task[r.task_id].append(r)
    logger.info("extracted %d rollouts with commands across %d tasks", len(rollouts), len(by_task))

    problems = {p["task_id"]: p for p in load_problems()}
    candidate_tasks = sorted(t for t in by_task if t in problems)
    rng = random.Random(args.seed)
    rng.shuffle(candidate_tasks)

    def has_verdict_mix(task_id: str) -> bool:
        fractions = {r.pass_fraction for r in by_task[task_id] if not r.dropped}
        return len(fractions) > 1

    # Stable sort after the shuffle: tasks with both pass and fail rollouts first,
    # so the fail path gets replay coverage whenever the data offers it.
    candidate_tasks.sort(key=lambda t: not has_verdict_mix(t))
    chosen = candidate_tasks[: args.tasks]

    plan = []
    for task_id in chosen:
        # Verdict mix: alternate the highest and lowest pass_fraction rollouts so
        # parity is checked on both pass and fail paths, not just easy successes.
        undropped = [r for r in by_task[task_id] if not r.dropped]
        ranked = sorted(undropped or by_task[task_id], key=lambda r: -(r.pass_fraction or 0.0))
        sampled: list[RecordedRollout] = []
        lo, hi = 0, len(ranked) - 1
        while len(sampled) < args.rollouts_per_task and lo <= hi:
            sampled.append(ranked[lo] if len(sampled) % 2 == 0 else ranked[hi])
            if len(sampled) % 2 == 1:
                lo += 1
            else:
                hi -= 1
        plan.append((task_id, sampled))

    for task_id, sampled in plan:
        logger.info(
            "plan: task %s -> %d rollouts, commands=%s pass_fractions=%s",
            task_id, len(sampled), [len(r.commands) for r in sampled],
            [r.pass_fraction for r in sampled],
        )
    if args.dry_run:
        return

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"exp_dir": str(args.exp_dir), "modes": modes, "tasks": {}}
    for task_id, sampled in plan:
        task = problems[task_id]
        task_result: dict[str, Any] = {"oracle": {}, "rollouts": []}
        for mode in modes:
            task_result["oracle"][mode] = asdict(run_leg(args, task, mode, args.cache_dir, commands=[]))
        for rollout in sampled:
            entry: dict[str, Any] = {
                "group_id": rollout.group_id,
                "rollout_index": rollout.rollout_index,
                "model_version": rollout.model_version,
                "n_commands": len(rollout.commands),
                "recorded": {
                    "reward": rollout.reward,
                    "verifier_pass": rollout.verifier_pass,
                    "pass_fraction": rollout.pass_fraction,
                    "dropped": rollout.dropped,
                },
                "replay": {},
            }
            legs = {}
            for mode in modes:
                legs[mode] = run_leg(args, task, mode, args.cache_dir, rollout.commands)
                entry["replay"][mode] = asdict(legs[mode])
            if len(modes) == 2:
                a, b = (legs[m] for m in modes)
                entry["parity"] = {
                    "start_agree": (a.started, a.init_ok) == (b.started, b.init_ok),
                    "command_count_agree": len(a.commands) == len(b.commands),
                    "command_success_agreement": command_agreement(a, b),
                    "final_verdict_agree": (a.final_passed == b.final_passed),
                    "tests_agree": (a.passed_tests, a.total_tests) == (b.passed_tests, b.total_tests),
                }
            task_result["rollouts"].append(entry)
            logger.info("task %s rollout %s done: %s", task_id, rollout.rollout_index, entry.get("parity"))
        results["tasks"][task_id] = task_result

    if len(modes) == 2:
        rollout_entries = [e for t in results["tasks"].values() for e in t["rollouts"] if "parity" in e]
        agreements = [e["parity"]["command_success_agreement"] for e in rollout_entries]
        agreements = [a for a in agreements if a is not None]
        latencies = {
            mode: [c["seconds"] for t in results["tasks"].values() for e in t["rollouts"]
                   for c in e["replay"][mode]["commands"]]
            for mode in modes
        }
        results["summary"] = {
            "n_tasks": len(results["tasks"]),
            "n_rollouts": len(rollout_entries),
            "mean_command_success_agreement": sum(agreements) / len(agreements) if agreements else None,
            "final_verdict_agreement": (
                sum(e["parity"]["final_verdict_agree"] for e in rollout_entries) / len(rollout_entries)
                if rollout_entries else None
            ),
            "mean_command_seconds": {
                mode: (sum(vals) / len(vals) if vals else None) for mode, vals in latencies.items()
            },
        }
        logger.info("SUMMARY: %s", json.dumps(results["summary"], indent=2))

    args.out.write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
