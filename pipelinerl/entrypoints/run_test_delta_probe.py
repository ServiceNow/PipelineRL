"""Run the offline terminal test-delta replay prober against an env fleet."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import aiohttp
from omegaconf import OmegaConf

from pipelinerl.domains.terminal.load_tasks import load_problems
from pipelinerl.domains.terminal.test_delta_probe import (
    compute_summary,
    probe_rollout,
    read_jsonl,
    result_to_dict,
    select_probe_rows,
    summary_to_dict,
    task_index,
    write_results_jsonl,
)

logger = logging.getLogger(__name__)


def _load_terminal_tasks(config_path: Path) -> dict[str, dict[str, Any]]:
    if config_path.exists():
        cfg = OmegaConf.load(config_path)
        params = OmegaConf.to_container(getattr(cfg, "dataset_loader_params", {}), resolve=True)
    else:
        params = {}
    return task_index(load_problems(**params))


async def _run(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.audit_jsonl)
    tasks = _load_terminal_tasks(args.config)
    selected = select_probe_rows(rows, tasks, sample=args.sample, seed=args.seed, max_turns=args.max_turns)
    logger.info("selected %d/%d probeable audit rows", len(selected), len(rows))

    semaphore = asyncio.Semaphore(args.concurrency)
    base_url = args.fleet_base_url.rstrip("/")
    timeout = aiohttp.ClientTimeout(total=args.request_timeout, connect=10)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def post_json(route: str, payload: dict[str, Any], request_timeout: float) -> dict[str, Any]:
            async with session.post(f"{base_url}{route}", json=payload, timeout=request_timeout) as response:
                if response.status != 200:
                    text = await response.text()
                    raise RuntimeError(f"{route} -> HTTP {response.status}: {text}")
                return await response.json()

        async def run_one(row: dict[str, Any]):
            async with semaphore:
                return await probe_rollout(row, tasks[row["task_id"]], post_json, args.request_timeout)

        results = await asyncio.gather(*(run_one(row) for row in selected))

    write_results_jsonl(results, args.out)
    summary = compute_summary(results)
    summary_data = summary_to_dict(summary)
    args.summary_out.write_text(json.dumps(summary_data, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary_data, indent=2, sort_keys=True))

    if summary.replay_fidelity is not None and summary.replay_fidelity < args.min_fidelity:
        logger.error("replay fidelity %.3f below threshold %.3f", summary.replay_fidelity, args.min_fidelity)
        return 2
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_jsonl", nargs="+", type=Path, help="rollout_audit jsonl path(s) with commands")
    parser.add_argument("--fleet-base-url", required=True, help="base URL for one terminal env server, e.g. http://host:7777")
    parser.add_argument("--out", type=Path, required=True, help="output jsonl path")
    parser.add_argument("--summary-out", type=Path, default=None, help="summary JSON path; default <out>.summary.json")
    parser.add_argument("--config", type=Path, default=Path("conf/terminal.yaml"), help="terminal config used to load task set")
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--min-fidelity", type=float, default=0.8)
    args = parser.parse_args()
    if args.summary_out is None:
        args.summary_out = args.out.with_suffix(args.out.suffix + ".summary.json")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
