"""Run offline terminal twin mining over rollout audit jsonl files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipelinerl.domains.terminal.twin_mining import (
    mine_twins,
    read_jsonl,
    write_twins_jsonl,
    yield_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_jsonl", nargs="+", type=Path, help="rollout_audit jsonl path(s) with commands")
    parser.add_argument("--out", type=Path, required=True, help="output twins jsonl path")
    parser.add_argument("--min-prefix", type=int, default=3)
    args = parser.parse_args()

    rows = read_jsonl(args.audit_jsonl)
    twins = mine_twins(rows, min_prefix=args.min_prefix)
    write_twins_jsonl(twins, args.out)
    print(json.dumps(yield_report(rows, twins), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
