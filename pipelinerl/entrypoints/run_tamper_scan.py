"""Scan terminal rollout audits for verifier-tampering signals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipelinerl.domains.terminal.tamper_scan import read_jsonl, scan_rows, write_evidence_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_jsonl", nargs="+", type=Path, help="rollout_audit jsonl path(s) with commands")
    parser.add_argument("--out", type=Path, required=True, help="matched-command evidence jsonl path")
    parser.add_argument("--summary-out", type=Path, default=None, help="summary JSON path; default <out>.summary.json")
    args = parser.parse_args()
    if args.summary_out is None:
        args.summary_out = args.out.with_suffix(args.out.suffix + ".summary.json")

    evidence, summary = scan_rows(read_jsonl(args.audit_jsonl))
    write_evidence_jsonl(evidence, args.out)
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
