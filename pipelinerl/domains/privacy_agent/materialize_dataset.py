import argparse
import json
from pathlib import Path

from .dataset import DEFAULT_DATASET_NAME, load_problems


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize a privacy_agent dataset slice into a single JSONL file.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Named dataset slice to materialize. Defaults to seed20.",
    )
    parser.add_argument(
        "--annotations-path",
        required=True,
        help="Path to the chain-review annotations JSON file.",
    )
    parser.add_argument(
        "--curated-path",
        required=True,
        help="Path to the curated chains JSONL file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Maximum number of accepted chains to materialize.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap applied after materialization.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    problems = load_problems(
        dataset_names=[args.dataset_name],
        annotations_path=args.annotations_path,
        curated_path=args.curated_path,
        sample_size=args.sample_size,
        max_examples=args.max_examples,
    )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in problems:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")

    print(f"Wrote {len(problems)} problems to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
