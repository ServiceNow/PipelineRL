import argparse
import json
from pathlib import Path

from .dataset import DEFAULT_DATASET_NAME, load_problems
from .settings import PrivacyAgentSettings


def _settings_from_args(args: argparse.Namespace) -> PrivacyAgentSettings:
    default_settings = PrivacyAgentSettings()
    return PrivacyAgentSettings(
        annotations_path=(
            Path(args.annotations_path).expanduser()
            if args.annotations_path
            else default_settings.annotations_path
        ),
        curated_path=(
            Path(args.curated_path).expanduser()
            if args.curated_path
            else default_settings.curated_path
        ),
        sample_size=args.sample_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the privacy_agent seed dataset to JSONL.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--annotations-path", default=None)
    parser.add_argument("--curated-path", default=None)
    args = parser.parse_args()

    settings = _settings_from_args(args)
    problems = load_problems(
        [args.dataset_name],
        **settings.dataset_loader_kwargs(),
    )

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for problem in problems:
            handle.write(json.dumps(problem, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
