import argparse
from pathlib import Path

from pipelinerl.domains.tau2.prerun import (
    PreRunManifest,
    PreRunSpec,
    finalize_prerun_manifest,
    require_ready_manifest,
)


def finalize(spec_path: Path, evidence_dir: Path, output_path: Path) -> PreRunManifest:
    spec = PreRunSpec.model_validate_json(spec_path.read_text())
    manifest = finalize_prerun_manifest(spec, evidence_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    require_ready_manifest(manifest)
    return manifest


def validate(manifest_path: Path) -> PreRunManifest:
    manifest = PreRunManifest.model_validate_json(manifest_path.read_text())
    require_ready_manifest(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize or validate Tau2/Gemma pre-run gate evidence",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--spec", type=Path, required=True)
    finalize_parser.add_argument("--evidence-dir", type=Path, required=True)
    finalize_parser.add_argument("--output", type=Path, required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--manifest", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "finalize":
        finalize(args.spec, args.evidence_dir, args.output)
    else:
        validate(args.manifest)


if __name__ == "__main__":
    main()
