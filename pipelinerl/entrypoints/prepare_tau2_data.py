"""Prepare attested Tau2 data from pinned source checkouts."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any

from pipelinerl.domains.tau2.client import NEMO_GYM_SHA, TAU2_DATA_SHA
from pipelinerl.domains.tau2.dataset import write_tau2_prepared_data


logger = logging.getLogger(__name__)


def _git_output(checkout: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_checkout(checkout: Path, revision: str, label: str) -> None:
    if not checkout.is_dir():
        raise ValueError(f"{label} checkout does not exist: {checkout}")
    head = _git_output(checkout, "rev-parse", "HEAD")
    if head != revision:
        raise ValueError(f"{label} checkout is {head}, expected {revision}")
    status = _git_output(checkout, "status", "--short", "--untracked-files=no")
    if status:
        raise ValueError(f"{label} checkout has tracked changes:\n{status}")


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "pipelinerl_pinned_tau2_prepare_utils",
        path,
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load pinned Gym normalizer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_upstream_normalizer(
    nemo_gym_checkout: Path,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    path = (
        nemo_gym_checkout
        / "benchmarks"
        / "tau2"
        / "prepare_utils"
        / "__init__.py"
    )
    if not path.is_file():
        raise ValueError(f"Pinned Gym normalizer does not exist: {path}")
    normalize_row = getattr(_load_module(path), "normalize_row", None)
    if not callable(normalize_row):
        raise ValueError("Pinned Gym normalize_row is not callable")
    return normalize_row


def prepare_tau2_data(
    tau2_source_checkout: Path,
    nemo_gym_source_checkout: Path,
    output_dir: Path,
) -> Path:
    _validate_checkout(tau2_source_checkout, TAU2_DATA_SHA, "Tau2 data source")
    _validate_checkout(nemo_gym_source_checkout, NEMO_GYM_SHA, "NeMo Gym source")
    normalizer = _load_upstream_normalizer(nemo_gym_source_checkout)

    dump_script = tau2_source_checkout / "dump_nemo_gym_data.sh"
    if not dump_script.is_file():
        raise ValueError(f"Tau2 dump script does not exist: {dump_script}")

    provision_env = os.environ.copy()
    provision_env["UV_FROZEN"] = "1"
    subprocess.run(
        ["uv", "sync", "--frozen", "--extra", "knowledge"],
        cwd=tau2_source_checkout,
        env=provision_env,
        check=True,
    )
    dump_python = tau2_source_checkout / ".venv" / "bin" / "python"
    if not dump_python.is_file():
        raise ValueError(f"Tau2 dump Python does not exist: {dump_python}")

    with TemporaryDirectory(prefix="pipelinerl-tau2-data-") as temp_dir:
        raw_root = Path(temp_dir) / "raw"
        dump_env = provision_env.copy()
        dump_env["TAU2_SKIP_UV_SYNC"] = "1"
        dump_env["TAU2_DUMP_PYTHON"] = str(dump_python)
        command = ["bash", str(dump_script)]
        for dataset in ("airline", "retail", "telecom"):
            command.extend(["--dataset", dataset])
        command.extend(["--output-root", str(raw_root)])
        subprocess.run(
            command,
            cwd=tau2_source_checkout,
            env=dump_env,
            check=True,
        )
        _validate_checkout(
            tau2_source_checkout,
            TAU2_DATA_SHA,
            "Tau2 data source",
        )
        _validate_checkout(
            nemo_gym_source_checkout,
            NEMO_GYM_SHA,
            "NeMo Gym source",
        )
        manifest_path = write_tau2_prepared_data(
            raw_root,
            output_dir,
            normalizer,
        )

    logger.info("Prepared attested Tau2 data at %s", manifest_path)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tau2-source-checkout",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--nemo-gym-source-checkout",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    prepare_tau2_data(
        args.tau2_source_checkout.resolve(),
        args.nemo_gym_source_checkout.resolve(),
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
