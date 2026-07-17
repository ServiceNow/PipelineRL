import hashlib
import os
import py_compile
import subprocess
from pathlib import Path

import pytest

from pipelinerl.domains.tau2.client import NEMO_GYM_SHA, NEMO_GYM_TITO_PATCH_SHA
from pipelinerl.entrypoints.run_tau2_gym import (
    _GYM_APP_PATH,
    _GYM_APP_POST_PATCH_SHA256,
    _GYM_TITO_PATCH_PATH,
    apply_gym_tito_patch,
    assert_dedicated_gym_checkout,
    assert_gym_checkout,
)


def test_gym_checkout_must_live_inside_its_run_directory(tmp_path: Path):
    run_dir = tmp_path / "run"
    dedicated = run_dir / "nemo-gym"

    assert_dedicated_gym_checkout(dedicated, run_dir)
    with pytest.raises(RuntimeError, match="must be dedicated"):
        assert_dedicated_gym_checkout(tmp_path / "shared-gym", run_dir)


def test_patch_artifact_matches_pinned_digest():
    assert hashlib.sha256(_GYM_TITO_PATCH_PATH.read_bytes()).hexdigest() == NEMO_GYM_TITO_PATCH_SHA


@pytest.mark.skipif(
    "NEMO_GYM_SOURCE_CHECKOUT" not in os.environ,
    reason="requires the pinned NeMo Gym source checkout",
)
def test_patch_applies_to_dedicated_pinned_checkout(tmp_path: Path):
    source = Path(os.environ["NEMO_GYM_SOURCE_CHECKOUT"]).resolve()
    run_dir = tmp_path / "run"
    checkout = run_dir / "nemo-gym"
    run_dir.mkdir()
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(source), str(checkout)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "checkout", "--quiet", NEMO_GYM_SHA],
        check=True,
    )

    assert_dedicated_gym_checkout(checkout, run_dir)
    assert_gym_checkout(checkout)
    apply_gym_tito_patch(checkout)
    apply_gym_tito_patch(checkout)

    target = checkout / _GYM_APP_PATH
    assert hashlib.sha256(target.read_bytes()).hexdigest() == _GYM_APP_POST_PATCH_SHA256
    py_compile.compile(target, doraise=True)
    status = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain=v1", "--untracked-files=no"],
        text=True,
    )
    assert [line[3:] for line in status.splitlines()] == [str(_GYM_APP_PATH)]

    target.write_text(target.read_text() + "\n")
    with pytest.raises(RuntimeError, match="does not match the pinned base or patched content"):
        apply_gym_tito_patch(checkout)
