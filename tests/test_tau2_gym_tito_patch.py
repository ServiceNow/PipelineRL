import ast
import copy
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


@pytest.mark.skipif(
    "NEMO_GYM_SOURCE_CHECKOUT" not in os.environ,
    reason="requires the pinned NeMo Gym source checkout",
)
def test_patched_strict_tito_guards_execute_and_are_used(tmp_path: Path):
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
    apply_gym_tito_patch(checkout)

    target = checkout / _GYM_APP_PATH
    tree = ast.parse(target.read_text())
    helper_name = "_validate_pipelinerl_token_capture"
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    chat_completions = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "chat_completions"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == helper_name
        for node in ast.walk(chat_completions)
    )

    namespace = {}
    helper_module = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
    exec(compile(helper_module, str(target), "exec"), namespace)
    validate = namespace[helper_name]

    valid_chat = {
        "prompt_token_ids": [10, 11],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2},
    }
    valid_choice = {"token_ids": [20, 21]}
    valid_logprobs = [
        {"token": "token_id:20", "logprob": -0.1},
        {"token": "token_id:21", "logprob": -0.2},
    ]
    assert validate(valid_chat, valid_choice, valid_logprobs) == (
        [10, 11],
        [20, 21],
        [-0.1, -0.2],
    )

    cases = [
        (valid_chat, valid_choice, [{"token": "bad", "logprob": -0.1}], "malformed logprob"),
        ({"usage": valid_chat["usage"]}, valid_choice, valid_logprobs, "valid prompt_token_ids"),
        (valid_chat, {"token_ids": ["20", 21]}, valid_logprobs, "valid completion token_ids"),
        (valid_chat, {"token_ids": [20]}, valid_logprobs, "do not match logprob token IDs"),
        (
            {**valid_chat, "usage": {"prompt_tokens": 1, "completion_tokens": 2}},
            valid_choice,
            valid_logprobs,
            "usage.prompt_tokens",
        ),
        (
            {**valid_chat, "usage": {"prompt_tokens": 2, "completion_tokens": 1}},
            valid_choice,
            valid_logprobs,
            "usage.completion_tokens",
        ),
    ]
    for chat, choice, logprobs, message in cases:
        with pytest.raises(RuntimeError, match=message):
            validate(copy.deepcopy(chat), copy.deepcopy(choice), copy.deepcopy(logprobs))
