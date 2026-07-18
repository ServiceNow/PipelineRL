import ast
import copy
import hashlib
import os
import py_compile
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipelinerl.domains.tau2.client import NEMO_GYM_SHA, NEMO_GYM_TITO_PATCH_SHA
from pipelinerl.entrypoints.run_tau2_gym import (
    _GYM_APP_PATH,
    _GYM_APP_POST_PATCH_SHA256,
    _GYM_PATCH_TARGETS,
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

    for path, (_, post_sha) in _GYM_PATCH_TARGETS.items():
        target = checkout / path
        assert hashlib.sha256(target.read_bytes()).hexdigest() == post_sha
        py_compile.compile(target, doraise=True)
    assert _GYM_PATCH_TARGETS[_GYM_APP_PATH][1] == _GYM_APP_POST_PATCH_SHA256

    status = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain=v1", "--untracked-files=no"],
        text=True,
    )
    assert {line[3:] for line in status.splitlines()} == {
        str(path) for path in _GYM_PATCH_TARGETS
    }

    target = checkout / _GYM_APP_PATH
    target.write_text(target.read_text() + "\n")
    with pytest.raises(RuntimeError, match="patch targets do not match"):
        apply_gym_tito_patch(checkout)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


@pytest.mark.skipif(
    "NEMO_GYM_SOURCE_CHECKOUT" not in os.environ,
    reason="requires the pinned NeMo Gym source checkout",
)
def test_patched_strict_tito_and_provenance_guards_execute_and_are_used(tmp_path: Path):
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
    token_helper_name = "_validate_pipelinerl_token_capture"
    version_helper_name = "_validate_pipelinerl_model_versions"
    token_helper = _function(tree, token_helper_name)
    version_helper = _function(tree, version_helper_name)
    chat_completions = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "chat_completions"
    )
    called_names = {
        node.func.id
        for node in ast.walk(chat_completions)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {token_helper_name, version_helper_name} <= called_names
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_chat_completion_with_headers"
        for node in ast.walk(chat_completions)
    )

    namespace = {
        "PIPELINERL_VERSION_START_HEADER": "X-PipelineRL-Version-Start",
        "PIPELINERL_VERSION_END_HEADER": "X-PipelineRL-Version-End",
    }
    helper_module = ast.fix_missing_locations(
        ast.Module(body=[token_helper, version_helper], type_ignores=[])
    )
    exec(compile(helper_module, str(target), "exec"), namespace)
    validate_tokens = namespace[token_helper_name]
    validate_versions = namespace[version_helper_name]

    valid_chat = {
        "prompt_token_ids": [10, 11],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2},
    }
    valid_choice = {"token_ids": [20, 21]}
    valid_logprobs = [
        {"token": "token_id:20", "logprob": -0.1},
        {"token": "token_id:21", "logprob": -0.2},
    ]
    assert validate_tokens(valid_chat, valid_choice, valid_logprobs) == (
        [10, 11],
        [20, 21],
        [-0.1, -0.2],
    )
    assert validate_versions(
        {
            "X-PipelineRL-Version-Start": "8",
            "X-PipelineRL-Version-End": "10",
        },
        "http://actor:8000/v1/",
    ) == (8, 10, "http://actor:8000/v1")

    token_cases = [
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
    for chat, choice, logprobs, message in token_cases:
        with pytest.raises(RuntimeError, match=message):
            validate_tokens(copy.deepcopy(chat), copy.deepcopy(choice), copy.deepcopy(logprobs))

    version_cases = [
        ({"X-PipelineRL-Version-End": "2"}, "missing valid"),
        (
            {
                "X-PipelineRL-Version-Start": "-1",
                "X-PipelineRL-Version-End": "2",
            },
            "negative",
        ),
        (
            {
                "X-PipelineRL-Version-Start": "3",
                "X-PipelineRL-Version-End": "2",
            },
            "decreased",
        ),
    ]
    for headers, message in version_cases:
        with pytest.raises(RuntimeError, match=message):
            validate_versions(headers, "http://actor:8000/v1")

    openai_tree = ast.parse((checkout / "nemo_gym/openai_utils.py").read_text())
    client_class = next(
        node
        for node in openai_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NeMoGymAsyncOpenAI"
    )
    assert any(
        isinstance(node, ast.AsyncFunctionDef)
        and node.name == "create_chat_completion_with_headers"
        for node in client_class.body
    )

    converter_tree = ast.parse((checkout / "nemo_gym/responses_converter.py").read_text())
    converter_source = ast.unparse(converter_tree)
    for field in ("model_version_start", "model_version_end", "policy_endpoint"):
        assert field in converter_source

    tau_target = checkout / "responses_api_agents/tau2/app.py"
    tau_tree = ast.parse(tau_target.read_text())
    attach_helper = _function(tau_tree, "_attach_pipelinerl_provenance")
    tau_namespace = {
        "Mapping": Mapping,
        "_PIPELINERL_PROVENANCE_FIELDS": (
            "model_version_start",
            "model_version_end",
            "policy_endpoint",
        ),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[attach_helper], type_ignores=[])),
            str(tau_target),
            "exec",
        ),
        tau_namespace,
    )
    attach = tau_namespace["_attach_pipelinerl_provenance"]
    raw_message = {
        "prompt_token_ids": [10],
        "generation_token_ids": [20],
        "model_version_start": 8,
        "model_version_end": 10,
        "policy_endpoint": "http://actor:8000/v1",
    }
    message = SimpleNamespace(
        role="assistant",
        raw_data={"choices": [{"message": raw_message}]},
    )
    item = SimpleNamespace(
        prompt_token_ids=[10],
        generation_token_ids=[20],
        model_version_start=None,
        model_version_end=None,
        policy_endpoint=None,
    )
    attach([message], [item])
    assert (item.model_version_start, item.model_version_end, item.policy_endpoint) == (
        8,
        10,
        "http://actor:8000/v1",
    )

    del raw_message["model_version_end"]
    with pytest.raises(RuntimeError, match="missing PipelineRL provenance"):
        attach([message], [item])
