import ast
import copy
import hashlib
import importlib.util
import json
import os
import py_compile
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipelinerl.domains.tau2.client import (
    NEMO_GYM_SHA,
    NEMO_GYM_TITO_PATCH_SHA,
    TAU2_RUNTIME_SHA,
)
from pipelinerl.entrypoints.run_tau2_gym import (
    _GYM_APP_PATH,
    _GYM_APP_POST_PATCH_SHA256,
    _GYM_PATCH_TARGETS,
    _GYM_TAU2_REQUIREMENTS_PATH,
    _GYM_TAU2_SOURCE_PATH,
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
        if target.suffix == ".py":
            py_compile.compile(target, doraise=True)
    assert _GYM_PATCH_TARGETS[_GYM_APP_PATH][1] == _GYM_APP_POST_PATCH_SHA256

    requirements = (checkout / _GYM_TAU2_REQUIREMENTS_PATH).read_text().splitlines()
    assert (
        "tau2[knowledge] @ git+https://github.com/bxyu-nvidia/tau2-bench@"
        f"{TAU2_RUNTIME_SHA}"
    ) in requirements

    status = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain=v1", "--untracked-files=no"],
        text=True,
    )
    assert {line[3:] for line in status.splitlines()} == {
        str(path) for path in _GYM_PATCH_TARGETS
    }

    for path in _GYM_PATCH_TARGETS:
        target = checkout / path
        original = target.read_bytes()
        target.write_bytes(original + b"\n")
        with pytest.raises(RuntimeError, match="patch targets do not match"):
            apply_gym_tito_patch(checkout)
        target.write_bytes(original)


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
def test_patched_strict_tito_and_provenance_guards_execute_and_are_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
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

    source_target = checkout / _GYM_TAU2_SOURCE_PATH
    source_spec = importlib.util.spec_from_file_location("patched_tau2_source", source_target)
    assert source_spec is not None and source_spec.loader is not None
    source_module = importlib.util.module_from_spec(source_spec)
    source_spec.loader.exec_module(source_module)

    def install_direct_url(payload: str | None) -> None:
        monkeypatch.setattr(
            source_module,
            "distribution",
            lambda name: SimpleNamespace(
                read_text=lambda filename: payload
            )
            if name == "tau2"
            else None,
        )

    pinned_vcs_info = {
        "vcs": "git",
        "commit_id": TAU2_RUNTIME_SHA,
        "requested_revision": TAU2_RUNTIME_SHA,
    }
    install_direct_url(json.dumps({"vcs_info": pinned_vcs_info}))
    source_module.assert_tau2_runtime_revision()

    def missing_distribution(name: str):
        raise source_module.PackageNotFoundError(name)

    monkeypatch.setattr(source_module, "distribution", missing_distribution)
    with pytest.raises(RuntimeError, match="not installed"):
        source_module.assert_tau2_runtime_revision()

    invalid_direct_urls = [
        (None, "no PEP 610"),
        ("{", "malformed"),
        (json.dumps({}), "runtime provenance"),
        (
            json.dumps(
                {
                    "vcs_info": {
                        "vcs": "git",
                        "requested_revision": TAU2_RUNTIME_SHA,
                    }
                }
            ),
            "runtime provenance",
        ),
        (
            json.dumps(
                {
                    "vcs_info": {
                        "vcs": "git",
                        "commit_id": TAU2_RUNTIME_SHA,
                    }
                }
            ),
            "runtime provenance",
        ),
        (
            json.dumps({"vcs_info": {**pinned_vcs_info, "vcs": "hg"}}),
            "runtime provenance",
        ),
        (
            json.dumps({"vcs_info": {**pinned_vcs_info, "commit_id": "0" * 40}}),
            "runtime provenance",
        ),
        (
            json.dumps(
                {
                    "vcs_info": {
                        **pinned_vcs_info,
                        "requested_revision": "bxyu/nemo_gym_stable",
                    }
                }
            ),
            "runtime provenance",
        ),
    ]
    for payload, message in invalid_direct_urls:
        install_direct_url(payload)
        with pytest.raises(RuntimeError, match=message):
            source_module.assert_tau2_runtime_revision()

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
    runtime_attestation = next(
        node
        for node in tau_tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "assert_tau2_runtime_revision"
    )
    tau2_import_lines = [
        node.lineno
        for node in tau_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("tau2.")
    ]
    assert tau2_import_lines
    assert runtime_attestation.lineno < min(tau2_import_lines)

    tau_agent = next(
        node
        for node in tau_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "Tau2Agent"
    )
    run_method = next(
        node
        for node in tau_agent.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "run"
    )
    seed_move = next(
        node
        for node in ast.walk(run_method)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "input_items_1"
    )
    assert ast.unparse(seed_move.value) == "output_items[:1]"
    assert any(
        ast.unparse(argument) == "output_items[1:]"
        for node in ast.walk(run_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "split_responses_input_output_items"
        for argument in node.args
    )
    count_loop = next(
        node
        for node in ast.walk(run_method)
        if isinstance(node, ast.For)
        and any(
            isinstance(child, ast.AugAssign)
            and isinstance(child.target, ast.Name)
            and child.target.id == "num_agent_calls"
            for child in ast.walk(node)
        )
    )
    assert ast.unparse(count_loop.iter) == "result.messages"
    assert any(
        isinstance(node, ast.Compare)
        and ast.unparse(node) == "message.role == 'assistant'"
        for node in ast.walk(count_loop)
    )

    adapter_tree = ast.parse(
        (
            Path(__file__).parents[1]
            / "pipelinerl/domains/tau2/rollouts.py"
        ).read_text()
    )
    extract_policy_calls = _function(
        adapter_tree,
        "_extract_policy_calls",
    )
    expected_calls = next(
        node
        for node in ast.walk(extract_policy_calls)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "expected_calls"
            for target in node.targets
        )
    )
    assert ast.unparse(expected_calls.value) == (
        "max(run_response.num_agent_calls - 1, 0)"
    )

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
