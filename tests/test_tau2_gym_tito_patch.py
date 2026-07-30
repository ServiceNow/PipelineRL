import ast
import asyncio
import copy
import hashlib
import importlib.util
import json
import math
import os
import py_compile
import subprocess
from collections.abc import Mapping
from contextvars import ContextVar
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

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


    auxiliary_names = {
        "_JudgeOutputInvalid",
        "_JudgeRewardInvalid",
        "_require_auxiliary_state",
        "_validate_token_count",
        "_response_metadata",
        "_validate_judge_content",
        "_next_user_call_position",
        "_append_auxiliary_call",
        "_pipelinerl_auxiliary_generate",
        "_pipelinerl_evaluate_nl_assertions",
        "_pipelinerl_calculate_nl_reward",
        "_judge_invalid_marker",
        "_rewardless_boundary_result",
        "_new_auxiliary_state",
        "_judge_sampling",
    }
    auxiliary_nodes = [
        node
        for node in tau_tree.body
        if isinstance(
            node,
            (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        )
        and node.name in auxiliary_names
    ]
    assert {node.name for node in auxiliary_nodes} == auxiliary_names

    class FakeRewardInfo:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeRewardTypeValue(str):
        @property
        def value(self):
            return str(self)

    auxiliary_context = ContextVar("test_auxiliary_state", default=None)
    judge_attempt_context = ContextVar("test_judge_attempt", default=None)
    auxiliary_namespace = {
        "Any": Any,
        "Mapping": Mapping,
        "ContextVar": ContextVar,
        "asyncio": asyncio,
        "hashlib": hashlib,
        "json": json,
        "math": math,
        "perf_counter": perf_counter,
        "NLAssertionsEvaluator": object,
        "Task": object,
        "SimulationRun": object,
        "RewardInfo": FakeRewardInfo,
        "RewardType": SimpleNamespace(
            NL_ASSERTION=FakeRewardTypeValue("NL_ASSERTION")
        ),
        "_AUXILIARY_RUN_STATE": auxiliary_context,
        "_JUDGE_ATTEMPT_STATE": judge_attempt_context,
        "_PIPELINERL_JUDGE_MARKER_KEY": "pipelinerl_judge_reward_invalid",
        "_PIPELINERL_JUDGE_REASON": "judge_reward_invalid",
        "_PIPELINERL_JUDGE_STAGE": "judge",
        "_USER_CALL_NAME": "user_simulator_response",
        "_JUDGE_CALL_NAME": "nl_assertions_eval",
        "_ORIGINAL_USER_GENERATE": None,
        "_ORIGINAL_JUDGE_GENERATE": None,
        "_ORIGINAL_NL_EVALUATE": None,
        "_ORIGINAL_NL_CALCULATE": None,
    }
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=auxiliary_nodes, type_ignores=[])
            ),
            str(tau_target),
            "exec",
        ),
        auxiliary_namespace,
    )

    validate_judge = auxiliary_namespace["_validate_judge_content"]
    judge_output_error = auxiliary_namespace["_JudgeOutputInvalid"]
    valid_duplicate_output = json.dumps(
        {
            "results": [
                {
                    "expectedOutcome": "same assertion",
                    "metExpectation": True,
                    "reasoning": "first occurrence",
                },
                {
                    "expectedOutcome": "same assertion",
                    "metExpectation": False,
                    "reasoning": "second occurrence",
                },
            ]
        }
    )
    validate_judge(
        valid_duplicate_output,
        ("same assertion", "same assertion"),
    )

    invalid_judge_outputs = [
        ("", ("a",), "missing_content"),
        ("not json", ("a",), "invalid_json"),
        (json.dumps([]), ("a",), "invalid_object"),
        (json.dumps({}), ("a",), "missing_results"),
        (json.dumps({"results": {}}), ("a",), "invalid_results"),
        (json.dumps({"results": []}), ("a",), "result_count_mismatch"),
        (
            json.dumps({"results": ["not-an-object"]}),
            ("a",),
            "invalid_result",
        ),
        (
            json.dumps(
                {
                    "results": [
                        {
                            "expectedOutcome": "a",
                            "metExpectation": True,
                            "reasoning": "ok",
                        },
                        {
                            "expectedOutcome": "b",
                            "metExpectation": True,
                            "reasoning": "extra",
                        },
                    ]
                }
            ),
            ("a",),
            "result_count_mismatch",
        ),
        (
            json.dumps(
                {
                    "results": [
                        {
                            "expectedOutcome": "b",
                            "metExpectation": True,
                            "reasoning": "reordered",
                        },
                        {
                            "expectedOutcome": "a",
                            "metExpectation": True,
                            "reasoning": "reordered",
                        },
                    ]
                }
            ),
            ("a", "b"),
            "expected_outcome_mismatch",
        ),
        (
            json.dumps(
                {
                    "results": [
                        {
                            "expectedOutcome": "a",
                            "metExpectation": True,
                            "reasoning": "ok",
                        },
                        {
                            "expectedOutcome": "a",
                            "metExpectation": True,
                            "reasoning": "duplicate",
                        },
                    ]
                }
            ),
            ("a", "b"),
            "expected_outcome_mismatch",
        ),
        (
            json.dumps(
                {
                    "results": [
                        {
                            "expectedOutcome": "wrong",
                            "metExpectation": True,
                            "reasoning": "wrong",
                        }
                    ]
                }
            ),
            ("a",),
            "expected_outcome_mismatch",
        ),
    ]
    invalid_judge_outputs.extend(
        (
            json.dumps(
                {
                    "results": [
                        {
                            "expectedOutcome": "a",
                            "metExpectation": value,
                            "reasoning": "wrong type",
                        }
                    ]
                }
            ),
            ("a",),
            "invalid_met_expectation",
        )
        for value in (0, 1, "true")
    )
    invalid_judge_outputs.append(
        (
            json.dumps(
                {
                    "results": [
                        {
                            "expectedOutcome": "a",
                            "metExpectation": True,
                            "reasoning": " ",
                        }
                    ]
                }
            ),
            ("a",),
            "missing_reasoning",
        )
    )
    for content, expected_outcomes, detail_code in invalid_judge_outputs:
        with pytest.raises(judge_output_error, match=detail_code):
            validate_judge(content, expected_outcomes)

    wrapper = auxiliary_namespace["_pipelinerl_auxiliary_generate"]

    def message(
        *,
        content: str | None,
        model: str = "qwen-user-primary",
        finish_reason: str = "stop",
        usage: Mapping[str, int] | None = None,
        tool_calls: list[Any] | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            content=content,
            tool_calls=tool_calls,
            usage=usage
            if usage is not None
            else {"prompt_tokens": 11, "completion_tokens": 7},
            raw_data={
                "model": model,
                "choices": [{"finish_reason": finish_reason}],
            },
        )

    def run_async(coro):
        return asyncio.run(coro)

    base_state = {
        "user_api_base": "http://user-proxy/v1",
        "judge_api_base": "http://judge-proxy/v1",
        "user_model_name": "qwen-user-primary",
        "judge_model_name": "qwen-judge-secondary",
        "approved_model_aliases": (
            "qwen-user-primary",
            "qwen-judge-secondary",
        ),
        "judge_temperature": 0.6,
        "judge_top_p": 0.95,
        "judge_top_k": 20,
        "judge_seed": 17,
        "judge_initial_max_tokens": 64,
        "judge_retry_max_tokens": 128,
        "auxiliary_model_timeout_s": 1.0,
        "next_user_call_index": 1,
        "auxiliary_model_calls": [],
    }
    valid_single_output = json.dumps(
        {
            "results": [
                {
                    "expectedOutcome": "a",
                    "metExpectation": True,
                    "reasoning": "supported",
                }
            ]
        }
    )
    captured_judge_requests = []

    async def successful_judge_generate(**kwargs):
        captured_judge_requests.append(kwargs)
        return message(content=valid_single_output)

    auxiliary_namespace["_ORIGINAL_JUDGE_GENERATE"] = successful_judge_generate
    state = copy.deepcopy(base_state)
    auxiliary_token = auxiliary_context.set(state)
    attempt_token = judge_attempt_context.set(
        {
            "expected_outcomes": ("a",),
            "attempt_index": 1,
            "completion_budget": 64,
        }
    )
    try:
        run_async(
            wrapper(
                model="ignored-upstream-judge",
                messages=[],
                call_name="nl_assertions_eval",
            )
        )
    finally:
        judge_attempt_context.reset(attempt_token)
        auxiliary_context.reset(auxiliary_token)
    assert captured_judge_requests[0]["model"] == "qwen-judge-secondary"
    assert captured_judge_requests[0]["api_base"] == "http://judge-proxy/v1"
    assert {
        key: captured_judge_requests[0][key]
        for key in ("temperature", "top_p", "seed", "max_tokens")
    } == {
        "temperature": 0.6,
        "top_p": 0.95,
        "seed": 17,
        "max_tokens": 64,
    }
    assert state["auxiliary_model_calls"] == [
        {
            "role": "judge",
            "call_index": 1,
            "attempt_index": 1,
            "requested_model_alias": "qwen-judge-secondary",
            "response_model_alias": "qwen-user-primary",
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "latency_s": state["auxiliary_model_calls"][0]["latency_s"],
            "finish_reason": "stop",
            "completion_budget": 64,
            "output_character_count": len(valid_single_output),
            "output_sha256": hashlib.sha256(
                valid_single_output.encode("utf-8")
            ).hexdigest(),
            "failure_code": None,
        }
    ]

    async def length_limited_judge_generate(**kwargs):
        return message(
            content=valid_single_output,
            finish_reason="length",
        )

    auxiliary_namespace[
        "_ORIGINAL_JUDGE_GENERATE"
    ] = length_limited_judge_generate
    length_state = copy.deepcopy(base_state)
    auxiliary_token = auxiliary_context.set(length_state)
    attempt_token = judge_attempt_context.set(
        {
            "expected_outcomes": ("a",),
            "attempt_index": 1,
            "completion_budget": 64,
        }
    )
    try:
        with pytest.raises(
            judge_output_error,
            match="finish_reason_length",
        ):
            run_async(
                wrapper(
                    model="ignored",
                    messages=[],
                    call_name="nl_assertions_eval",
                )
            )
    finally:
        judge_attempt_context.reset(attempt_token)
        auxiliary_context.reset(auxiliary_token)
    assert length_state["auxiliary_model_calls"][0][
        "failure_code"
    ] == "finish_reason_length"

    async def unapproved_alias_generate(**kwargs):
        return message(content=valid_single_output, model="rogue-model")

    auxiliary_namespace["_ORIGINAL_JUDGE_GENERATE"] = unapproved_alias_generate
    state = copy.deepcopy(base_state)
    auxiliary_token = auxiliary_context.set(state)
    attempt_token = judge_attempt_context.set(
        {
            "expected_outcomes": ("a",),
            "attempt_index": 1,
            "completion_budget": 64,
        }
    )
    try:
        with pytest.raises(RuntimeError, match="unapproved alias"):
            run_async(
                wrapper(
                    model="ignored",
                    messages=[],
                    call_name="nl_assertions_eval",
                )
            )
    finally:
        judge_attempt_context.reset(attempt_token)
        auxiliary_context.reset(auxiliary_token)
    assert state["auxiliary_model_calls"] == []

    auxiliary_token = auxiliary_context.set(copy.deepcopy(base_state))
    try:
        with pytest.raises(RuntimeError, match="Unexpected Tau2 auxiliary call_name"):
            run_async(
                wrapper(
                    model="ignored",
                    messages=[],
                    call_name="future_auxiliary_role",
                )
            )
    finally:
        auxiliary_context.reset(auxiliary_token)

    async def invalid_usage_generate(**kwargs):
        return message(content=valid_single_output, usage={})

    auxiliary_namespace["_ORIGINAL_JUDGE_GENERATE"] = invalid_usage_generate
    auxiliary_token = auxiliary_context.set(copy.deepcopy(base_state))
    attempt_token = judge_attempt_context.set(
        {
            "expected_outcomes": ("a",),
            "attempt_index": 1,
            "completion_budget": 64,
        }
    )
    try:
        with pytest.raises(RuntimeError, match="prompt_tokens"):
            run_async(
                wrapper(
                    model="ignored",
                    messages=[],
                    call_name="nl_assertions_eval",
                )
            )
    finally:
        judge_attempt_context.reset(attempt_token)
        auxiliary_context.reset(auxiliary_token)

    async def transport_failure_generate(**kwargs):
        raise ConnectionError("service unavailable")

    auxiliary_namespace["_ORIGINAL_JUDGE_GENERATE"] = transport_failure_generate
    auxiliary_token = auxiliary_context.set(copy.deepcopy(base_state))
    attempt_token = judge_attempt_context.set(
        {
            "expected_outcomes": ("a",),
            "attempt_index": 1,
            "completion_budget": 64,
        }
    )
    try:
        with pytest.raises(ConnectionError, match="service unavailable"):
            run_async(
                wrapper(
                    model="ignored",
                    messages=[],
                    call_name="nl_assertions_eval",
                )
            )
    finally:
        judge_attempt_context.reset(attempt_token)
        auxiliary_context.reset(auxiliary_token)

    async def stalled_generate(**kwargs):
        await asyncio.sleep(10)

    auxiliary_namespace["_ORIGINAL_JUDGE_GENERATE"] = stalled_generate
    timeout_state = copy.deepcopy(base_state)
    timeout_state["auxiliary_model_timeout_s"] = 0.001
    auxiliary_token = auxiliary_context.set(timeout_state)
    attempt_token = judge_attempt_context.set(
        {
            "expected_outcomes": ("a",),
            "attempt_index": 1,
            "completion_budget": 64,
        }
    )
    try:
        with pytest.raises(asyncio.TimeoutError):
            run_async(
                wrapper(
                    model="ignored",
                    messages=[],
                    call_name="nl_assertions_eval",
                )
            )
    finally:
        judge_attempt_context.reset(attempt_token)
        auxiliary_context.reset(auxiliary_token)
    assert timeout_state["auxiliary_model_calls"] == []

    user_messages = [
        message(content=None),
        message(content="retry succeeded"),
        message(content="next turn"),
    ]

    async def user_generate(**kwargs):
        return user_messages.pop(0)

    auxiliary_namespace["_ORIGINAL_USER_GENERATE"] = user_generate
    user_state = copy.deepcopy(base_state)
    auxiliary_token = auxiliary_context.set(user_state)
    try:
        for _ in range(3):
            run_async(
                wrapper(
                    model="ignored",
                    messages=[],
                    call_name="user_simulator_response",
                )
            )
    finally:
        auxiliary_context.reset(auxiliary_token)
    assert [
        (
            call["call_index"],
            call["attempt_index"],
            call["failure_code"],
        )
        for call in user_state["auxiliary_model_calls"]
    ] == [
        (1, 1, "empty_message"),
        (1, 2, None),
        (2, 1, None),
    ]

    malformed_requests = []

    async def malformed_judge_generate(**kwargs):
        malformed_requests.append(kwargs)
        return message(content=json.dumps({"results": []}))

    auxiliary_namespace["_ORIGINAL_JUDGE_GENERATE"] = malformed_judge_generate

    async def original_evaluate(trajectory, nl_assertions):
        return await wrapper(
            model="hard-coded-upstream-judge",
            messages=[],
            call_name="nl_assertions_eval",
        )

    auxiliary_namespace["_ORIGINAL_NL_EVALUATE"] = original_evaluate
    retry_state = copy.deepcopy(base_state)
    auxiliary_token = auxiliary_context.set(retry_state)
    reward_invalid_error = auxiliary_namespace["_JudgeRewardInvalid"]
    try:
        with pytest.raises(reward_invalid_error) as exc_info:
            run_async(
                auxiliary_namespace["_pipelinerl_evaluate_nl_assertions"](
                    object,
                    [],
                    ["a"],
                )
            )
    finally:
        auxiliary_context.reset(auxiliary_token)
    assert exc_info.value.detail_code == "result_count_mismatch"
    assert exc_info.value.attempt_count == 2
    assert [
        request["max_tokens"] for request in malformed_requests
    ] == [64, 128]
    assert all(
        (
            request["model"],
            request["temperature"],
            request["top_p"],
            request["seed"],
        )
        == ("qwen-judge-secondary", 0.6, 0.95, 17)
        for request in malformed_requests
    )
    assert [
        call["attempt_index"] for call in retry_state["auxiliary_model_calls"]
    ] == [1, 2]
    assert all(
        call["failure_code"] == "result_count_mismatch"
        for call in retry_state["auxiliary_model_calls"]
    )

    async def invalid_calculate(task, trajectory):
        raise reward_invalid_error("result_count_mismatch", 2)

    auxiliary_namespace["_ORIGINAL_NL_CALCULATE"] = invalid_calculate
    reward_info = run_async(
        auxiliary_namespace["_pipelinerl_calculate_nl_reward"](
            object,
            object(),
            [],
        )
    )
    marker = reward_info.info["pipelinerl_judge_reward_invalid"]
    assert reward_info.reward == 0.0
    assert marker == {
        "reason": "judge_reward_invalid",
        "producer_stage": "judge",
        "detail_code": "result_count_mismatch",
        "attempt_count": 2,
        "diagnostic": "judge output failed strict assertion validation",
    }

    generation_assignments = {
        ast.unparse(target): ast.unparse(node.value)
        for node in tau_tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
        and target.attr == "generate"
    }
    assert generation_assignments == {
        "tau2_user_simulator.generate": "_pipelinerl_auxiliary_generate",
        "tau2_nl_assertions.generate": "_pipelinerl_auxiliary_generate",
    }
    tau_config = next(
        node
        for node in tau_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "Tau2Config"
    )
    config_validator = copy.deepcopy(
        next(
            node
            for node in tau_config.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "validate_auxiliary_models"
        )
    )
    config_validator.decorator_list = []
    config_namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[config_validator], type_ignores=[])
            ),
            str(tau_target),
            "exec",
        ),
        config_namespace,
    )
    validate_config = config_namespace["validate_auxiliary_models"]
    valid_config = SimpleNamespace(
        model_server=SimpleNamespace(name="policy"),
        user_model_server=SimpleNamespace(name="user"),
        judge_model_server=SimpleNamespace(name="judge"),
        user_model_name="user-alias",
        judge_model_name="judge-alias",
        judge_initial_max_tokens=64,
        judge_retry_max_tokens=128,
    )
    assert validate_config(valid_config) is valid_config
    invalid_config = copy.deepcopy(valid_config)
    invalid_config.judge_model_server.name = "user"
    with pytest.raises(ValueError, match="server refs must differ"):
        validate_config(invalid_config)

    assert _GYM_PATCH_TARGETS[Path("nemo_gym/openai_utils.py")][1] == (
        "06210bba5847f52e71bdd8aba33390be3aec63bcc85ccc6210d42cc58b7f7fc6"
    )

    run_method_copy = copy.deepcopy(run_method)
    run_method_copy.decorator_list = []

    class FakeTau2RunRequest:
        model_fields = (
            "responses_create_params",
            "config",
            "task",
            "seed",
            "evaluation_type",
            "save_dir",
            "user_voice_settings",
            "user_persona_config",
            "verbose_logs",
            "audio_debug",
            "audio_taps",
            "auto_review",
            "review_mode",
            "hallucination_feedback",
        )

    class FakeTau2VerifyResponse(dict):
        def __init__(self, **kwargs):
            super().__init__(schema_version=1, outcome="completed", **kwargs)

    class FakeJSONResponse:
        def __init__(self, *, status_code, content):
            self.status_code = status_code
            self.content = content

    class FakeConverter:
        def __init__(self, *, return_token_id_information):
            assert return_token_id_information

        def chat_completions_messages_to_responses_items(self, messages):
            assert messages == []
            return []

    response_namespace = dict(auxiliary_namespace)
    response_namespace.update(
        {
            "Tau2RunRequest": FakeTau2RunRequest,
            "Tau2VerifyResponse": FakeTau2VerifyResponse,
            "JSONResponse": FakeJSONResponse,
            "jsonable_encoder": lambda value: value,
            "VLLMConverter": FakeConverter,
            "split_responses_input_output_items": lambda items: ([], []),
            "to_litellm_messages": lambda messages: [],
            "_attach_pipelinerl_provenance": lambda messages, items: None,
            "get_server_url": lambda ref: ref,
            "time": lambda: 123,
        }
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[run_method_copy], type_ignores=[])
            ),
            str(tau_target),
            "exec",
        ),
        response_namespace,
    )

    marker_info = {
        "reason": "judge_reward_invalid",
        "producer_stage": "judge",
        "detail_code": "result_count_mismatch",
        "attempt_count": 2,
        "diagnostic": "judge output failed strict assertion validation",
    }
    active_result = SimpleNamespace(
        messages=[],
        reward_info=SimpleNamespace(
            reward=0.0,
            info={
                "nl": {
                    "pipelinerl_judge_reward_invalid": marker_info,
                }
            },
        ),
        duration=2.0,
        model_dump=lambda mode: {
            "messages": [],
            "termination_reason": "agent_stop",
            "reward_info": {
                "reward": 0.0,
                "reward_breakdown": {
                    "NL_ASSERTION": 0.0,
                    "DB": 1.0,
                },
                "info": {
                    "nl": {
                        "pipelinerl_judge_reward_invalid": marker_info,
                    }
                },
            },
        },
    )

    async def run_single_task(**kwargs):
        return active_result

    response_namespace["run_single_task"] = run_single_task
    request_config = SimpleNamespace(
        domain="retail",
        llm_user=None,
        llm_args_user={},
        llm_agent=None,
        llm_args_agent={},
        max_steps=0,
    )
    request_task = SimpleNamespace(id="task-1")
    request_params = SimpleNamespace(
        input=[],
        model="qwen-policy",
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        model_dump=lambda exclude_unset: {},
    )
    request = SimpleNamespace(
        responses_create_params=request_params,
        config=request_config,
        task=request_task,
        seed=3,
        evaluation_type="all",
        save_dir=None,
        user_voice_settings=None,
        user_persona_config=None,
        verbose_logs=False,
        audio_debug=False,
        audio_taps=False,
        auto_review=False,
        review_mode="full",
        hallucination_feedback=None,
    )
    fake_agent = SimpleNamespace(
        config=SimpleNamespace(
            model_server=SimpleNamespace(name="policy"),
            user_model_server=SimpleNamespace(name="user"),
            judge_model_server=SimpleNamespace(name="judge"),
            user_model_name="qwen-user-primary",
            judge_model_name="qwen-judge-secondary",
            user_llm_args={},
            max_steps=200,
            judge_temperature=0.6,
            judge_top_p=0.95,
            judge_top_k=20,
            judge_seed=17,
            judge_initial_max_tokens=64,
            judge_retry_max_tokens=128,
            auxiliary_model_timeout_s=1.0,
        ),
        base_url_for_run=lambda url, body: f"http://{url}",
    )
    patched_run = response_namespace["run"]
    boundary_response = run_async(patched_run(fake_agent, request))
    assert boundary_response.status_code == 409
    assert boundary_response.content["outcome"] == "boundary_failure"
    assert "reward" not in boundary_response.content
    boundary_result = boundary_response.content["result"]
    assert "reward" not in boundary_result["reward_info"]
    assert "NL_ASSERTION" not in boundary_result["reward_info"][
        "reward_breakdown"
    ]
    assert boundary_result["reward_info"]["reward_breakdown"]["DB"] == 1.0
    assert boundary_response.content["auxiliary_model_calls"] == []

    active_result.reward_info = SimpleNamespace(reward=0.75, info={})
    completed_response = run_async(patched_run(fake_agent, request))
    assert completed_response["outcome"] == "completed"
    assert completed_response["reward"] == 0.75
    assert completed_response["result"] is active_result
