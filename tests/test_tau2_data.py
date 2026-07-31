"""Tests for attested Tau2 prepared data."""

from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from pipelinerl.domains.tau2.client import NEMO_GYM_SHA, TAU2_DATA_SHA
from pipelinerl.domains.tau2.dataset import (
    NEMO_GYM_REPOSITORY,
    TAU2_DATA_DOMAINS,
    TAU2_DATA_REPOSITORY,
    TAU2_EXPECTED_NL_REWARD_ROWS,
    TAU2_EXPECTED_NONEMPTY_NL_ASSERTION_ROWS,
    TAU2_EXPECTED_ROW_COUNTS,
    load_tau2_problems,
    validate_tau2_prepared_data,
    write_tau2_prepared_data,
)
from pipelinerl.entrypoints.prepare_tau2_data import (
    _validate_checkout,
    prepare_tau2_data,
)


def _task_id(dataset: str, index: int) -> str:
    if dataset == "telecom":
        return f"telecom-{index}"
    return str(index)


def _raw_row(dataset: str, index: int) -> dict:
    reward_basis = ["DB"]
    if dataset == "retail" and index < 112:
        reward_basis.append("NL_ASSERTION")
    if dataset == "airline" or (dataset == "retail" and index < 40):
        nl_assertions = [{"description": f"assertion-{index}"}]
    else:
        nl_assertions = None
    return {
        "config": {
            "domain": dataset,
            "task_split_name": "base",
            "timeout": None,
            "save_to": "/tmp/unreviewed",
            "llm_args_user": {"temperature": 0.0},
        },
        "task": {
            "id": _task_id(dataset, index),
            "evaluation_criteria": {
                "reward_basis": reward_basis,
                "nl_assertions": nl_assertions,
            },
        },
        "seed": index,
        "evaluation_type": "all",
        "responses_create_params": {"input": [], "tools": []},
    }


def _write_raw_rows(root: Path) -> None:
    for dataset in TAU2_DATA_DOMAINS:
        directory = root / dataset
        directory.mkdir(parents=True)
        for index in range(TAU2_EXPECTED_ROW_COUNTS[dataset]):
            (directory / f"{index}.json").write_text(
                json.dumps(_raw_row(dataset, index)) + "\n"
            )


def _upstream_normalize_row(row: dict) -> dict:
    row["config"]["save_to"] = ""
    row["evaluation_type"] = "all"
    row["config"].get("llm_args_user", {}).pop("temperature", None)
    basis = row["task"]["evaluation_criteria"]["reward_basis"]
    row["task"]["evaluation_criteria"]["reward_basis"] = [
        item for item in basis if item != "NL_ASSERTION"
    ]
    return row


def _prepared(tmp_path: Path):
    raw_root = tmp_path / "raw"
    _write_raw_rows(raw_root)
    manifest_path = write_tau2_prepared_data(
        raw_root,
        tmp_path / "prepared",
        _upstream_normalize_row,
    )
    return manifest_path, validate_tau2_prepared_data(manifest_path)


def test_prepared_data_preserves_reward_and_loads_natural_population(tmp_path):
    manifest_path, validated = _prepared(tmp_path)

    assert validated.manifest.source_repository == TAU2_DATA_REPOSITORY
    assert validated.manifest.source_revision == TAU2_DATA_SHA
    assert validated.manifest.reference_normalizer_repository == (
        NEMO_GYM_REPOSITORY
    )
    assert validated.manifest.reference_normalizer_revision == NEMO_GYM_SHA
    assert validated.manifest.total_row_count == 278
    assert [item.row_count for item in validated.manifest.files] == [50, 114, 114]
    assert [item.nl_reward_basis_rows for item in validated.manifest.files] == [
        0,
        112,
        0,
    ]
    assert [
        item.nonempty_nl_assertion_rows for item in validated.manifest.files
    ] == [50, 40, 0]

    problems = load_tau2_problems(
        TAU2_DATA_DOMAINS,
        data_files={
            dataset: str(path)
            for dataset, path in validated.data_files.items()
        },
        prepared_data_manifest=str(manifest_path),
        seed=17,
    )
    assert len(problems) == 278
    assert Counter(problem["dataset"] for problem in problems) == {
        "airline": 50,
        "retail": 114,
        "telecom": 114,
    }
    assert {problem["domain"] for problem in problems} == {"tau2"}
    composite = {
        (problem["dataset"], problem["task_id"]) for problem in problems
    }
    assert len(composite) == 278
    assert len({problem["task_id"] for problem in problems}) == 228


def test_normalization_rejects_any_second_delta(tmp_path):
    raw_root = tmp_path / "raw"
    _write_raw_rows(raw_root)

    def divergent_upstream(row: dict) -> dict:
        row = _upstream_normalize_row(row)
        row["config"]["unexpected"] = True
        return row

    with pytest.raises(ValueError, match="outside reward_basis"):
        write_tau2_prepared_data(
            raw_root,
            tmp_path / "prepared",
            divergent_upstream,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda row: row["config"].update(domain="retail"), "config.domain"),
        (
            lambda row: row["config"].update(task_split_name="full"),
            "base",
        ),
        (lambda row: row["config"].update(timeout=1.0), "timeout"),
        (lambda row: row.update(evaluation_type="env"), "evaluation_type"),
        (
            lambda row: row["task"]["evaluation_criteria"][
                "reward_basis"
            ].extend(["NL_ASSERTION", "NL_ASSERTION"]),
            "occurrences",
        ),
    ),
)
def test_preparation_rejects_row_contract_drift(
    tmp_path,
    mutation,
    message,
):
    raw_root = tmp_path / "raw"
    _write_raw_rows(raw_root)
    path = raw_root / "airline" / "0.json"
    row = json.loads(path.read_text())
    mutation(row)
    path.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match=message):
        write_tau2_prepared_data(
            raw_root,
            tmp_path / "prepared",
            _upstream_normalize_row,
        )


def test_preparation_rejects_duplicate_within_domain_identity(tmp_path):
    raw_root = tmp_path / "raw"
    _write_raw_rows(raw_root)
    first = json.loads((raw_root / "telecom" / "0.json").read_text())
    second_path = raw_root / "telecom" / "1.json"
    second = json.loads(second_path.read_text())
    second["task"]["id"] = first["task"]["id"]
    second_path.write_text(json.dumps(second) + "\n")

    with pytest.raises(ValueError, match="duplicate task identities"):
        write_tau2_prepared_data(
            raw_root,
            tmp_path / "prepared",
            _upstream_normalize_row,
        )


def test_validation_rejects_path_substitution_and_content_tamper(tmp_path):
    manifest_path, validated = _prepared(tmp_path)
    configured = {
        dataset: str(path) for dataset, path in validated.data_files.items()
    }
    substituted = dict(configured)
    substituted["retail"] = configured["airline"]
    with pytest.raises(ValueError, match="does not match"):
        validate_tau2_prepared_data(manifest_path, data_files=substituted)

    airline = validated.data_files["airline"]
    airline.write_bytes(airline.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_tau2_prepared_data(manifest_path, data_files=configured)


def test_validation_rejects_canonical_self_reported_source_drift(tmp_path):
    manifest_path, _ = _prepared(tmp_path)
    payload = json.loads(manifest_path.read_text())
    payload["source_revision"] = "0" * 40
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )

    with pytest.raises(ValueError, match="source or normalization identity"):
        validate_tau2_prepared_data(manifest_path)


def test_checkout_validation_rejects_revision_and_tracked_drift(
    tmp_path,
    monkeypatch,
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    outputs = iter(["wrong-revision"])
    monkeypatch.setattr(
        "pipelinerl.entrypoints.prepare_tau2_data._git_output",
        lambda *_args: next(outputs),
    )
    with pytest.raises(ValueError, match="expected"):
        _validate_checkout(checkout, TAU2_DATA_SHA, "Tau2 data source")

    outputs = iter([TAU2_DATA_SHA, " M uv.lock"])
    monkeypatch.setattr(
        "pipelinerl.entrypoints.prepare_tau2_data._git_output",
        lambda *_args: next(outputs),
    )
    with pytest.raises(ValueError, match="tracked changes"):
        _validate_checkout(checkout, TAU2_DATA_SHA, "Tau2 data source")


def test_preparer_uses_pinned_environment_and_dump_hooks(
    tmp_path,
    monkeypatch,
):
    tau2_checkout = tmp_path / "tau2"
    gym_checkout = tmp_path / "gym"
    output_dir = tmp_path / "prepared"
    dump_script = tau2_checkout / "dump_nemo_gym_data.sh"
    dump_python = tau2_checkout / ".venv" / "bin" / "python"
    dump_script.parent.mkdir()
    dump_script.write_text("#!/usr/bin/env bash\n")
    dump_python.parent.mkdir(parents=True)
    dump_python.write_text("")
    gym_checkout.mkdir()

    validations = []
    monkeypatch.setattr(
        "pipelinerl.entrypoints.prepare_tau2_data._validate_checkout",
        lambda checkout, revision, label: validations.append(
            (checkout, revision, label)
        ),
    )
    normalizer = object()
    monkeypatch.setattr(
        "pipelinerl.entrypoints.prepare_tau2_data._load_upstream_normalizer",
        lambda _checkout: normalizer,
    )
    manifest_path = output_dir / "manifest.json"
    writes = []

    def record_write(raw_root, output, normalize):
        writes.append((raw_root, output, normalize))
        return manifest_path

    monkeypatch.setattr(
        "pipelinerl.entrypoints.prepare_tau2_data.write_tau2_prepared_data",
        record_write,
    )
    calls = []

    def record_run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(subprocess, "run", record_run)
    monkeypatch.delenv("TAU2_SKIP_UV_SYNC", raising=False)
    monkeypatch.delenv("TAU2_DUMP_PYTHON", raising=False)

    assert (
        prepare_tau2_data(tau2_checkout, gym_checkout, output_dir)
        == manifest_path
    )
    assert validations == [
        (tau2_checkout, TAU2_DATA_SHA, "Tau2 data source"),
        (gym_checkout, NEMO_GYM_SHA, "NeMo Gym source"),
        (tau2_checkout, TAU2_DATA_SHA, "Tau2 data source"),
        (gym_checkout, NEMO_GYM_SHA, "NeMo Gym source"),
    ]
    assert len(calls) == 2
    sync_command, sync_kwargs = calls[0]
    assert sync_command == ["uv", "sync", "--frozen", "--extra", "knowledge"]
    assert sync_kwargs["cwd"] == tau2_checkout
    assert sync_kwargs["check"] is True
    assert sync_kwargs["env"]["UV_FROZEN"] == "1"
    assert "TAU2_SKIP_UV_SYNC" not in sync_kwargs["env"]
    assert "TAU2_DUMP_PYTHON" not in sync_kwargs["env"]

    dump_command, dump_kwargs = calls[1]
    assert dump_command[:2] == ["bash", str(dump_script)]
    assert dump_command[2:9] == [
        "--dataset",
        "airline",
        "--dataset",
        "retail",
        "--dataset",
        "telecom",
        "--output-root",
    ]
    assert Path(dump_command[9]).name == "raw"
    assert dump_kwargs["cwd"] == tau2_checkout
    assert dump_kwargs["check"] is True
    assert dump_kwargs["env"]["UV_FROZEN"] == "1"
    assert dump_kwargs["env"]["TAU2_SKIP_UV_SYNC"] == "1"
    assert dump_kwargs["env"]["TAU2_DUMP_PYTHON"] == str(dump_python)
    assert len(writes) == 1
    assert writes[0][0] == Path(dump_command[9])
    assert writes[0][1:] == (output_dir, normalizer)


@pytest.mark.skipif(
    not {
        "TAU2_DATA_SOURCE_CHECKOUT",
        "NEMO_GYM_SOURCE_CHECKOUT",
    }.issubset(os.environ),
    reason="pinned Tau2 data and NeMo Gym source checkouts are unavailable",
)
def test_pinned_source_preparation_matches_measured_contract(tmp_path):
    checkouts = {}
    for env_name, revision in (
        ("TAU2_DATA_SOURCE_CHECKOUT", TAU2_DATA_SHA),
        ("NEMO_GYM_SOURCE_CHECKOUT", NEMO_GYM_SHA),
    ):
        checkout = tmp_path / env_name.lower()
        subprocess.run(
            ["git", "clone", "--shared", os.environ[env_name], str(checkout)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", revision],
            check=True,
        )
        checkouts[env_name] = checkout

    manifest_path = prepare_tau2_data(
        checkouts["TAU2_DATA_SOURCE_CHECKOUT"],
        checkouts["NEMO_GYM_SOURCE_CHECKOUT"],
        tmp_path / "prepared",
    )
    validated = validate_tau2_prepared_data(manifest_path)
    assert validated.manifest.total_row_count == 278
    assert {
        item.dataset: item.row_count for item in validated.manifest.files
    } == TAU2_EXPECTED_ROW_COUNTS
    assert {
        item.dataset: item.nl_reward_basis_rows
        for item in validated.manifest.files
    } == TAU2_EXPECTED_NL_REWARD_ROWS
    assert {
        item.dataset: item.nonempty_nl_assertion_rows
        for item in validated.manifest.files
    } == TAU2_EXPECTED_NONEMPTY_NL_ASSERTION_ROWS
