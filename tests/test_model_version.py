"""Unit tests for the per-token model_version plumbing.

Covers the two pure pieces of the (otherwise cluster-only) feature: parsing the optional
`:v<version>` suffix off a `token_id:<id>` string, and the model_version padding / fallback in
`convert_to_fast_llm_format`.
"""

import pytest

from pipelinerl.llm import parse_token_id_and_version
from pipelinerl.preprocess import convert_to_fast_llm_format


@pytest.mark.parametrize(
    "token, expected",
    [
        ("token_id:1271", (1271, None)),  # no version suffix (backward compatible)
        ("token_id:1271:v5", (1271, 5)),
        ("token_id:1271:v0", (1271, 0)),  # version 0 is a real version, not "absent"
        ("token_id:50257:v123456", (50257, 123456)),
    ],
)
def test_parse_token_id_and_version(token, expected):
    assert parse_token_id_and_version(token) == expected


def test_convert_model_version_per_token_left_padded():
    # Completion versions are left-padded to the full sequence with the per-rollout scalar.
    entry = {"input_ids": [10, 11, 12, 13, 14, 15], "model_version": 1, "token_versions": [2, 3]}
    assert convert_to_fast_llm_format(entry)["model_version"] == [1, 1, 1, 1, 2, 3]


def test_convert_model_version_per_token_pads_with_first_when_no_scalar():
    entry = {"input_ids": [10, 11, 12, 13], "token_versions": [7, 8]}
    assert convert_to_fast_llm_format(entry)["model_version"] == [7, 7, 7, 8]


def test_convert_model_version_scalar_broadcast_fallback():
    # No per-token versions: broadcast the per-rollout scalar across the sequence.
    entry = {"input_ids": [10, 11, 12], "model_version": 4, "token_versions": []}
    assert convert_to_fast_llm_format(entry)["model_version"] == [4, 4, 4]


def test_convert_model_version_absent():
    entry = {"input_ids": [10, 11, 12]}
    assert "model_version" not in convert_to_fast_llm_format(entry)


def test_convert_model_version_full_completion_no_prompt():
    entry = {"input_ids": [10, 11, 12], "model_version": 9, "token_versions": [2, 3, 4]}
    assert convert_to_fast_llm_format(entry)["model_version"] == [2, 3, 4]
