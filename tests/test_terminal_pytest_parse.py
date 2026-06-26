"""Unit tests for the graded-reward pytest summary parser.

Covers the cases Codex flagged: all-passed, mixed failed/passed, error, and
skipped-only / no-tests (which must yield total 0 -> reward_fail, not 0/0).
"""
from pipelinerl.domains.terminal.proot_env import ProotTerminalEnvironment

parse = ProotTerminalEnvironment._parse_pytest_counts


def test_all_passed():
    assert parse("3 passed in 0.05s") == (3, 3)


def test_mixed_failed_passed():
    assert parse("1 failed, 2 passed in 0.06s") == (2, 3)


def test_all_failed():
    assert parse("2 failed in 0.04s") == (0, 2)


def test_error_counts_as_non_pass():
    # Collection/setup error: 0 passed, denominator counts the error.
    assert parse("1 error in 0.03s") == (0, 1)


def test_no_tests_ran():
    assert parse("no tests ran in 0.01s") == (0, 0)


def test_empty_output():
    assert parse("") == (0, 0)


def test_skipped_and_xfail_excluded():
    # Skipped / xfail / xpass are excluded from the denominator and never passes.
    assert parse("2 passed, 1 skipped, 1 xfailed, 1 xpassed in 0.07s") == (2, 2)


def test_failed_passed_with_error():
    assert parse("1 failed, 3 passed, 2 errors in 0.09s") == (3, 6)
