from pipelinerl.domains.terminal.rollouts import (
    _COMMAND_AUDIT_MAX_CHARS,
    _COMMAND_TRUNCATION_SUFFIX,
    _cap_commands,
    _terminal_audit_base,
)


def test_cap_commands_short_passthrough() -> None:
    commands = ["pwd", "echo hello", "x" * _COMMAND_AUDIT_MAX_CHARS]

    assert _cap_commands(commands) == commands


def test_cap_commands_truncates_long_command() -> None:
    command = "x" * (_COMMAND_AUDIT_MAX_CHARS + 5)

    assert _cap_commands([command]) == ["x" * _COMMAND_AUDIT_MAX_CHARS + _COMMAND_TRUNCATION_SUFFIX]


def test_cap_commands_empty() -> None:
    assert _cap_commands([]) == []


def test_terminal_audit_base_starts_with_empty_command_lists() -> None:
    audit = _terminal_audit_base({"task_id": "task-1"})

    assert audit["commands"] == []
    assert audit["command_errors"] == []
