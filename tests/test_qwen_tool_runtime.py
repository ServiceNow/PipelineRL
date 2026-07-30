import hashlib
import json
import os
from pathlib import Path

import pytest

from pipelinerl.prerun_evidence import (
    QWEN35_9B_MODEL_DESCRIPTOR,
    QWEN35_CHAT_TEMPLATE_SHA256,
    QWEN35_TOKENIZER_CONFIG_SHA256,
    QWEN35_TOOL_CALL_PARSER,
)


_SNAPSHOT_ENV = "QWEN35_RUNTIME_SNAPSHOT"
pytestmark = pytest.mark.skipif(
    not os.environ.get(_SNAPSHOT_ENV),
    reason=f"set {_SNAPSHOT_ENV} to execute staged Qwen runtime checks",
)


def _snapshot() -> Path:
    path = Path(os.environ[_SNAPSHOT_ENV])
    if not path.is_dir():
        pytest.fail(f"{_SNAPSHOT_ENV}={path} is not a directory")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up an item.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                    "required": ["item", "count"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "confirm",
                "description": "Confirm a choice.",
                "parameters": {
                    "type": "object",
                    "properties": {"accepted": {"type": "boolean"}},
                    "required": ["accepted"],
                },
            },
        },
    ]


def test_qwen_runtime_surface_matches_reviewed_artifacts():
    from transformers import AutoTokenizer

    snapshot = _snapshot()
    artifacts = dict(QWEN35_9B_MODEL_DESCRIPTOR.artifact_sha256)
    assert _sha256(snapshot / "tokenizer_config.json") == (
        QWEN35_TOKENIZER_CONFIG_SHA256
    )
    assert _sha256(snapshot / "chat_template.jinja") == (
        QWEN35_CHAT_TEMPLATE_SHA256
    )
    assert artifacts["tokenizer_config.json"] == QWEN35_TOKENIZER_CONFIG_SHA256
    assert artifacts["chat_template.jinja"] == QWEN35_CHAT_TEMPLATE_SHA256

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    assert tokenizer.chat_template == (snapshot / "chat_template.jinja").read_text()


def test_qwen_template_makes_thinking_explicit():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_snapshot(), local_files_only=True)
    messages = [{"role": "user", "content": "Hello"}]
    thinking = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    no_thinking = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    assert thinking.endswith("<|im_start|>assistant\n<think>\n")
    assert no_thinking.endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def test_native_qwen_parser_resolves_and_parses_real_xml():
    from transformers import AutoTokenizer
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.tool_parsers import ToolParserManager

    tokenizer = AutoTokenizer.from_pretrained(_snapshot(), local_files_only=True)
    parser_cls = ToolParserManager.get_tool_parser(QWEN35_TOOL_CALL_PARSER)
    assert parser_cls.__name__ == "Qwen3XMLToolParser"
    with pytest.raises(KeyError):
        ToolParserManager.get_tool_parser("rl_tool")

    request = ChatCompletionRequest(
        model="qwen3.5-9b-policy",
        messages=[{"role": "user", "content": "Use the tools"}],
        tools=_tools(),
        tool_choice="auto",
    )
    parser = parser_cls(tokenizer, request.tools)

    plain = parser.extract_tool_calls("ordinary text", request)
    assert plain.tools_called is False
    assert plain.tool_calls == []
    assert plain.content == "ordinary text"

    parsed = parser.extract_tool_calls(
        "<think>check both</think>\n"
        "<tool_call><function=lookup><parameter=item>widget</parameter>"
        "<parameter=count>2</parameter></function></tool_call>"
        "<tool_call><function=confirm><parameter=accepted>true</parameter>"
        "</function></tool_call>",
        request,
    )
    assert parsed.tools_called is True
    assert parsed.content == "<think>check both</think>\n"
    assert [call.function.name for call in parsed.tool_calls] == [
        "lookup",
        "confirm",
    ]
    assert json.loads(parsed.tool_calls[0].function.arguments) == {
        "item": "widget",
        "count": 2,
    }
    assert json.loads(parsed.tool_calls[1].function.arguments) == {
        "accepted": True,
    }


def test_vllm_normalizes_openai_tool_history_before_qwen_template():
    from transformers import AutoTokenizer
    from vllm.entrypoints.chat_utils import _postprocess_messages

    tokenizer = AutoTokenizer.from_pretrained(_snapshot(), local_files_only=True)
    history = [
        {"role": "user", "content": "Look it up"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"item":"widget","count":2}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "available"},
    ]

    _postprocess_messages(history)
    assert history[1]["tool_calls"][0]["function"]["arguments"] == {
        "item": "widget",
        "count": 2,
    }
    rendered = tokenizer.apply_chat_template(
        history,
        tools=_tools(),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    assert "<function=lookup>" in rendered
    assert "<parameter=item>\nwidget\n</parameter>" in rendered
    assert "<parameter=count>\n2\n</parameter>" in rendered
