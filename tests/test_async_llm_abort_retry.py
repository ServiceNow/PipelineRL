import asyncio
from types import SimpleNamespace

import pytest
from transformers.tokenization_utils_base import BatchEncoding

from pipelinerl.async_llm import (
    RetryableAbortedCompletionError,
    llm_async_generate,
    make_training_text,
)
from pipelinerl.finetune.data import MASKED_TOKEN_ID
from pipelinerl.llm import LLMCall, LLMOutput, Prompt, TokenLogprob
from pipelinerl.vllm1 import WeightUpdateManager, resolve_weight_update_name


class DummyResponse:
    def __init__(self, payload: dict, ok: bool = True):
        self.payload = payload
        self.ok = ok

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self):
        return str(self.payload)

    async def json(self):
        return self.payload

    def raise_for_status(self):
        raise RuntimeError(self.payload)


class DummySession:
    def __init__(self, payloads: list[dict]):
        self.payloads = list(payloads)
        self.calls = 0

    def post(self, **kwargs):
        payload = self.payloads[self.calls]
        self.calls += 1
        return DummyResponse(payload)


class DummyLLM:
    base_url = "http://test"
    api_token = None
    model_name = "dummy-model"
    collect_logprobs = True
    parameters = {}
    chat_template_kwargs = None

    def load_tokenizer(self):
        return None

    def log_output(self, prompt, output, count_tokens=False):
        return LLMCall(
            prompt=prompt,
            output=output,
            cached=False,
            llm_info={},
        )


def _abort_payload() -> dict:
    return {
        "id": "chatcmpl-abort",
        "choices": [
            {
                "message": {"role": "assistant", "content": ""},
                "logprobs": {"content": []},
                "finish_reason": "abort",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0},
    }


def _good_payload() -> dict:
    return {
        "id": "chatcmpl-good",
        "choices": [
            {
                "message": {"role": "assistant", "content": "ok"},
                "logprobs": {
                    "content": [{"token": "token_id:42", "logprob": -0.1}]
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1},
    }


def test_llm_async_generate_retries_once_on_abort_response() -> None:
    prompt = Prompt.from_user_message("hello")
    llm = DummyLLM()
    session = DummySession([_abort_payload(), _good_payload()])

    llm_call = asyncio.run(llm_async_generate(llm, prompt, session))

    assert session.calls == 2
    assert llm_call.output.content == "ok"
    assert [lp.token_id for lp in llm_call.logprobs] == [42]


def test_llm_async_generate_raises_after_repeated_abort_responses() -> None:
    prompt = Prompt.from_user_message("hello")
    llm = DummyLLM()
    session = DummySession([_abort_payload(), _abort_payload()])

    with pytest.raises(RetryableAbortedCompletionError):
        asyncio.run(llm_async_generate(llm, prompt, session))

    assert session.calls == 2


class CapturingSession(DummySession):
    def post(self, **kwargs):
        self.last_payload = kwargs["json"]
        return super().post(**kwargs)


def test_llm_async_generate_keeps_tool_call_history_arguments_as_wire_strings() -> None:
    prompt = Prompt(messages=_tool_history_messages('{"command": "ls"}'), tools=_bash_tool_definition())
    llm = DummyLLM()
    session = CapturingSession([_good_payload()])

    llm_call = asyncio.run(llm_async_generate(llm, prompt, session))

    sent_args = session.last_payload["messages"][2]["tool_calls"][0]["function"]["arguments"]
    stored_args = llm_call.prompt.messages[2]["tool_calls"][0]["function"]["arguments"]
    assert sent_args == '{"command": "ls"}'
    assert stored_args == '{"command": "ls"}'


def test_llm_async_generate_stringifies_dict_tool_call_history_for_wire() -> None:
    prompt = Prompt(messages=_tool_history_messages({"command": "ls"}), tools=_bash_tool_definition())
    llm = DummyLLM()
    session = CapturingSession([_good_payload()])

    asyncio.run(llm_async_generate(llm, prompt, session))

    sent_args = session.last_payload["messages"][2]["tool_calls"][0]["function"]["arguments"]
    assert sent_args == '{"command": "ls"}'


def test_make_training_text_rejects_abort_responses_even_with_logprobs() -> None:
    llm_call = LLMCall(
        prompt=Prompt.from_user_message("hello"),
        output=LLMOutput(content="partial"),
        cached=False,
        llm_info={"finish_reason": "abort"},
        logprobs=[TokenLogprob(token_id=42, logprob=-0.1, generated=1)],
    )
    llm = SimpleNamespace()

    with pytest.raises(RetryableAbortedCompletionError):
        make_training_text(llm, llm_call)


class ChatTemplateTokenizer:
    bos_token = None
    eos_token = "<eos>"

    def apply_chat_template(self, conversation, tokenize=True, return_dict=True, **kwargs):
        if not tokenize:
            if conversation[-1]["role"] == "assistant":
                return "<user>hello<assistant>ok"
            return "<user>hello<assistant>"
        token_ids = [11, 12, 13]
        if return_dict:
            return BatchEncoding({"input_ids": token_ids})
        return token_ids


def test_make_training_text_requests_token_id_list_from_chat_template() -> None:
    llm_call = LLMCall(
        prompt=Prompt.from_user_message("hello"),
        output=LLMOutput(content="ok"),
        cached=False,
        llm_info={"finish_reason": "stop"},
        logprobs=[TokenLogprob(token_id=42, logprob=-0.1, generated=1)],
    )
    llm = SimpleNamespace(
        model_name="dummy-model",
        tokenizer=ChatTemplateTokenizer(),
        chat_template_kwargs=None,
    )

    training_text = make_training_text(llm, llm_call)

    assert training_text.input_ids == [11, 12, 13, 42]
    assert training_text.labels == [MASKED_TOKEN_ID, MASKED_TOKEN_ID, MASKED_TOKEN_ID, 42]


def test_resolve_weight_update_name_maps_qwen35_wrapper_text_weights() -> None:
    model = SimpleNamespace(language_model=SimpleNamespace())

    assert (
        resolve_weight_update_name(model, "model.layers.0.input_layernorm.weight")
        == "language_model.model.layers.0.input_layernorm.weight"
    )
    assert (
        resolve_weight_update_name(model, "lm_head.weight")
        == "language_model.lm_head.weight"
    )


def test_resolve_weight_update_name_keeps_native_and_checkpoint_names() -> None:
    causal_model = SimpleNamespace(model=SimpleNamespace())
    wrapper_model = SimpleNamespace(language_model=SimpleNamespace())

    assert (
        resolve_weight_update_name(causal_model, "model.layers.0.input_layernorm.weight")
        == "model.layers.0.input_layernorm.weight"
    )
    assert (
        resolve_weight_update_name(wrapper_model, "model.language_model.layers.0.input_layernorm.weight")
        == "model.language_model.layers.0.input_layernorm.weight"
    )
    assert (
        resolve_weight_update_name(wrapper_model, "model.visual.blocks.0.norm1.weight")
        == "model.visual.blocks.0.norm1.weight"
    )


class DummyEngine:
    def __init__(self):
        self.pause_calls = []
        self.resume_calls = 0

    async def pause_generation(self, **kwargs):
        self.pause_calls.append(kwargs)

    async def resume_generation(self):
        self.resume_calls += 1


class DummyEngineClient:
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.calls = []

    async def collective_rpc_async(self, method, args=(), kwargs=None, timeout=None):
        self.calls.append((method, args, kwargs, timeout))
        if self.should_fail:
            raise RuntimeError("rpc failed")
        return []


class DummyWeightUpdateRequest:
    version = 7

    def model_dump_json(self):
        return '{"version": 7}'


def test_weight_update_manager_uses_keep_without_clearing_cache() -> None:
    manager = WeightUpdateManager(args=SimpleNamespace(), engine=DummyEngine(), engine_client=DummyEngineClient())

    asyncio.run(manager.receive_weight_update(DummyWeightUpdateRequest()))

    assert manager.engine.pause_calls == [{"mode": "keep", "clear_cache": False}]
    assert manager.engine.resume_calls == 1
    assert manager.engine_client.calls[0][0] == "receive_weight_update"


def test_weight_update_manager_resumes_generation_on_rpc_failure() -> None:
    manager = WeightUpdateManager(
        args=SimpleNamespace(),
        engine=DummyEngine(),
        engine_client=DummyEngineClient(should_fail=True),
    )

    with pytest.raises(RuntimeError, match="rpc failed"):
        asyncio.run(manager.receive_weight_update(DummyWeightUpdateRequest()))

    assert manager.engine.resume_calls == 1


class ToolTemplateTokenizer:
    bos_token = None
    eos_token = "<eos>"

    def _render(self, conversation, add_generation_prompt=False, **kwargs):
        parts = []
        for message in conversation:
            role = message["role"]
            parts.append(f"<{role}>")
            content = message.get("content") or ""
            if content:
                parts.append(content)
            for tool_call in message.get("tool_calls") or []:
                function = tool_call["function"]
                arguments = function["arguments"]
                if not isinstance(arguments, dict):
                    raise TypeError("tool arguments must be a mapping")
                parts.append(f"<tool_call:{function['name']}:{arguments['command']}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def apply_chat_template(self, conversation, tokenize=True, return_dict=False, add_generation_prompt=False, **kwargs):
        rendered = self._render(conversation, add_generation_prompt=add_generation_prompt, **kwargs)
        token_ids = [ord(ch) % 251 for ch in rendered]
        if tokenize:
            if return_dict:
                return BatchEncoding({"input_ids": token_ids})
            return token_ids
        return rendered


def _tool_history_messages(arguments):
    return [
        {"role": "system", "content": "use tools"},
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "bash", "arguments": arguments},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "file.txt"},
        {"role": "user", "content": "second"},
    ]


def _bash_tool_definition():
    return [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]


def test_make_training_text_normalizes_multiturn_tool_call_history():
    tokenizer = ToolTemplateTokenizer()
    prompt = Prompt(messages=_tool_history_messages('{"command": "ls"}'), tools=_bash_tool_definition())
    output = LLMOutput(content="")
    output.tool_calls = [
        SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="bash", arguments='{"command": "pwd"}'),
        )
    ]
    llm_call = LLMCall(
        prompt=prompt,
        output=output,
        cached=False,
        llm_info={"finish_reason": "stop"},
        logprobs=[TokenLogprob(token_id=77, logprob=-0.1, generated=1)],
        prompt_length_tokens=0,
        output_length_tokens=1,
    )
    llm = SimpleNamespace(
        model_name="dummy-model",
        tokenizer=tokenizer,
        chat_template_kwargs={"enable_thinking": False},
    )

    training_text = make_training_text(llm, llm_call)

    canonical_prompt = _tool_history_messages({"command": "ls"})
    expected_prompt_ids = tokenizer.apply_chat_template(
        canonical_prompt,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        tools=_bash_tool_definition(),
        enable_thinking=False,
    )
    assert training_text.input_ids[:-1] == expected_prompt_ids
    assert training_text.labels[:-1] == [MASKED_TOKEN_ID] * len(expected_prompt_ids)
    assert training_text.input_ids[-1] == 77
