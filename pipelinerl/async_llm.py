import base64
import io
import json
import logging

import aiohttp
import litellm
import numpy as np
from PIL import Image
from pipelinerl.llm import LLMCall, LLMOutput, Prompt, TokenLogprob, TrainableLLM

from pipelinerl.finetune.data import MASKED_TOKEN_ID
from pipelinerl.rollouts import TrainingText, apply_rollout_reward
from pipelinerl.processor_factory import get_processor
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)


class RetryableLLMResponseError(aiohttp.ClientPayloadError):
    """Raised when an LLM server returns a transiently invalid response body."""

    def __init__(self, message: str, *, status: int | None = None, body: str | None = None):
        self.status = status
        self.body = body
        super().__init__(message)


def _preview_response_body(body: str | None, limit: int = 1000) -> str:
    if body is None:
        return "<unavailable>"
    if len(body) <= limit:
        return body
    return body[:limit] + "...<truncated>"


def _raise_invalid_llm_response(
    reason: str,
    *,
    status: int | None = None,
    body: str | None = None,
) -> None:
    preview = _preview_response_body(body)
    raise RetryableLLMResponseError(
        f"Invalid LLM response ({reason}); status={status}; body={preview!r}",
        status=status,
        body=body,
    )


def extract_images_from_messages(messages: list[dict]) -> list[Image.Image]:
    """Extract PIL Images from multimodal messages."""

    images = []
    for message in messages:
        if isinstance(message.get("content"), list):
            for content_item in message["content"]:
                if content_item is None:
                    continue
                if content_item.get("type") == "image" and "image" in content_item:
                    images.append(content_item["image"])
                elif (
                    content_item.get("type") == "image_url"
                    and "image_url" in content_item
                ):
                    # Handle base64 format
                    url = content_item["image_url"]["url"]
                    if url.startswith("data:image;base64,"):
                        base64_data = url.split("data:image;base64,")[1]
                        image_data = base64.b64decode(base64_data)
                        image = Image.open(io.BytesIO(image_data))
                        images.append(image)

    return images


def _to_plain_obj(value):
    """convert OmegaConf containers into Python types"""

    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {key: _to_plain_obj(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_obj(item) for item in value]
    return value


def normalize_chat_template_messages(messages: list) -> list[dict]:
    normalized = []
    for message in messages:
        if isinstance(message, dict):
            message_dict = dict(message)
        elif hasattr(message, "model_dump"):
            message_dict = message.model_dump(exclude_none=True)
        elif hasattr(message, "dict"):
            message_dict = message.dict(exclude_none=True)
        else:
            message_dict = {
                key: getattr(message, key)
                for key in ("role", "content", "tool_calls", "reasoning_content")
                if hasattr(message, key)
            }

        message_dict.setdefault("reasoning_content", None)
        normalized.append(_to_plain_obj(message_dict))
    return normalized


async def llm_async_generate(
    llm: TrainableLLM,
    prompt: Prompt,
    session: aiohttp.ClientSession,
    max_tokens_override: int | None = None,
) -> LLMCall:
    llm.load_tokenizer()
    headers = {"Content-Type": "application/json"}
    if llm.api_token:
        headers |= {"Authorization": f"Bearer {llm.api_token}"}
    data = {
        "model": llm.model_name,
        "messages": prompt.messages,
    }
    if llm.collect_logprobs:
        data.update(
            {
                "logprobs": 1,
                "include_stop_str_in_output": True,
                "skip_special_tokens": False,
            }
        )

    extra_parameters = llm.parameters
    if isinstance(extra_parameters, (DictConfig, ListConfig)):
        extra_parameters = OmegaConf.to_container(extra_parameters, resolve=True)
    if extra_parameters is None:
        extra_parameters = {}
    if not isinstance(extra_parameters, dict):
        raise TypeError(
            f"LLM parameters must serialize to a mapping, got {type(extra_parameters)}"
        )

    logger.debug(f"POST request to {llm.base_url}/v1/chat/completions")

    if prompt.tools:
        data["tools"] = _to_plain_obj(prompt.tools)

    if max_tokens_override is not None:
        data["max_completion_tokens"] = max_tokens_override

    # Merge extra_parameters first so that data (model, messages, logprobs settings) takes precedence
    payload = _to_plain_obj({**extra_parameters, **data})
    async with session.post(
        url=f"{llm.base_url}/v1/chat/completions",
        json=payload,
        headers=headers,
        ssl=False,
    ) as response:
        response_text = await response.text()
        if not response.ok:
            logger.error(f"Failed to get completion: {response_text}")
            response.raise_for_status()
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            logger.exception(
                "Failed to decode LLM response JSON: status=%s body=%r",
                response.status,
                _preview_response_body(response_text),
            )
            _raise_invalid_llm_response(
                "response body is not valid JSON",
                status=response.status,
                body=response_text,
            )

    if data is None:
        logger.warning(
            "LLM server returned JSON null: status=%s body=%r",
            response.status,
            _preview_response_body(response_text),
        )
        _raise_invalid_llm_response(
            "response body is JSON null",
            status=response.status,
            body=response_text,
        )
    if not isinstance(data, dict):
        _raise_invalid_llm_response(
            f"expected JSON object, got {type(data).__name__}",
            status=response.status,
            body=response_text,
        )

    try:
        content = data["choices"][0]["message"]["content"]
        raw_tool_calls = data["choices"][0]["message"].get("tool_calls", [])
        if not content and not raw_tool_calls:
            logger.warning(f"Empty completion {data}")

        parsed_logprobs = []
        finish_reason = None
        if llm.collect_logprobs:
            completion_logprobs = data["choices"][0]["logprobs"]["content"]
            for logprob in completion_logprobs:
                if logprob:
                    try:
                        # We assume that the server was launched with --return-tokens-as-token-ids
                        # and that the tokens are provided as: ['token_id:1271', 'token_id:1505', '
                        parsed_logprobs.append(
                            TokenLogprob(
                                token_id=int(logprob["token"].split(":")[-1]),
                                logprob=logprob["logprob"],
                                generated=1,
                            )
                        )
                    except Exception as e:
                        logger.error(f"Failed to process logprobs: {logprob}")
                        logger.error(e)
        finish_reason = data["choices"][0].get("finish_reason")
    except Exception:
        logger.exception(
            "Failed to parse LLM response: status=%s body=%r parsed=%r",
            response.status,
            _preview_response_body(response_text),
            data,
        )
        _raise_invalid_llm_response(
            "response does not match OpenAI chat completion schema",
            status=response.status,
            body=response_text,
        )

    output = LLMOutput(content=content or "")
    if raw_tool_calls:
        output.tool_calls = [litellm.ChatCompletionMessageToolCall(**tc) for tc in raw_tool_calls]
    llm_call = llm.log_output(prompt, output, count_tokens=False)
    llm_call.prompt_length_tokens = data["usage"]["prompt_tokens"]
    llm_call.output_length_tokens = data["usage"]["completion_tokens"]
    if finish_reason:
        llm_call.llm_info["finish_reason"] = finish_reason
    assert llm_call is not None, "llm_call is None"
    llm_call.logprobs = parsed_logprobs
    return llm_call


def make_training_text(llm: TrainableLLM, llm_call: LLMCall) -> TrainingText:
    # Extract visual features if present
    images = []
    use_processor = False
    visual_features = None
    assistant_msg: dict = {"role": "assistant", "content": llm_call.output.content or ""}
    if llm_call.output.tool_calls:
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in llm_call.output.tool_calls
        ]
    prompt_messages = normalize_chat_template_messages(llm_call.prompt.messages)
    full_messages = prompt_messages + [assistant_msg]

    if hasattr(llm_call.prompt, "messages"):
        images = extract_images_from_messages(prompt_messages)
        if images:
            use_processor = True

    if use_processor:
        # Use processor for vision-language models
        processor = get_processor(llm.model_name)

        try:
            # Apply chat template using processor for proper image token handling
            prompt_text = processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Create full conversation with assistant response
            text = processor.apply_chat_template(
                full_messages,
                tokenize=False,
            )

            # Process prompt with images to get token IDs with image placeholders
            prompt_inputs = processor(
                text=processor.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                ),
                images=images,
                return_tensors=None,
            )

            # prompt_inputs["input_ids"] is a list of list
            prompt_token_ids = prompt_inputs["input_ids"][0]

            # Process images to get visual features
            processed = processor(
                text=[prompt_text], images=images, padding=True, return_tensors=None
            )
            visual_features = {
                key: value
                for key, value in processed.items()
                if isinstance(value, np.ndarray)
                and key not in ["input_ids", "attention_mask"]
            }

        except Exception as e:
            raise ValueError(f"Failed to process with vision-language processor: {e}")
    else:
        tools_kwarg = {"tools": llm_call.prompt.tools} if llm_call.prompt.tools else {}
        prompt_text = llm.tokenizer.apply_chat_template(
            conversation=prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            **tools_kwarg,
        )
        text = llm.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            **tools_kwarg,
        )
        prompt_token_ids = llm.tokenizer.apply_chat_template(
            prompt_messages,
            add_special_tokens=True,
            add_generation_prompt=True,
            **tools_kwarg,
        )

    output_text = text[len(prompt_text) :]

    tokenizer = processor.tokenizer if use_processor else llm.tokenizer

    if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
        text = text[len(tokenizer.bos_token) :]

    if not llm_call.logprobs:
        raise ValueError("Logprobs are required to make training data for RL")

    # We add the exact token ids and logprobs to "training_text" to ensure inference/training consistency
    labels = [lp.token_id for lp in llm_call.logprobs]
    input_ids = prompt_token_ids + labels
    # Apply masking to input tokens that aren't generated
    labels = [MASKED_TOKEN_ID] * len(prompt_token_ids) + labels
    logprobs = [lp.logprob for lp in llm_call.logprobs]
    finish_reason = llm_call.llm_info.get("finish_reason")
    if finish_reason is not None:
        finished = finish_reason != "length"
    else:
        eos_token = tokenizer.eos_token or ""
        finished = bool(eos_token) and (llm_call.output.content or "").endswith(eos_token)
    prompt_tokens = llm_call.prompt_length_tokens
    output_tokens = llm_call.output_length_tokens

    return TrainingText(
        text=text,
        n_predicted=len(output_text),
        input_ids=input_ids,
        labels=labels,
        logprobs=logprobs,
        finished=finished,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        visual_features=visual_features,
    )


def make_training_texts_from_llm_calls(
    llm: TrainableLLM,
    llm_calls: list[LLMCall],
    reward: float | None = None,
) -> list[TrainingText]:
    training_texts = [make_training_text(llm, llm_call) for llm_call in llm_calls]
    if reward is not None:
        training_texts = apply_rollout_reward(training_texts, reward)
    return training_texts
