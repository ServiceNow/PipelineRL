import base64
import io
import logging

import aiohttp
import numpy as np
from PIL import Image
from tapeagents.core import LLMCall, LLMOutput, Prompt, TokenLogprob
from tapeagents.llms.trainable import TrainableLLM

from pipelinerl.finetune.data import MASKED_TOKEN_ID
from pipelinerl.rollouts import TrainingText
from pipelinerl.processor_factory import get_processor

logger = logging.getLogger(__name__)


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
                        try:
                            base64_data = url.split("data:image;base64,")[1]
                            image_data = base64.b64decode(base64_data)
                            image = Image.open(io.BytesIO(image_data))
                            images.append(image)
                        except Exception as e:
                            raise e

    return images


async def llm_async_generate(
    llm: TrainableLLM, prompt: Prompt, session: aiohttp.ClientSession
) -> LLMCall:
    llm.load_tokenizer()
    headers = {"Content-Type": "application/json"}
    if llm.api_token:
        headers |= {"Authorization": f"Bearer {llm.api_token}"}
    data = {
        "model": llm.model_name,
        "messages": prompt.messages,
        "stream": llm.stream,
    }
    if llm.collect_logprobs:
        data.update(
            {
                "logprobs": 1,
                "include_stop_str_in_output": True,
                "skip_special_tokens": False,
            }
        )

    logger.debug(f"POST request to {llm.base_url}/v1/chat/completions")

    async with session.post(
        url=f"{llm.base_url}/v1/chat/completions",
        json=data | llm.parameters,
        headers=headers,
        ssl=False,
    ) as response:
        if not response.ok:
            error_text = await response.text()
            logger.error(f"Failed to get completion: {error_text}")
            response.raise_for_status()
        data = await response.json()

    try:
        content = data["choices"][0]["message"]["content"]
        if not content:
            logger.warning(f"Empty completion {data}")

        parsed_logprobs = []
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
    except Exception as e:
        logger.exception(f"Failed to parse llm response: {data}")
        raise e

    output = LLMOutput(content=content)
    llm_call = llm.log_output(prompt, output, count_tokens=False)
    llm_call.prompt_length_tokens = data["usage"]["prompt_tokens"]
    llm_call.output_length_tokens = data["usage"]["completion_tokens"]
    assert llm_call is not None, "llm_call is None"
    llm_call.logprobs = parsed_logprobs
    return llm_call


def make_training_text(llm: TrainableLLM, llm_call: LLMCall) -> TrainingText:
    # Extract visual features if present
    images = []
    use_processor = False
    visual_features = None
    full_messages = llm_call.prompt.messages + [
        {"role": "assistant", "content": llm_call.output.content}
    ]

    if hasattr(llm_call.prompt, "messages"):
        images = extract_images_from_messages(llm_call.prompt.messages)
        if images:
            use_processor = True

    if use_processor:
        # Use processor for vision-language models
        processor = get_processor(llm.model_name)

        try:
            # Apply chat template using processor for proper image token handling
            prompt_text = processor.apply_chat_template(
                llm_call.prompt.messages,
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
                    llm_call.prompt.messages, tokenize=False, add_generation_prompt=True
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
        # Use tokenizer for text-only models
        prompt_text = llm.tokenizer.apply_chat_template(
            conversation=llm_call.prompt.messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        text = llm.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
        )
        prompt_token_ids = llm.tokenizer.apply_chat_template(
            llm_call.prompt.messages,
            add_special_tokens=True,
            add_generation_prompt=True,
        )

    output_text = text[len(prompt_text) :]

    # Get the appropriate tokenizer (from processor if using vision model)
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
    finished = llm_call.output.content.endswith(tokenizer.eos_token)
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
