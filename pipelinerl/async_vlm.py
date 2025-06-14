import logging
import aiohttp
from typing import Optional
import torch
import base64
import io

from pipelinerl.finetune.data import MASKED_TOKEN_ID

from tapeagents.core import LLMCall, LLMOutput, Prompt, TokenLogprob, TrainingText
from tapeagents.llms.trainable import TrainableLLM
from transformers import AutoProcessor
from PIL import Image

logger = logging.getLogger(__name__)


class VLMCall(LLMCall):
    """Extended LLMCall with vision-language model metrics."""
    
    num_images: int = 0
    image_sizes: Optional[list] = None


class MultimodalTrainingText(TrainingText):
    """Extended TrainingText class for multimodal inputs with visual features."""
    model_config = {"arbitrary_types_allowed": True}
    
    pixel_values: Optional[list] = None
    image_thw: Optional[list] = None


class TrainableVLM(TrainableLLM):
    """Extended TrainableLLM with vision-language model support."""
    
    processor: Optional[object] = None
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def load_processor(self):
        """Load AutoProcessor for vision-language models."""
        if self.processor is None:
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                logger.info(f"Loaded processor for {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to load processor for {self.model_name}: {e}")
                self.processor = None


async def vlm_async_generate(vlm: TrainableVLM, prompt: Prompt, session: aiohttp.ClientSession) -> VLMCall:
    """Async generate function for vision-language models."""
    vlm.load_processor()
    
    # Extract images and calculate metrics
    images = extract_images_from_messages(prompt.messages)
    num_images = len(images)
    image_sizes = [img.size for img in images] if images else None
    
    headers = {"Content-Type": "application/json"}
    if vlm.api_token:
        headers |= {"Authorization": f"Bearer {vlm.api_token}"}
    data = {
        "model": vlm.model_name,
        "messages": prompt.messages,
        "stream": vlm.stream,
    }
    if vlm.collect_logprobs:
        logprob_data = {
            "logprobs": 1,
            "include_stop_str_in_output": True,
            "skip_special_tokens": False,
            "echo": True,  
        }
        data.update(logprob_data)

    logger.debug(f"POST request to {vlm.base_url}/v1/chat/completions")

    async with session.post(
        url=f"{vlm.base_url}/v1/chat/completions",
        json=data | vlm.parameters,
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
        if vlm.collect_logprobs:
            prompt_logprobs = data['prompt_logprobs'][1:]
            for prompt_logprob in prompt_logprobs:
                for k, v in prompt_logprob.items():
                    parsed_logprobs.append(
                        TokenLogprob(
                            token_id=int(k),
                            logprob=v["logprob"],
                            generated=0,
                        )
                    )
            
            completion_logprobs = data["choices"][0]["logprobs"]["content"]
            for completion_logprob in completion_logprobs:
                parsed_logprobs.append(
                    TokenLogprob(
                        token_id=int(completion_logprob["token"].split(":")[-1]),
                        logprob=completion_logprob["logprob"],
                        generated=1,
                    )
                )
            
    except Exception as e:
        logger.exception(f"Failed to parse vlm response: {data}")
        raise e

    output = LLMOutput(content=content)
    llm_call = vlm.log_output(prompt, output, count_tokens=False)
    assert llm_call is not None, "llm_call is None"
    llm_call.prompt_length_tokens = data['usage']['prompt_tokens']
    llm_call.output_length_tokens = data['usage']['completion_tokens']
    
    # Convert to VLMCall with image metrics
    vlm_call = VLMCall(
        **llm_call.model_dump(),
        num_images=num_images,
        image_sizes=image_sizes,
        logprobs=parsed_logprobs
    )
    
    return vlm_call


def extract_images_from_messages(messages):
    """Extract PIL Images from multimodal messages."""
    if messages is None:
        raise ValueError("Messages cannot be None")
    
    images = []
    for message in messages:
        if isinstance(message.get('content'), list):
            for content_item in message['content']:
                if content_item is None:
                    continue
                if content_item.get('type') == 'image' and 'image' in content_item:
                    images.append(content_item['image'])
                elif content_item.get('type') == 'image_url' and 'image_url' in content_item:
                    # Handle base64 format
                    url = content_item['image_url']['url']
                    if url.startswith('data:image;base64,'):
                        try:
                            base64_data = url.split('data:image;base64,')[1]
                            image_data = base64.b64decode(base64_data)
                            image = Image.open(io.BytesIO(image_data))
                            images.append(image)
                        except Exception as e:
                            logger.warning(f"Failed to decode base64 image: {e}")
    
    if not images:
        raise ValueError("No images found in messages")
    
    return images


def make_multimodal_training_text(vlm: TrainableVLM, vlm_call: VLMCall) -> MultimodalTrainingText:
    """Create multimodal training text with visual features."""
    from pipelinerl.async_llm import make_training_text
    
    # Start with regular training text
    training_text = make_training_text(vlm, vlm_call)
    
    # Extract visual features if present
    pixel_values = None
    image_thw = None
    
    if hasattr(vlm_call.prompt, 'messages'):
        images = extract_images_from_messages(vlm_call.prompt.messages)
        if images:
            # Load processor if not already loaded
            if vlm.processor is None:
                vlm.load_processor()
            
            if vlm.processor is not None:
                try:
                    # Process images and text with the model's processor
                    # Apply chat template to get text representation
                    text = vlm.processor.apply_chat_template(
                        vlm_call.prompt.messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    
                    # Process with both text and images
                    processed = vlm.processor(
                        text=[text],
                        images=images,
                        padding=True,
                        return_tensors=None
                    )
                    # Convert numpy arrays to lists for JSON serialization
                    pixel_values = processed.pixel_values.tolist() # num_channels, image_size, image_size
                    image_thw = processed.image_grid_thw.tolist() # 3
                except Exception as e:
                    raise ValueError(f"Failed to process images with processor: {e}")
    
    # Create multimodal training text with visual features
    assert pixel_values is not None or image_thw is not None, "No visual features found in llm_call"
    return MultimodalTrainingText(
        **training_text.model_dump(),
        pixel_values=pixel_values,
        image_thw=image_thw
    )