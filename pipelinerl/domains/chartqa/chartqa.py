import base64
import io
import logging
import time
from typing import Dict, Any

import aiohttp
from omegaconf import DictConfig
from pydantic import BaseModel
from PIL import Image

from pipelinerl.rollouts import RolloutResult
from tapeagents.core import Prompt
from pipelinerl.async_vlm import TrainableVLM

from pipelinerl.async_vlm import vlm_async_generate, make_multimodal_training_text
from .evaluation import evaluate_answer

logger = logging.getLogger(__name__)


class ChartQARewardTable(BaseModel):
    wrong_answer_not_finished: float
    wrong_answer_finished: float
    no_answer_not_finished: float
    no_answer_finished: float
    unparsable_not_finished: float
    unparsable_finished: float
    correct_answer_not_finished: float
    correct_answer_finished: float


def encode_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image;base64,{img_str}"


def create_multimodal_message(image: Image.Image, question: str) -> Dict[str, Any]:
    """Create a multimodal message with image and text."""
    image_base64 = encode_image_to_base64(image)
    
    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_base64
                }
            },
            {
                "type": "text", 
                "text": question
            }
        ]
    }


async def generate_chartqa_rollout(
    cfg: DictConfig,
    llm: TrainableVLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    """Generate a rollout for ChartQA domain."""
    messages = []
    
    # Add system prompt if specified
    if cfg.actor.system_prompt:
        messages.append({"role": "system", "content": cfg.actor.system_prompt})
    
    # Create the multimodal user message with chart image and question
    question_text = cfg.actor.task_template.format(question=problem["question"])
    multimodal_message = create_multimodal_message(problem["image"], question_text)
    messages.append(multimodal_message)
    
    prompt = Prompt(messages=messages)

    time_start = time.time()
    vlm_call = await vlm_async_generate(llm, prompt, session)
    latency = time.time() - time_start

    assert vlm_call.output.content is not None
    rewards = ChartQARewardTable(**dict(cfg.rewards))
    discount_factor = cfg.actor.discount_factor

    # Evaluate the answer using our custom evaluation logic
    try:
        answer_status = evaluate_answer(vlm_call.output.content, problem["answer"])
    except Exception as e:
        logger.error(f"Error evaluating answer: {e}")
        answer_status = "unparsable"

    try:
        trace = make_multimodal_training_text(llm, vlm_call)
        # Check if the generation is finished (ended with EOS token)
        finished = 1 if trace.input_ids[-1] == llm.tokenizer.eos_token_id else 0
    except Exception as e:
        logger.error(f"Error creating training text: {e}")
        raise

    # Determine reward based on answer status and finished state
    try:
        match (answer_status, finished):
            case ("wrong", 0):
                reward = rewards.wrong_answer_not_finished
            case ("wrong", 1):
                reward = rewards.wrong_answer_finished
            case ("no_answer", 0):
                reward = rewards.no_answer_not_finished
            case ("no_answer", 1):
                reward = rewards.no_answer_finished
            case ("unparsable", 0):
                reward = rewards.unparsable_not_finished
            case ("unparsable", 1):
                reward = rewards.unparsable_finished
            case ("correct", 0):
                reward = rewards.correct_answer_not_finished
            case ("correct", 1):
                reward = rewards.correct_answer_finished
            case _:
                raise ValueError(f"Invalid answer_status/finished combination: {answer_status}/{finished}")

        # Apply discount factor based on output length
        reward *= discount_factor**vlm_call.output_length_tokens
        trace.reward = reward
    except Exception as e:
        logger.error(f"Error calculating reward: {e}")
        raise

    metrics = {
        "reward": reward,
        "success": answer_status == "correct",
        "no_error": answer_status != "unparsable",
        "no_answer": answer_status == "no_answer",
        "overflow": 0 if finished else 1,
    }

    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        dataset_name=problem.get("dataset"),
        prompt_tokens=[vlm_call.prompt_length_tokens],
        output_tokens=[vlm_call.output_length_tokens],
        latency=latency,
    )