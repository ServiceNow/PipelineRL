import base64
import io
import logging
import time
from typing import Dict, Any

import aiohttp
from omegaconf import DictConfig
from pydantic import BaseModel
from PIL import Image

from pipelinerl.rollouts import RolloutResult, BaseMetrics
from tapeagents.core import Prompt
from tapeagents.llms.trainable import TrainableLLM
from pipelinerl.async_llm import llm_async_generate, make_training_text
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
    llm: TrainableLLM,
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
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - time_start

    assert llm_call.output.content is not None
    rewards = ChartQARewardTable(**dict(cfg.rewards))
    discount_factor = cfg.actor.discount_factor

    # Evaluate the answer using our custom evaluation logic
    if llm.tokenizer.eos_token is not None and llm_call.output.content.endswith(llm.tokenizer.eos_token):
        content = llm_call.output.content[:-len(llm.tokenizer.eos_token)] 
    else:
        content = llm_call.output.content
    try:
        answer_status = evaluate_answer(content, problem["answer"])
    except Exception as e:
        logger.error(f"Error evaluating answer: {e}")
        answer_status = "unparsable"

    try:
        trace = make_training_text(llm, llm_call)
        # Check if the generation is finished (ended with EOS token)
    except Exception as e:
        logger.error(f"Error creating training text: {e}")
        raise

    # Determine reward based on answer status and finished state
    try:
        match (answer_status, trace.finished):
            case ("wrong", False):
                reward = rewards.wrong_answer_not_finished
            case ("wrong", True):
                reward = rewards.wrong_answer_finished
            case ("no_answer", False):
                reward = rewards.no_answer_not_finished
            case ("no_answer", True):
                reward = rewards.no_answer_finished
            case ("unparsable", False):
                reward = rewards.unparsable_not_finished
            case ("unparsable", True):
                reward = rewards.unparsable_finished
            case ("correct", False):
                reward = rewards.correct_answer_not_finished
            case ("correct", True):
                reward = rewards.correct_answer_finished
            case _:
                raise ValueError(f"Invalid answer_status/finished combination: {answer_status}/{trace.finished}")

        # Apply discount factor based on output length
        reward *= discount_factor**llm_call.output_length_tokens
        trace.reward = reward
    except Exception as e:
        logger.error(f"Error calculating reward: {e}")
        raise

    metrics = BaseMetrics(
        reward=reward,
        success=answer_status == "correct",
        no_error=answer_status != "unparsable",
        no_answer=answer_status == "no_answer",
    )

    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        dataset_name=problem.get("dataset"),
        latency=latency,
    )