import logging
import aiohttp

from pipelinerl.finetune.data import MASKED_TOKEN_ID

from tapeagents.core import LLMCall, LLMOutput, Prompt, TokenLogprob, TrainingText
from tapeagents.llms.trainable import TrainableLLM


logger = logging.getLogger(__name__)


async def llm_async_generate(llm: TrainableLLM, prompt: Prompt, session: aiohttp.ClientSession) -> LLMCall:
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
        logprob_data = {
            "logprobs": 1,
            "include_stop_str_in_output": True,
            "skip_special_tokens": False,
            "echo": True,  
        }
        data.update(logprob_data)

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
            prompt_logprobs = data['prompt_logprobs'][1:]
            for prompt_logprob in prompt_logprobs:
                for k, v in prompt_logprob.items():
                    parsed_logprobs.append(
                        TokenLogprob(
                            token_id=int(k),
                            logprob=v["logprob"],
                            generated=0,  # Prompt tokens are not generated
                        )
                    )
            
            completion_logprobs = data["choices"][0]["logprobs"]["content"]
            for completion_logprob in completion_logprobs:
                parsed_logprobs.append(
                    TokenLogprob(
                        token_id=int(completion_logprob["token"].split(":")[-1]),  # Extract token ID from 'token_id:1271' format
                        logprob=completion_logprob["logprob"],
                        generated=1,  # Completion tokens are generated
                    )
                )
            
    except Exception as e:
        logger.exception(f"Failed to parse llm response: {data}")
        raise e

    output = LLMOutput(content=content)
    llm_call = llm.log_output(prompt, output, count_tokens=False)
    llm_call.prompt_length_tokens = data['usage']['prompt_tokens']
    llm_call.output_length_tokens = data['usage']['completion_tokens']
    assert llm_call is not None, "llm_call is None"
    llm_call.logprobs = parsed_logprobs
    return llm_call


def make_training_text(llm: TrainableLLM, llm_call: LLMCall) -> TrainingText:
    training_text = llm.make_training_text(llm_call.prompt, llm_call.output)
    if not llm_call.logprobs:
        raise ValueError("Logprobs are required to make training data for RL")
    
    # Extract prompt tokens from logprobs if available (when echo=True)
    # This ensures we use the exact same tokenization as VLLM
    prompt_logprobs = [lp for lp in llm_call.logprobs if lp.generated == 0]
    completion_logprobs = [lp for lp in llm_call.logprobs if lp.generated == 1]
    
    # Use prompt tokens from VLLM (consistent with server tokenization)
    prompt_token_ids = [lp.token_id for lp in prompt_logprobs]
    labels = [lp.token_id for lp in completion_logprobs]
    logprobs = [lp.logprob for lp in completion_logprobs]
    
    input_ids = prompt_token_ids + labels
    # Apply masking to input tokens that aren't generated
    labels = [MASKED_TOKEN_ID] * len(prompt_token_ids) + labels
    training_text.input_ids = input_ids
    training_text.labels = labels
    training_text.logprobs = logprobs
    return training_text
