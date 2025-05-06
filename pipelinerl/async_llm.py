import logging
import aiohttp

from tapeagents.core import LLMCall, LLMOutput, Prompt, TokenLogprob
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
        data.update({
            "logprobs": 1,
            "include_stop_str_in_output": True,
            "skip_special_tokens": False,
        })
    
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

        logprobs = None
        if llm.collect_logprobs:
            prompt_token_ids = llm.tokenizer.apply_chat_template(
                prompt.messages, add_special_tokens=True, add_generation_prompt=True
            )
            completion_logprobs = data["choices"][0]["logprobs"]["content"]
            logprobs = llm.make_llm_call_logprobs(prompt_token_ids, completion_logprobs)
            # <end_of_turn> is the end of message for Gemma2B, eos_token is wrong for this model
            for eos_str in [llm.tokenizer.eos_token, "<end_of_turn>"]:
                if content.endswith(eos_str):
                    # the eos was added in the case where self.collect_logprobs is True
                    # TapeAgents is not expecting the eos token in the completion
                    content = content[: -len(eos_str)]
    except Exception as e:
        logger.exception(f"Failed to parse llm response: {data}")
        raise e
        
    output = LLMOutput(content=content)
    llm_call = llm.log_output(prompt, output)
    assert llm_call is not None, "llm_call is None"
    llm_call.logprobs = logprobs
    return llm_call