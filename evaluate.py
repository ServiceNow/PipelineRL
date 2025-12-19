import json
import asyncio
import aiohttp
from datasets import load_dataset, Dataset
from i3_logic.task2verifier import verifier_classes
from i3_logic.base import Data
import os
import random
from tqdm import tqdm

random.seed(42)

# ************************
# LLM Inference Module


async def llm_inference(prompt: list) -> str:
    """
    LLM inference function.
    Calls the vLLM OpenAI-compatible API endpoint.
    """
    

    VLLM_ENDPOINT = "http://localhost:8000/v1/chat/completions"
    VLLM_MODEL_NAME = "apriel"

    async def _infer():
        headers = {"Content-Type": "application/json"}
        data = {
            "model": VLLM_MODEL_NAME,
            "messages": prompt,
            "max_tokens": 16384,  # Set maximum tokens to 16K
        }
        timeout = aiohttp.ClientTimeout(total=6000) 
        async with aiohttp.ClientSession(timeout=timeout) as session:
            print('*** Entered queue ****')
            async with session.post(
                VLLM_ENDPOINT, headers=headers, data=json.dumps(data)
            ) as resp:
                resp_json = await resp.json()
                # Extract the response text from the API response
                return resp_json["choices"][0]["message"]["content"]

    return await _infer()


def post_process(generation: str) -> str:
    try:
        if "[BEGIN FINAL RESPONSE]" in generation:
            generation = generation.split("[BEGIN FINAL RESPONSE]")[1]
            generation = generation.replace("[END FINAL RESPONSE]", "")
    except Exception as e:
        print(f"Postprocessing failed: {e}")

    return generation.replace("<|end|>", "").strip()


async def _infer_one(prompt, idx, sem, results):
    async with sem:
        response = await llm_inference(prompt)
        response = post_process(response)
        results[idx] = response


def infer_large_batch(prompts: list) -> list:
    """
    Perform LLM inference on a batch of prompts in parallel, with a cap of 100 concurrent inferences.
    Order of results is preserved. Shows tqdm progress bar.
    """

    async def _main():
        sem = asyncio.Semaphore(25)
        results = [None] * len(prompts)
        pbar = tqdm(total=len(prompts), desc="LLM Inference", ncols=80)

        async def _infer_one_with_pbar(prompt, idx, sem, results):
            await _infer_one(prompt, idx, sem, results)
            pbar.update(1)

        tasks = [
            asyncio.create_task(_infer_one_with_pbar(prompt, idx, sem, results))
            for idx, prompt in enumerate(prompts)
        ]
        await asyncio.gather(*tasks)
        pbar.close()
        return results

    return asyncio.run(_main())


# *****************
# Data processing
def load_data() -> list:
    dataset = load_dataset(
        "ServiceNow-AI/mixed-training-text-datasets",
        "prime-intellect-logic",
        token=os.environ.get("HF_TOKEN"),
    )["train"]

    dataset = dataset.filter(
        lambda sample: sample["ability"] == "logic"
        and json.loads(sample["extra_info"]).get("task", "") in verifier_classes
    )
    dataset = dataset.to_list()

    processed_dataset = []
    for sample in dataset:
        reward_data = json.loads(sample["reward_model"]["ground_truth"])
        processed_dataset.append(
            {
                "dataset": "logic",
                # Single turn only.
                "task": sample["prompt"][0]["content"],
                "reward_context": reward_data,
                "extra_info": json.loads(sample["extra_info"]),
            }
        )

    return processed_dataset

def get_verifier(extra_info):
    return verifier_classes.get(extra_info.get('task'))()

def verify(request, generation):
    reward_context = request['reward_context']
    extra_info = request.get('extra_info', {})
    verifier = get_verifier(extra_info)
    try:
        return verifier.verify(Data.from_json_dict(reward_context), generation)
    except Exception as e:
        breakpoint()
        print(f"Verification failed: {e}")
        raise e

def __main():
    dataset = load_data()
    random.shuffle(dataset)
    dataset = dataset[:250]

    print(f"Loaded {len(dataset)} samples for inference.")
    prompts = [[{"role": "user", "content": item["task"]}] for item in dataset]
    results = infer_large_batch(prompts)

    success = [verify(dataset[i], results[i]) for i in range(len(dataset))]
    print(success)
    print(f"Accuracy: {sum([1.0 if x else 0.0 for x in success])/len(success)}")
    breakpoint()




if __name__ == "__main__":
    __main()
