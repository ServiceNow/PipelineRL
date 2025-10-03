import json
import os
import time

import numpy as np
import ray
import requests
from tapeagents.llms import TrainableLLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"

llm_url = "http://localhost:8080"
# llm_model = "Qwen/Qwen3-8B"
llm_model = "Qwen/Qwen2.5-7B"
# exp_name = "qwen3-8b"
exp_name = "qwen2.5-7b"

def llm_quick_response(prompt: str):
    t = time.perf_counter()
    r = requests.post(
        url=f"{llm_url}/v1/chat/completions",
        json={
            "model": llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        headers={"Content-Type": "application/json"},
        stream=False,
        verify=False,
    )
    d = r.json()
    dt = time.perf_counter() - t
    return d["choices"][0]["message"]["content"], dt


llm = TrainableLLM(base_url=llm_url, model_name=llm_model)
response = llm.quick_response("Hello, how are you?")
response2, _ = llm_quick_response("Hello, how are you?")
assert len(response) > 0
assert len(response2) > 0
assert llm.tokenizer is not None
print("LLM is ready")


with open("debug_training_texts.jsonl", "r", encoding="utf-8") as f:
    all_dicts = [json.loads(line) for line in f if line.strip()]
total_tokens = 0
for d in all_dicts:
    text = d["text"]
    n_predicted = d["n_predicted"]
    prompt = text[:-n_predicted]
    response = text[-n_predicted:]
    tokens = llm.tokenizer.encode(text)
    total_tokens += len(tokens)
print(f"Loaded {len(all_dicts)} texts, total tokens: {total_tokens}")

prompts = [d["text"][:-d["n_predicted"]] for d in all_dicts]
chunk_size = 4
prompts_chunks = [prompts[i:i+chunk_size] for i in range(0, len(prompts), chunk_size)]
print(f"Chunked to {len(prompts_chunks)} chunks")


def benchmark_llm(n_workers: int):
    ray.shutdown()
    ray.init(num_cpus=n_workers)

    def get_responses(prompts: str):
        responses = []
        # local_llm = TrainableLLM(base_url=llm_url, model_name=llm_model)
        for i, prompt in enumerate(prompts):
            r, dt = llm_quick_response(prompt)
            responses.append((prompt + r, dt))
        return responses

    remote_fn = ray.remote(get_responses)

    t = time.perf_counter()

    chunks = prompts_chunks
    if n_workers > len(chunks):
        multiplier = n_workers // len(chunks) + 1
        chunks = chunks * multiplier
        print(f"Multiplied to {len(chunks)} chunks")
    unfinished_tasks = []
    for chunk in chunks:
        unfinished_tasks.append(remote_fn.remote(chunk))

    responses = []
    total_tokens = 0
    total_finished = 0
    latencies = []
    print(f"Submitted {len(unfinished_tasks)} tasks")
    while unfinished_tasks:
        finished_tasks, unfinished_tasks = ray.wait(unfinished_tasks, num_returns=len(unfinished_tasks), timeout=0.1)
        for finished_task in finished_tasks:
            responses = ray.get(finished_task)
            total_finished += 1
            for response, dt in responses:
                latencies.append(dt)
                tokens = llm.tokenizer.encode(response)
                total_tokens += len(tokens)
        dt = time.perf_counter() - t
        if len(finished_tasks) > 0:
            print(f"t: {dt:.2f}s, {total_finished} finished, Total tokens: {total_tokens}, tokens/sec: {total_tokens / dt:.2f}")
        # if dt > 600:
        #     print("Timeout 10 minutes, stopping")
        #     break
        time.sleep(1.0)

    final_time = time.perf_counter() - t
    print(f"Final, workers:{n_workers}, t:{final_time:.2f}s, total tokens: {total_tokens}, tokens/sec: {total_tokens / final_time:.2f}")
    ray.shutdown()
    mean_latency = np.mean(latencies)
    return total_tokens, final_time, mean_latency

stats = {}
for n_workers in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    print(f"Benchmarking {n_workers} workers..")
    tokens, dt, mean_latency = benchmark_llm(n_workers)
    print(f"Done {n_workers} workers: {tokens} tokens, {dt:.2f}s, speed {tokens / dt:.2f} tokens/sec, mean latency: {mean_latency:.2f}s")
    stats[n_workers] = {"tokens": tokens, "dt": dt, "mean_latency": mean_latency}
    with open(f"llm_token_stats_chunk{chunk_size}_{exp_name}.jsonl", "a") as f:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        row = json.dumps({"ts": ts, "n_workers": n_workers, "tokens": tokens, "dt": dt, "mean_latency": mean_latency})
        f.write(row + "\n")

print("Benchmarking done")
with open(f"llm_token_stats_all_chunk{chunk_size}_{exp_name}.json", "w") as f:
    json.dump(stats, f, indent=4)
print("All stats saved")