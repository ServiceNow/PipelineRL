import asyncio
import json
import os
import random
import time

import aiohttp
import numpy as np
from tapeagents.llms import TrainableLLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"

llm_url = "http://localhost:8080"
llm_model = "Qwen/Qwen3-8B"
# llm_model = "Qwen/Qwen2.5-7B"
exp_name = "qwen3-8b-v1"
# exp_name = "qwen2.5-7b"
max_tokens = 8192


async def llm_quick_response_async(session: aiohttp.ClientSession, prompt: str):
    """Async version of LLM quick response"""
    async with session.post(
        url=f"{llm_url}/v1/chat/completions",
        json={
            "model": llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        headers={"Content-Type": "application/json"},
        ssl=False,
    ) as response:
        d = await response.json()
        return d["choices"][0]["message"]["content"]




# Initial LLM test (synchronous)
llm = TrainableLLM(base_url=llm_url, model_name=llm_model)
response = llm.quick_response("Hello, how are you?")
assert len(response) > 0
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
random.seed(42)
random.shuffle(prompts)
chunk_size = 4
prompts_chunks = [prompts[i:i+chunk_size] for i in range(0, len(prompts), chunk_size)]
print(f"Chunked to {len(prompts_chunks)} chunks")
too_many_chunks = prompts_chunks * 20


async def get_responses_async(session: aiohttp.ClientSession, prompts: list[str], tokenizer):
    """Process a chunk of prompts asynchronously"""
    responses = []
    for prompt in prompts:
        t = time.perf_counter()
        try:
            r = await llm_quick_response_async(session, prompt)
            dt = time.perf_counter() - t
            responses.append((prompt + r, dt))
        except Exception as e:
            print(f"Error processing prompt: {e}")
            dt = time.perf_counter() - t
            responses.append((prompt, dt))
    return responses


async def benchmark_llm_async(n_workers: int):
    """Benchmark LLM using async/await with controlled concurrency"""
    print(f"Starting async benchmark with {n_workers} concurrent workers")
    
    start_time = time.perf_counter()
    
    n_chunks = max(200, n_workers * 2)
    chunks = too_many_chunks[:n_chunks]
    print(f"Multiplied to {len(chunks)} chunks")
    random.seed(42)
    random.shuffle(chunks)
    
    total_tokens = 0
    total_finished = 0
    latencies = []
    
    # Create shared aiohttp session with connection pooling
    connector = aiohttp.TCPConnector(limit=n_workers, limit_per_host=n_workers)
    timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Create all tasks
        tasks = []
        for chunk in chunks:
            task = asyncio.create_task(get_responses_async(session, chunk, llm.tokenizer))
            tasks.append(task)
        
        print(f"Created {len(tasks)} tasks")
        
        # Process tasks with controlled concurrency
        pending = set(tasks)
        active = set()
        
        while pending or active:
            # Fill up active tasks up to n_workers limit
            while len(active) < n_workers and pending:
                task = pending.pop()
                active.add(task)
            
            if not active:
                break
            
            # Wait for at least one task to complete
            done, active = await asyncio.wait(active, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
            
            # Process completed tasks
            for finished_task in done:
                try:
                    responses = await finished_task
                    total_finished += 1
                    for response, latency in responses:
                        latencies.append(latency)
                        tokens = llm.tokenizer.encode(response)
                        total_tokens += len(tokens)
                except Exception as e:
                    print(f"Task failed with error: {e}")
                    total_finished += 1
            
            # Log progress
            dt = time.perf_counter() - start_time
            if len(done) > 0:
                print(f"t: {dt:.2f}s, {total_finished} finished, Total tokens: {total_tokens}, tokens/sec: {total_tokens / dt:.2f}, last 10 latency: {np.mean(latencies[-10:]) if latencies else 0:.2f}s")
                with open(f"llm_token_stats_chunk{chunk_size}_{exp_name}_log.jsonl", "a") as f:
                    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    row = json.dumps({
                        "ts": ts,
                        "exp_name": exp_name,
                        "n_workers": n_workers,
                        "tokens": total_tokens,
                        "dt": dt,
                        "mean_latency": np.mean(latencies) if latencies else 0,
                        "last_10_latency": np.mean(latencies[-10:]) if latencies else 0,
                        "total_finished": total_finished,
                        "token_speed": total_tokens / dt if dt > 0 else 0
                    })
                    f.write(row + "\n")
            
            # Check saturation mode
            if len(pending) + len(active) < n_workers:
                print(f"Saturation mode ended, stopping")
                # Cancel remaining tasks
                for task in active:
                    task.cancel()
                break
            
            await asyncio.sleep(2.0)
    
    final_time = time.perf_counter() - start_time
    print(f"Final, workers:{n_workers}, t:{final_time:.2f}s, total tokens: {total_tokens}, tokens/sec: {total_tokens / final_time:.2f}")
    mean_latency = np.mean(latencies) if latencies else 0
    return total_tokens, final_time, mean_latency


async def run_benchmarks():
    """Run benchmarks for different worker counts"""
    stats = {}
    for n_workers in [128]:  # [64, 256, 128, 32, 4, 8, 16, 512, 1024]: # most optimal first
        print(f"Benchmarking {n_workers} workers..")
        tokens, dt, mean_latency = await benchmark_llm_async(n_workers)
        print(f"Done {n_workers} workers: {tokens} tokens, {dt:.2f}s, speed {tokens / dt:.2f} tokens/sec, mean latency: {mean_latency:.2f}s")
        stats[n_workers] = {"tokens": tokens, "dt": dt, "mean_latency": mean_latency}
        with open(f"llm_token_stats_chunk{chunk_size}_{exp_name}.jsonl", "a") as f:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            row = json.dumps({"ts": ts, "n_workers": n_workers, "tokens": tokens, "dt": dt, "mean_latency": mean_latency})
            f.write(row + "\n")
        await asyncio.sleep(3.0)
    
    print("Benchmarking done")
    with open(f"llm_token_stats_all_chunk{chunk_size}_{exp_name}.json", "w") as f:
        json.dump(stats, f, indent=4)
    print("All stats saved")


if __name__ == "__main__":
    # Run the async benchmarks
    asyncio.run(run_benchmarks())

