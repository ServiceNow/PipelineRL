from __future__ import annotations

import time


class SlidingWindowAggregator:
    def __init__(self, window_size: int):
        self.window_size = max(1, int(window_size))
        self.prompt_tokens_window: list[list[int]] = []
        self.output_tokens_window: list[list[int]] = []
        self.timestamps: list[float] = []

    def update(self, prompt_tokens: list[int], output_tokens: list[int]) -> None:
        self.prompt_tokens_window.append(prompt_tokens)
        self.output_tokens_window.append(output_tokens)
        self.timestamps.append(time.time())
        if len(self.prompt_tokens_window) > self.window_size:
            self.prompt_tokens_window.pop(0)
            self.output_tokens_window.pop(0)
            self.timestamps.pop(0)

    def get_stats(self) -> dict[str, float] | None:
        if len(self.prompt_tokens_window) < self.window_size:
            return None

        null_stats = {
            "samples_per_second": 0.0,
            "output_tokens_per_second": 0.0,
            "prompt_tokens_per_second": 0.0,
            "total_tokens_per_second": 0.0,
        }
        if not self.timestamps:
            return null_stats

        time_span = self.timestamps[-1] - self.timestamps[0]
        if time_span < 1e-6:
            return null_stats

        num_samples = sum(len(tokens) for tokens in self.prompt_tokens_window)
        total_output_tokens = sum(sum(tokens) for tokens in self.output_tokens_window)
        total_prompt_tokens = sum(sum(tokens) for tokens in self.prompt_tokens_window)

        return {
            "samples_per_second": num_samples / time_span,
            "output_tokens_per_second": total_output_tokens / time_span,
            "prompt_tokens_per_second": total_prompt_tokens / time_span,
            "total_tokens_per_second": (total_output_tokens + total_prompt_tokens) / time_span,
        }
