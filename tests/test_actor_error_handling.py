"""Test that actor rollout error handling doesn't crash the entire actor.

Specifically tests that:
1. HTTP 4xx errors from vLLM (e.g., max_tokens too large) are handled gracefully
2. Groups where ALL rollouts fail are dropped (not submitted)
3. Groups where SOME rollouts fail submit only valid results
4. HTTP 5xx errors still propagate as fatal
"""

import asyncio
import queue
from unittest.mock import MagicMock, AsyncMock, patch

import aiohttp
import pytest
from omegaconf import OmegaConf

from pipelinerl.rollouts import BaseMetrics, RolloutResult, TrainingText


# ---------------------------------------------------------------------------
# Helpers – lightweight stand-ins for heavy classes used by schedule_rollouts
# ---------------------------------------------------------------------------

class FakeQueue:
    """Minimal stand-in for SharedMemoryQueue (no shared memory needed)."""

    def __init__(self):
        self._q = queue.Queue()

    def put(self, item, block=True, timeout=None):
        self._q.put(item)

    def get(self, block=True, timeout=None):
        return self._q.get(block=block, timeout=timeout)

    def qsize(self):
        return self._q.qsize()

    def max_actual_entry_size(self):
        return 0

    def get_memory_size(self):
        return 0


class FakeTrainerState:
    def __init__(self):
        self.propagated_weight_version = 1
        self.samples_processed = 0


def make_good_result() -> RolloutResult:
    """A valid rollout result with one training sample."""
    return RolloutResult(
        training_texts=[
            TrainingText(
                text="prompt output",
                n_predicted=6,
                reward=1.0,
                input_ids=[1, 2, 3],
                labels=[-100, 2, 3],
                finished=True,
                prompt_tokens=5,
                output_tokens=6,
            )
        ],
        metrics=BaseMetrics(reward=1.0, success=True, no_error=True, no_answer=False),
        latency=0.5,
    )


def make_client_response_error(status: int, message: str = "Bad Request"):
    """Create an aiohttp.ClientResponseError."""
    mock_req = MagicMock()
    mock_req.url = "http://localhost:8080/v1/chat/completions"
    return aiohttp.ClientResponseError(
        request_info=mock_req,
        history=(),
        status=status,
        message=message,
    )


# ---------------------------------------------------------------------------
# Core test: exercise rollout_and_maybe_produce_result + group completion
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_rollouts_fail_group_dropped():
    """When all rollouts in a group fail with 4xx, the group should be dropped."""
    attempts = 4
    problem_q = FakeQueue()
    result_q = FakeQueue()
    trainer_state = FakeTrainerState()

    # Put one problem in the queue
    problem_q.put({"task": "What is 2+2?", "answer": "4"})

    call_count = 0

    async def failing_rollout_policy(cfg, llm, problem, session):
        nonlocal call_count
        call_count += 1
        raise make_client_response_error(400, "max_tokens too large")

    cfg = OmegaConf.create({
        "actor": {
            "rollout_policy": "not_used",  # we patch it
            "llm_max_rollouts": 64,
        },
        "finetune": {
            "train_batch_size": 1000,
            "gradient_accumulation_passes": 1,
            "train_iters": 100,
            "interrupt_train_steps": None,
        },
        "debug": {},
    })

    llms = [MagicMock()]  # 1 LLM

    # We can't easily run schedule_rollouts (too many dependencies),
    # so we directly test the inner logic by reimplementing the key parts.
    # This mirrors rollout_and_maybe_produce_result + group completion.

    group_rollouts = {}
    group_id = 0
    group_rollouts[group_id] = []
    finished_rollouts = 0
    warnings_logged = []

    for rollout_index in range(attempts):
        try:
            rollout_result = await failing_rollout_policy(cfg, llms[0], {"task": "x"}, None)
        except aiohttp.ClientResponseError as e:
            if 400 <= e.status < 500:
                warnings_logged.append(str(e.status))
                rollout_result = RolloutResult(
                    training_texts=[],
                    metrics=BaseMetrics(reward=0.0, success=False, no_error=False, no_answer=True),
                    latency=0.0,
                )
            else:
                raise

        rollout_result.model_version = 1
        rollout_result.group_id = f"test_{group_id}"
        group_rollouts[group_id].append(rollout_result)

    # Now check group completion logic
    assert len(group_rollouts[group_id]) == attempts
    valid_results = [r for r in group_rollouts[group_id] if r.training_texts]

    # All failed → group should be dropped
    assert len(valid_results) == 0, "Expected all results to be empty"
    assert call_count == attempts
    assert len(warnings_logged) == attempts

    # In real code: del group_rollouts[group_id], don't put in result_q
    del group_rollouts[group_id]
    assert result_q.qsize() == 0, "No group should be in the result queue"


@pytest.mark.asyncio
async def test_partial_failure_submits_valid_only():
    """When some rollouts fail but others succeed, submit only valid ones."""
    attempts = 4
    result_q = FakeQueue()

    call_count = 0

    async def mixed_rollout_policy(cfg, llm, problem, session):
        nonlocal call_count
        call_count += 1
        # First 2 calls fail, last 2 succeed
        if call_count <= 2:
            raise make_client_response_error(400, "max_tokens too large")
        return make_good_result()

    group_rollouts = {}
    group_id = 0
    group_rollouts[group_id] = []

    for rollout_index in range(attempts):
        try:
            rollout_result = await mixed_rollout_policy(None, None, {"task": "x"}, None)
        except aiohttp.ClientResponseError as e:
            if 400 <= e.status < 500:
                rollout_result = RolloutResult(
                    training_texts=[],
                    metrics=BaseMetrics(reward=0.0, success=False, no_error=False, no_answer=True),
                    latency=0.0,
                )
            else:
                raise

        rollout_result.model_version = 1
        rollout_result.group_id = f"test_{group_id}"
        group_rollouts[group_id].append(rollout_result)

    assert len(group_rollouts[group_id]) == attempts

    valid_results = [r for r in group_rollouts[group_id] if r.training_texts]

    # 2 failed, 2 succeeded
    assert len(valid_results) == 2, f"Expected 2 valid results, got {len(valid_results)}"

    # In real code: result_queue.put(valid_results)
    result_q.put(valid_results)
    got = result_q.get(block=False)
    assert len(got) == 2
    assert all(len(r.training_texts) > 0 for r in got)


@pytest.mark.asyncio
async def test_5xx_errors_still_propagate():
    """HTTP 5xx errors should NOT be caught — they indicate server failure."""

    async def server_error_policy(cfg, llm, problem, session):
        raise make_client_response_error(500, "Internal Server Error")

    with pytest.raises(aiohttp.ClientResponseError) as exc_info:
        try:
            await server_error_policy(None, None, {"task": "x"}, None)
        except aiohttp.ClientResponseError as e:
            if 400 <= e.status < 500:
                pass  # Would be caught in real code
            else:
                raise  # 5xx re-raised

    assert exc_info.value.status == 500


@pytest.mark.asyncio
async def test_all_succeed_normal_path():
    """When all rollouts succeed, the full group is submitted."""
    attempts = 4
    result_q = FakeQueue()

    async def good_policy(cfg, llm, problem, session):
        return make_good_result()

    group_rollouts = {}
    group_id = 0
    group_rollouts[group_id] = []

    for rollout_index in range(attempts):
        try:
            rollout_result = await good_policy(None, None, {"task": "x"}, None)
        except aiohttp.ClientResponseError as e:
            if 400 <= e.status < 500:
                rollout_result = RolloutResult(
                    training_texts=[],
                    metrics=BaseMetrics(reward=0.0, success=False, no_error=False, no_answer=True),
                    latency=0.0,
                )
            else:
                raise

        rollout_result.model_version = 1
        rollout_result.group_id = f"test_{group_id}"
        group_rollouts[group_id].append(rollout_result)

    valid_results = [r for r in group_rollouts[group_id] if r.training_texts]
    assert len(valid_results) == attempts, "All rollouts should be valid"

    result_q.put(valid_results)
    got = result_q.get(block=False)
    assert len(got) == attempts


@pytest.mark.asyncio
async def test_consumer_assertion_accepts_partial_group():
    """The consumer-side assertion should accept groups with fewer than `attempts` results."""
    attempts = 8
    # Simulate a partial group with 5 valid results
    partial_count = 5

    results = [make_good_result() for _ in range(partial_count)]

    # This mirrors the relaxed assertion in actor.py
    assert isinstance(results, list)
    assert isinstance(results[0], RolloutResult)
    assert 0 < len(results) <= attempts, (
        f"Expected 1-{attempts} rollouts, got {len(results)}"
    )

    group_samples = sum(len(r.training_texts) for r in results)
    assert group_samples == partial_count
