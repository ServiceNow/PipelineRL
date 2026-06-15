"""In-training hint re-mining for hint-conditioned cube RL (Exp 1.5 / v2).

Moves the JefHinter mine->inject loop inside PipelineRL training: cube workers
report rendered trajectory transcripts after each train rollout, and a single
:class:`HintRefreshActor` periodically re-mines the per-task ``good`` hint map
with the JefHinter miner prompt (``cube_harness.jefhinter``), rebuilding the
``distractor`` map as a seeded shuffle of the good hints (same rule as
``cube-harness/scripts/build_hint_bank.py``).

Versioning / GRPO group consistency: every refresh that changes the good map
bumps an integer version and keeps a short per-version history of the maps.
The train scheduler (``launch.py``) pins one hint version per GRPO group and
passes it through ``rollout(hint_version=...)``; workers fetch the map for
exactly that version, so all ``attempts`` rollouts of a group share the same
hints even if a refresh lands mid-group. Version 0 means "no refresh yet" —
workers keep the statically seeded bank hints from the cube config.

The pure logic lives in :class:`HintRefreshState` (thread-safe, Ray-free, unit
testable with a fake LLM); :class:`HintRefreshActor` is a thin threaded Ray
wrapper so ``report``/``get_hints``/``get_version`` stay responsive while a
slow ``refresh`` (one miner LLM call per task) is running.

Research thread: ``plans_rl/research/ongoing/hint_conditioned_rl``.
"""

from __future__ import annotations

import logging
import random
import threading
from collections import deque

import ray
from cube_harness.jefhinter import MINER_SYSTEM_PROMPT, MINER_SYSTEM_PROMPT_GENERAL, mine_hint_from_transcripts
from cube_harness.llm import LLM, LLMConfig

logger = logging.getLogger(__name__)

_HISTORY_LIMIT = 8  # versions of (good, distractor) maps kept for pinned fetches


def build_miner_llm_config(llm_info: dict) -> LLMConfig:
    """Plain (non-routed) cube-harness LLMConfig from the actor llm kwargs.

    Same fields ``pipelinerl.cube_rl.domain.set_agent_llm_config`` uses; fixed
    miner sampling params (temperature 0.6, 1024 completion tokens) matching the
    frozen JefHinter harness defaults.
    """
    api_base = str(llm_info["base_url"])
    if not api_base.endswith("/v1"):
        api_base += "/v1"
    model_name = str(llm_info.get("served_model_name") or llm_info["model_name"])
    if not model_name.startswith("openai/"):
        model_name = f"openai/{model_name}"
    return LLMConfig(
        model_name=model_name,
        api_base=api_base,
        api_key="EMPTY",
        temperature=0.6,
        max_completion_tokens=1024,
    )


class HintRefreshState:
    """Rolling transcript buffers + versioned good/distractor hint maps."""

    def __init__(
        self,
        llm: LLM,
        refresh_min_episodes: int,
        max_per_task: int,
        seed: int,
        general_prompt: bool = False,
        reject_literals: bool = False,
    ) -> None:
        self._llm = llm
        self._refresh_min_episodes = int(refresh_min_episodes)
        self._max_per_task = int(max_per_task)
        self._seed = int(seed)
        # Instance-general miner knobs (default off -> original behavior). general_prompt
        # selects the literal-forbidding system prompt; reject_literals drops any mined hint
        # that still carries instance-specific tokens.
        self._system_prompt = MINER_SYSTEM_PROMPT_GENERAL if general_prompt else MINER_SYSTEM_PROMPT
        self._reject_literals = bool(reject_literals)
        self._lock = threading.Lock()
        self._buffers: dict[str, deque[tuple[float, str]]] = {}
        self._version = 0
        self._history: dict[int, dict[str, dict[str, str]]] = {0: {"good": {}, "distractor": {}}}
        self._refreshing = False

    def report(self, task_id: str, reward: float, transcript: str) -> None:
        with self._lock:
            buffer = self._buffers.setdefault(task_id, deque(maxlen=self._max_per_task))
            buffer.append((float(reward), transcript))

    def get_version(self) -> int:
        with self._lock:
            return self._version

    def get_hints(self, condition: str, version: int | None = None) -> tuple[int, dict[str, str]]:
        """(version, per-task hint map) for one condition; ``none`` -> (version, {}).

        ``version`` pins a historical map (GRPO group consistency); unknown or
        unset versions resolve to the latest.
        """
        with self._lock:
            resolved = version if version in self._history else self._version
            if condition == "none":
                return resolved, {}
            if condition not in ("good", "distractor"):
                raise ValueError(f"unknown hint condition {condition!r} (expected good | none | distractor)")
            return resolved, dict(self._history[resolved][condition])

    def _mining_items(self) -> list[tuple[str, str, str | None]]:
        """(task_id, failed_transcript, passed_transcript) for each eligible task."""
        items: list[tuple[str, str, str | None]] = []
        with self._lock:
            for task_id, buffer in self._buffers.items():
                if len(buffer) < self._refresh_min_episodes:
                    continue
                episodes = list(buffer)
                failed = [t for r, t in episodes if r <= 0]
                passed = [t for r, t in episodes if r > 0]
                if not failed:
                    continue  # nothing to learn from on this task this round
                items.append((task_id, failed[-1], passed[-1] if passed else None))
        return items

    def refresh(self) -> int:
        """Re-mine eligible tasks; on any good-map change bump + record a new version."""
        with self._lock:
            if self._refreshing:
                logger.info("hint refresh already in progress; skipping (version=%d)", self._version)
                return self._version
            self._refreshing = True
        try:
            items = self._mining_items()
            mined: dict[str, str] = {}
            for task_id, failed_transcript, passed_transcript in items:
                hint = mine_hint_from_transcripts(
                    self._llm,
                    task_id,
                    failed_transcript,
                    passed_transcript,
                    system_prompt=self._system_prompt,
                    reject_literals=self._reject_literals,
                )
                if hint is not None and hint.text.strip():
                    mined[task_id] = hint.text.strip()
            with self._lock:
                good = dict(self._history[self._version]["good"])
                changed = {t: h for t, h in mined.items() if good.get(t) != h}
                good.update(changed)
                if changed:
                    self._version += 1
                    self._history[self._version] = {"good": good, "distractor": _build_distractor(good, self._seed)}
                    while len(self._history) > _HISTORY_LIMIT:
                        del self._history[min(self._history)]
                logger.info(
                    "hint refresh: %d tasks eligible, %d hints mined, %d changed -> version %d (%d good hints)",
                    len(items),
                    len(mined),
                    len(changed),
                    self._version,
                    len(good),
                )
                return self._version
        finally:
            with self._lock:
                self._refreshing = False


def _build_distractor(good: dict[str, str], seed: int) -> dict[str, str]:
    """A real hint for the WRONG task, per task — same rule as build_hint_bank.build_bank."""
    task_ids = sorted(good)
    rng = random.Random(seed)
    distractor: dict[str, str] = {}
    for tid in task_ids:
        others = [t for t in task_ids if t != tid]
        if others:
            distractor[tid] = good[rng.choice(others)]
    return distractor


@ray.remote(max_restarts=0, max_task_retries=0, num_cpus=0, max_concurrency=8)
class HintRefreshActor:
    """Threaded Ray wrapper over :class:`HintRefreshState` (state is lock-guarded)."""

    def __init__(
        self,
        llm_info: dict,
        refresh_min_episodes: int,
        max_per_task: int,
        seed: int,
        general_prompt: bool = False,
        reject_literals: bool = False,
    ) -> None:
        self._state = HintRefreshState(
            llm=build_miner_llm_config(llm_info).make(),
            refresh_min_episodes=refresh_min_episodes,
            max_per_task=max_per_task,
            seed=seed,
            general_prompt=general_prompt,
            reject_literals=reject_literals,
        )

    def report(self, task_id: str, reward: float, transcript: str) -> None:
        self._state.report(task_id, reward, transcript)

    def refresh(self) -> int:
        return self._state.refresh()

    def get_version(self) -> int:
        return self._state.get_version()

    def get_hints(self, condition: str, version: int | None = None) -> tuple[int, dict[str, str]]:
        return self._state.get_hints(condition, version)
