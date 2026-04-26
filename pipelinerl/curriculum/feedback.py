"""Curriculum feedback computation and message types."""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Literal, Union

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CategoryFeedback(BaseModel):
    """Feedback message from preprocessor to actor about category performance."""

    kind: Literal["category_feedback"] = "category_feedback"

    # Map from category -> mean advantage for that category in this batch
    category_advantages: Dict[str, float] = Field(default_factory=dict)

    # Map from category -> number of samples with non-zero advantages in this batch
    category_counts: Dict[str, int] = Field(default_factory=dict)

    # Map from category -> total number of rollouts in this batch
    category_total_rollouts: Dict[str, int] = Field(default_factory=dict)

    # Map from category -> set of problem IDs in this batch
    category_problem_ids: Dict[str, List[str]] = Field(default_factory=dict)

    # Map from category -> success rate in this batch
    category_success_rates: Dict[str, float] = Field(default_factory=dict)

    # Map from problem_id -> success rate (for estimated mode)
    # Used to update per-problem success rate estimates in the actor
    problem_success_rates: Dict[str, float] = Field(default_factory=dict)

    # Timestamp for ordering/staleness detection
    timestamp: float = Field(default_factory=time.time)

    # Model version these stats correspond to
    model_version: int = 0


class BanditStateUpdate(BaseModel):
    """Full bandit state update (for synchronization)."""

    kind: Literal["bandit_state_update"] = "bandit_state_update"

    q_values: Dict[str, float] = Field(default_factory=dict)
    sample_counts: Dict[str, int] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


CurriculumMessage = Union[CategoryFeedback, BanditStateUpdate]


class CurriculumFeedbackTracker:
    """Tracks curriculum feedback stats over multiple batches for periodic logging."""

    def __init__(self, log_every_n_samples: int = 128):
        self.log_every_n_samples = log_every_n_samples
        self.cumulative_rollouts_per_category: Dict[str, int] = defaultdict(int)
        self.cumulative_problems_per_category: Dict[str, set] = defaultdict(set)
        self.reset()

    def reset(self):
        """Reset batch stats (not cumulative)."""
        self.batch_nonzero_adv_per_category: Dict[str, int] = defaultdict(int)
        self.batch_rollouts_per_category: Dict[str, int] = defaultdict(int)
        self.batch_problems_per_category: Dict[str, set] = defaultdict(set)
        self.batch_count = 0

    def update(self, feedback: CategoryFeedback):
        """Accumulate stats from a feedback message."""
        for category, count in feedback.category_counts.items():
            self.batch_nonzero_adv_per_category[category] += count
        for category, count in feedback.category_total_rollouts.items():
            self.batch_rollouts_per_category[category] += count
            self.cumulative_rollouts_per_category[category] += count
        for category, problem_ids in feedback.category_problem_ids.items():
            self.batch_problems_per_category[category].update(problem_ids)
            self.cumulative_problems_per_category[category].update(problem_ids)
        self.batch_count += 1

    def has_data(self) -> bool:
        """Check if there are accumulated stats to report."""
        return self.batch_count > 0

    def time_to_log(self) -> bool:
        """Check if enough samples have accumulated for periodic logging."""
        total_rollouts = sum(self.batch_rollouts_per_category.values())
        return total_rollouts >= self.log_every_n_samples

    def get_summary_and_reset(self) -> str:
        """Get a formatted summary string and reset batch stats."""
        parts = [f"{self.batch_count} chunks"]
        for cat in sorted(self.batch_rollouts_per_category.keys()):
            total = self.batch_rollouts_per_category[cat]
            nonzero = self.batch_nonzero_adv_per_category.get(cat, 0)
            zero = total - nonzero
            problems = len(self.batch_problems_per_category.get(cat, set()))
            cum_total = self.cumulative_rollouts_per_category[cat]
            cum_problems = len(self.cumulative_problems_per_category[cat])
            parts.append(
                f"{cat}: {total} rollouts ({zero} zero-adv), {problems} problems "
                f"(cumulative: {cum_total} rollouts, {cum_problems} problems)"
            )
        self.reset()
        return " | ".join(parts)


def compute_category_feedback(
    dataset: List[Dict],
    category_fields: Union[str, List[str]] = None,
) -> CategoryFeedback:
    """
    Compute feedback statistics from a preprocessed batch.

    Called in preprocess.py after populate_rl_data() computes advantages.

    Args:
        dataset: List of processed entries with 'advantages', 'metadata', etc.
        category_fields: Field(s) to use for category grouping. Can be a single
            field name or a list of fields for joint categories. If None or empty,
            uses "_selected_category" from metadata only.

    Returns:
        CategoryFeedback message with aggregated statistics
    """
    # Normalize category_fields to list
    if category_fields is None:
        fields = []
    elif isinstance(category_fields, str):
        fields = [category_fields] if category_fields else []
    else:
        fields = list(category_fields)

    def get_category_from_entry(entry: Dict) -> str:
        """Extract category from entry using configured fields."""
        values = [str(entry[field]) for field in fields if entry.get(field)]
        return "|".join(values)

    # Aggregate by category
    category_advantages: Dict[str, List[float]] = defaultdict(list)
    category_nonzero_counts: Dict[str, int] = defaultdict(int)  # For logging
    category_rewards: Dict[str, List[float]] = defaultdict(list)
    category_problem_ids: Dict[str, set] = defaultdict(set)

    # Per-problem rewards for estimated mode
    problem_rewards: Dict[str, List[float]] = defaultdict(list)

    for entry in dataset:
        # Get category from metadata (set by bandit iterator) or entry itself
        metadata = entry.get("metadata", {})
        category = metadata.get("_selected_category")
        problem_id = metadata.get("id")  # For estimated mode

        if category is None and fields:
            # Fallback to category_fields in entry
            category = get_category_from_entry(entry)

        # Track rewards for success rate estimation (do this even without category)
        rewards = entry.get("rewards", [])
        reward = None
        if rewards:
            # Use the first reward value (they're typically all the same for a rollout)
            reward = rewards[0] if isinstance(rewards, list) else rewards
            if np.isfinite(reward):
                # Track per-problem for estimated mode (independent of category)
                if problem_id is not None:
                    problem_rewards[str(problem_id)].append(reward)
                # Track per-category if we have one
                if category is not None:
                    category_rewards[category].append(reward)
                    category_problem_ids[category].add(str(problem_id))

        # Skip category-based stats if no category available
        if category is None:
            # This shouldn't happen in estimated mode if iterator is working correctly
            logger.debug(f"Entry missing _selected_category, problem_id={problem_id}")
            continue

        # Get advantages (per-token, use mean of absolute values like SEC)
        advantages = entry.get("advantages", [])
        labels = entry.get("labels", [])
        if advantages and labels:
            adv_arr = np.abs(np.array(advantages))
            mask = np.array(labels) != -100  # Response mask (like SEC)
            if mask.any():
                # Masked mean of absolute advantages (like SEC's masked_mean)
                mean_abs_adv = float(adv_arr[mask].mean())
                if np.isfinite(mean_abs_adv):
                    category_advantages[category].append(mean_abs_adv)
                    if mean_abs_adv > 0:
                        category_nonzero_counts[category] += 1

    # Compute category aggregates
    result_advantages = {}
    result_counts = {}
    result_total_rollouts = {}
    result_problem_ids = {}
    result_success_rates = {}

    all_categories = set(category_advantages.keys()) | set(category_rewards.keys())

    for category in all_categories:
        advs = category_advantages.get(category, [])
        rewards = category_rewards.get(category, [])
        problem_ids = category_problem_ids.get(category, set())

        if advs:
            # Mean includes zero-advantage samples (like SEC)
            result_advantages[category] = float(np.mean(advs))
        # Count of non-zero advantage samples (for logging/tracking)
        result_counts[category] = category_nonzero_counts.get(category, 0)
        result_total_rollouts[category] = len(rewards)
        result_problem_ids[category] = list(problem_ids)
        if rewards:
            # Success = positive reward
            success_rate = sum(1 for r in rewards if r > 0) / len(rewards)
            result_success_rates[category] = float(success_rate)

    # Compute per-problem success rates (for estimated mode)
    result_problem_success_rates = {}
    for problem_id, rewards in problem_rewards.items():
        if rewards:
            success_rate = sum(1 for r in rewards if r > 0) / len(rewards)
            result_problem_success_rates[problem_id] = float(success_rate)

    model_version = 0
    if dataset:
        # Get model version from first entry's metadata
        model_version = dataset[0].get("metadata", {}).get("model_version", 0)

    logger.debug(
        f"Computed feedback for {len(all_categories)} categories, "
        f"{len(result_problem_success_rates)} problems: "
        f"advantages={result_advantages}, counts={result_counts}"
    )

    return CategoryFeedback(
        category_advantages=result_advantages,
        category_counts=result_counts,
        category_total_rollouts=result_total_rollouts,
        category_problem_ids=result_problem_ids,
        category_success_rates=result_success_rates,
        problem_success_rates=result_problem_success_rates,
        model_version=model_version,
    )
