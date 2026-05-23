"""Multi-armed bandit for curriculum learning."""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class BanditConfig(BaseModel):
    """Configuration for bandit-based curriculum learning."""

    # Whether curriculum learning is enabled
    enabled: bool = Field(default=False, description="Enable curriculum learning")

    # How difficulty is determined for curriculum learning:
    # - "field": use the value from difficulty_field directly as category
    # - "estimated": estimate difficulty from per-group success rates
    #   (requires GRPO-like policy with attempts > 1)
    difficulty_source: str = Field(
        default="field",
        description="How difficulty is determined: 'field' (from difficulty_field) or 'estimated' (from success rates)",
    )

    # Field name for difficulty/level (e.g., "level", "difficulty")
    # Used when difficulty_source="field". If None, uses category_fields only.
    difficulty_field: Optional[str] = Field(
        default=None,
        description="Field in problem dict containing difficulty/level. "
        "Takes precedence over category_fields when set.",
    )

    # Additional field(s) for categorization beyond difficulty
    # Can be a single string or list of strings for joint categories
    # e.g., "dataset" or ["dataset", "type"] for joint category "math|algebra"
    category_fields: Union[str, List[str]] = Field(
        default_factory=list,
        description="Additional field(s) in problem dict for categorization. "
        "Can be a single string or list of strings.",
    )

    # Bandit algorithm parameters
    temperature: float = Field(
        default=1.0, description="Softmax temperature (higher = more exploration)"
    )
    learning_rate: float = Field(
        default=0.1, description="Q-value update learning rate"
    )
    initial_q_value: float = Field(
        default=0.0, description="Initial Q-value for new categories"
    )

    # Update signal: which metric to use
    update_signal: str = Field(
        default="advantage",
        description="Signal for Q-update: 'advantage', 'reward', or 'success'",
    )

    # Number of difficulty buckets for estimated mode
    # Problems are assigned to buckets based on their estimated success rate
    # e.g., 5 buckets: [0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]
    num_difficulty_buckets: int = Field(
        default=5,
        description="Number of difficulty buckets for estimated mode",
    )

    # How often to re-assign problems to buckets based on updated success rates
    # Only applies when difficulty_source="estimated"
    # Each preprocessor batch sends one feedback message, so this is roughly
    # "reindex every N optimization steps"
    reindex_interval: int = Field(
        default=5,
        description="Re-index problems every N preprocessor feedback messages",
    )

    @model_validator(mode="after")
    def validate_category_fields(self) -> "BanditConfig":
        """Validate that category fields are properly configured."""
        if not self.enabled:
            return self

        has_difficulty_field = bool(self.difficulty_field)
        has_category_fields = bool(self.category_fields)

        if self.difficulty_source == "field":
            # When using field-based difficulty, must have at least one field
            if not has_difficulty_field and not has_category_fields:
                raise ValueError(
                    "When difficulty_source='field', either difficulty_field or "
                    "category_fields must be provided"
                )
        # When difficulty_source="estimated", no fields required (difficulty comes from success rates)
        # but category_fields can still be used for grouping

        return self

    def get_all_category_fields(self) -> List[str]:
        """Get all category fields as a list (difficulty_field + category_fields).

        Converts category_fields to list if it's a string, and prepends
        difficulty_field if set.
        """
        fields = []

        # Add difficulty_field first if set
        if self.difficulty_field:
            fields.append(self.difficulty_field)

        # Add category_fields (normalize string to list)
        if isinstance(self.category_fields, str):
            if self.category_fields:  # non-empty string
                fields.append(self.category_fields)
        else:
            fields.extend(self.category_fields)

        return fields

    def success_rate_to_bucket(self, success_rate: float) -> str:
        """Convert a success rate to a difficulty bucket name.

        Args:
            success_rate: Success rate in [0, 1]

        Returns:
            Bucket name like "bucket_0" (hardest) to "bucket_4" (easiest)
        """
        # Clamp to [0, 1]
        success_rate = max(0.0, min(1.0, success_rate))
        # Bucket index: 0 = hardest (low success), num_buckets-1 = easiest (high success)
        bucket_idx = min(
            int(success_rate * self.num_difficulty_buckets),
            self.num_difficulty_buckets - 1
        )
        return f"bucket_{bucket_idx}"

    def get_bucket_names(self) -> List[str]:
        """Get all bucket names for estimated mode."""
        return [f"bucket_{i}" for i in range(self.num_difficulty_buckets)]


class BanditState(BaseModel):
    """Serializable state for the bandit."""

    q_values: Dict[str, float] = Field(default_factory=dict)
    sample_counts: Dict[str, int] = Field(default_factory=dict)


class SuccessRateTracker:
    """Tracks per-problem success rates for estimated difficulty mode.

    Uses exponential moving average to track success rates over time.
    """

    def __init__(self, learning_rate: float = 0.3, default_rate: float = 0.5):
        """
        Args:
            learning_rate: EMA learning rate for success rate updates
            default_rate: Default success rate for unseen problems
        """
        self.learning_rate = learning_rate
        self.default_rate = default_rate
        # problem_id -> estimated success rate
        self._success_rates: Dict[str, float] = {}
        # problem_id -> number of observations
        self._observation_counts: Dict[str, int] = {}

    def get_success_rate(self, problem_id: str) -> float:
        """Get estimated success rate for a problem."""
        return self._success_rates.get(problem_id, self.default_rate)

    def update(self, problem_id: str, success_rate: float) -> None:
        """Update success rate estimate for a problem.

        Args:
            problem_id: Unique problem identifier
            success_rate: Observed success rate (e.g., from batch)
        """
        if problem_id not in self._success_rates:
            # First observation - use it directly
            self._success_rates[problem_id] = success_rate
            self._observation_counts[problem_id] = 1
        else:
            # EMA update
            old_rate = self._success_rates[problem_id]
            new_rate = (1 - self.learning_rate) * old_rate + self.learning_rate * success_rate
            self._success_rates[problem_id] = new_rate
            self._observation_counts[problem_id] += 1

    def get_state(self) -> Dict[str, Dict[str, float]]:
        """Get serializable state."""
        return {
            "success_rates": dict(self._success_rates),
            "observation_counts": {k: float(v) for k, v in self._observation_counts.items()},
        }

    def get_stats(self, config: "BanditConfig") -> Dict[str, float]:
        """Get statistics for wandb logging.

        Args:
            config: BanditConfig to compute bucket assignments

        Returns:
            Dict of stat_name -> value
        """
        stats = {}

        if not self._success_rates:
            return stats

        rates = list(self._success_rates.values())
        stats["estimated/num_problems_tracked"] = len(rates)
        stats["estimated/mean_success_rate"] = np.mean(rates)
        stats["estimated/std_success_rate"] = np.std(rates) if len(rates) > 1 else 0.0
        stats["estimated/min_success_rate"] = np.min(rates)
        stats["estimated/max_success_rate"] = np.max(rates)

        # Distribution across buckets
        bucket_counts: Dict[str, int] = {name: 0 for name in config.get_bucket_names()}
        for rate in rates:
            bucket = config.success_rate_to_bucket(rate)
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        total = len(rates)
        for bucket, count in bucket_counts.items():
            stats[f"estimated/bucket_fraction/{bucket}"] = count / total if total > 0 else 0.0

        # Average observations per problem
        if self._observation_counts:
            obs_counts = list(self._observation_counts.values())
            stats["estimated/mean_observations"] = np.mean(obs_counts)

        return stats

    def set_state(self, state: Dict[str, Dict[str, float]]) -> None:
        """Set state from serialized form."""
        self._success_rates = dict(state.get("success_rates", {}))
        self._observation_counts = {k: int(v) for k, v in state.get("observation_counts", {}).items()}

    def __len__(self) -> int:
        return len(self._success_rates)


class BanditCurriculum:
    """
    Multi-armed bandit for curriculum learning.

    Uses softmax/Boltzmann selection over Q-values representing category value.
    Q-values are updated based on mean advantage feedback from the preprocessor.
    """

    def __init__(self, config: BanditConfig, categories: Optional[List[str]] = None):
        self.config = config
        self.state = BanditState()

        # Initialize categories if provided
        if categories:
            for cat in categories:
                self._ensure_category(cat)

    def _ensure_category(self, category: str) -> None:
        """Ensure category exists in state."""
        if category not in self.state.q_values:
            self.state.q_values[category] = self.config.initial_q_value
            self.state.sample_counts[category] = 0

    def get_selection_probabilities(self) -> Dict[str, float]:
        """
        Compute softmax selection probabilities over categories.

        Returns dict mapping category -> probability
        """
        if not self.state.q_values:
            return {}

        categories = list(self.state.q_values.keys())
        q_vals = np.array([self.state.q_values[c] for c in categories])

        # Softmax with temperature
        # Subtract max for numerical stability
        q_vals = q_vals - np.max(q_vals)
        exp_vals = np.exp(q_vals / max(self.config.temperature, 1e-8))
        probs = exp_vals / exp_vals.sum()

        return {cat: float(prob) for cat, prob in zip(categories, probs)}

    def select_category(self, category_to_problems: Optional[Dict[str, list]] = None) -> Optional[str]:
        """
        Select a category using softmax/Boltzmann selection.

        Args:
            category_to_problems: If provided, only select from non-empty categories.

        Returns selected category name or None if no categories available.
        """
        probs = self.get_selection_probabilities()
        if not probs:
            return None

        # Filter to non-empty categories
        if category_to_problems is not None:
            probs = {c: p for c, p in probs.items() if category_to_problems.get(c)}
            if not probs:
                return None
            total = sum(probs.values())
            probs = {c: p / total for c, p in probs.items()}

        categories = list(probs.keys())
        probabilities = [probs[c] for c in categories]

        return np.random.choice(categories, p=probabilities)

    def update_q_value(self, category: str, signal: float) -> None:
        """
        Update Q-value for a category based on feedback signal.

        Uses exponential moving average: Q = (1-lr)*Q + lr*signal

        Args:
            category: The category to update
            signal: The feedback signal (e.g., mean advantage)
        """
        if not np.isfinite(signal):
            logger.warning(f"Ignoring non-finite signal {signal} for category {category}")
            return

        self._ensure_category(category)

        old_q = self.state.q_values[category]
        lr = self.config.learning_rate
        new_q = (1 - lr) * old_q + lr * signal

        self.state.q_values[category] = new_q
        self.state.sample_counts[category] += 1

        logger.debug(
            f"Updated Q[{category}]: {old_q:.4f} -> {new_q:.4f} (signal={signal:.4f})"
        )

    def get_state(self) -> BanditState:
        """Get serializable state for persistence/communication."""
        return self.state.model_copy(deep=True)

    def set_state(self, state: BanditState) -> None:
        """Set state from serialized form."""
        self.state = state.model_copy(deep=True)

    def get_stats(self) -> Dict[str, float]:
        """Get statistics for logging."""
        stats = {}
        probs = self.get_selection_probabilities()

        for cat in self.state.q_values:
            stats[f"curriculum/q_value/{cat}"] = self.state.q_values[cat]
            stats[f"curriculum/samples/{cat}"] = self.state.sample_counts[cat]
            if cat in probs:
                stats[f"curriculum/prob/{cat}"] = probs[cat]

        # Also log aggregate stats
        if probs:
            # Entropy of selection distribution (higher = more uniform)
            probs_arr = np.array(list(probs.values()))
            entropy = -np.sum(probs_arr * np.log(probs_arr + 1e-10))
            stats["curriculum/selection_entropy"] = entropy

        return stats
