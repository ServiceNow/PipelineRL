"""Bandit-based problem iterator for curriculum learning."""

import logging
import random
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional

from .bandit import BanditConfig, BanditCurriculum, SuccessRateTracker

logger = logging.getLogger(__name__)


class BanditIterator:
    """
    Iterator that selects problems based on bandit-learned category distribution.

    Supports two modes:
    - difficulty_source="field": Categories from data fields (e.g., "level")
    - difficulty_source="estimated": Categories from estimated success rates
    """

    def __init__(
        self,
        dataset: List[Dict[str, Any]],
        bandit: BanditCurriculum,
        config: BanditConfig,
        success_tracker: Optional[SuccessRateTracker] = None,
    ):
        self.dataset = dataset
        self.bandit = bandit
        self.config = config
        self.success_tracker = success_tracker

        # Index problems by category for efficient sampling
        self.category_to_problems: Dict[str, List[int]] = defaultdict(list)  # category -> list of indices
        self._index_by_category()

        # Initialize bandit with discovered categories
        for category in self.category_to_problems:
            self.bandit._ensure_category(category)

        logger.info(
            f"BanditIterator initialized with {len(self.category_to_problems)} categories"
        )

    def _get_problem_id(self, problem: Dict[str, Any], index: int) -> str:
        """Get unique identifier for a problem."""
        return str(problem.get("id", f"idx_{index}"))

    def _get_category(self, problem: Dict[str, Any], index: int = -1) -> Optional[str]:
        """Extract category from a problem based on config."""
        if self.config.difficulty_source == "estimated":
            # Use estimated success rate to determine bucket
            problem_id = self._get_problem_id(problem, index)

            if self.success_tracker:
                success_rate = self.success_tracker.get_success_rate(problem_id)
                return self.config.success_rate_to_bucket(success_rate)
            else:
                # No tracker yet, use default bucket (middle)
                return self.config.success_rate_to_bucket(0.5)

        # Field-based mode
        fields = self.config.get_all_category_fields()
        if not fields:
            return None

        values = []
        for field in fields:
            value = problem.get(field)
            if value is not None:
                values.append(str(value))
            else:
                values.append("unknown")

        if values and any(v != "unknown" for v in values):
            return "|".join(values)

        return "unknown"

    def _index_by_category(self) -> None:
        """Build index mapping categories to problem indices."""
        if self.config.difficulty_source == "estimated":
            # For estimated mode, initialize all buckets
            for bucket_name in self.config.get_bucket_names():
                self.category_to_problems[bucket_name] = []

            # Index problems by their current estimated bucket
            for i, problem in enumerate(self.dataset):
                category = self._get_category(problem, i)
                if category:
                    self.category_to_problems[category].append(i)

            logger.info(
                f"Indexed {len(self.dataset)} problems into {len(self.category_to_problems)} "
                f"difficulty buckets (estimated mode)"
            )
            for cat in sorted(self.category_to_problems.keys()):
                problems = self.category_to_problems[cat]
                logger.info(f"  {cat}: {len(problems)} problems")
        else:
            # Field-based mode
            fields = self.config.get_all_category_fields()
            if not fields:
                logger.info("No category fields configured")
                return

            for i, problem in enumerate(self.dataset):
                category = self._get_category(problem, i)
                if category is not None:
                    self.category_to_problems[category].append(i)

            logger.info(
                f"Indexed {len(self.dataset)} problems into {len(self.category_to_problems)} categories"
            )
            for cat, indices in self.category_to_problems.items():
                logger.info(f"  Category '{cat}': {len(indices)} problems")

    def reindex_by_estimated_difficulty(self) -> None:
        """Re-index problems by their current estimated success rates.

        Should be called periodically when success rate estimates have changed.
        """
        if self.config.difficulty_source != "estimated":
            return

        # Clear and rebuild
        self.category_to_problems.clear()
        for bucket_name in self.config.get_bucket_names():
            self.category_to_problems[bucket_name] = []

        for i, problem in enumerate(self.dataset):
            category = self._get_category(problem, i)
            if category:
                self.category_to_problems[category].append(i)

        # Ensure bandit knows about all buckets
        for category in self.category_to_problems:
            self.bandit._ensure_category(category)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Infinite iterator that samples problems based on bandit selection."""
        while True:
            # Pass category_to_problems so bandit excludes empty categories
            category = self.bandit.select_category(self.category_to_problems)
            logger.debug(f"Bandit selected category: {category}")

            if category is not None and category in self.category_to_problems:
                indices = self.category_to_problems[category]
                problem_idx = random.choice(indices)
            else:
                # Fallback to uniform random if no valid categories
                problem_idx = random.randint(0, len(self.dataset) - 1)
                category = self._get_category(self.dataset[problem_idx], problem_idx)

            problem = dict(self.dataset[problem_idx])
            problem["_selected_category"] = category

            yield problem
