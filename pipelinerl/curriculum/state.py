"""Curriculum state manager with feedback listener."""

import logging
import threading
from typing import Any, Dict, Iterator, List, Optional

from pydantic import TypeAdapter

from pipelinerl.streams import SingleStreamSpec, read_stream

from .bandit import BanditConfig, BanditCurriculum, BanditState, SuccessRateTracker
from .feedback import BanditStateUpdate, CategoryFeedback, CurriculumMessage
from .iterator import BanditIterator

logger = logging.getLogger(__name__)


class CurriculumState:
    """
    Implements Self-Evolving Curriculum (SEC): https://arxiv.org/abs/2505.14970
    
    Manages all curriculum learning state: bandit, success tracker, and feedback listener.
    """

    def __init__(
        self,
        config: BanditConfig,
        feedback_stream: SingleStreamSpec,
    ):
        self.config = config
        self.bandit = BanditCurriculum(config)

        # Create success tracker for estimated mode
        self.success_tracker: Optional[SuccessRateTracker] = None
        if config.difficulty_source == "estimated":
            self.success_tracker = SuccessRateTracker(learning_rate=config.learning_rate)
            logger.info("Created success rate tracker for estimated difficulty mode")

        self.feedback_stream = feedback_stream
        self._iterator: Optional[BanditIterator] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._message_adapter = TypeAdapter(CurriculumMessage)
        self._updates_since_reindex = 0

    def create_iterator(self, dataset: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """Create a bandit-based iterator for the dataset.

        Args:
            dataset: List of problem dicts

        Returns:
            Iterator that yields problems based on bandit selection
        """
        self._iterator = BanditIterator(
            dataset, self.bandit, self.config, self.success_tracker
        )
        return iter(self._iterator)

    def start_listening(self):
        """Start background thread to listen for feedback."""

        def listen():
            logger.info(
                f"Starting curriculum feedback listener on stream {self.feedback_stream.topic}"
            )
            try:
                with read_stream(self.feedback_stream) as reader:
                    for line in reader.read():
                        if self._stop_event.is_set():
                            break
                        try:
                            message = self._message_adapter.validate_python(line)
                            self._process_curriculum_feedback(message)
                        except Exception as e:
                            logger.warning(f"Failed to parse curriculum message: {e}")
            except Exception as e:
                logger.error(f"Curriculum feedback listener error: {e}")

        self._thread = threading.Thread(target=listen, daemon=True)
        self._thread.start()
        logger.info("Started curriculum feedback listener")

    def stop_listening(self):
        """Stop the feedback listener thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            logger.info("Stopped curriculum feedback listener")

    def _process_curriculum_feedback(self, message: CurriculumMessage):
        """Process incoming curriculum feedback."""
        if isinstance(message, CategoryFeedback):
            # Update Q-values based on advantages
            for category, advantage in message.category_advantages.items():
                self.bandit.update_q_value(category, advantage)

            # Update per-problem success rates (for estimated mode)
            if self.success_tracker and message.problem_success_rates:
                for problem_id, success_rate in message.problem_success_rates.items():
                    self.success_tracker.update(problem_id, success_rate)

                self._updates_since_reindex += 1
                # Periodically trigger re-indexing when success rates change
                if self._updates_since_reindex >= self.config.reindex_interval:
                    if self._iterator is not None:
                        self._iterator.reindex_by_estimated_difficulty()
                        logger.debug("Re-indexed problems by estimated difficulty")
                    self._updates_since_reindex = 0

            logger.debug(
                f"Processed feedback for {len(message.category_advantages)} categories, "
                f"{len(message.problem_success_rates)} problems "
                f"(model_version={message.model_version})"
            )

        elif isinstance(message, BanditStateUpdate):
            # Full state synchronization
            state = BanditState(
                q_values=message.q_values,
                sample_counts=message.sample_counts,
            )
            self.bandit.set_state(state)
            logger.info("Synchronized bandit state from external update")

    def track_rollout_results(self, rollout_results):
        """Extract curriculum metadata from rollout results.

        Args:
            rollout_results: List of RolloutResult objects

        Returns:
            Tuple of (categories, feature_values, feature_categories)
        """
        categories = []
        feature_values: Dict[str, list] = {}
        feature_categories: Dict[str, list] = {}

        category_fields = self.config.get_all_category_fields()

        for result in rollout_results:
            for text in result.training_texts:
                metadata = text.metadata or {}
                # Track selected category
                if "_selected_category" in metadata:
                    categories.append(metadata["_selected_category"])
                # Track each configured feature field
                for field in category_fields:
                    key = f"_curriculum_{field}"
                    if key in metadata:
                        value = metadata[key]
                        # Try numeric first (for averages)
                        try:
                            if field not in feature_values:
                                feature_values[field] = []
                            feature_values[field].append(float(value))
                        except (ValueError, TypeError):
                            pass
                        # Always track as categorical (for distribution)
                        if field not in feature_categories:
                            feature_categories[field] = []
                        feature_categories[field].append(str(value))

        return categories, feature_values, feature_categories

    def compute_batch_stats(
        self,
        categories: list,
        feature_values: Dict[str, list],
        feature_categories: Dict[str, list],
    ) -> Dict[str, float]:
        """Compute curriculum statistics for logging.

        Args:
            categories: List of selected categories in this batch
            feature_values: Dict of field -> list of numeric values
            feature_categories: Dict of field -> list of categorical values

        Returns:
            Dict of stat_name -> value
        """
        from collections import Counter

        stats = {}

        # Add bandit Q-value stats
        stats.update(self.bandit.get_stats())

        # Add success tracker stats (for estimated mode)
        if self.success_tracker:
            stats.update(self.success_tracker.get_stats(self.config))

        # Average feature values (like SEC's loader/average_difficulty)
        for field, values in feature_values.items():
            if values:
                stats[f"curriculum/average_{field}"] = sum(values) / len(values)

        # Categorical distribution for each feature field (fraction and count)
        for field, values in feature_categories.items():
            if values:
                value_counts = Counter(values)
                total = len(values)
                for value, count in value_counts.items():
                    stats[f"curriculum/{field}/{value}"] = count / total
                    stats[f"curriculum/{field}_count/{value}"] = count

        # For estimated mode, track bucket distribution from selected categories
        # (categories contains bucket names like "bucket_0", "bucket_1", etc.)
        if self.config.difficulty_source == "estimated" and categories:
            bucket_counts = Counter(categories)
            total = len(categories)
            for bucket_name, count in bucket_counts.items():
                # Extract bucket number from "bucket_0" -> "0"
                bucket_num = bucket_name.replace("bucket_", "")
                stats[f"curriculum/bucket/{bucket_num}"] = count / total
                stats[f"curriculum/bucket_count/{bucket_num}"] = count

        return stats
