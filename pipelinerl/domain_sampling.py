from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Mapping

logger = logging.getLogger(__name__)

# Minimum completions before dynamic adjustment kicks in
_MIN_COMPLETIONS_FOR_ADJUSTMENT = 50
# Clamp adjustment factors to avoid extreme swings
_MIN_ADJUSTMENT = 0.1
_MAX_ADJUSTMENT = 10.0


class DomainWeightedSampler:
    """Randomly samples problems according to per-domain weights.

    Supports dynamic weight adjustment based on completion tracking to maintain
    target domain ratios in the output stream despite varying processing speeds.
    """

    def __init__(
        self,
        samples: list[dict],
        weights: Mapping[str, float],
        rng: random.Random | None = None,
        adaptive: bool = True,
    ):
        if not weights:
            raise ValueError("domain_mix cannot be empty when provided")
        self.random = rng or random
        self.adaptive = adaptive
        samples_by_domain: dict[str, list[dict]] = defaultdict(list)
        for sample in samples:
            domain = sample.get("domain")
            if not domain:
                raise ValueError("Each sample must include a 'domain' field for domain_mix to work")
            samples_by_domain[str(domain)].append(sample)

        provided_domains = {str(domain) for domain in weights}
        cleaned_weights: dict[str, float] = {}
        for domain, value in weights.items():
            val = float(value)
            if val < 0:
                raise ValueError(f"domain_mix weight for '{domain}' must be non-negative")
            if val == 0:
                continue
            cleaned_weights[str(domain)] = val

        if not cleaned_weights:
            raise ValueError("domain_mix must include at least one positive weight")

        # accept zero weights but require the domain to be declared.
        missing = set(samples_by_domain) - provided_domains
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(
                "domain_mix is missing weights for dataset domains: " + missing_list
            )

        unused = provided_domains - set(samples_by_domain)
        if unused:
            unused_list = ", ".join(sorted(unused))
            raise ValueError(
                "domain_mix specifies domains not present in dataset: " + unused_list
            )

        self.samples_by_domain = samples_by_domain
        self.domains: list[str] = []
        self.base_weights: dict[str, float] = {}
        self.thresholds: list[float] = []
        total = 0.0
        for domain, weight in cleaned_weights.items():
            total += weight
            self.domains.append(domain)
            self.base_weights[domain] = weight
            self.thresholds.append(total)
        if total <= 0:
            raise ValueError("Sum of domain_mix weights must be positive")
        self.total_weight = total

        # Target ratios (normalized weights)
        self.target_ratios = {d: w / total for d, w in self.base_weights.items()}

        # Completion tracking for adaptive sampling
        self.completion_counts: dict[str, int] = {d: 0 for d in self.domains}
        self.total_completions = 0
        self._last_log_completions = 0

    def record_completion(self, domain: str) -> None:
        """Record that a sample from the given domain has completed processing.

        This enables adaptive weight adjustment to maintain target domain ratios
        in the output stream despite varying processing speeds per domain.
        """
        if domain in self.completion_counts:
            self.completion_counts[domain] += 1
            self.total_completions += 1

            # Log periodically
            if self.total_completions - self._last_log_completions >= 500:
                self._log_domain_stats()
                self._last_log_completions = self.total_completions

    def _log_domain_stats(self) -> None:
        """Log current domain distribution vs targets."""
        if self.total_completions == 0:
            return
        parts = []
        for domain in self.domains:
            actual = self.completion_counts[domain] / self.total_completions
            target = self.target_ratios[domain]
            parts.append(f"{domain}={actual:.1%}(target={target:.1%})")
        logger.info(f"Domain completion stats ({self.total_completions} total): {', '.join(parts)}")

    def _pick_domain_static(self) -> str:
        """Pick domain using static weights (original behavior)."""
        r = self.random.random() * self.total_weight
        for domain, threshold in zip(self.domains, self.thresholds):
            if r < threshold:
                return domain
        return self.domains[-1]

    def _pick_domain_adaptive(self) -> str:
        """Pick domain using dynamically adjusted weights based on completion ratios."""
        # Calculate current completion ratios
        current_ratios = {
            d: self.completion_counts[d] / self.total_completions
            for d in self.domains
        }

        # Calculate adjusted weights: boost under-represented, reduce over-represented
        adjusted_weights: dict[str, float] = {}
        for domain in self.domains:
            target = self.target_ratios[domain]
            current = current_ratios[domain]

            if current > 0:
                # adjustment = target / current
                # If current=46%, target=30% → adjustment=0.65 (sample less)
                # If current=18%, target=30% → adjustment=1.67 (sample more)
                adjustment = target / current
                adjustment = max(_MIN_ADJUSTMENT, min(_MAX_ADJUSTMENT, adjustment))
            else:
                # No completions yet for this domain, boost sampling
                adjustment = _MAX_ADJUSTMENT

            adjusted_weights[domain] = self.base_weights[domain] * adjustment

        # Sample based on adjusted weights
        total = sum(adjusted_weights.values())
        r = self.random.random() * total
        cumsum = 0.0
        for domain in self.domains:
            cumsum += adjusted_weights[domain]
            if r < cumsum:
                return domain
        return self.domains[-1]

    def _pick_domain(self) -> str:
        """Pick a domain for the next sample."""
        if not self.adaptive or self.total_completions < _MIN_COMPLETIONS_FOR_ADJUSTMENT:
            return self._pick_domain_static()
        return self._pick_domain_adaptive()

    def sample(self) -> dict:
        domain = self._pick_domain()
        return self.random.choice(self.samples_by_domain[domain])
