from __future__ import annotations

import random
from collections import defaultdict
from typing import Mapping


class DomainWeightedSampler:
    """Randomly samples problems according to per-domain weights."""

    def __init__(self, samples: list[dict], weights: Mapping[str, float], rng: random.Random | None = None):
        if not weights:
            raise ValueError("domain_mix cannot be empty when provided")
        self.random = rng or random
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
        self.thresholds: list[float] = []
        total = 0.0
        for domain, weight in cleaned_weights.items():
            total += weight
            self.domains.append(domain)
            self.thresholds.append(total)
        if total <= 0:
            raise ValueError("Sum of domain_mix weights must be positive")
        self.total_weight = total

    def _pick_domain(self) -> str:
        """keep samples independent and proportional to the weights"""
        r = self.random.random() * self.total_weight
        for domain, threshold in zip(self.domains, self.thresholds):
            if r < threshold:
                return domain
        return self.domains[-1]

    def sample(self) -> dict:
        domain = self._pick_domain()
        return self.random.choice(self.samples_by_domain[domain])
