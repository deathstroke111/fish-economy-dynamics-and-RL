"""Softmax/Boltzmann exploration bandit."""

import math

from .base import BaseBandit


class SoftmaxBandit(BaseBandit):
    def __init__(self, arms, temperature: float = 0.5, seed: int = 0) -> None:
        super().__init__(arms=arms, seed=seed)
        self.temperature = temperature

    def select_action(self) -> int:
        untried = [arm_id for arm_id, count in self.counts.items() if count == 0]
        if untried:
            return self.random.choice(untried)

        max_value = max(self.value_estimates.values())
        scaled_values = []
        for arm_id in self.arm_ids:
            shifted = (self.value_estimates[arm_id] - max_value) / self.temperature
            scaled_values.append(math.exp(shifted))
        total = sum(scaled_values)
        threshold = self.random.random() * total
        cumulative = 0.0
        for arm_id, weight in zip(self.arm_ids, scaled_values):
            cumulative += weight
            if cumulative >= threshold:
                return arm_id
        return self.arm_ids[-1]
