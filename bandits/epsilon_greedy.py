"""Epsilon-greedy bandit."""

from .base import BaseBandit


class EpsilonGreedyBandit(BaseBandit):
    def __init__(self, arms, epsilon: float = 0.1, seed: int = 0) -> None:
        super().__init__(arms=arms, seed=seed)
        self.epsilon = epsilon

    def select_action(self) -> int:
        untried = [arm_id for arm_id, count in self.counts.items() if count == 0]
        if untried:
            return self.random.choice(untried)
        if self.random.random() < self.epsilon:
            return self.random.choice(self.arm_ids)
        return max(self.arm_ids, key=lambda arm_id: self.value_estimates[arm_id])
