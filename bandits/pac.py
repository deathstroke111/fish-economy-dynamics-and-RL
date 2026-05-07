"""A lightweight PAC-style pure exploration bandit.

This uses a successive-elimination strategy: it cycles through active arms,
updates empirical means, and periodically removes arms whose optimistic bound
is worse than another arm's pessimistic bound.
"""

import math
from collections import deque

from .base import BaseBandit


class PACSuccessiveEliminationBandit(BaseBandit):
    def __init__(
        self,
        arms,
        min_pulls_per_round: int = 3,
        delta: float = 0.1,
        seed: int = 0,
    ) -> None:
        super().__init__(arms=arms, seed=seed)
        self.min_pulls_per_round = min_pulls_per_round
        self.delta = delta
        self.active_arm_ids = list(self.arm_ids)
        self._queue = deque(self.active_arm_ids)

    def select_action(self) -> int:
        if not self._queue:
            self._queue.extend(self.active_arm_ids)
        return self._queue[0]

    def update(self, arm_id: int, reward: float) -> None: 
        super().update(arm_id, reward)
        if self._queue and self._queue[0] == arm_id:
            self._queue.popleft()
        self._maybe_eliminate()

    def _maybe_eliminate(self) -> None:
        if len(self.active_arm_ids) <= 1:
            return

        if any(self.counts[arm_id] < self.min_pulls_per_round for arm_id in self.active_arm_ids):
            return

        confidence = {
            arm_id: math.sqrt(
                math.log((4.0 * len(self.active_arm_ids) * (self.total_updates + 1)) / self.delta)
                / (2.0 * self.counts[arm_id])
            )
            for arm_id in self.active_arm_ids
        }

        best_arm = max(self.active_arm_ids, key=lambda arm_id: self.value_estimates[arm_id])
        best_lower_bound = self.value_estimates[best_arm] - confidence[best_arm]

        survivors = []
        for arm_id in self.active_arm_ids:
            upper_bound = self.value_estimates[arm_id] + confidence[arm_id]
            if upper_bound >= best_lower_bound or arm_id == best_arm:
                survivors.append(arm_id)

        self.active_arm_ids = survivors
        self._queue = deque(self.active_arm_ids)
