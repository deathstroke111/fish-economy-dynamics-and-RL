"""UCB1 bandit."""

import math

from .base import BaseBandit


class UCB1Bandit(BaseBandit):
    def select_action(self) -> int:
        untried = [arm_id for arm_id, count in self.counts.items() if count == 0]
        if untried:
            return self.random.choice(untried)

        total_count = sum(self.counts.values())
        return max(
            self.arm_ids,
            key=lambda arm_id: self.value_estimates[arm_id]
            + math.sqrt((2.0 * math.log(total_count)) / self.counts[arm_id]),
        )
