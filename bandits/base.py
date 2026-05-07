"""Common bandit interface."""

import random
from typing import Dict, Iterable, List

from fishery.state import PolicyArm


class BaseBandit:
    """Base class shared by all action-selection strategies."""

    def __init__(self, arms: Iterable[PolicyArm], seed: int = 0) -> None:
        arm_list = list(arms)
        if not arm_list:
            raise ValueError("Bandits require at least one arm")
        self.arms: List[PolicyArm] = arm_list
        self.arm_ids = [arm.arm_id for arm in arm_list]
        self.random = random.Random(seed)
        self.counts: Dict[int, int] = {arm.arm_id: 0 for arm in arm_list}
        self.value_estimates: Dict[int, float] = {arm.arm_id: 0.0 for arm in arm_list}
        self.total_updates = 0

    def select_action(self) -> int:
        raise NotImplementedError

    def update(self, arm_id: int, reward: float) -> None:
        count = self.counts[arm_id] + 1
        estimate = self.value_estimates[arm_id]
        self.value_estimates[arm_id] = estimate + ((reward - estimate) / count)
        self.counts[arm_id] = count
        self.total_updates += 1

    def arm_lookup(self) -> Dict[int, PolicyArm]:
        return {arm.arm_id: arm for arm in self.arms}
