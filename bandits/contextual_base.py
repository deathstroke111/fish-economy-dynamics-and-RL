"""Common interface for contextual bandits."""

import random
from typing import Dict, Iterable, List

from fishery.state import PolicyArm


class BaseContextualBandit:
    """Base class for contextual action-selection strategies."""

    def __init__(self, arms: Iterable[PolicyArm], seed: int = 0) -> None:
        arm_list = list(arms)
        if not arm_list:
            raise ValueError("Contextual bandits require at least one arm")
        self.arms: List[PolicyArm] = arm_list
        self.arm_ids = [arm.arm_id for arm in arm_list]
        self.random = random.Random(seed)
        self.total_updates = 0

    def select_action(self, context: List[float]) -> int:
        raise NotImplementedError

    def update(self, context: List[float], arm_id: int, reward: float) -> None:
        self.total_updates += 1

    def arm_lookup(self) -> Dict[int, PolicyArm]:
        return {arm.arm_id: arm for arm in self.arms}
