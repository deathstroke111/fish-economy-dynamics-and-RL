"""Discretized contextual bandit."""

from typing import Dict, List, Tuple

from .contextual_base import BaseContextualBandit


BucketKey = Tuple[int, int, int]


class DiscretizedContextualBandit(BaseContextualBandit):
    def __init__(self, arms, epsilon: float = 0.1, seed: int = 0) -> None:
        super().__init__(arms=arms, seed=seed)
        self.epsilon = epsilon
        self.bucket_counts: Dict[BucketKey, Dict[int, int]] = {}
        self.bucket_values: Dict[BucketKey, Dict[int, float]] = {}

    def bucket_for_context(self, context: List[float]) -> BucketKey:
        fish_norm = context[1]
        ships_norm = context[2]
        time_norm = context[3]
        return (
            self._bucket_index(fish_norm, (0.25, 0.5, 0.75)),
            self._bucket_index(ships_norm, (0.2, 0.5)),
            self._bucket_index(time_norm, (1.0 / 3.0, 2.0 / 3.0)),
        )

    def select_action(self, context: List[float]) -> int:
        bucket = self.bucket_for_context(context)
        counts = self.bucket_counts.setdefault(bucket, {arm_id: 0 for arm_id in self.arm_ids})
        values = self.bucket_values.setdefault(bucket, {arm_id: 0.0 for arm_id in self.arm_ids})

        untried = [arm_id for arm_id in self.arm_ids if counts[arm_id] == 0]
        if untried:
            return self.random.choice(untried)
        if self.random.random() < self.epsilon:
            return self.random.choice(self.arm_ids)
        return max(self.arm_ids, key=lambda arm_id: values[arm_id])

    def update(self, context: List[float], arm_id: int, reward: float) -> None:
        bucket = self.bucket_for_context(context)
        counts = self.bucket_counts.setdefault(bucket, {arm_id: 0 for arm_id in self.arm_ids})
        values = self.bucket_values.setdefault(bucket, {arm_id: 0.0 for arm_id in self.arm_ids})

        count = counts[arm_id] + 1
        values[arm_id] = values[arm_id] + ((reward - values[arm_id]) / count)
        counts[arm_id] = count
        super().update(context, arm_id, reward)

    @staticmethod
    def _bucket_index(value: float, thresholds: Tuple[float, ...]) -> int:
        for index, threshold in enumerate(thresholds):
            if value < threshold:
                return index
        return len(thresholds)
