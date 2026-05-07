"""Reward functions for the fishery simulator."""

from .config import FisheryConfig


def compute_step_reward(profit: float, fish_population: float, config: FisheryConfig) -> float:
    """Balanced reward used for V1 bandit training."""

    collapse_penalty = 10.0 if fish_population < config.collapse_threshold() else 0.0
    return (profit / 1000.0) + (fish_population / config.carrying_capacity) - collapse_penalty
