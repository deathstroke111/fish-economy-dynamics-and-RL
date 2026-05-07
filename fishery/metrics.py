"""Summary helpers for completed rollouts."""

from statistics import mean
from typing import List, Optional

from .config import FisheryConfig
from .state import EpisodeSummary, PolicyArm, TrajectoryPoint


def summarize_trajectory(
    trajectory: List[TrajectoryPoint],
    policy: PolicyArm,
    config: FisheryConfig,
) -> EpisodeSummary:
    """Compute episode-level metrics from the rollout trajectory."""

    end_fish_values = [point.end_fish_population for point in trajectory]
    profits = [point.profit for point in trajectory]
    rewards = [point.reward for point in trajectory]
    collapse_threshold = config.collapse_threshold()

    collapse_step: Optional[int] = None
    for point in trajectory:
        if point.end_fish_population < collapse_threshold:
            collapse_step = point.step
            break

    return EpisodeSummary(
        arm_id=policy.arm_id,
        arm_name=policy.name,
        cumulative_reward=sum(rewards),
        cumulative_profit=sum(profits),
        mean_fish_stock=mean(end_fish_values) if end_fish_values else 0.0,
        min_fish_stock=min(end_fish_values) if end_fish_values else 0.0,
        collapse_flag=collapse_step is not None,
        time_to_collapse=collapse_step,
        final_ships=trajectory[-1].end_ships if trajectory else 0.0,
        final_fish_population=trajectory[-1].end_fish_population if trajectory else 0.0,
        steps=len(trajectory),
    )
