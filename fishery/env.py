"""Rollout helpers for fishery simulations."""

from typing import List, Tuple

from .config import FisheryConfig
from .dynamics import step
from .metrics import summarize_trajectory
from .state import EpisodeSummary, FisheryState, PolicyArm, TrajectoryPoint


def initial_state(config: FisheryConfig) -> FisheryState:
    return FisheryState(
        time=0,
        fish_population=config.initial_fish_population,
        ships=config.initial_ships,
    )


def rollout(
    initial: FisheryState,
    policy: PolicyArm,
    config: FisheryConfig,
    horizon_steps: int = None,
) -> Tuple[EpisodeSummary, List[TrajectoryPoint]]:
    """Run a full deterministic episode under one fixed policy."""

    state = initial
    steps = horizon_steps if horizon_steps is not None else config.horizon_steps
    trajectory: List[TrajectoryPoint] = []

    for step_index in range(steps):
        next_state, metrics = step(state, policy, config)
        trajectory.append(
            TrajectoryPoint(
                step=step_index,
                arm_id=policy.arm_id,
                arm_name=policy.name,
                start_fish_population=state.fish_population,
                start_ships=state.ships,
                end_fish_population=next_state.fish_population,
                end_ships=next_state.ships,
                hatch=metrics.hatch,
                death=metrics.death,
                catch=metrics.catch,
                revenue=metrics.revenue,
                cost=metrics.cost,
                profit=metrics.profit,
                ship_building=metrics.ship_building,
                ship_scrap=metrics.ship_scrap,
                density=metrics.density,
                reward=metrics.reward,
            )
        )
        state = next_state

    summary = summarize_trajectory(trajectory, policy, config)
    return summary, trajectory
