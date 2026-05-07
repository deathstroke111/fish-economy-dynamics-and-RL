"""Core step equations for the fishery model."""

from typing import Tuple

from .config import FisheryConfig
from .lookup import linear_interpolate
from .rewards import compute_step_reward
from .state import FisheryState, PolicyArm, StepMetrics


def step(
    state: FisheryState,
    policy: PolicyArm,
    config: FisheryConfig,
) -> Tuple[FisheryState, StepMetrics]:
    """Advance the simulator by one discrete timestep."""

    fish_population = state.fish_population
    ships = state.ships

    hatch_rate_percent = config.hatch_rate_percent + (
        policy.breeding_rate_bonus * config.hatch_rate_percent / 100.0
    )
    hatch = (hatch_rate_percent / 100.0) * fish_population

    death_fraction_percent = linear_interpolate(
        fish_population / config.carrying_capacity,
        config.death_fraction_points,
    )
    death = (death_fraction_percent / 100.0) * fish_population

    density = (fish_population / config.area) * (1.0 - policy.reserved_area / 100.0)
    catch_per_ship = linear_interpolate(density, config.catch_in_density_points)
    catch = min(catch_per_ship * ships, config.catch_max)

    effective_fish_price = config.fish_price * (1.0 - policy.fish_tax / 100.0)
    revenue = catch * effective_fish_price
    cost = config.ship_cost_per_month * ships
    profit = revenue - cost

    ship_build_time = (policy.ship_tax / 100.0 * config.baseline_ship_build_time) + (
        config.baseline_ship_build_time
    )
    ship_building = 0.0
    if ships < config.system_ship_max and profit > 0:
        ship_building = (profit * config.investment_fraction) / ship_build_time
    ship_scrap = ships / config.ship_lifetime_months

    next_fish_population = max(
        0.0,
        fish_population + config.dt_months * (hatch - death - catch),
    )
    next_ships = max(0.0, ships + config.dt_months * (ship_building - ship_scrap))
    if policy.ship_cap is not None:
        next_ships = min(next_ships, policy.ship_cap)

    next_state = FisheryState(
        time=state.time + 1,
        fish_population=next_fish_population,
        ships=next_ships,
    )
    reward = compute_step_reward(profit, next_fish_population, config)
    metrics = StepMetrics(
        hatch=hatch,
        death=death,
        catch=catch,
        revenue=revenue,
        cost=cost,
        profit=profit,
        ship_building=ship_building,
        ship_scrap=ship_scrap,
        density=density,
        reward=reward,
        death_fraction_percent=death_fraction_percent,
        catch_per_ship=catch_per_ship,
    )
    return next_state, metrics
