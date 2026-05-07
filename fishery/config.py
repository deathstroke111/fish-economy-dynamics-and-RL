"""Configuration for the fishery simulator."""

from dataclasses import dataclass, field
from typing import Tuple


Point = Tuple[float, float]


@dataclass(frozen=True)
class FisheryConfig:
    """Central source of truth for simulator constants."""

    dt_months: float = 1.0
    horizon_steps: int = 120
    initial_fish_population: float = 1000.0
    initial_ships: float = 1.0

    carrying_capacity: float = 1200.0
    area: float = 100.0
    catch_max: float = 10000.0
    system_ship_max: float = 100000.0
    ship_lifetime_months: float = 12.0

    hatch_rate_percent: float = 6.0
    fish_price: float = 20.0
    ship_cost_per_month: float = 250.0
    investment_fraction: float = 0.2
    baseline_ship_build_time: float = 300.0
    collapse_threshold_fraction: float = 0.1

    death_fraction_points: Tuple[Point, ...] = field(
        default_factory=lambda: (
            (0.0, 5.22),
            (0.2, 5.23),
            (0.4, 5.255),
            (0.6, 5.345),
            (0.8, 5.665),
            (1.0, 6.0),
            (1.2, 6.44),
            (1.4, 7.13),
            (1.6, 7.97),
            (1.8, 9.32),
            (2.0, 11.0),
        )
    )
    catch_in_density_points: Tuple[Point, ...] = field(
        default_factory=lambda: (
            (0.0, 0.0),
            (1.0, 5.0),
            (2.0, 10.4),
            (3.0, 15.9),
            (4.0, 20.2),
            (5.0, 22.1),
            (6.0, 23.2),
            (7.0, 23.8),
            (8.0, 24.2),
            (9.0, 24.6),
            (10.0, 25.0),
        )
    )

    def collapse_threshold(self) -> float:
        return self.carrying_capacity * self.collapse_threshold_fraction
