"""Dataclasses used by the fishery simulator."""

from dataclasses import asdict, dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class FisheryState:
    time: int
    fish_population: float
    ships: float


@dataclass(frozen=True)
class PolicyArm:
    arm_id: int
    name: str
    fish_tax: float = 0.0
    ship_tax: float = 0.0
    reserved_area: float = 0.0
    ship_cap: Optional[float] = None
    breeding_rate_bonus: float = 0.0


@dataclass(frozen=True)
class StepMetrics:
    hatch: float
    death: float
    catch: float
    revenue: float
    cost: float
    profit: float
    ship_building: float
    ship_scrap: float
    density: float
    reward: float
    death_fraction_percent: float
    catch_per_ship: float


@dataclass(frozen=True)
class TrajectoryPoint:
    step: int
    arm_id: int
    arm_name: str
    start_fish_population: float
    start_ships: float
    end_fish_population: float
    end_ships: float
    hatch: float
    death: float
    catch: float
    revenue: float
    cost: float
    profit: float
    ship_building: float
    ship_scrap: float
    density: float
    reward: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeSummary:
    arm_id: int
    arm_name: str
    cumulative_reward: float
    cumulative_profit: float
    mean_fish_stock: float
    min_fish_stock: float
    collapse_flag: bool
    time_to_collapse: Optional[int]
    final_ships: float
    final_fish_population: float
    steps: int

    def to_dict(self) -> Dict[str, float]:
        data = asdict(self)
        data["collapse_flag"] = int(self.collapse_flag)
        return data
