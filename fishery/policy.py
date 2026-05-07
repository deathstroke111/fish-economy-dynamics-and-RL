"""Policy definitions for V1 experiments."""

from typing import List

from .state import PolicyArm


def get_v1_policy_arms() -> List[PolicyArm]:
    """Return the exact V1 arms aligned to the AnyLogic comparison set."""

    return [
        PolicyArm(arm_id=1, name="baseline_laissez_faire"),
        PolicyArm(arm_id=3, name="moderate_taxation", ship_tax=30.0),
        PolicyArm(arm_id=4, name="high_taxation", ship_tax=60.0),
        PolicyArm(arm_id=5, name="hard_capacity_control", ship_cap=10.0),
        PolicyArm(arm_id=7, name="weak_capacity_control", ship_cap=25.0),
        PolicyArm(arm_id=10, name="moderate_marine_reserve", reserved_area=25.0),
        PolicyArm(arm_id=11, name="strong_marine_reserve", reserved_area=40.0),
    ]
