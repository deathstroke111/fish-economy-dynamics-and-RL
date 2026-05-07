import unittest

from fishery.config import FisheryConfig
from fishery.dynamics import step
from fishery.env import initial_state, rollout
from fishery.policy import get_v1_policy_arms
from fishery.state import FisheryState, PolicyArm


class DynamicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = FisheryConfig()

    def test_death_fraction_is_treated_as_percentage(self) -> None:
        state = FisheryState(time=0, fish_population=1200.0, ships=0.0)
        policy = PolicyArm(arm_id=1, name="baseline")
        _, metrics = step(state, policy, self.config)
        self.assertAlmostEqual(metrics.death, 72.0, places=5)

    def test_zero_ships_implies_zero_catch(self) -> None:
        state = FisheryState(time=0, fish_population=1000.0, ships=0.0)
        policy = PolicyArm(arm_id=1, name="baseline")
        _, metrics = step(state, policy, self.config)
        self.assertEqual(metrics.catch, 0.0)

    def test_population_and_ships_never_go_negative(self) -> None:
        state = FisheryState(time=0, fish_population=1.0, ships=1000.0)
        policy = PolicyArm(arm_id=1, name="baseline")
        next_state, _ = step(state, policy, self.config)
        self.assertGreaterEqual(next_state.fish_population, 0.0)
        self.assertGreaterEqual(next_state.ships, 0.0)

    def test_higher_reserved_area_reduces_catch(self) -> None:
        state = FisheryState(time=0, fish_population=1000.0, ships=5.0)
        open_water = PolicyArm(arm_id=1, name="open", reserved_area=0.0)
        reserve = PolicyArm(arm_id=2, name="reserve", reserved_area=40.0)
        _, open_metrics = step(state, open_water, self.config)
        _, reserve_metrics = step(state, reserve, self.config)
        self.assertLessEqual(reserve_metrics.catch, open_metrics.catch)

    def test_higher_ship_tax_reduces_ship_building(self) -> None:
        state = FisheryState(time=0, fish_population=1000.0, ships=5.0)
        low_tax = PolicyArm(arm_id=1, name="low", ship_tax=0.0)
        high_tax = PolicyArm(arm_id=2, name="high", ship_tax=60.0)
        _, low_metrics = step(state, low_tax, self.config)
        _, high_metrics = step(state, high_tax, self.config)
        self.assertLessEqual(high_metrics.ship_building, low_metrics.ship_building)

    def test_ship_cap_is_enforced(self) -> None:
        state = FisheryState(time=0, fish_population=1000.0, ships=9.9)
        cap_policy = PolicyArm(arm_id=5, name="cap", ship_cap=10.0)
        next_state, _ = step(state, cap_policy, self.config)
        self.assertLessEqual(next_state.ships, 10.0)

    def test_policy_rollouts_show_expected_relative_behavior(self) -> None:
        summaries = {}
        for policy in get_v1_policy_arms():
            summary, _ = rollout(initial_state(self.config), policy, self.config)
            summaries[policy.name] = summary

        self.assertTrue(summaries["baseline_laissez_faire"].collapse_flag)
        self.assertFalse(summaries["moderate_taxation"].collapse_flag)
        self.assertFalse(summaries["high_taxation"].collapse_flag)
        self.assertTrue(
            summaries["high_taxation"].final_fish_population
            > summaries["moderate_taxation"].final_fish_population
        )
        self.assertTrue(summaries["strong_marine_reserve"].final_fish_population > summaries["moderate_marine_reserve"].final_fish_population)


if __name__ == "__main__":
    unittest.main()
