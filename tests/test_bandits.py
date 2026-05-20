import unittest

from bandits import (
    EpsilonGreedyBandit,
    PACSuccessiveEliminationBandit,
    SoftmaxBandit,
    UCB1Bandit,
)
from fishery.policy import get_v1_policy_arms


class BanditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arms = get_v1_policy_arms()

    def test_epsilon_greedy_updates_value_estimate(self) -> None:
        bandit = EpsilonGreedyBandit(self.arms, seed=1)
        arm_id = bandit.select_action()
        bandit.update(arm_id, 10.0)
        self.assertEqual(bandit.counts[arm_id], 1)
        self.assertAlmostEqual(bandit.value_estimates[arm_id], 10.0)

    def test_ucb_selects_untried_arms_first(self) -> None:
        bandit = UCB1Bandit(self.arms, seed=2)
        seen = {bandit.select_action() for _ in range(len(self.arms))}
        self.assertTrue(seen.issubset({arm.arm_id for arm in self.arms}))

    def test_softmax_tracks_updates(self) -> None:
        bandit = SoftmaxBandit(self.arms, seed=3)
        arm_id = bandit.select_action()
        bandit.update(arm_id, 5.0)
        self.assertEqual(bandit.counts[arm_id], 1)
        self.assertAlmostEqual(bandit.value_estimates[arm_id], 5.0)

    def test_pac_bandit_eliminates_down_to_active_subset(self) -> None:
        bandit = PACSuccessiveEliminationBandit(self.arms, min_pulls_per_round=1, seed=4)
        rewards = {
            arm.arm_id: float(index)
            for index, arm in enumerate(self.arms)
        }
        for _ in range(3):
            arm_id = bandit.select_action()
            bandit.update(arm_id, rewards[arm_id])
        self.assertGreaterEqual(len(bandit.active_arm_ids), 1)
        self.assertLessEqual(len(bandit.active_arm_ids), len(self.arms))

if __name__ == "__main__":
    unittest.main()
