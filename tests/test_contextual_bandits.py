import tempfile
import unittest
from pathlib import Path

from bandits import (
    ContextualThompsonBandit,
    DiscretizedContextualBandit,
    LinUCBBandit,
)
from experiments.compare_contextual_bandits import run_contextual_comparison
from fishery.config import FisheryConfig
from fishery.env import contextual_state_features
from fishery.state import FisheryState
from experiments.common import ensure_directory


class ContextualBanditTests(unittest.TestCase):
    def setUp(self) -> None:
        from fishery.policy import get_v1_policy_arms

        self.arms = get_v1_policy_arms()
        self.config = FisheryConfig()

    def test_context_extraction_uses_expected_feature_order(self) -> None:
        state = FisheryState(time=30, fish_population=600.0, ships=3.0)
        context = contextual_state_features(state, self.config)
        self.assertEqual(len(context), 4)
        self.assertEqual(context[0], 1.0)
        self.assertAlmostEqual(context[1], 0.5)
        self.assertAlmostEqual(context[2], 0.3)
        self.assertAlmostEqual(context[3], 30.0 / self.config.horizon_steps)

    def test_context_ships_feature_is_capped(self) -> None:
        state = FisheryState(time=0, fish_population=1000.0, ships=50.0)
        context = contextual_state_features(state, self.config)
        self.assertEqual(context[2], 1.0)

    def test_linucb_updates_selected_arm_only(self) -> None:
        bandit = LinUCBBandit(self.arms, seed=1)
        context = [1.0, 0.5, 0.2, 0.1]
        before_selected = [row[:] for row in bandit.A_matrices[1]]
        before_other = [row[:] for row in bandit.A_matrices[3]]
        bandit.update(context, 1, 5.0)
        self.assertNotEqual(bandit.A_matrices[1], before_selected)
        self.assertEqual(bandit.A_matrices[3], before_other)

    def test_contextual_thompson_updates_selected_arm_only(self) -> None:
        bandit = ContextualThompsonBandit(self.arms, seed=1)
        context = [1.0, 0.5, 0.2, 0.1]
        before_selected = list(bandit.b_vectors[1])
        before_other = list(bandit.b_vectors[3])
        bandit.update(context, 1, 5.0)
        self.assertNotEqual(bandit.b_vectors[1], before_selected)
        self.assertEqual(bandit.b_vectors[3], before_other)

    def test_discretized_contextual_updates_selected_bucket_only(self) -> None:
        bandit = DiscretizedContextualBandit(self.arms, seed=1)
        context = [1.0, 0.8, 0.3, 0.2]
        bucket = bandit.bucket_for_context(context)
        bandit.update(context, 1, 3.0)
        self.assertEqual(bandit.bucket_counts[bucket][1], 1)
        self.assertAlmostEqual(bandit.bucket_values[bucket][1], 3.0)

    def test_unseen_bucket_explores_untried_arms_first(self) -> None:
        bandit = DiscretizedContextualBandit(self.arms, seed=2)
        context = [1.0, 0.6, 0.1, 0.1]
        seen = {bandit.select_action(context) for _ in range(len(self.arms))}
        self.assertTrue(seen.issubset({arm.arm_id for arm in self.arms}))

    def test_contextual_smoke_outputs_and_shared_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = ensure_directory(Path(temp_dir) / "contextual")
            run_contextual_comparison(
                output_dir=output_dir,
                episodes=2,
                seeds=1,
                decision_interval=10,
                horizon_steps=120,
            )

            episode_lines = output_dir.joinpath("contextual_episode_summaries.csv").read_text().strip().splitlines()
            window_lines = output_dir.joinpath("contextual_window_summaries.csv").read_text().strip().splitlines()
            self.assertEqual(len(episode_lines) - 1, 3 * 1 * 2)
            self.assertEqual(len(window_lines) - 1, 3 * 1 * 2 * 12)

            header = window_lines[0].split(",")
            self.assertIn("context_bias", header)
            self.assertIn("context_fish_norm", header)
            self.assertIn("context_ships_norm", header)
            self.assertIn("context_time_norm", header)

            rows = [line.split(",") for line in window_lines[1:]]
            column_index = {name: header.index(name) for name in header}
            warmup = {}
            for row in rows:
                if row[column_index["seed"]] == "0" and row[column_index["episode"]] == "0":
                    warmup.setdefault(row[column_index["algorithm"]], []).append(
                        row[column_index["arm_id"]]
                    )
            first_passes = [sequence[:7] for sequence in warmup.values()]
            self.assertTrue(all(sequence == first_passes[0] for sequence in first_passes[1:]))
            divergence = {tuple(sequence[7:12]) for sequence in warmup.values()}
            self.assertGreater(len(divergence), 1)


if __name__ == "__main__":
    unittest.main()
