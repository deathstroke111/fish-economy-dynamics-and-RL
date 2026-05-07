"""Compare bandit algorithms on the fishery environment."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import DefaultDict, Dict, Iterable, List, Tuple, Type

from bandits import (
    EpsilonGreedyBandit,
    PACSuccessiveEliminationBandit,
    SoftmaxBandit,
    UCB1Bandit,
)
from fishery.config import FisheryConfig
from fishery.env import initial_state, rollout
from fishery.policy import get_v1_policy_arms
from plots.plotting import bar_chart, line_chart

from .common import ensure_directory, write_csv


BanditSpec = Tuple[str, Type]


def _build_bandits() -> List[BanditSpec]:
    return [
        ("epsilon_greedy", EpsilonGreedyBandit),
        ("ucb1", UCB1Bandit),
        ("softmax", SoftmaxBandit),
        ("pac_successive_elimination", PACSuccessiveEliminationBandit),
    ]


def run_comparison(output_dir: Path, episodes: int, seeds: int) -> None:
    config = FisheryConfig()
    policies = get_v1_policy_arms()
    policy_lookup = {policy.arm_id: policy for policy in policies}
    output_dir = ensure_directory(output_dir)

    episode_rows: List[Dict[str, object]] = []
    action_counts: DefaultDict[str, Dict[str, int]] = defaultdict(
        lambda: {policy.name: 0 for policy in policies}
    )
    reward_history: DefaultDict[str, Dict[int, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    reward_by_arm: DefaultDict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    collapse_by_arm: DefaultDict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for algorithm_name, bandit_type in _build_bandits():
        for seed in range(seeds):
            bandit = bandit_type(policies, seed=seed)
            for episode in range(episodes):
                arm_id = bandit.select_action()
                policy = policy_lookup[arm_id]
                summary, _ = rollout(initial_state(config), policy, config)
                bandit.update(arm_id, summary.cumulative_reward)

                episode_rows.append(
                    {
                        "algorithm": algorithm_name,
                        "seed": seed,
                        "episode": episode,
                        "arm_id": summary.arm_id,
                        "arm_name": summary.arm_name,
                        "cumulative_reward": summary.cumulative_reward,
                        "cumulative_profit": summary.cumulative_profit,
                        "mean_fish_stock": summary.mean_fish_stock,
                        "min_fish_stock": summary.min_fish_stock,
                        "collapse_flag": int(summary.collapse_flag),
                        "time_to_collapse": summary.time_to_collapse,
                        "final_ships": summary.final_ships,
                        "final_fish_population": summary.final_fish_population,
                    }
                )
                action_counts[algorithm_name][policy.name] += 1
                reward_history[algorithm_name][episode].append(summary.cumulative_reward)
                reward_by_arm[algorithm_name][policy.name].append(summary.cumulative_reward)
                collapse_by_arm[algorithm_name][policy.name].append(
                    1.0 if summary.collapse_flag else 0.0
                )

    write_csv(output_dir / "bandit_episode_summaries.csv", episode_rows)

    reward_series = []
    for algorithm_name, episode_map in reward_history.items():
        reward_series.append(
            (
                algorithm_name,
                [mean(episode_map[index]) for index in range(episodes)],
            )
        )
    line_chart(
        title="Mean Episode Reward by Algorithm",
        x_label="Episode",
        y_label="Mean Reward",
        series=reward_series,
        output_path=output_dir / "reward_by_episode.svg",
    )

    for algorithm_name, counts in action_counts.items():
        bar_chart(
            title=f"Action Selection Frequency - {algorithm_name}",
            x_label="Policy Arm",
            y_label="Selections",
            values=counts,
            output_path=output_dir / f"{algorithm_name}_action_frequency.svg",
        )
        bar_chart(
            title=f"Average Reward by Arm - {algorithm_name}",
            x_label="Policy Arm",
            y_label="Average Reward",
            values={
                arm_name: mean(values) if values else 0.0
                for arm_name, values in reward_by_arm[algorithm_name].items()
            },
            output_path=output_dir / f"{algorithm_name}_average_reward_by_arm.svg",
        )
        bar_chart(
            title=f"Collapse Rate by Arm - {algorithm_name}",
            x_label="Policy Arm",
            y_label="Collapse Rate",
            values={
                arm_name: mean(values) if values else 0.0
                for arm_name, values in collapse_by_arm[algorithm_name].items()
            },
            output_path=output_dir / f"{algorithm_name}_collapse_rate_by_arm.svg",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per seed")
    parser.add_argument("--seeds", type=int, default=20, help="Number of random seeds")
    parser.add_argument(
        "--output-dir",
        default="outputs/bandits",
        help="Directory for CSV and SVG outputs",
    )
    args = parser.parse_args()
    run_comparison(Path(args.output_dir), episodes=args.episodes, seeds=args.seeds)


if __name__ == "__main__":
    main()
