"""Compare bandit algorithms on the fishery environment."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import DefaultDict, Dict, List, Tuple, Type

from bandits import (
    EpsilonGreedyBandit,
    PACSuccessiveEliminationBandit,
    SoftmaxBandit,
    UCB1Bandit,
)
from fishery.config import FisheryConfig
from fishery.env import initial_state, rollout_window
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


def run_comparison(output_dir: Path, episodes: int, seeds: int, decision_interval: int = 10) -> None:
    config = FisheryConfig()
    policies = get_v1_policy_arms()
    policy_lookup = {policy.arm_id: policy for policy in policies}
    output_dir = ensure_directory(output_dir)

    if config.horizon_steps % decision_interval != 0:
        raise ValueError("decision_interval must divide the episode horizon exactly")

    episode_rows: List[Dict[str, object]] = []
    window_rows: List[Dict[str, object]] = []
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

    bandit_specs = _build_bandits()
    for seed in range(seeds):
        seeded_bandits = {
            algorithm_name: bandit_type(policies, seed=seed)
            for algorithm_name, bandit_type in bandit_specs
        }
        for episode in range(episodes):
            episode_totals = {
                algorithm_name: {
                    "reward": 0.0,
                    "profit": 0.0,
                    "min_fish": None,
                    "collapse_flag": False,
                    "time_to_collapse": None,
                }
                for algorithm_name in seeded_bandits
            }
            current_states = {
                algorithm_name: initial_state(config)
                for algorithm_name in seeded_bandits
            }

            num_windows = config.horizon_steps // decision_interval
            for window_index in range(num_windows):
                for algorithm_name, bandit in seeded_bandits.items():
                    arm_id = bandit.select_action()
                    policy = policy_lookup[arm_id]
                    next_state, summary, _ = rollout_window(
                        current_states[algorithm_name],
                        policy,
                        config,
                        horizon_steps=decision_interval,
                    )
                    current_states[algorithm_name] = next_state
                    bandit.update(arm_id, summary.cumulative_reward)

                    totals = episode_totals[algorithm_name]
                    totals["reward"] += summary.cumulative_reward
                    totals["profit"] += summary.cumulative_profit
                    totals["min_fish"] = (
                        summary.min_fish_stock
                        if totals["min_fish"] is None
                        else min(totals["min_fish"], summary.min_fish_stock)
                    )
                    if summary.collapse_flag and totals["time_to_collapse"] is None:
                        totals["collapse_flag"] = True
                        totals["time_to_collapse"] = (window_index * decision_interval) + (
                            summary.time_to_collapse or 0
                        )

                    window_rows.append(
                        {
                            "algorithm": algorithm_name,
                            "seed": seed,
                            "episode": episode,
                            "window_index": window_index,
                            "window_start_step": window_index * decision_interval,
                            "window_end_step": ((window_index + 1) * decision_interval) - 1,
                            "arm_id": summary.arm_id,
                            "arm_name": summary.arm_name,
                            "window_cumulative_reward": summary.cumulative_reward,
                            "window_cumulative_profit": summary.cumulative_profit,
                            "window_mean_fish_stock": summary.mean_fish_stock,
                            "window_min_fish_stock": summary.min_fish_stock,
                            "window_collapse_flag": int(summary.collapse_flag),
                            "window_time_to_collapse": summary.time_to_collapse,
                            "end_ships": summary.final_ships,
                            "end_fish_population": summary.final_fish_population,
                        }
                    )

                    action_counts[algorithm_name][policy.name] += 1
                    reward_by_arm[algorithm_name][policy.name].append(summary.cumulative_reward)
                    collapse_by_arm[algorithm_name][policy.name].append(
                        1.0 if summary.collapse_flag else 0.0
                    )

            for algorithm_name, state in current_states.items():
                totals = episode_totals[algorithm_name]
                reward_history[algorithm_name][episode].append(totals["reward"])
                episode_rows.append(
                    {
                        "algorithm": algorithm_name,
                        "seed": seed,
                        "episode": episode,
                        "cumulative_reward": totals["reward"],
                        "cumulative_profit": totals["profit"],
                        "min_fish_stock": totals["min_fish"],
                        "collapse_flag": int(totals["collapse_flag"]),
                        "time_to_collapse": totals["time_to_collapse"],
                        "final_ships": state.ships,
                        "final_fish_population": state.fish_population,
                        "decision_windows": num_windows,
                    }
                )

    write_csv(output_dir / "bandit_episode_summaries.csv", episode_rows)
    write_csv(output_dir / "bandit_window_summaries.csv", window_rows)

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
        "--decision-interval",
        type=int,
        default=10,
        help="Number of simulator steps between bandit decisions",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/bandits",
        help="Directory for CSV and SVG outputs",
    )
    args = parser.parse_args()
    run_comparison(
        Path(args.output_dir),
        episodes=args.episodes,
        seeds=args.seeds,
        decision_interval=args.decision_interval,
    )


if __name__ == "__main__":
    main()
