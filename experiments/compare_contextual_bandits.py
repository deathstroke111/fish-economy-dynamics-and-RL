"""Compare contextual bandit algorithms on the fishery environment."""

import argparse
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import DefaultDict, Dict, List, Optional, Tuple, Type

from bandits import (
    ContextualThompsonBandit,
    DiscretizedContextualBandit,
    LinUCBBandit,
)
from fishery.config import FisheryConfig
from fishery.env import contextual_state_features, initial_state, rollout_window
from fishery.policy import get_v1_policy_arms
from plots.plotting import bar_chart, line_chart

from .common import ensure_directory, write_csv


ContextualSpec = Tuple[str, Type]


def _build_contextual_bandits() -> List[ContextualSpec]:
    return [
        ("linucb", LinUCBBandit),
        ("contextual_thompson", ContextualThompsonBandit),
        ("discretized_contextual", DiscretizedContextualBandit),
    ]


def _warmup_arm_sequence(arm_ids: List[int], seed: int) -> List[int]:
    shared_random = random.Random(seed)
    remaining = list(arm_ids)
    ordered: List[int] = []
    while remaining:
        arm_id = shared_random.choice(remaining)
        ordered.append(arm_id)
        remaining.remove(arm_id)
    return ordered


def run_contextual_comparison(
    output_dir: Path,
    episodes: int,
    seeds: int,
    decision_interval: int = 10,
    horizon_steps: Optional[int] = None,
) -> None:
    config = FisheryConfig()
    if horizon_steps is not None:
        config = replace(config, horizon_steps=horizon_steps)
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

    contextual_specs = _build_contextual_bandits()
    for seed in range(seeds):
        warmup_sequence = _warmup_arm_sequence([policy.arm_id for policy in policies], seed=seed)
        seeded_bandits = {
            algorithm_name: bandit_type(policies, seed=seed)
            for algorithm_name, bandit_type in contextual_specs
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
                    context = contextual_state_features(current_states[algorithm_name], config)
                    if episode == 0 and window_index < len(warmup_sequence):
                        arm_id = warmup_sequence[window_index]
                    else:
                        arm_id = bandit.select_action(context)
                    policy = policy_lookup[arm_id]
                    next_state, summary, _ = rollout_window(
                        current_states[algorithm_name],
                        policy,
                        config,
                        horizon_steps=decision_interval,
                    )
                    current_states[algorithm_name] = next_state
                    bandit.update(context, arm_id, summary.cumulative_reward)

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
                            "context_bias": context[0],
                            "context_fish_norm": context[1],
                            "context_ships_norm": context[2],
                            "context_time_norm": context[3],
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

    write_csv(output_dir / "contextual_episode_summaries.csv", episode_rows)
    write_csv(output_dir / "contextual_window_summaries.csv", window_rows)

    reward_series = []
    for algorithm_name, episode_map in reward_history.items():
        reward_series.append(
            (
                algorithm_name,
                [mean(episode_map[index]) for index in range(episodes)],
            )
        )
    line_chart(
        title="Mean Episode Reward by Contextual Algorithm",
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
        help="Number of simulator steps between contextual decisions",
    )
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=None,
        help="Optional override for the episode horizon in simulator steps",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/contextual_bandits",
        help="Directory for contextual bandit CSV and SVG outputs",
    )
    args = parser.parse_args()
    run_contextual_comparison(
        Path(args.output_dir),
        episodes=args.episodes,
        seeds=args.seeds,
        decision_interval=args.decision_interval,
        horizon_steps=args.horizon_steps,
    )


if __name__ == "__main__":
    main()
