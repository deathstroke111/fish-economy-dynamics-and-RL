"""Run one deterministic episode for each V1 policy arm."""

import argparse
from pathlib import Path
from typing import Dict, List

from fishery.config import FisheryConfig
from fishery.env import initial_state, rollout
from fishery.policy import get_v1_policy_arms
from plots.plotting import bar_chart, line_chart

from .common import ensure_directory, write_csv


def run_baseline(output_dir: Path) -> None:
    config = FisheryConfig()
    output_dir = ensure_directory(output_dir)
    policies = get_v1_policy_arms()

    summary_rows: List[Dict[str, object]] = []
    trajectory_rows: List[Dict[str, object]] = []
    fish_series = []
    ship_series = []
    reward_by_arm = {}
    collapse_by_arm = {}

    for policy in policies:
        summary, trajectory = rollout(initial_state(config), policy, config)
        summary_rows.append(summary.to_dict())
        trajectory_rows.extend(point.to_dict() for point in trajectory)
        fish_series.append((policy.name, [point.end_fish_population for point in trajectory]))
        ship_series.append((policy.name, [point.end_ships for point in trajectory]))
        reward_by_arm[policy.name] = summary.cumulative_reward
        collapse_by_arm[policy.name] = 1.0 if summary.collapse_flag else 0.0

    write_csv(output_dir / "baseline_summaries.csv", summary_rows)
    write_csv(output_dir / "baseline_trajectories.csv", trajectory_rows)

    line_chart(
        title="Fish Stock Over Time by Policy Arm",
        x_label="Month",
        y_label="Fish Population",
        series=fish_series,
        output_path=output_dir / "fish_stock_over_time.svg",
    )
    line_chart(
        title="Ships Over Time by Policy Arm",
        x_label="Month",
        y_label="Ships",
        series=ship_series,
        output_path=output_dir / "ships_over_time.svg",
    )
    bar_chart(
        title="Cumulative Reward by Policy Arm",
        x_label="Policy Arm",
        y_label="Cumulative Reward",
        values=reward_by_arm,
        output_path=output_dir / "reward_by_arm.svg",
    )
    bar_chart(
        title="Collapse Flag by Policy Arm",
        x_label="Policy Arm",
        y_label="Collapse",
        values=collapse_by_arm,
        output_path=output_dir / "collapse_rate_by_arm.svg",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/baseline",
        help="Directory for CSV and SVG outputs",
    )
    args = parser.parse_args()
    run_baseline(Path(args.output_dir))


if __name__ == "__main__":
    main()
