"""CLI entry point for FishEconomy experiments."""

import argparse
from pathlib import Path

from experiments.baseline import run_baseline
from experiments.compare_bandits import run_comparison
from experiments.compare_contextual_bandits import run_contextual_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline_parser = subparsers.add_parser("baseline", help="Run fixed-policy baselines")
    baseline_parser.add_argument(
        "--output-dir",
        default="outputs/baseline",
        help="Directory for baseline CSV and SVG outputs",
    )

    compare_parser = subparsers.add_parser(
        "compare-bandits",
        help="Run bandit comparisons over repeated episodes",
    )
    compare_parser.add_argument("--episodes", type=int, default=200, help="Episodes per seed")
    compare_parser.add_argument("--seeds", type=int, default=20, help="Number of seeds")
    compare_parser.add_argument(
        "--decision-interval",
        type=int,
        default=10,
        help="Number of simulator steps between bandit decisions",
    )
    compare_parser.add_argument(
        "--horizon-steps",
        type=int,
        default=None,
        help="Optional override for the episode horizon in simulator steps",
    )
    compare_parser.add_argument(
        "--output-dir",
        default="outputs/bandits",
        help="Directory for bandit CSV and SVG outputs",
    )

    contextual_parser = subparsers.add_parser(
        "compare-contextual-bandits",
        help="Run contextual bandit comparisons over repeated episodes",
    )
    contextual_parser.add_argument("--episodes", type=int, default=200, help="Episodes per seed")
    contextual_parser.add_argument("--seeds", type=int, default=20, help="Number of seeds")
    contextual_parser.add_argument(
        "--decision-interval",
        type=int,
        default=10,
        help="Number of simulator steps between contextual decisions",
    )
    contextual_parser.add_argument(
        "--horizon-steps",
        type=int,
        default=None,
        help="Optional override for the episode horizon in simulator steps",
    )
    contextual_parser.add_argument(
        "--output-dir",
        default="outputs/contextual_bandits",
        help="Directory for contextual bandit CSV and SVG outputs",
    )

    args = parser.parse_args()
    if args.command == "baseline":
        run_baseline(Path(args.output_dir))
        return
    if args.command == "compare-bandits":
        run_comparison(
            Path(args.output_dir),
            episodes=args.episodes,
            seeds=args.seeds,
            decision_interval=args.decision_interval,
            horizon_steps=args.horizon_steps,
        )
        return
    if args.command == "compare-contextual-bandits":
        run_contextual_comparison(
            Path(args.output_dir),
            episodes=args.episodes,
            seeds=args.seeds,
            decision_interval=args.decision_interval,
            horizon_steps=args.horizon_steps,
        )
        return


if __name__ == "__main__":
    main()
