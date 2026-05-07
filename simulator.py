"""CLI entry point for FishEconomy experiments."""

import argparse
from pathlib import Path

from experiments.baseline import run_baseline
from experiments.compare_bandits import run_comparison


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
        "--output-dir",
        default="outputs/bandits",
        help="Directory for bandit CSV and SVG outputs",
    )

    args = parser.parse_args()
    if args.command == "baseline":
        run_baseline(Path(args.output_dir))
        return
    if args.command == "compare-bandits":
        run_comparison(Path(args.output_dir), episodes=args.episodes, seeds=args.seeds)
        return


if __name__ == "__main__":
    main()
