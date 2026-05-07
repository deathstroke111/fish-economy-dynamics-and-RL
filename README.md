# FishEconomy

FishEconomy is a Python project for studying a classic tragedy-of-the-commons problem: a shared fishery where ecological recovery, harvesting pressure, and economic incentives interact over time.

At its core, the project asks a simple but important question:

> If a fisheries department can only change policy levers such as taxes or protected area, which policies keep the system both economically useful and ecologically alive?

The current version implements:

- a deterministic fishery simulator based on a system dynamics model
- a fixed set of policy arms inspired by the original AnyLogic-style setup
- episode-level multi-armed bandit experiments
- CSV logging and SVG plotting
- tests for lookup tables, dynamics, and bandit behavior

## Who This README Is For

This document is written for two kinds of readers:

1. Someone who wants to understand the project idea, its conclusions, and why it matters.
2. Someone who wants to understand how the codebase is wired, how data flows through it, and how to run or extend it.

If you only want the broad picture, read:

- [Project in One Page](#project-in-one-page)
- [What the Project Currently Shows](#what-the-project-currently-shows)
- [Broad Learnings](#broad-learnings)

If you want the implementation view, read:

- [How the Model Works](#how-the-model-works)
- [Repository Structure](#repository-structure)
- [Execution Flow](#execution-flow)
- [How to Run the Project](#how-to-run-the-project)
- [How to Extend It](#how-to-extend-it)

## Project in One Page

This project models a fishery as a coupled bioeconomic system.

The ecological side:

- fish reproduce
- fish die naturally
- fish are harvested by ships

The economic side:

- ships generate revenue by catching fish
- profit encourages fleet growth
- ships also decay and leave the system over time

This creates the central tragedy-of-the-commons tension:

- more fish enables more catch
- more catch can increase profit
- more profit can increase fishing effort
- more fishing effort can deplete fish
- depleted fish eventually undermine the whole economy

The policy-maker in this project is not controlling fish directly. Instead, it chooses indirect levers such as:

- fish taxes
- ship taxes
- reserved area / marine protection
- ship caps

The learning layer treats each policy bundle as an action. A bandit algorithm chooses one policy arm for a full episode, observes the resulting reward, and gradually learns which arms tend to perform well.

## Why This Project Exists

There are two overlapping goals:

1. Scientific intuition:
   understand how short-term exploitation interacts with long-term ecological collapse.
2. Decision-making intuition:
   understand where simple bandits help, where they mislead, and why this kind of problem eventually wants reinforcement learning.

This matters because the problem is not a pure static optimization problem. A policy changes the future state of the fish stock, so the environment is stateful and path-dependent.

That is exactly why this repo starts with bandits for intuition, but is intentionally structured so it can later evolve toward an RL environment.

## What the Project Currently Shows

The current implementation uses:

- monthly timesteps
- 120-month episodes
- deterministic dynamics
- piecewise-linear lookup tables for mortality and catch efficiency
- 7 fixed policy arms
- a balanced reward combining profit and sustainability

From the current baseline outputs in [`outputs/baseline_cli`](./outputs/baseline_cli), the qualitative story is already visible:

- `baseline_laissez_faire` collapses, with collapse detected around month 25.
- `moderate_taxation` and `high_taxation` avoid collapse over the 120-month horizon.
- `moderate_marine_reserve` and `strong_marine_reserve` also avoid collapse.
- `strong_marine_reserve` achieves the highest cumulative reward in the current reward design.
- `high_taxation` produces the highest cumulative raw profit, but not the highest reward.

One important caveat is also visible:

- `hard_capacity_control` and `weak_capacity_control` currently behave almost identically to baseline, because under the present calibration the fleet never grows enough for the cap to bind.

That caveat is not a bug in the project structure. It is a calibration result, and it highlights an important modeling lesson: policy intent only matters if it meaningfully couples into the system dynamics.

## Broad Learnings

Even in its current V1 form, the project already supports several useful conclusions.

### 1. Reward design changes what “best” means

If you optimize for raw profit alone, the system may reward policies that are ecologically risky. Once fish-stock health is included in reward, the ranking of policies changes.

In the current setup:

- `high_taxation` has the highest raw profit
- `strong_marine_reserve` has the highest cumulative reward

That is a very practical lesson: the choice of objective function is a policy decision, not just a technical detail.

### 2. Bandits are useful, but this is not truly a bandit world

The bandit framing is still useful because it lets us compare policy bundles quickly. But the underlying environment is stateful:

- today’s action affects tomorrow’s fish stock
- tomorrow’s fish stock affects future catch
- future catch affects future profits and ships

So the bandit layer is best understood as a simplified experimentation lens, not the final theory of control for this problem.

### 3. Conservation can outperform naive exploitation on the chosen reward

The current outputs suggest that stronger ecological protection can dominate laissez-faire behavior once long-term system health is valued.

### 4. Some policy ideas need calibration, not just code

The capacity-cap policies are implemented correctly in V1, but do not yet produce interesting behavior because the simulated fleet stays too small. This is a reminder that a policy mechanism can exist in code without becoming behaviorally relevant in the model.

## Model Overview

The simulator is based on two stocks:

- `F`: fish population
- `S`: ships / fishing effort

With continuous-time inspiration but discrete-time implementation:

- fish dynamics: `dF/dt = hatch - death - catch`
- ship dynamics: `dS/dt = ship_building - ship_scrap`

In practice, V1 uses Euler stepping with `dt = 1 month`.

### Ecological side

- fish hatch at a base percentage rate
- natural death depends on `F / K` through a lookup table
- catch depends on effective density through another lookup table

### Economic side

- catch generates revenue
- revenue minus operating cost gives profit
- positive profit drives ship-building
- ships are scrapped over their lifetime

### Policy side

Each policy arm changes one or more levers:

- `fish_tax`
- `ship_tax`
- `reserved_area`
- `ship_cap`

## V1 Policy Arms

The current experiment set includes exactly these seven arms:

| Arm ID | Name | Policy |
| --- | --- | --- |
| 1 | `baseline_laissez_faire` | No intervention |
| 3 | `moderate_taxation` | `ship_tax = 30` |
| 4 | `high_taxation` | `ship_tax = 60` |
| 5 | `hard_capacity_control` | `ship_cap = 10` |
| 7 | `weak_capacity_control` | `ship_cap = 25` |
| 10 | `moderate_marine_reserve` | `reserved_area = 25` |
| 11 | `strong_marine_reserve` | `reserved_area = 40` |

These are defined in [`fishery/policy.py`](./fishery/policy.py).

## Reward Design

The training reward for V1 is:

`reward_t = (profit_t / 1000) + (fish_t / K) - 10 * I(fish_t < 0.1 * K)`

This means the learner is rewarded for:

- making profit
- keeping fish stock high relative to carrying capacity

And strongly penalized for collapse:

- collapse threshold: `fish_population < 0.1 * carrying_capacity`

Episode reward is the sum of per-step rewards across 120 months.

## Repository Structure

```text
FishEconomy/
├── bandits/
│   ├── base.py
│   ├── epsilon_greedy.py
│   ├── softmax.py
│   ├── ucb.py
│   └── pac.py
├── experiments/
│   ├── baseline.py
│   ├── common.py
│   └── compare_bandits.py
├── fishery/
│   ├── config.py
│   ├── dynamics.py
│   ├── env.py
│   ├── lookup.py
│   ├── metrics.py
│   ├── policy.py
│   ├── rewards.py
│   └── state.py
├── plots/
│   └── plotting.py
├── tests/
│   ├── test_bandits.py
│   ├── test_dynamics.py
│   └── test_lookup.py
├── outputs/
├── IMPLEMENTATION_PLAN.md
├── pyproject.toml
├── simulator.py
└── README.md
```

## How the Code Is Wired

The project is intentionally split into small layers.

### `fishery/`: simulation core

This is the heart of the project.

- [`fishery/config.py`](./fishery/config.py)
  holds the full parameter set, including lookup-table data and default horizon values.
- [`fishery/state.py`](./fishery/state.py)
  defines the dataclasses used throughout the project:
  `FisheryState`, `PolicyArm`, `StepMetrics`, `TrajectoryPoint`, and `EpisodeSummary`.
- [`fishery/lookup.py`](./fishery/lookup.py)
  implements piecewise-linear interpolation.
- [`fishery/dynamics.py`](./fishery/dynamics.py)
  implements one simulation step.
- [`fishery/rewards.py`](./fishery/rewards.py)
  computes the per-step reward.
- [`fishery/metrics.py`](./fishery/metrics.py)
  converts a trajectory into an episode summary.
- [`fishery/env.py`](./fishery/env.py)
  provides `initial_state(...)` and `rollout(...)`.

### `bandits/`: action-selection algorithms

Each bandit selects a policy arm ID and updates its estimate from episode reward.

- [`bandits/base.py`](./bandits/base.py)
  defines the shared interface and running-average update rule.
- [`bandits/epsilon_greedy.py`](./bandits/epsilon_greedy.py)
  explores randomly with probability `epsilon`.
- [`bandits/softmax.py`](./bandits/softmax.py)
  samples actions using Boltzmann probabilities.
- [`bandits/ucb.py`](./bandits/ucb.py)
  uses optimism under uncertainty.
- [`bandits/pac.py`](./bandits/pac.py)
  uses a successive-elimination pure-exploration strategy.

### `experiments/`: runnable workflows

- [`experiments/baseline.py`](./experiments/baseline.py)
  runs each policy arm once and writes trajectories, summaries, and comparison plots.
- [`experiments/compare_bandits.py`](./experiments/compare_bandits.py)
  runs multiple seeds and episodes for each bandit algorithm and writes summary outputs.
- [`experiments/common.py`](./experiments/common.py)
  contains shared CSV and directory helpers.

### `plots/`: dependency-light chart rendering

[`plots/plotting.py`](./plots/plotting.py) writes SVG charts directly. This avoids requiring `matplotlib` for the current environment.

### `tests/`: guardrails

The tests focus on correctness of:

- lookup interpolation
- percentage interpretation for mortality
- key monotonicity properties in the dynamics
- basic bandit update behavior
- coarse qualitative policy behavior

## Execution Flow

The project’s runtime flow looks like this:

```text
Policy arm or bandit choice
        ->
Initial state
        ->
step(...)
        ->
TrajectoryPoint logging
        ->
EpisodeSummary aggregation
        ->
CSV + SVG outputs
```

For a baseline run:

1. Load config and the 7 V1 policy arms.
2. Reset to the same initial state for each arm.
3. Roll out 120 monthly steps.
4. Save trajectory and summary CSVs.
5. Render comparison plots.

For a bandit run:

1. Build a bandit with the 7 policy arms.
2. Reset the simulator for each episode.
3. Let the bandit choose one arm for the full episode.
4. Roll out 120 monthly steps.
5. Update the bandit with episode reward.
6. Aggregate results over many episodes and seeds.

## How to Run the Project

### Requirements

- Python 3.10+

The current code path is intentionally light on dependencies and runs with the standard library in this environment. The `pyproject.toml` currently lists `numpy`, but the main simulator, experiments, and tests do not rely on it.

### Run tests

```bash
python3 -m unittest discover -s tests
```

### Run the baseline policy comparison

```bash
python3 simulator.py baseline --output-dir outputs/baseline_cli
```

This writes:

- `baseline_summaries.csv`
- `baseline_trajectories.csv`
- `fish_stock_over_time.svg`
- `ships_over_time.svg`
- `reward_by_arm.svg`
- `collapse_rate_by_arm.svg`

### Run the bandit comparison

```bash
python3 simulator.py compare-bandits --episodes 200 --seeds 20 --output-dir outputs/bandits
```

For faster smoke runs:

```bash
python3 simulator.py compare-bandits --episodes 10 --seeds 3 --output-dir outputs/bandits_smoke
```

This writes:

- `bandit_episode_summaries.csv`
- `reward_by_episode.svg`
- one action-frequency chart per algorithm
- one average-reward-by-arm chart per algorithm
- one collapse-rate-by-arm chart per algorithm

## Understanding the Output Files

### Baseline outputs

The most useful file for quick interpretation is:

- [`outputs/baseline_cli/baseline_summaries.csv`](./outputs/baseline_cli/baseline_summaries.csv)

It tells you, per policy arm:

- total reward
- total profit
- average fish stock
- minimum fish stock
- whether the system collapsed
- when collapse first occurred
- final ships
- final fish population

### Bandit outputs

The main file is:

- [`outputs/bandits_cli_v2/bandit_episode_summaries.csv`](./outputs/bandits_cli_v2/bandit_episode_summaries.csv)

It lets you inspect:

- which arm each algorithm selected in each episode
- how reward evolved by episode
- which arms each algorithm preferred over time

## Current Results Snapshot

Using the checked-in outputs from the current repo state:

- Baseline collapses with negative cumulative reward.
- Both reserve arms and both taxation arms avoid collapse over 120 months.
- `strong_marine_reserve` currently has the best cumulative reward: about `48.94`.
- `high_taxation` currently has the best cumulative raw profit: about `7215.47`.
- In the sample bandit runs in [`outputs/bandits_cli_v2`](./outputs/bandits_cli_v2), the better-performing algorithms tend to concentrate on `strong_marine_reserve`.

The sample bandit averages from that output set are roughly:

| Algorithm | Mean episode reward in sample run |
| --- | --- |
| `softmax` | `-248.653` |
| `ucb1` | `-248.653` |
| `epsilon_greedy` | `-249.163` |
| `pac_successive_elimination` | `-349.074` |

These values are from a small checked-in sample run, so they should be interpreted as workflow confirmation, not final scientific results.

## Assumptions and Known Limitations

This version makes a few deliberate choices:

- deterministic dynamics only
- no stochastic ecological shocks
- no observation noise
- no adaptive within-episode control
- no RL environment yet
- no direct AnyLogic parity guarantee

There are also two especially important modeling notes:

### 1. This is episode-level control, not step-level control

Bandits choose one policy for the whole episode. That is a good simplification for V1, but it is not the same as adaptive governance.

### 2. Capacity caps are currently structurally present but behaviorally inactive

Because fleet growth remains tiny in the current calibration, ship caps of `10` and `25` do not bind. If caps are meant to matter, the next step is calibration work, not a rewrite of the architecture.

## How to Extend It

Some natural next steps are:

### Modeling extensions

- calibrate parameters against the original AnyLogic behavior
- add stochastic shocks
- add explicit quota mechanics
- add technology-change arms
- add adaptive investment mechanisms

### Learning extensions

- move from episode-level bandits to control-window bandits
- build a Gymnasium-style RL wrapper
- allow state-dependent policies
- compare myopic rewards versus discounted long-horizon returns

### Analysis extensions

- sensitivity analysis over key parameters
- policy robustness under noise
- reward-ablation studies
- better experiment reporting and tables

## Recommended Reading Order for New Contributors

If you are new to the codebase, this order works well:

1. Read this README.
2. Read [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md).
3. Read [`fishery/config.py`](./fishery/config.py) and [`fishery/policy.py`](./fishery/policy.py).
4. Read [`fishery/dynamics.py`](./fishery/dynamics.py) and [`fishery/env.py`](./fishery/env.py).
5. Run the baseline workflow.
6. Inspect the CSV summaries and SVG outputs.
7. Then move to [`bandits/`](./bandits) and [`experiments/compare_bandits.py`](./experiments/compare_bandits.py).

## Final Takeaway

FishEconomy is not just a simulator. It is a small experimental lab for thinking about control in ecological-economic systems.

Its main value is that it makes a subtle point concrete:

> A policy can look attractive in the short run while quietly degrading the state that makes future success possible.

That is the heart of the tragedy of the commons, and it is exactly the kind of setting where learning, control, and system dynamics need to be studied together rather than in isolation.
