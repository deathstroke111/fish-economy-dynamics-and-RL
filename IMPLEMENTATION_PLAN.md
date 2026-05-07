# FishEconomy Implementation Plan

## 1. Goal

Build a Python simulation environment for a fishery commons system, then attach learning-based policy selection so we can study:

- tragedy of the commons dynamics
- effects of taxes, quotas, subsidies, and conservation policies
- exploration vs exploitation under ecological feedback
- when simple bandits fail and RL becomes necessary

The first usable version should be:

- deterministic
- discrete-time
- easy to inspect and plot
- modular enough to swap in bandits first, RL later

## 2. Recommended V1 Scope

Keep the first version intentionally small:

- one simulator
- one set of state equations
- one action interface for policy levers
- a few fixed policy bundles as bandit arms
- batch experiment runner
- plotting and CSV logging

Do not start with:

- stochastic shocks
- deep RL
- multi-agent fishing fleets
- UI-heavy dashboards

Those are good V2+ extensions once the simulator is trusted.

## 3. Core Modeling Decisions

### 3.1 Time Representation

Your original SD model is continuous, but V1 should use fixed discrete timesteps with Euler integration:

- `dt = 1.0` for the first version
- later allow smaller `dt` such as `0.25` if stability is an issue

State update form:

- `F_next = max(0, F + dt * (H - D - C))`
- `S_next = max(0, S + dt * (B - R))`

### 3.2 Lookup Tables

Implement these as piecewise-linear interpolation functions:

- `DeathFraction(F / K)`
- `CatchInDensity(rho)`

This preserves the SD model behavior more faithfully than replacing them too early with fitted formulas.

### 3.3 Policy Timing

This is the most important design choice for bandits.

Because actions change future fish stock, this is not a pure one-step bandit. For V1, use one of these two formulations:

1. Episode-level bandit
   - choose one policy arm for the entire simulation run
   - reward is total discounted return or final score over the full horizon
   - simplest and most stable starting point

2. Control-window bandit
   - choose a policy every `N` timesteps, such as every 12 steps
   - reward is the accumulated score over that window
   - closer to control, but more state-dependent

Recommended V1:

- start with episode-level bandit
- move to control-window bandit only after baseline plots look correct

## 4. State, Actions, and Reward

### 4.1 State Variables

Minimum state:

- `fish_population`
- `ships`
- `time`

Derived variables to log every step:

- `hatch`
- `death`
- `catch`
- `revenue`
- `cost`
- `profit`
- `ship_building`
- `ship_scrap`
- `density`

### 4.2 Action Space

Treat the agent as the fisheries department. It changes policy levers, not fish or ships directly.

Recommended policy levers:

- `fish_tax`
- `ship_tax`
- `reserved_area`
- `breeding_rate_bonus`

For bandits, define a small discrete action set of policy bundles such as:

- `baseline`
- `high_tax`
- `moderate_tax`
- `quota_like_low_catch`
- `high_conservation`
- `subsidized_growth`
- `balanced_policy`

Even if quota is not explicit in the original equations, you can represent it cleanly by capping `C` with a policy-adjusted maximum catch.

### 4.3 Reward Design

Do not use raw short-term profit as the only reward, or the learner will likely collapse the fish stock.

Recommended reward options:

1. Economic reward
   - `reward = profit`

2. Sustainability-adjusted reward
   - `reward = profit - alpha * collapse_penalty`

3. Multi-objective score
   - `reward = w1 * normalized_profit + w2 * normalized_fish_stock - w3 * volatility_penalty`

Recommended V1:

- log all three
- train the bandit on one chosen reward
- make the training reward configurable

Also define a collapse threshold, for example:

- collapse if `fish_population < 0.1 * K`

## 5. Proposed Project Structure

```text
FishEconomy/
├── IMPLEMENTATION_PLAN.md
├── simulator.py                  # temporary entry point during early build
├── fishery/
│   ├── __init__.py
│   ├── config.py                 # constants and default parameters
│   ├── lookup.py                 # interpolation helpers
│   ├── state.py                  # state dataclasses
│   ├── policy.py                 # policy/action definitions
│   ├── dynamics.py               # step equations
│   ├── env.py                    # simulator wrapper / reset / rollout
│   ├── rewards.py                # reward functions
│   └── metrics.py                # summary metrics
├── bandits/
│   ├── __init__.py
│   ├── base.py
│   ├── epsilon_greedy.py
│   ├── softmax.py
│   ├── ucb.py
│   └── pac.py
├── experiments/
│   ├── baseline.py
│   ├── compare_policies.py
│   └── compare_bandits.py
├── plots/
│   └── plotting.py
└── tests/
    ├── test_lookup.py
    ├── test_dynamics.py
    ├── test_rewards.py
    └── test_bandits.py
```

If you want to stay lighter initially, keep `simulator.py` and add folders only after the first baseline run works.

## 6. Phase-by-Phase Build Plan

## Phase 1: Formalize the Model

Goal:

- translate the SD equations into explicit Python functions

Tasks:

- define all constants in one config object
- encode lookup tables
- document units and interpretation for each variable
- decide timestep `dt`
- decide whether policy acts every step or per episode

Deliverable:

- a written parameter table
- lookup functions with tests

Acceptance criteria:

- lookup functions interpolate correctly
- no hidden constants scattered across the code

## Phase 2: Build the Deterministic Simulator

Goal:

- produce a trustworthy no-agent simulation

Tasks:

- create a `State` dataclass
- implement one `step(state, policy, params)` function
- clip invalid values such as negative fish or ships
- return both next state and per-step diagnostics
- implement `rollout(horizon, initial_state, policy)`

Deliverable:

- simulator that runs from `F(0)=1000`, `S(0)=1`

Acceptance criteria:

- simulation runs for a fixed horizon without crashing
- fish and ship trajectories are sensible
- results are reproducible

## Phase 3: Baseline Analysis Without Learning

Goal:

- understand the natural system before adding bandits

Tasks:

- run baseline policy
- run a few hand-crafted policies
- plot fish, ships, catch, and profit over time
- identify collapse regimes and stable regimes

Deliverable:

- baseline notebook or script producing plots

Acceptance criteria:

- you can visually explain the reinforcing and balancing loops
- you can point to at least one overfishing scenario and one stabilized scenario

## Phase 4: Add the Policy Interface

Goal:

- make the simulator controllable by algorithms

Tasks:

- define a `PolicyAction` dataclass or simple dict
- map each policy arm to concrete lever values
- ensure policies feed into catch, costs, regeneration, or build time
- optionally add quota as a catch cap modifier

Deliverable:

- discrete action catalog usable by all algorithms

Acceptance criteria:

- any policy can be applied without changing simulator internals
- policy effects are visible in logged trajectories

## Phase 5: Add Reward and Metrics

Goal:

- separate simulator physics from optimization targets

Tasks:

- implement configurable reward functions
- implement episode summary metrics
- log:
  - cumulative profit
  - average fish stock
  - min fish stock
  - collapse count
  - final ships
  - reward

Deliverable:

- metrics module used by experiments

Acceptance criteria:

- the same rollout can be scored multiple ways
- collapse and sustainability metrics are easy to compare

## Phase 6: Integrate Bandit Algorithms

Goal:

- compare simple decision rules on top of the simulator

Tasks:

- create a common `select_action()` / `update()` interface
- implement:
  - epsilon-greedy
  - softmax
  - UCB
  - one PAC-style pure exploration variant
- run many episodes per algorithm

Recommended evaluation loop:

1. reset simulator
2. bandit selects one policy arm
3. run full episode
4. compute episode reward
5. update bandit statistics

Deliverable:

- experiment script comparing bandit learning curves

Acceptance criteria:

- each algorithm improves action estimates over repeated episodes
- results can be reproduced with a fixed random seed

## Phase 7: Experiment Harness and Visualization

Goal:

- make results easy to compare and explain

Tasks:

- batch-run many seeds
- save episode-level results to CSV
- compute mean and confidence intervals
- plot:
  - fish stock over time
  - ship count over time
  - reward by episode
  - action selection frequency
  - regret or pseudo-regret if applicable

Deliverable:

- repeatable experiment scripts and charts

Acceptance criteria:

- one command can regenerate key figures
- plots clearly differentiate algorithms and policies

## Phase 8: Prepare for RL

Goal:

- reshape the code so RL can be added without rewriting the simulator

Tasks:

- expose `reset()` and `step(action)` style environment methods
- return `(observation, reward, done, info)`
- decide observation vector
- decide action cadence for RL

Suggested observation vector:

- normalized fish stock
- normalized ships
- recent profit
- recent catch
- current policy values

Deliverable:

- environment wrapper that can later plug into Gymnasium-like APIs

Acceptance criteria:

- the simulator core is separate from the learning algorithm
- moving from bandit to RL only changes the controller layer

## 7. Mathematical Mapping Into Code

Translate the SD equations directly.

### Fish dynamics

- `H = (F_h + (B_r * F_h / 100)) * F`
- `D = death_fraction(F / K) * F`
- `rho = (F / A) * (1 - reserved_area / 100)`
- `c_s = catch_in_density(rho)`
- `C = min(c_s * S, C_max)`
- `F_next = max(0, F + dt * (H - D - C))`

### Ship dynamics

- `revenue = C * (P_f - T_f * P_f / 100)`
- `cost = 250 * S`
- `profit = revenue - cost`
- `T_c = (ship_tax / 100) * 300 + 300`
- `B = (profit * F_i / T_c) if S < S_max else 0`
- `R = S / L`
- `S_next = max(0, S + dt * (B - R))`

### Important implementation guardrails

- clamp negative profits only if the SD logic requires it
- otherwise allow negative profit and let ship growth go to zero naturally if desired
- verify whether death percentages should be divided by `100`

That last point is critical. The provided death lookup values look like percentages in SD-style tables. Before locking the simulator, verify whether `5.22` means:

- `5.22`
- or `5.22%`
- or `0.0522`

This single interpretation changes system behavior drastically.

## 8. Key Risks to Resolve Early

### Risk 1: Unit ambiguity

The equations mix values that look like percentages, rates, and absolute flows. Resolve:

- whether lookup outputs are fractions or percentages
- what one timestep represents
- whether hatch and death are annualized

### Risk 2: Bandit non-stationarity

Bandits assume reward distributions do not drift too much. In your system, they do because ecological state changes over time.

Mitigation:

- use episode-level arms first
- keep horizons fixed
- compare to control-window bandits only later

### Risk 3: Reward hacking

A learner may discover high short-term profit by depleting fish.

Mitigation:

- add collapse penalties
- report sustainability metrics separately from reward

### Risk 4: Numerical instability

If `dt` is too large, trajectories may look unrealistic.

Mitigation:

- start with `dt=1`
- test smaller `dt`
- compare plots qualitatively

## 9. Testing Plan

Minimum tests for V1:

- interpolation returns exact table values at knot points
- fish and ships never go negative
- zero ships implies zero catch
- higher reserved area lowers effective density and catch
- higher fish tax lowers revenue
- bandit action counts and update logic behave as expected

Also add one regression test:

- a fixed seed and fixed policy should produce a known summary output

## 10. Suggested Milestones

### Milestone 1

- simulator core works with one fixed baseline policy

### Milestone 2

- plots show commons collapse under aggressive exploitation

### Milestone 3

- discrete policy catalog is implemented

### Milestone 4

- epsilon-greedy and UCB compare across many episodes

### Milestone 5

- control-window formulation is tested

### Milestone 6

- environment is ready for RL experiments

## 11. Recommended Build Order for This Repo

Given the repo is nearly empty, the fastest practical order is:

1. put constants and lookup tables into code
2. implement one deterministic `step()` function
3. run and plot one baseline trajectory
4. add 5 to 7 policy bundles
5. add episode-level reward computation
6. implement epsilon-greedy and UCB first
7. batch-run experiments across seeds
8. only then add softmax, PAC, and RL scaffolding

## 12. Immediate Next Tasks

If you want the most efficient path forward, do these next:

1. Confirm unit conventions for the lookup tables and rates.
2. Build the deterministic simulator before any learning code.
3. Define 5 to 7 discrete policy arms for the bandits.
4. Choose one training reward and two evaluation-only metrics.
5. Generate baseline plots and verify the system behaves intuitively.

## 13. Recommended V1 Deliverable

The first project checkpoint should be:

- a Python simulator that runs end-to-end
- one script that compares fixed policies
- one script that compares bandit algorithms over many episodes
- saved plots showing ecology, economy, and reward outcomes

That will give you a solid base for both research-style experiments and the later RL upgrade.
