# Contextual Bandits in FishEconomy

This guide explains the contextual-bandit experiment from first principles and then follows the implementation through the repository. It is intended to be read alongside the code, especially [`experiments/compare_contextual_bandits.py`](../experiments/compare_contextual_bandits.py).

## 1. The decision problem

FishEconomy models a shared fishery in which ecological and economic dynamics interact:

- fish reproduce and die;
- ships harvest fish and generate revenue;
- profit can encourage fleet growth;
- excessive fishing can deplete the stock and cause collapse.

The learner does not select a low-level fishing action. It selects one of seven complete policy arms, such as taxation, ship-capacity control, or a marine reserve. The selected arm controls the simulator for a 10-step window.

The learning loop is:

```text
observe state
  -> construct context
  -> select policy arm
  -> simulate 10 steps
  -> observe cumulative window reward
  -> update the learner
```

The default episode has 120 simulator steps, so there are 12 contextual decisions per episode.

## 2. Why ordinary bandits are limited

An ordinary multi-armed bandit stores one estimate per arm:

```text
estimated_reward[arm]
```

Epsilon-greedy, softmax, UCB1, and successive elimination all use variations of this idea. They can learn which policy performs best on average, but they cannot distinguish between situations such as:

```text
healthy fish stock + many ships
depleted fish stock + many ships
```

The same arm may be appropriate in one situation and harmful in the other. A stateless estimate averages these cases together.

A contextual bandit instead tries to learn:

```text
expected reward = f(arm, current state)
```

That gives the learner a chance to choose conservatively when the fishery is fragile and more permissively when it is healthy.

## 3. The context used by the experiment

The context is created by [`fishery/env.py`](../fishery/env.py):

```python
[
    fish_population / carrying_capacity,
    min(ships / 10.0, 1.0),
]
```

The two values are:

1. `fish_norm`: fish population relative to the carrying capacity.
2. `ships_norm`: fishing effort normalized to a reference fleet size and capped at `1.0`.

Time is deliberately excluded. Two states that have similar fish and ship values are therefore treated as similar even if they occur at different episode positions.

This is a compact state summary, not the full simulator state. It leaves out quantities such as recent catches, profit history, and explicit time-to-horizon.

## 4. The 10-step control window

The experiment runner performs the following operations for every window:

```python
context = contextual_state_features(current_state, config)
arm_id = bandit.select_action(context)
policy = policy_lookup[arm_id]

next_state, summary, _ = rollout_window(
    current_state,
    policy,
    config,
    horizon_steps=decision_interval,
)

bandit.update(context, arm_id, summary.cumulative_reward)
```

The learner receives the cumulative reward from the entire 10-step window. It does not receive a separate supervised label for each individual simulator step.

The first episode has a shared warmup: the first pass across all seven arms uses the same seeded arm order for every contextual algorithm. This ensures that every algorithm gets initial observations for every arm before it begins choosing freely.

## 5. Discretized contextual bandits

[`DiscretizedContextualBandit`](../bandits/discretized_contextual.py) divides the continuous state into coarse regions. Fish stock uses four buckets and ships use three buckets, producing keys such as:

```text
(fish_bucket, ships_bucket)
```

For each key, the algorithm stores a count and an average reward for every arm. It then applies epsilon-greedy selection inside that bucket.

This approach can represent nonlinear behavior because every region has its own arm estimates. It is also easy to inspect. The cost is data sparsity: observations in nearby buckets are not shared, and every bucket-arm combination needs experience.

## 6. Neural state embedding

The two linear contextual algorithms pass the raw context through [`NeuralStateEncoder`](../bandits/neural_encoder.py). Its structure is:

```text
2 raw state features
        -> tanh hidden layer of size 8
        -> tanh embedding of size 6
```

The forward pass is:

```text
h = tanh(W1 x + b1)
z = tanh(W2 h + b2)
```

Here `x` is the two-feature context and `z` is the six-dimensional embedding used by the contextual bandit.

The purpose is not to create a neural policy. The policy arms remain explicit and interpretable. The encoder provides a richer representation so the arm model can respond to nonlinear combinations such as high fishing effort being especially dangerous when fish stock is low.

The resulting architecture is:

```text
raw state -> neural state embedding -> arm-specific linear reward model
```

## 7. Online encoder updates

After an arm produces a reward, the encoder is updated online. The selected arm has a current parameter vector `theta`, and its prediction is:

```text
prediction = theta · embedding
```

The prediction error is:

```text
error = prediction - observed_reward
```

The encoder uses this error to update its weights through manually implemented backpropagation and gradient descent. The implementation clips gradients and the prediction error to reduce the effect of unusually large rewards.

This creates a useful but important complication: the encoder and the arm models are learning simultaneously. The embedding changes over time, so the linear models are estimating rewards in a feature space that is also moving. That can make learning less stable than ordinary linear contextual bandits.

## 8. LinUCB

[`LinUCBBandit`](../bandits/linucb.py) maintains a separate regression model for every arm. For arm `a`, it stores `A_a` and `b_a`, then estimates:

```text
theta_a = A_a^-1 b_a
```

For an embedding `z`, the estimated reward is:

```text
exploitation = theta_a · z
```

LinUCB adds an uncertainty bonus:

```text
exploration = alpha * sqrt(z^T A_a^-1 z)
```

The selected arm maximizes:

```text
exploitation + exploration
```

After observing reward `r`, only the selected arm is updated:

```text
A_a <- A_a + z z^T
b_a <- b_a + r z
```

Repeated observations reduce uncertainty in regions of the embedding space that the algorithm has visited often.

## 9. Contextual Thompson sampling

[`ContextualThompsonBandit`](../bandits/contextual_thompson.py) maintains the same per-arm regression statistics but explores differently. It computes an estimated mean parameter vector and samples a plausible parameter vector using an approximate posterior:

```text
sampled_theta_a ~ approximate Normal(theta_a, uncertainty_a)
```

It scores each arm using the sampled vector and chooses the arm with the highest sampled score. Uncertain arms sometimes receive optimistic samples and are therefore explored without an explicit UCB bonus.

The current implementation uses only the diagonal of the covariance approximation. This is inexpensive and keeps the project dependency-light, but it ignores correlations between embedding dimensions.

## 10. Why contextual performance can still be worse

More information does not automatically produce better results. Contextual algorithms can underperform ordinary epsilon-greedy for several reasons:

- They have more parameters to estimate.
- Each 10-step window provides only one training sample.
- Contexts can be sparse, especially for discretized models.
- Exploration mistakes can cause large ecological penalties.
- The chosen arm changes the future fish and ship state.
- Removing time makes some distinct situations look identical.
- The algorithms optimize immediate window reward, not full-episode value.
- The neural embedding and linear heads are both adapting online.

The last two points are especially important. This environment is sequential: actions affect future states. A contextual bandit uses the current state, but it does not explicitly model the long-term value of changing that state. The setup is therefore a useful bridge toward reinforcement learning, but it is not yet a full RL controller.

## 11. What to measure

Episode reward is only one view of performance. The contextual outputs also make it possible to examine:

- reward per decision window;
- reward by arm;
- action-selection frequency;
- fish stock and ship count at each decision;
- collapse rate;
- time to collapse;
- final fish population;
- final fleet size;
- performance across random seeds.

The window-level CSV records `context_fish_norm` and `context_ships_norm` alongside the selected arm and resulting reward. This allows questions such as:

```text
Which policies were selected when fish stock was low?
Did the algorithm become more conservative as ship pressure increased?
Did high reward come at the cost of later collapse?
```

## 12. Contextual bandits versus reinforcement learning

The current system is best described as windowed contextual control:

```text
state -> arm -> short rollout -> reward
```

A reinforcement-learning formulation would additionally learn how actions affect future value:

```text
state -> action -> next state -> long-term value
```

The next evolution could include discounted returns, value functions, model predictive control, or constrained reinforcement learning. Those approaches would be better suited to explicitly trading immediate profit against future stock health.

## 13. Running the experiment

```bash
python3 simulator.py compare-contextual-bandits \
  --episodes 200 \
  --seeds 20 \
  --decision-interval 10 \
  --horizon-steps 120 \
  --output-dir outputs/contextual_bandits
```

For a quick smoke run:

```bash
python3 simulator.py compare-contextual-bandits \
  --episodes 2 \
  --seeds 1 \
  --decision-interval 10 \
  --horizon-steps 120 \
  --output-dir outputs/contextual_bandits_smoke
```

The main implementation files are:

- [`fishery/env.py`](../fishery/env.py): state features and rollouts;
- [`experiments/compare_contextual_bandits.py`](../experiments/compare_contextual_bandits.py): experiment loop;
- [`bandits/discretized_contextual.py`](../bandits/discretized_contextual.py): bucketed learner;
- [`bandits/linucb.py`](../bandits/linucb.py): UCB learner;
- [`bandits/contextual_thompson.py`](../bandits/contextual_thompson.py): Thompson learner;
- [`bandits/neural_encoder.py`](../bandits/neural_encoder.py): online state representation.

