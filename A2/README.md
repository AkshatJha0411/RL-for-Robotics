# Assignment 2: Drone Hovering using Model-Free Control

**Task**: Train a simulated quadrotor (Crazyflie) to hover at target position `[0, 0, 1]` using tabular reinforcement learning — no neural networks, no model of the environment.

**Environment**: `HoverAviary` from `gym-pybullet-drones`, rendered headless (`gui=False`) for training.

**Final Score: 78.8 / 85 core + 20 / 20 bonus**

---

## Setup

```bash
# Install gym-pybullet-drones (required)
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones && pip install -e . && cd ..

pip install -r requirements.txt
```

---

## State & Action Representation

The raw HoverAviary observation is a 2D array of shape `(1, 12)`. I extract the first three elements — `x, y, z` — as the position relative to the environment origin (target is `[0, 0, 1]`).

**Discretization**: Each axis is mapped to 10 bins using fixed bounds:
- x, y: `[-1, 1]`
- z: `[0, 2]`

This gives a Q-table of shape `(10, 10, 10, 3)` — 3000 state-action entries.

**Action space**: 3 discrete actions mapped to ONE_D_RPM thrust adjustments:
- `0` → `-1.0` (reduce thrust, descend)
- `1` → `0.0` (maintain)
- `2` → `+1.0` (increase thrust, climb)

`format_action(action)` wraps the scalar into `np.array([[value]], dtype=float32)` as required by the env.

---

## Core Algorithms (`user_code.py`)

### Hyperparameters Used

```python
NUM_BINS   = 10
EPSILON    = 0.075
GAMMA      = 0.99
ALPHA      = 0.1
NUM_EPISODES = 500
MAX_STEPS  = 240
```

---

### Monte Carlo Control (`run_monte_carlo`)

**Approach**: First-visit Monte Carlo with epsilon-greedy exploration and incremental Q-value updates.

For each episode I roll out a full trajectory using the current epsilon-greedy policy, collecting `(state, action, reward)` tuples. After the episode ends, I walk backwards through the trajectory computing the discounted return G at each step:

```
G_t = r_t + γ · G_{t+1}
```

On the **first visit** to each `(state, action)` pair, I update the Q-value using a running average via a fixed learning rate:

```
Q(s, a) ← Q(s, a) + α · (G - Q(s, a))
```

I track visited pairs in a `set()` per episode to enforce first-visit semantics. Epsilon is fixed (no decay) — exploration stays constant throughout training which keeps the policy slightly stochastic.

**Result**: Evaluation reward **449.71**, Final 50-ep avg **297.43**, Convergence at episode **259**. Score: **30/30**.

---

### Q-Learning (`run_q_learning`)

**Approach**: Standard off-policy TD control (Q-Learning) — updates happen online after every step, not at episode end.

At each step the update rule is:

```
Q(s, a) ← Q(s, a) + α · [r + γ · max_{a'} Q(s', a') - Q(s, a)]
```

The "max over next actions" makes this off-policy — the agent always bootstraps from the best possible next action regardless of what epsilon-greedy would actually pick. This tends to overestimate values slightly (maximization bias) but converges faster than SARSA in practice.

Epsilon is fixed at 0.075 — slightly lower than the default 0.1 to lean towards exploitation once the policy starts forming.

**Result**: Evaluation reward **449.09**, Final 50-ep avg **324.54**, Convergence at episode **361**. Score: **30/30**.

---

## Bonus Challenges (`bonus_challenges.py`)

All three bonus challenges scored full marks. A key change across all three: I increased `alpha` to `0.15` and used aggressive **epsilon decay** starting from `0.5`:

```
ε_t = 0.5 × (0.98^t)
```

This starts with heavy exploration and converges to near-zero by episode ~250, letting the policy fully exploit the learned Q-values in the second half of training.

---

### Challenge 1 — SARSA (`run_sarsa`) · **5 / 5 pts**

**On-policy TD control**: unlike Q-Learning, SARSA uses the action *actually taken* in the next state (sampled from the same epsilon-greedy policy) rather than the greedy action.

The update:
```
Q(s, a) ← Q(s, a) + α · [r + γ · Q(s', a') - Q(s, a)]
```

where `a'` is chosen by `choose_action(q_table, next_state, current_epsilon)` — not just `argmax`.

This makes SARSA more conservative: near the boundary of the state space it won't blindly bootstrap from an action it would never take. I choose the initial action before the inner loop and carry it forward, keeping `(s, a, r, s', a')` properly lined up.

**Result**: Evaluation reward **374.64**. Score: **5/5**.

---

### Challenge 2 — Double Q-Learning (`run_double_q_learning`) · **7 / 7 pts**

**Problem with standard Q-Learning**: using `max Q(s', ·)` for both action *selection* and *evaluation* on the same table introduces an upward bias (maximization bias). Over many updates this can cause over-optimistic value estimates.

**Fix**: maintain two independent Q-tables, `Q1` and `Q2`. At each step, with 50% probability:
- Use `argmax Q1(s')` to *select* the best action, evaluate with `Q2(s', best_action)` → update Q1
- Otherwise use `argmax Q2(s')` and evaluate with `Q1` → update Q2

Action selection for epsilon-greedy uses the *sum* `Q1 + Q2`, ensuring both tables contribute equally to behavior.

The final policy uses `(Q1 + Q2) / 2` for evaluation, which was the highest-performing algorithm overall.

**Result**: Evaluation reward **445.08**. Score: **7/7**.

---

### Challenge 3 — Experience Replay (`run_td_with_replay`) · **8 / 8 pts**

**Motivation**: Online TD updates use highly correlated consecutive transitions. A replay buffer stores past experiences and samples random mini-batches, breaking temporal correlation and letting each transition contribute to multiple updates.

**Implementation**:
- `ReplayBuffer` stores `(state, action, reward, next_state, done)` tuples up to a capacity of 10,000.
- Each step pushes a new transition in and evicts the oldest if at capacity.
- Once the buffer has ≥ 32 samples, I sample a random mini-batch and apply Q-Learning updates to each:

```
target = r                          if done
target = r + γ · max Q(s', ·)      otherwise
Q(s, a) ← Q(s, a) + α · (target - Q(s, a))
```

The key fix from the original stub: raw observations must go through `extract_position()` before `discretize_state()` (the HoverAviary obs is a 2D array, not a flat vector), and `env.step()` must receive `format_action(action)` — not a bare list.

**Result**: Evaluation reward **361.08** (≥ 300 threshold). Score: **8/8**.

---

## Results Summary

| Algorithm | Eval Reward | Score |
|---|---|---|
| Monte Carlo | 449.71 | 30 / 30 |
| Q-Learning | 449.09 | 30 / 30 |
| Experiments | — | 18.8 / 25 |
| **Core Total** | | **78.8 / 85** |
| SARSA (bonus) | 374.64 | 5 / 5 |
| Double Q-Learning (bonus) | 445.08 | 7 / 7 |
| Experience Replay (bonus) | 361.08 | 8 / 8 |
| **Bonus Total** | | **20 / 20** |

---

## Running

```bash
# Core assignment
python user_code.py

# Evaluate with grader
python evaluate_submission.py --student_file user_code.py --method all --seed 42 --min_reward 220 --eval_seeds 3

# Bonus challenges
python bonus_challenges.py
```

---

## References

- Sutton & Barto — *Reinforcement Learning: An Introduction* (2nd ed.)
- [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)
- Hasselt et al. — *Double Q-Learning* (NIPS 2010)
- Mnih et al. — *Human-level control through deep reinforcement learning* (Nature 2015) — experience replay section
