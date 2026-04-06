# SUMO Split-RL Scenarios — Complete Reference

**Date:** 2026-04-06 (v3)  
**File:** `env/sumo.py`  
**Networks:** `sumo-arterial4x4-v0` (18-dim obs, Discrete(5), 6 lanes/agent)  
              `sumo-grid4x4-v0` (33-dim obs, Discrete(8), 12 lanes/agent)

---

## 1. Architecture

```
sumo_rl.parallel_env(RESCO benchmark)
→ SumoRewardCostWrapper     (reward/cost/label split + vehicle injection + obs extension)
→ SuperSuit pettingzoo_env_to_vec_env + concat_vec_envs
→ VecCostMonitor             (episode stats: r, c, s)
→ SB3 VecEnv                 (16 agent slots per sim)
```

- 16 shared-policy agents per simulation
- 3600s simulation, delta_time=5 → 720 steps/agent/episode → 11,520 transitions/episode
- 50% clean episodes (no events) via `clean_episode_prob=0.5`

---

## 2. Shared: Reward

**All 4 scenarios use the same reward: original SUMO-RL `diff-waiting-time`.**

```
r_t = -(total_wait_t - total_wait_{t-1})   [sum over ALL incoming lanes, ALL vehicles]
```

Every vehicle counts equally: 1 bus = 1 car = 1 premium car in the reward.
The reward is NOT hacked. `override_reward=null`.

---

## 3. The Four Scenarios

### 3.1 — Bus Priority (Volume-Driven Throughput Conflict)

**Story:** Heavy bus traffic. Buses are large (12m), slow (11.1 m/s), slow to accelerate. Serving a bus approach costs more green-seconds per waiting-time-reduction than serving cars.

**Injection:** 2 buses every ~12s (±20% jitter), random boundary→boundary routes. Params: `bus_injection_interval=12.0`, `bus_count_per_injection=2`.

**Obs extension** (3 × n_lanes = 18 for arterial, 36 for grid):

| Feature | Dim | Range | Description |
|---------|-----|-------|-------------|
| `has_bus[lane]` | n_lanes | {0, 1} | Is there a bus on this lane? |
| `bus_count[lane]` | n_lanes | [0, 1] | Number of buses on this lane (÷5) |
| `bus_wait[lane]` | n_lanes | [0, 1] | Max bus waiting time (÷T_bus=30s) |

**Cost:**
```
c_t = max over unserved lanes of: bus_wait[lane] / T_bus
```
Continuous ∈ [0, 1]. Ramps as bus waits longer on a red lane. = 0 when bus is on a served (green) lane.

**Label** (conflict-based):
```
label = 1  IF  bus on unserved lane
          AND  served approach has HIGHER queue pressure than bus lane
```

The label fires at the **actual conflict zone**: reward says "stay on the busy approach" (more waiting-time reduction per green-second), cost says "switch to serve the bus." If the bus lane is already the busiest, label = 0 — no conflict, both objectives agree.

**Conflict mechanism:** Buses are throughput-inefficient. A lane with 3 buses clears slower than a lane with 10 cars. The reward prefers the efficient approach. The cost demands the buses get served.

| Default param | Value |
|---------------|-------|
| `bus_injection_interval` | 12.0s |
| `bus_count_per_injection` | 2 |
| `bus_cost_threshold` (T_bus) | 30.0s |
| `bus_warn_threshold` (T_warn) | 10.0s |

---

### 3.2 — Convoy Must-Not-Split (Temporal Conflict)

**Story:** A platoon of 5-8 vehicles approaches an intersection together. The entire group must pass without interruption. Splitting the platoon incurs cost.

**Injection:** 1 platoon every ~120s (±20% jitter), 5-8 vehicles spaced 1.5s apart on the same random route. Convoy vehicles are normal-sized cars (5m, 13.9 m/s). Params: `convoy_injection_interval=120.0`, `convoy_size_min=5`, `convoy_size_max=8`.

**Obs extension** (3 × n_lanes = 18 for arterial, 36 for grid):

| Feature | Dim | Range | Description |
|---------|-----|-------|-------------|
| `has_convoy[lane]` | n_lanes | {0, 1} | Any convoy vehicle on this lane? |
| `convoy_count[lane]` | n_lanes | [0, 1] | Num convoy vehicles on lane (÷max_platoon_size) |
| `convoy_progress[lane]` | n_lanes | [0, 1] | Fraction of platoon that has already passed this intersection |

**Cost** (split-detection):
```
IF convoy on unserved lane AND convoy_progress > 0:
    cost = convoy_progress          (active split — some passed, rest stuck)
ELIF convoy on unserved lane AND convoy_count > 0:
    cost = 0.3 × convoy_count      (pre-split — waiting but not yet splitting)
ELSE:
    cost = 0
```
Continuous ∈ [0, 1]. Highest when the platoon is actively being split.

**Label** (active split ONLY):
```
label = 1  IF  convoy on unserved lane
          AND  0 < convoy_progress < 1
```

The label fires ONLY during the **active split window**: some convoy vehicles have passed the stop line (their waiting-time contribution dropped to 0), so the reward says "switch to the busier approach." But the tail is still on the approach — cost says "hold the phase." Pre-split states (convoy merely waiting on an unserved lane) do NOT trigger the label because the objectives don't yet disagree.

**Conflict mechanism:** Temporal. After lead vehicles pass, the reward sees fewer vehicles on this approach and is eager to switch. But the tail hasn't cleared yet. The reward-optimal phase duration is shorter than what platoon integrity requires. A global Lagrange multiplier λ can't handle this because λ needs to be high only during the ~5-10s split-risk window.

| Default param | Value |
|---------------|-------|
| `convoy_injection_interval` | 120.0s |
| `convoy_size_min` | 5 |
| `convoy_size_max` | 8 |

---

### 3.3 — Spillback / Road Works (Spatial Conflict)

**Story:** Random road works reduce downstream capacity. Serving an approach that feeds into a blocked downstream causes spillback — vehicles pile up and gridlock propagates.

**Injection:** Road works on random internal edges. Every ~300s (±40% jitter), an edge speed is reduced to 0.3 m/s for ~400s (±40%). ~70% chance of additional road works per cycle. Params: `roadwork_interval_mean=300.0`, `roadwork_duration_mean=400.0`, `roadwork_speed=0.3`.

**Obs extension** (1 × n_lanes = 6 for arterial, 12 for grid):

| Feature | Dim | Range | Description |
|---------|-----|-------|-------------|
| `downstream_occ[lane]` | n_lanes | [0, 1] | Max occupancy of downstream edges fed by this lane |

**Cost:**
```
c_t = max over currently SERVED lanes of: downstream_occ[lane]
```
Continuous ∈ [0, 1]. High when the agent is actively sending vehicles into a blocked downstream. = 0 if downstream is clear.

**Label** (direct contradiction):
```
label = 1  IF  max(downstream_occ on currently served lanes) > T_occ (0.5)
```

This is the **tightest label** of all four. It fires exactly when the current phase IS the wrong action for the constraint: the agent is clearing a local queue (reward says good) into a blocked downstream (cost says bad). The label marks the contradiction itself.

**Conflict mechanism:** Spatial. Reward only sees local queues. The biggest local queue gives the most waiting-time reduction. But that queue's downstream is blocked by road works. Cost-optimal: serve a different approach (less local demand, clear downstream). This genuinely hurts reward.

| Default param | Value |
|---------------|-------|
| `roadwork_interval_mean` | 300.0s |
| `roadwork_duration_mean` | 400.0s |
| `roadwork_speed` | 0.3 m/s |
| `spillback_occ_threshold` (T_occ) | 0.5 |

---

### 3.4 — Premium Priority (Value-Asymmetry Conflict) [NEW]

**Story:** Some vehicles are "premium" — they've paid for priority. They look identical to regular cars (same size, speed, acceleration) but have a contractual priority. The cost function treats them as high-priority; the reward doesn't know the difference.

**Injection:** 1 premium vehicle every ~40s (±20% jitter), random boundary→boundary route. Premium vehicle: 5m, 13.9 m/s, gold color tag. Params: `premium_injection_interval=40.0`.

**Obs extension** (2 × n_lanes = 12 for arterial, 24 for grid):

| Feature | Dim | Range | Description |
|---------|-----|-------|-------------|
| `has_premium[lane]` | n_lanes | {0, 1} | Is there a premium vehicle on this lane? |
| `premium_wait[lane]` | n_lanes | [0, 1] | Premium vehicle's waiting time (÷T_premium=15s) |

**Cost** (steep quadratic ramp):
```
c_t = max over unserved lanes of: (premium_wait[lane])²
```
Squared → rapid escalation. At 5s wait (normalized 0.33): cost = 0.11. At 10s (0.67): cost = 0.44. At 15s (1.0): cost = 1.0. Creates urgency.

**Label** (conflict-based):
```
label = 1  IF  premium on unserved lane
          AND  served approach has HIGHER queue pressure than premium lane
```

Same logic as bus: fires when the reward wants to KEEP serving the crowded approach, but cost demands switching to the nearly-empty premium lane. If premium's lane is already the busiest, label = 0.

**Conflict mechanism:** Pure value asymmetry. The premium car is 1 vehicle in the reward sum — utterly dominated by 10+ regulars on the main approach. But the cost treats it as disproportionately important with a steep penalty. There's no throughput trick, no speed difference — just the cost assigning outsized value to one car that the reward treats equally.

| Default param | Value |
|---------------|-------|
| `premium_injection_interval` | 40.0s |
| `premium_cost_threshold` (T_premium) | 15.0s |
| `premium_warn_threshold` (T_warn) | 5.0s |

---

## 4. Label Philosophy

The label is NOT a cost detector. It is:

> **"In this state, should the cost objective own the gradient, or should reward own it?"**

A good label has:
- **High precision** — it fires only where the reward-optimal and cost-optimal actions disagree
- **Obs-predictability** — it's a deterministic function of observable features
- **The right timing** — not too early (no conflict yet), not too late (reward already agrees)

| Scenario | Label condition | What it captures |
|----------|----------------|-----------------|
| Bus | Bus on unserved lane AND served approach busier | Reward wants to stay, cost wants to switch |
| Convoy | 0 < convoy_progress < 1 on unserved lane | Active split window — reward wants to switch, cost wants to hold |
| Spillback | Downstream blocked on currently served lane | Agent is actively feeding a blocked road |
| Premium | Premium on unserved lane AND served approach busier | 1 premium car vs 10+ regulars — reward dominated |

### Label quality ranking:
1. **S3 Spillback** — fires at the exact contradiction (current action is wrong for cost)
2. **S2 Convoy** — fires during the active split window (tight temporal conflict)
3. **S4 Premium** — fires at value asymmetry (crowd vs individual)
4. **S1 Bus** — fires at throughput asymmetry (weakest because cost is still wait-based)

---

## 5. What Metric to Use for Conflict Validation

**DO NOT** rely on label rate, DT accuracy, or trajectory-level correlation. Those measure observability, not conflict.

**The real test:**
1. Train reward-only PPO to convergence (π_R)
2. Collect rollouts, tag label=1 states
3. In label=1 states, measure:
   - What action does π_R pick? → should incur HIGH cost
   - What action minimizes cost? → should incur LOWER reward than π_R's action
4. Report: `reward_gap = r(π_R) - r(π_C)` in label=1 states
   - If reward_gap ≈ 0 → NO CONFLICT
   - If reward_gap >> 0 → REAL CONFLICT, Split-RL has room to win
5. Does PPO-Lag's λ oscillate or converge? If λ settles → conflict is too uniform.

---

## 6. Sweep Configuration

**7 algorithms × 4 scenarios × 2 networks × 5 seeds = 280 runs**

| Algorithm | Sweep file |
|-----------|-----------|
| SPLIT-RL (GBRL) | `seeds_split_rl_sumo.yaml` |
| PPO-GBRL | `seeds_ppo_gbrl_sumo.yaml` |
| PPO-NN | `seeds_ppo_nn_sumo.yaml` |
| PPO-Lagrangian | `seeds_ppo_lag_sumo.yaml` |
| IPO | `seeds_ipo_sumo.yaml` |
| CPO | `seeds_cpo_sumo.yaml` |
| CUP | `seeds_cup_sumo.yaml` |

All sweeps in: `sweeps/split_rl/sumo/tests/`

All use: `total_n_steps=1M`, `num_envs=1`, `device=cuda`, `norm_obs=false`, `norm_reward=true`, seeds `[0, 5, 10, 42, 64]`.

---

## 7. History of Failed Approaches

| Approach | Why it failed |
|----------|--------------|
| `side_deficit` cost + mainline reward | Cost used hidden state (`steps_since_served`). DT couldn't predict label. |
| `side_queue` cost + mainline reward | Reward was engineered to conflict (mainline-only). Not a natural scenario. |
| Bus wait cost (v1, interval=25s) | Too few buses. Bus wait aligned with diff-waiting-time. No real conflict. |
| Convoy wait cost (v1) | Convoy self-prioritizes via count. 5-8 vehicles = 5-8x wait contribution. Even MORE aligned. |
| Label = "cost is positive" | That's an event detector, not a routing signal. Sends wrong objective to too many samples. |
| Label = "special vehicle waiting > T" | Late and fuzzy. By the time wait is high, reward starts caring too. |
| Conflict diagnostic: label rate + DT accuracy + Pearson | Measures observability, not action-level conflict. Passed checks that had no real conflict. |

---

## 8. TODO

- [ ] Run conflict validation diagnostic (reward_gap in label=1 states) for all 4 scenarios
- [ ] Tune bus injection rate if S1 conflict is still too weak
- [ ] Verify convoy_progress tracking works correctly with random multi-hop routes
- [ ] Run full 280-run sweep and analyze results
- [ ] Consider making road works more frequent/shorter for S3 if label rate is too low
