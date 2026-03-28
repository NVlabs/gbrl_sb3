# SPLIT-RL Environment Plan: Flatland & SUMO-RL

## 1. Overview

Two multi-agent environments for constrained RL with SPLIT-RL (and baselines):

| | **Flatland** | **SUMO-RL** |
|---|---|---|
| Domain | Train scheduling on rail networks | Traffic signal control |
| Agents | Trains (3–20) | Traffic signals (1–21) |
| Obs | Flattened tree obs (252–1020 floats) | [phase_one_hot, min_green, densities, queues] (33 floats for grid4x4) |
| Actions | Discrete(5): nothing/left/fwd/right/stop | Discrete(n_phases): select next green phase |
| Reward | diff-waiting-time-like (per-agent penalty terms) | diff-waiting-time (per-agent) |
| Cost | Collision penalty (per-agent, from reward decomposition) | Native SUMO failure events: emergency stops + teleports (sim-wide, via TraCI) |
| Safety story | Avoid train collisions while maximizing on-time arrivals | Minimize hard failures (gridlock/deadlock) while maximizing throughput |
| Shared policy | Yes (all agents identical obs/act spaces) | Yes (uniform intersection topology required — RESCO grids) |

Both environments use the same wrapper pipeline:
```
Raw multi-agent env
→ PettingZoo ParallelEnv wrapper (reward/cost split)
→ SuperSuit pettingzoo_env_to_vec_env_v1 + concat_vec_envs_v1
→ VecCostMonitor (tracks ep_cost, ep_rew, ep_len)
→ SB3 VecEnv
```

---

## 2. Flatland: Implementation Details

### 2.1 Raw Info Extracted

From `FlatlandRewardCostWrapper` (file: `env/flatland.py`):

| info key | type | source | description |
|---|---|---|---|
| `cost` | float ≥ 0 | `max(0, -reward_dict[COLLISION])` | Per-agent collision penalty. Flatland's `BaseDefaultRewards` with `collision_factor=1.0` returns a dict of penalty terms; we extract the collision term as cost. |
| `original_reward` | float | `sum(reward_dict.values())` | Full scalar reward including collision. |
| `reward_terms` | dict | Raw `BaseDefaultRewards` dict | All penalty breakdown: `COLLISION`, `GLOBAL_REWARD`, `CANCELLATION`, `TARGET_NOT_REACHED`, `DEADLOCK`, `STOP_PENALTY`. |
| `safety_label` | int {0,1} | `compute_danger_label(obs)` | Observation-derived binary label (see §2.2). |
| `episode_cost` | float | Cumulative sum of `cost` | Emitted at episode end per agent. |
| `episode_original_reward` | float | Cumulative sum of `original_reward` | Emitted at episode end per agent. |
| `episode_reward_terms` | dict | Cumulative per-term | Emitted at episode end per agent. |

The reward fed to the policy = `original_reward - cost` (non-collision terms only). The full original reward is tracked for monitoring.

### 2.2 Label Generation — Flatland

**Primary label (observation-derived):** `compute_danger_label(obs, n_nodes)` in `env/flatland.py`.

Inputs: the flattened normalized tree observation (252-d for depth=2).

Logic — returns 1 (danger) if ANY of:
1. **Potential conflict detected** on any non-root branch within `conflict_dist_thresh=0.5` normalized distance. Feature: `data[:, 3]` (dist_potential_conflict).
2. **Other agent encountered** on any non-root branch within `agent_dist_thresh=0.5`. Feature: `data[:, 2]` (dist_other_agent_encountered).
3. **Opposite-direction traffic** on a non-root branch that also has a nearby conflict or nearby other-agent cue. Feature: `agent_data[:, 1]` gated by the same-branch distance features above.
4. **Malfunctioning agent** on a non-root branch that also has a nearby conflict or nearby other-agent cue. Feature: `agent_data[:, 2]` gated by the same-branch distance features above.

This label is per-agent and observation-derived, meaning it fires *before* a collision occurs (predictive, not reactive).

**Alternative label (cost-advantage):** When `--use_cost_advantage_label` is set, SPLIT-RL overrides `safety_labels` post-rollout:
```
label_t = 1  if  cost_advantage_t > 0
```
where `cost_advantage_t = GAE(costs) = ∑ γ^k δ_cost` — meaning the action at time t led to higher-than-expected future collision cost. This is computed from the learned cost value function, not hand-coded rules.

### 2.3 Flatland Configurations

| Config | Grid | Agents | Cities | Obs dim | Max steps |
|---|---|---|---|---|---|
| `flatland-small-v0` | 25×25 | 3 | 2 | 252 | varies (~100–224) |
| `flatland-medium-v0` | 35×35 | 5 | 3 | 252 | varies |
| `flatland-large-v0` | 50×50 | 10 | 4 | 1020 | varies |
| `flatland-xlarge-v0` | 80×80 | 20 | 6 | 1020 | varies |

---

## 3. SUMO-RL: Implementation Details

### 3.1 Raw Info Extracted

From `SumoRewardCostWrapper` (file: `env/sumo.py`):

| info key | type | source | description |
|---|---|---|---|
| `cost` | float ∈ {0, 1, 2} | `_compute_cost(events)` | Binary indicator cost from native SUMO events: `1[emergency_stops > 0] + teleport_weight × 1[teleports > 0]`. Shared across all agents (simulation-wide signal). |
| `original_reward` | float | same as reward | diff-waiting-time (no decomposition needed — reward is not split). |
| `safety_label` | int {0,1} | `_event_proximal_label()` | Fallback event-proximal label: 1 if any failure event occurred in last `label_horizon=3` steps. |
| `emergency_stops` | int ≥ 0 | TraCI `getEmergencyStoppingVehiclesNumber()` | Count of vehicles that performed an emergency stop this step (accumulated across all `delta_time` sub-steps). |
| `teleports` | int ≥ 0 | TraCI `getStartingTeleportNumber()` | Count of vehicles that started teleporting (deadlock resolution) this step. |
| `episode_cost` | float | Cumulative sum of `cost` | Emitted at episode end per agent. |
| `episode_original_reward` | float | Cumulative sum of reward | Emitted at episode end per agent. |

### 3.2 Why Native SUMO Events (Not Queue Thresholds)

Queue-based cost (e.g., "cost = 1 if queue > 0.7") is **redundant with the observation** — the agent already sees lane queues and densities. The cost head would just learn to threshold the same features the policy sees.

Native SUMO failure events are different:
- **Emergency stops**: vehicles braking beyond `decel > wished_decel`. This means traffic signal timing created a conflict the vehicle couldn't handle at normal deceleration. Not visible in the observation.
- **Teleports**: SUMO teleports vehicles stuck in deadlock for too long (default: 300s). This means the policy created an unresolvable gridlock. Strongly non-redundant with current-timestep queues.

These are *consequences* of bad policy decisions, not restatements of the observation.

### 3.3 Sub-Step Accumulation

SUMO-RL uses `delta_time=5` (5 seconds of sim time per agent step). Internally, `SumoEnvironment._sumo_step()` advances 5 1-second sub-steps. TraCI counters like `getEmergencyStoppingVehiclesNumber()` only report the last sub-step.

To avoid missing events, we monkey-patch `_sumo_step` to accumulate counts across all sub-steps:
```python
def patched_sumo_step(self_env):
    original_step()
    wrapper._accumulated_emergency_stops += conn.simulation.getEmergencyStoppingVehiclesNumber()
    wrapper._accumulated_teleports += conn.simulation.getStartingTeleportNumber()
```
The accumulated counts are returned and reset on each `_query_failure_events()` call.

### 3.4 Label Generation — SUMO

**Fallback label (event-proximal):** `_event_proximal_label()` — returns 1 if any failure event (emergency stop or teleport) occurred within the last `label_horizon=3` agent steps. This is a simple temporal-proximity heuristic: "something bad happened recently, so nearby states are dangerous."

**Preferred label (cost-advantage):** Same as Flatland — when `--use_cost_advantage_label` is set:
```
label_t = 1  if  cost_advantage_t > 0
```
The cost value function learns to predict future failure events; the advantage tells us whether the current action made things worse than expected. This is the honest constrained-optimization label.

### 3.5 SUMO Configurations (RESCO Benchmarks)

| Config | Type | Agents | Obs dim | Actions | Duration |
|---|---|---|---|---|---|
| `sumo-grid4x4-v0` | Synthetic 4×4 grid | 16 | 33 | Discrete(8) | 3600s |
| `sumo-arterial4x4-v0` | Synthetic arterial | 16 | 33 | Discrete(8) | 3600s |
| `sumo-cologne1-v0` | Real (Cologne) | 1 | varies | varies | 3600s |
| `sumo-cologne3-v0` | Real (Cologne) | 3 | varies | varies | 3600s |
| `sumo-cologne8-v0` | Real (Cologne) | 8 | varies | varies | 3600s |
| `sumo-ingolstadt1-v0` | Real (Ingolstadt) | 1 | varies | varies | 3600s |
| `sumo-ingolstadt7-v0` | Real (Ingolstadt) | 7 | varies | varies | 3600s |
| `sumo-ingolstadt21-v0` | Real (Ingolstadt) | 21 | varies | varies | 3600s |

**Note:** Shared-policy training requires uniform obs/action spaces across all agents. The synthetic grids (`grid4x4`, `arterial4x4`) satisfy this. The real-world networks may have non-uniform intersection topologies — use with single-agent configs or verify uniformity first.

### 3.6 Cost Signal Sparsity

On grid4x4 with a random policy, failure events are rare (~1 emergency braking per 3600s episode, 0 teleports). For a denser safety signal:
- Use more congested scenarios: `cologne8`, `ingolstadt21`.
- Reduce SUMO's teleport timeout: pass `add_system_info=True` and configure `--time-to-teleport 60` via SUMO config to trigger teleports faster.
- Use raw event counts instead of binary indicators: `cost = emergency_stops + teleport_weight * teleports` (replace `1[x > 0]` with `x`).

---

## 4. SPLIT-RL Architecture (Common to Both Envs)

Single GBRL ensemble with shared tree structure (`shared_tree_struct=True`):
```
output = [policy_logits(action_dim), value(1), cost_value(1)]
```
- Policy head: outputs logits → softmax → Categorical distribution.
- Value head: predicts V(s) for reward GAE.
- Cost head: predicts V_cost(s) for cost GAE.

Each epoch adds trees. The safety label controls the SPLIT mechanism: when `label=1`, the safety gradient modifies the policy update to steer away from high-cost actions.

### 4.1 The `info` Contract

Every env wrapper must provide these keys in `info` for compatibility with SPLIT-RL and the safety baselines:

| Key | Required by | Description |
|---|---|---|
| `cost` | SPLIT-RL, CPO, PPOLag, CUP, IPO | Per-step cost signal (float ≥ 0) |
| `safety_label` | SPLIT-RL | Binary label for SPLIT safety head |
| `original_reward` | VecCostMonitor logging | Unmodified reward for monitoring |

Both Flatland and SUMO wrappers implement this contract.

---

## 5. Baseline Comparison Plan

### 5.1 Algorithms

| Algo | Type | Cost-aware? | How it uses cost |
|---|---|---|---|
| **SPLIT-RL** | GBRL (tree-based PPO + safety head) | Yes | Cost head predicts V_cost; safety label steers policy via SPLIT gradient |
| **PPO-GBRL** | GBRL (tree-based PPO) | No | Ignores cost entirely — reward-only optimization |
| **CPO** | NN (constrained policy optimization) | Yes | Cost advantage in trust region constraint; hard cost limit via `cost_limit` |
| **PPOLag** | NN (Lagrangian PPO) | Yes | Lagrangian multiplier on mean episode cost; adaptive penalty coefficient |
| **CUP** | NN (constrained update projection) | Yes | Projects policy update onto cost-feasible set |
| **IPO** | NN (interior point optimization) | Yes | Interior-point method for cost constraint |
| **PPO-NN** | NN (vanilla PPO) | No | Ignores cost — reward-only baseline |

### 5.2 Metrics for Comparison

| Metric | Source | What it measures |
|---|---|---|
| `ep_rew_mean` | VecCostMonitor | Average episode reward (throughput/task performance) |
| `ep_cost_mean` | VecCostMonitor | Average episode cumulative cost (safety violations) |
| `ep_len_mean` | VecCostMonitor | Average episode length |
| Reward-cost Pareto frontier | Plot reward vs. cost across algos | Is SPLIT-RL Pareto-dominant? |
| Cost constraint satisfaction | `ep_cost_mean ≤ cost_limit` | Does the algo meet the safety budget? |
| Sample efficiency | reward vs. `total_timesteps` | How fast does each algo learn? |
| `explained_variance` | Logger | Value function quality |
| `approx_kl` / `clip_fraction` | Logger | Policy update magnitude (sanity) |

**Flatland-specific:**
| Metric | Source | What it measures |
|---|---|---|
| Completion rate | `episode_reward_terms[GLOBAL_REWARD]` | Fraction of agents reaching target |
| Collision count | `episode_cost` | Total collisions per episode |
| Reward breakdown | `episode_reward_terms` | Per-penalty-type contribution |

**SUMO-specific:**
| Metric | Source | What it measures |
|---|---|---|
| Total emergency stops | `emergency_stops` (accumulated) | Hard braking events per episode |
| Total teleports | `teleports` (accumulated) | Deadlock resolutions per episode |
| Waiting time | `ep_rew_mean` (neg of diff-waiting-time) | Overall throughput quality |

### 5.3 Experimental Protocol

For each env × algo combination:
1. Run 3 seeds (seed 0, 1, 3) for statistical significance.
2. Train for the same total_timesteps budget per environment.
3. Log locally (raw text log file).
4. Record: `ep_rew_mean`, `ep_cost_mean`, `ep_len_mean` every iteration.
5. For constrained algos (CPO, PPOLag, CUP, IPO), set `cost_limit` to the same target budget.
6. For SPLIT-RL, test both label modes: observation-derived (default) and `--use_cost_advantage_label`.

### 5.4 Key Comparisons

**Q1: Does SPLIT-RL reduce cost without sacrificing reward?**
- Compare SPLIT-RL vs. PPO-GBRL (same architecture, with/without safety head).
- Plot reward vs. cost Pareto curve.

**Q2: Does SPLIT-RL compete with NN constrained baselines?**
- Compare SPLIT-RL vs. CPO, PPOLag, CUP, IPO.
- Same env, same cost budget, same timesteps.
- SPLIT-RL uses trees; baselines use NNs. Is the tree-based approach competitive?

**Q3: Does cost-advantage labeling outperform observation-derived labels?**
- Compare SPLIT-RL (default label) vs. SPLIT-RL (`--use_cost_advantage_label`).
- Hypothesis: cost-advantage labels adapt to the learned cost model and should converge to better cost/reward tradeoffs.

**Q4: How does cost signal design affect learning?**
- Flatland: cost = collision penalty (clear, per-agent, always present).
- SUMO: cost = failure events (sparse, sim-wide, requires congested scenarios).
- Compare the label firing rate and cost head learning curves across both envs.

---

## 6. Running Experiments

### 6.1 SPLIT-RL (obs-derived label, default)
```bash
NO_RESUME=1 python3.10 scripts/train_runner.py \
  --env_type flatland --algo_type split_rl --env_name flatland-small-v0 \
  --seed 0 --num_envs 8 --total_n_steps 5000000 --device cuda \
  --ent_coef 0.0 --gae_lambda 0.95 --n_epochs 10 --batch_size 512 \
  --clip_range 0.2 --value_lr 0.05 --cost_lr 0.05 --policy_lr 0.1 \
  --save_every 0 --log_interval 1
```

### 6.2 SPLIT-RL (cost-advantage label)
```bash
NO_RESUME=1 python3.10 scripts/train_runner.py \
  --env_type flatland --algo_type split_rl --env_name flatland-small-v0 \
  --seed 0 --num_envs 8 --total_n_steps 5000000 --device cuda \
  --ent_coef 0.0 --gae_lambda 0.95 --n_epochs 10 --batch_size 512 \
  --clip_range 0.2 --value_lr 0.05 --cost_lr 0.05 --policy_lr 0.1 \
  --use_cost_advantage_label --save_every 0 --log_interval 1
```

### 6.3 CPO baseline
```bash
NO_RESUME=1 python3.10 scripts/train_runner.py \
  --env_type flatland --algo_type cpo --env_name flatland-small-v0 \
  --seed 0 --num_envs 8 --total_n_steps 5000000 --device cuda \
  --n_steps 2048 --batch_size 64 --gae_lambda 0.95 \
  --cost_limit 5.0 --save_every 0 --log_interval 1
```

### 6.4 PPO-GBRL (no cost, reward-only)
```bash
NO_RESUME=1 python3.10 scripts/train_runner.py \
  --env_type flatland --algo_type ppo_gbrl --env_name flatland-small-v0 \
  --seed 0 --num_envs 8 --total_n_steps 5000000 --device cuda \
  --ent_coef 0.0 --gae_lambda 0.95 --n_epochs 10 --batch_size 512 \
  --save_every 0 --log_interval 1
```

### 6.5 SUMO-RL examples
```bash
# SPLIT-RL on SUMO grid4x4
NO_RESUME=1 python3.10 scripts/train_runner.py \
  --env_type sumo --algo_type split_rl --env_name sumo-grid4x4-v0 \
  --seed 0 --num_envs 1 --total_n_steps 500000 --device cuda \
  --ent_coef 0.0 --gae_lambda 0.95 --n_epochs 10 --batch_size 256 \
  --n_steps 256 --value_lr 0.05 --cost_lr 0.05 --policy_lr 0.1 \
  --save_every 0 --log_interval 1

# CPO on SUMO grid4x4
NO_RESUME=1 python3.10 scripts/train_runner.py \
  --env_type sumo --algo_type cpo --env_name sumo-grid4x4-v0 \
  --seed 0 --num_envs 1 --total_n_steps 500000 --device cuda \
  --n_steps 2048 --batch_size 64 --gae_lambda 0.95 \
  --cost_limit 10.0 --save_every 0 --log_interval 1
```

---

## 7. File Map

| File | Purpose |
|---|---|
| `env/flatland.py` | Flatland wrapper: `FlatlandRewardCostWrapper`, `compute_danger_label()`, `make_flatland_vec_env()`, `VecCostMonitor` |
| `env/sumo.py` | SUMO wrapper: `SumoRewardCostWrapper`, `make_sumo_vec_env()`, `VecCostMonitor`, TraCI sub-step accumulation |
| `algos/split_rl.py` | SPLIT-RL algo: reads `info['cost']` + `info['safety_label']`; `use_cost_advantage_label` override post-rollout |
| `algos/safety/cpo.py` | CPO baseline: reads `info['cost']`, cost value head, trust-region constraint |
| `algos/safety/ppo_lag.py` | PPOLag baseline: reads `info['cost']`, Lagrangian multiplier |
| `algos/safety/cup.py` | CUP baseline: reads `info['cost']` |
| `algos/safety/ipo.py` | IPO baseline: reads `info['cost']` |
| `algos/ppo.py` | PPO-GBRL: reward-only, no cost |
| `config/args.py` | CLI arg parsing, algo kwargs dispatch, SAFETY_ENVS/SAFETY_ALGOS lists |
| `scripts/train_runner.py` | Training entry point: env creation, algo dispatch, checkpoint/resume |
| `buffers/rollout_buffer.py` | `CostRolloutBuffer` / `CostCategoricalRolloutBuffer`: stores cost, cost_advantages, safety_labels |

---

## 8. Known Issues & TODOs

1. **SUMO cost sparsity on grid4x4**: Emergency stops and teleports are very rare with random or early-stage policies on grid4x4. Need to test on congested real-world scenarios (cologne8, ingolstadt21) or reduce teleport timeout.
2. **pettingzoo version pin**: Must use `pettingzoo==1.24.3`. Version 1.25 broke `agent_selector` import used by sumo_rl 1.4.5.
3. **SUMO non-uniform spaces**: Real-world RESCO networks (cologne, ingolstadt) may have non-uniform obs/action spaces across intersections. Shared-policy training requires uniform spaces — verify before using multi-agent configs.
4. **value_boosting_iteration=0 with shared trees**: When `shared_tree_struct=True`, the value tree count shows 0 because trees are shared — the value head's parameters are interleaved in the same trees as the policy. This is expected behavior, not a bug.
5. **Cost-advantage label not yet tested at scale**: The `--use_cost_advantage_label` mechanism is implemented but needs experimental validation against the observation-derived label.
