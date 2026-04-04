# SUMO Bus Priority & Emergency Preemption — Implementation & Results

## 1. Problem Statement

SPLIT-RL needs environments with **genuine, state-localized multi-objective conflicts**: states where the reward-optimal action differs from the constraint-optimal action, and a tree can identify these states from observation features.

Previous attempts failed:
- `mainline` reward + `side_queue` cost: worked but was "reward hacking" — the reward was engineered to conflict.
- `throughput` reward + `side_queue` cost: Pearson correlation was **positive** (+0.537) — no conflict.

**The fix**: keep the honest base reward (`diff-waiting-time`, the SUMO-RL default) and create conflict via **task-level constraints** — priority vehicles (buses, emergencies) that the agent must serve even when the reward says otherwise.

---

## 2. Design: Why This Creates a Real Conflict

`diff-waiting-time` sums per-vehicle waiting time changes. If 10 cars are waiting on the mainline and 1 bus is waiting on a side lane, the reward says "serve the 10 cars." The bus is just 1 vehicle — it barely moves the reward needle.

But the **bus priority constraint** says "serve the bus, it's been waiting too long." This is structurally unsatisfiable alongside the reward: green time is finite, serving the bus means NOT serving the 10 cars for at least one phase cycle.

Key properties:
1. **The conflict is state-localized**: only when a bus/emergency is present AND waiting on an unserved lane.
2. **The conflict is obs-predictable**: `has_bus[lane_i]` and `bus_wait[lane_i]` are directly in the observation.
3. **A tree can split on it**: depth-2 split (`has_bus > 0` then `bus_wait > threshold`) identifies the conflict region exactly.
4. **The conflict moves around**: buses take random routes, so the constraint fires from different directions across episodes. The tree must learn the feature, not memorize a direction.

---

## 3. What Was Implemented

All changes are in [`env/sumo.py`](../env/sumo.py).

### 3.1 New Cost Functions in Registry

```python
COST_FN_REGISTRY = {
    ...
    "bus_priority": "Per-agent: max(bus_wait/T_bus) across lanes with buses",
    "emergency_preemption": "Per-agent: max(emg_wait/T_emg) across lanes with emergencies",
}
```

### 3.2 New Parameters on `SumoRewardCostWrapper.__init__`

| Parameter | Default | Description |
|---|---|---|
| `bus_injection_interval` | 25.0 | Seconds between bus injections (with ±20% jitter) |
| `bus_cost_threshold` | 30.0 | T_bus: seconds of bus wait that yields cost=1.0 |
| `bus_warn_threshold` | 10.0 | T_warn: seconds of bus wait that triggers label=1 |
| `emergency_injection_interval` | 105.0 | Seconds between emergency injections |
| `emergency_cost_threshold` | 10.0 | T_emg: seconds of emergency wait that yields cost=1.0 |

Same parameters are exposed on `make_sumo_vec_env()`.

### 3.3 Observation Space Extension

Computed at `__init__` time from the obs layout formula:
```
original obs = [phase_one_hot(n_phases), min_green(1), density(n_lanes), queue(n_lanes)]
n_lanes = (obs_dim - n_phases - 1) / 2
extension = has_X[n_lanes] + X_wait[n_lanes]  → +2*n_lanes dims
```

| Network | Original obs | Extension | New obs dim |
|---|---|---|---|
| arterial4x4 | 18 (5+1+6+6) | +12 (6+6) | **30** |
| grid4x4 | 33 (8+1+12+12) | +24 (12+12) | **57** |

The extended space is set in `__init__` so SuperSuit sees the correct dim. The actual feature values are appended in `_extend_obs()` after each step/reset.

### 3.4 Vehicle Injection Pipeline

**In `reset()`:**
1. `_discover_boundary_edges()` — finds network entry/exit edges (nodes with ≤2 edge connections)
2. `_create_bus_vtype()` / `_create_emergency_vtype()` — defines vehicle types via `traci.vehicletype.copy()`
3. Initializes per-agent zero arrays for `has_bus`, `bus_wait`, `has_emg`, `emg_wait`
4. Schedules first injection at ~half the interval

**In `_process_step()` (each agent step = 5 sim seconds):**
1. `_inject_vehicles(sim_time)` — checks if it's time to inject, picks random origin→dest from boundary edges, calls `traci.route.add()` + `traci.vehicle.add()`
2. `_detect_special_vehicles()` — iterates each agent's incoming lanes via `traci.lane.getLastStepVehicleIDs()`, checks `traci.vehicle.getVehicleClass()` for "bus"/"emergency", records max wait per lane
3. `_extend_obs(obs)` — appends `has_bus` + `bus_wait` (or `has_emg` + `emg_wait`) to each agent's obs vector

### 3.5 Cost Functions

**Bus priority** ([`_compute_bus_priority_cost()`](../env/sumo.py#L706)):
```
cost = max over lanes of: bus_wait[lane] / T_bus   (for lanes with buses)
     = 0                                             (no buses present)
```
Continuous ∈ [0, 1]. `bus_wait` is already normalized by T_bus in `_detect_special_vehicles()`.

**Emergency preemption** ([`_compute_emergency_preemption_cost()`](../env/sumo.py#L755)):
```
cost = max over lanes of: emg_wait[lane] / T_emg   (for lanes with emergencies)
     = 0                                             (no emergencies)
```

### 3.6 Labels

**Bus priority** ([`_bus_priority_label()`](../env/sumo.py#L722)):
```
label = 1  IF  has_bus[lane_i] = 1
              AND  lane_i NOT in served_lanes(current_phase)
              AND  bus_wait[lane_i] > T_warn / T_bus
         0  otherwise
```
This is the **decision frontier**: the bus has been waiting past the warning threshold and the agent is NOT currently serving it. A depth-2 tree can split on `bus_wait > threshold` then `has_bus`.

**Emergency preemption** ([`_emergency_preemption_label()`](../env/sumo.py#L764)):
```
label = 1  IF  has_emg[lane_i] = 1
              AND  lane_i NOT in served_lanes(current_phase)
         0  otherwise
```
No wait threshold — emergency preemption is immediate. Depth-1 tree suffices.

### 3.7 Integration Points

In `_process_step()`:
- Vehicle injection + detection happen **once** at the top (before the per-agent loop)
- Cost dispatch includes `bus_priority` and `emergency_preemption` branches
- Label dispatch includes `_bus_priority_label` and `_emergency_preemption_label` branches
- Obs extension happens in the return path: `_transform_obs()` then `_extend_obs()`

---

## 4. Diagnostic Script

[`scripts/diagnose_conflict.py`](../scripts/diagnose_conflict.py) updated to support:
```bash
python3.10 scripts/diagnose_conflict.py --env sumo --scenario bus_priority --network arterial
python3.10 scripts/diagnose_conflict.py --env sumo --scenario bus_priority --network grid
python3.10 scripts/diagnose_conflict.py --env sumo --scenario emergency_preemption --network arterial
python3.10 scripts/diagnose_conflict.py --env sumo --scenario emergency_preemption --network grid
```

Runs 3 episodes under random policy and checks 5 criteria:

| # | Check | Criterion | Why it matters |
|---|---|---|---|
| 1 | Label rate | 10–40% | Enough conflict to train on, not always-on |
| 2 | Cost\|label=1 | >80% co-occurrence | Label correctly identifies high-cost states |
| 3 | Label predictability | DT depth-3 accuracy >0.95 | Tree CAN learn the partition |
| 4 | Reward-cost correlation | <0.3 | Cost isn't free to satisfy alongside reward |
| 5 | Cost fires | >1% | Signal isn't dead |

---

## 5. Results — Bus Priority (Scenarios A1 & G1)

### Tuning History

| Attempt | interval | T_warn | Label rate | Bus any lane | Verdict |
|---|---|---|---|---|---|
| 1 | 75s | 15s | 4.1% | 10.8% | Too rare |
| 2 | 35s | 10s | 8.9% | 21.9% | Close but under 10% |
| 3 (final) | **25s** | **10s** | **14.0%** (A1) / **11.4%** (G1) | 32-34% | Target hit |

### A1: arterial4x4 + bus_priority — **5/5 PASS**

```
--- CONFLICT STATISTICS (34560 total transitions) ---
  Cost > 0:      6928/34560 = 0.200
  Label = 1:     4840/34560 = 0.140
  Cost mean:     0.1398
  Cost std:      0.3201

  Cost > 0 when label=1: 4840/4840 = 1.000

  Pearson correlation(reward, cost): -0.0408
    WEAK: Near-zero correlation — independent signals.

  Label predictability (DecisionTree depth=3, 5-fold CV):
    Accuracy: 0.974 +/- 0.006
    EXCELLENT: Label is highly predictable from obs.
    Top features: [27, 24, 25]   ← bus_wait features
    Importances:  ['0.426', '0.390', '0.110']

  Extended features (bus):
    n_lanes=6, extension starts at obs[18]
    Fraction of steps with bus on ANY lane: 0.324

--- VERDICT ---
  [PASS] Label rate: 14.0%
  [PASS] Cost|label=1: 100.0%
  [PASS] Label predictability: 0.974
  [PASS] Reward-cost corr: -0.041
  [PASS] Cost fires: 20.0%

  Result: 5/5 checks passed
```

Per-episode breakdown:
| Episode | Steps | Mean Reward | Mean Cost | Label Rate |
|---|---|---|---|---|
| 1 | 720 | -36.79 | 98.13 | 0.137 |
| 2 | 720 | -40.88 | 110.82 | 0.154 |
| 3 | 720 | -27.72 | 93.09 | 0.129 |

### G1: grid4x4 + bus_priority — **5/5 PASS**

```
--- CONFLICT STATISTICS (34560 total transitions) ---
  Cost > 0:      5192/34560 = 0.150
  Label = 1:     3943/34560 = 0.114
  Cost mean:     0.1007
  Cost std:      0.2753

  Cost > 0 when label=1: 3943/3943 = 1.000

  Pearson correlation(reward, cost): -0.1046
    GOOD: Negative correlation — reward and cost are in tension.

  Label predictability (DecisionTree depth=3, 5-fold CV):
    Accuracy: 0.952 +/- 0.008
    EXCELLENT: Label is highly predictable from obs.
    Top features: [55, 52, 49]   ← bus_wait features
    Importances:  ['0.339', '0.337', '0.319']

  Extended features (bus):
    n_lanes=12, extension starts at obs[33]
    Fraction of steps with bus on ANY lane: 0.341

--- VERDICT ---
  [PASS] Label rate: 11.4%
  [PASS] Cost|label=1: 100.0%
  [PASS] Label predictability: 0.952
  [PASS] Reward-cost corr: -0.105
  [PASS] Cost fires: 15.0%

  Result: 5/5 checks passed
```

### Key Observations

1. **The tree uses bus features, not traffic features.** Top-3 feature importances are all `bus_wait[lane_i]` — the tree learned to split on the bus extension obs, not on the base traffic state. This is exactly what we want for SPLIT-RL gradient routing.

2. **Cost|label=1 is 100%.** Every time the label fires, cost is also non-zero. The label is a strict subset of high-cost states (as designed: label requires `wait > T_warn`, cost fires for any `wait > 0`).

3. **Reward-cost correlation is near-zero to slightly negative.** On arterial (-0.041) and grid (-0.105). This means the cost signal is genuinely orthogonal to the reward — you can't satisfy the bus constraint "for free" by optimizing diff-waiting-time.

4. **Label rate is stable across episodes.** Arterial: 12.9–15.4%. Grid: 9.8–12.3%. Random routes + stochastic injection produce consistent conflict frequency.

---

## 6. What Remains: Emergency Preemption (A2, G2)

The implementation is **already in `env/sumo.py`** — `_create_emergency_vtype()`, `_compute_emergency_preemption_cost()`, `_emergency_preemption_label()` are all coded. Just needs:

1. Run A2 diagnostic: `--scenario emergency_preemption --network arterial`
2. Run G2 diagnostic: `--scenario emergency_preemption --network grid`
3. Tune `emergency_injection_interval` if label rate is outside 5-15% target
4. Verify 5/5 checks pass

Expected differences from bus:
- **Lower label rate** (~5-10%): emergencies are rarer
- **Sharper partition**: label fires immediately (no wait threshold), so depth-1 tree suffices
- **Tighter cost deadline**: T_emg=10s vs T_bus=30s

---

## 7. Files Changed

| File | What changed |
|---|---|
| [`env/sumo.py`](../env/sumo.py) | Added `bus_priority` + `emergency_preemption` to `COST_FN_REGISTRY`; new params on `__init__` and `make_sumo_vec_env`; obs space extension in `__init__`; bus/emergency vtype creation, injection, detection, obs extension, cost, and label methods; wired into `reset()` and `_process_step()` |
| [`scripts/diagnose_conflict.py`](../scripts/diagnose_conflict.py) | Rewrote `diagnose_sumo()` to accept `--scenario` and `--network` args; added bus/emergency extended feature analysis; added 5-check verdict with pass/fail |
| [`scripts/tune_bus_rate.py`](../scripts/tune_bus_rate.py) | Throwaway tuning script (tested intervals 35/25/20s) |

---

## 8. How to Run

### Quick smoke test (1 episode)

```bash
cd /swgwork/bfuhrer/projects/gbrl_project/nvlabs/gbrl_sb3
PYTHONPATH="$PWD:$PYTHONPATH" LIBSUMO_AS_TRACI=1 python3.10 -c "
from env.sumo import make_sumo_vec_env
import numpy as np
env = make_sumo_vec_env(env_name='sumo-arterial4x4-v0', override_reward=None, cost_fn='bus_priority')
obs = env.reset()
print(f'Obs shape: {obs.shape}')  # (16, 30) for arterial
for _ in range(10):
    obs, r, d, info = env.step(np.array([env.action_space.sample() for _ in range(env.num_envs)]))
    print(f'reward={r[0]:.2f}, cost={info[0][\"cost\"]:.3f}, label={info[0][\"safety_label\"]}')
env.close()
"
```

### Full diagnostic (3 episodes, ~8 min per scenario)

```bash
# A1: arterial + bus
PYTHONPATH="$PWD:$PYTHONPATH" LIBSUMO_AS_TRACI=1 python3.10 scripts/diagnose_conflict.py \
    --env sumo --scenario bus_priority --network arterial --episodes 3

# G1: grid + bus
PYTHONPATH="$PWD:$PYTHONPATH" LIBSUMO_AS_TRACI=1 python3.10 scripts/diagnose_conflict.py \
    --env sumo --scenario bus_priority --network grid --episodes 3

# A2: arterial + emergency (not yet validated)
PYTHONPATH="$PWD:$PYTHONPATH" LIBSUMO_AS_TRACI=1 python3.10 scripts/diagnose_conflict.py \
    --env sumo --scenario emergency_preemption --network arterial --episodes 3

# G2: grid + emergency (not yet validated)
PYTHONPATH="$PWD:$PYTHONPATH" LIBSUMO_AS_TRACI=1 python3.10 scripts/diagnose_conflict.py \
    --env sumo --scenario emergency_preemption --network grid --episodes 3
```

---

## 9. Default Configuration for Training

When ready to train SPLIT-RL with bus priority:

```python
env = make_sumo_vec_env(
    env_name="sumo-arterial4x4-v0",  # or "sumo-grid4x4-v0"
    override_reward=None,              # use original diff-waiting-time
    cost_fn="bus_priority",            # bus constraint
    bus_injection_interval=25.0,       # ~1 bus every 25s (with jitter)
    bus_cost_threshold=30.0,           # T_bus: cost=1.0 when bus waits 30s
    bus_warn_threshold=10.0,           # T_warn: label fires at 10s wait
)
```

The old defaults (`override_reward="mainline"`, `cost_fn="side_queue"`) are still available for backward compatibility but should NOT be used for new experiments.
