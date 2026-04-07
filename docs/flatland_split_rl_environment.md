# Flatland Split-RL Scenarios — Complete Reference

**Date:** 2026-04-07 (v3)  
**File:** `env/flatland.py`  
**Configs:** `flatland-small-v0` through `flatland-medium-malf-v0` (8 registered)

---

## 1. What Is Flatland?

Flatland is a **multi-agent train routing** environment on procedurally generated grid-based rail networks. Each agent controls a single train navigating from its start station to its destination. Trains share a fixed rail topology and must coordinate at switches and single-track segments to avoid deadlocks and collisions.

Key properties:
- **Grid world**: `x_dim × y_dim` cells, with cities connected by rail corridors.
- **Procedural generation**: Each reset produces a new random grid layout via `env_generator()`.
- **Shared infrastructure**: Trains compete for track capacity at switches and single-track corridors — classic resource contention.
- **Timetables**: Each agent has an `earliest_departure`, `latest_arrival`, and can compute its current *delay* and *slack* (time margin before becoming late).
- **Malfunctions**: Trains can randomly break down, blocking the track for multiple steps. Controlled by `malfunction_interval` (mean steps between events).

---

## 2. Architecture

```
env_generator(n_agents, x_dim, y_dim, n_cities, ...)
→ RailEnv  (with FlattenedNormalizedTreeObsForRailEnv + ShortestPathPredictorForRailEnv)
→ PettingzooFlatland                (Flatland's official PettingZoo parallel adapter)
→ FlatlandRewardCostWrapper          (our wrapper: reward/cost/label split + obs extension)
→ SuperSuit pettingzoo_env_to_vec_env_v1 + concat_vec_envs_v1
→ VecCostMonitor                     (episode stats: r, l, c, normalized_score, completion_rate)
→ SB3 VecEnv
```

- Each agent gets its own VecEnv slot. With `n_agents=5` and `n_envs=4`, the VecEnv has 20 slots.
- **Episode length**: `max_episode_steps = 4 × (x_dim + y_dim + n_agents)`. E.g. small = 4×(25+25+3) = 212 steps.
- SuperSuit's `MarkovVectorEnv` terminates agents individually as they reach their target, but our wrapper overrides this: **termination/truncation signals only fire when ALL agents are done**, preventing spurious 1-step "episodes" from flooding SB3's stats.
- A shared policy controls all agents (parameter sharing).

---

## 3. Observations

**Builder**: `FlattenedNormalizedTreeObsForRailEnv` with `ShortestPathPredictorForRailEnv`.

A **ternary tree** (branching factor 3: L/F/R) rooted at the agent's current cell, expanded via DFS pre-order to `max_depth`. Each node represents a cell reachable by one of the 3 choices at each switch.

### 3.1 Tree Shape

The flattened observation is structured as 12 groups of `n_nodes` features each:

```
obs = [ data(6 × n_nodes) | 1 × n_nodes (separator) | agent_data(5 × n_nodes) ]
     = 12 × n_nodes floats total
```

| max_depth | n_nodes | base obs_dim (12 × n_nodes) |
|-----------|---------|------------------------------|
| 2         | 21      | 252                          |
| 3         | 85      | 1020                         |

The tree includes the root node and all children out to `max_depth`, with 3-way branching (L/F/R) at each non-leaf.

### 3.2 Per-Node Features

**Data segment** (6 features per node):

| Index | Feature                      | Description                                        |
|-------|------------------------------|----------------------------------------------------|
| 0     | `dist_own_target`            | Distance to this agent's target                    |
| 1     | `dist_other_target`          | Distance to nearest other agent's target           |
| 2     | `dist_other_agent_encountered` | Distance to nearest other agent on this branch   |
| 3     | `dist_potential_conflict`    | Distance to nearest potential conflict/deadlock    |
| 4     | `dist_unusable_switch`       | Distance to nearest unusable (occupied) switch     |
| 5     | `dist_to_next_branch`        | Distance to next decision point (switch)           |

**Agent-data segment** (5 features per node):

| Index | Feature                       | Description                                        |
|-------|-------------------------------|----------------------------------------------------|
| 0     | `num_agents_same_direction`   | Agents moving same direction on this branch        |
| 1     | `num_agents_opposite_direction` | Agents moving opposite direction                 |
| 2     | `num_agents_malfunctioning`   | Malfunctioning (blocked) agents on this branch     |
| 3     | `speed_min_fractional`        | Min fractional speed of agents on this branch      |
| 4     | `num_agents_ready_to_depart`  | Agents waiting to depart on this branch            |

**Separator** (1 feature per node): `dist_min_to_target` — shortest-path distance to any target from this node.

All distance features are normalized by `observation_radius` (default = `max_depth`) and clipped to [-1, 1]. Missing branches have all features set to 0.

### 3.3 Slack Feature Extension (Always Active)

4 extra floats are **always** appended to the base tree observation, regardless of `cost_fn`. This ensures all algorithms (PPO baseline, Split-RL, etc.) see the **same 256-dim observation**.

| Feature | Range | Description |
|---------|-------|-------------|
| `own_slack` | [-1, 1] | Agent's delay, normalized by `max_episode_steps/4`. Positive = ahead of schedule. |
| `min_conflicting_slack_L` | [-1, 1] | Min slack of conflicting agents detected on the **left** branch. 1.0 if no conflict. |
| `min_conflicting_slack_F` | [-1, 1] | Min slack of conflicting agents detected on the **forward** branch. 1.0 if no conflict. |
| `min_conflicting_slack_R` | [-1, 1] | Min slack of conflicting agents detected on the **right** branch. 1.0 if no conflict. |

Total obs dim: 252 + 4 = **256** (all configs use `max_depth=2`).

---

## 4. Actions

**Discrete(5)**:

| Value | Name          | Effect                                |
|-------|---------------|---------------------------------------|
| 0     | `DO_NOTHING`  | Continue on current trajectory        |
| 1     | `MOVE_LEFT`   | Take left branch at next switch       |
| 2     | `MOVE_FORWARD`| Take forward branch at next switch    |
| 3     | `MOVE_RIGHT`  | Take right branch at next switch      |
| 4     | `STOP_MOVING` | Brake and wait (yield the right-of-way) |

Actions only have effect at switch points. On straight track, `MOVE_LEFT` / `MOVE_RIGHT` are equivalent to `MOVE_FORWARD`. `STOP_MOVING` halts the train wherever it is.

---

## 5. Reward

Flatland's `BaseDefaultRewards` provides a **per-agent decomposed reward dictionary** with named penalty terms (from `DefaultPenalties` enum):

| Penalty Term          | When                                                    | Typical Value |
|-----------------------|---------------------------------------------------------|---------------|
| `INVALID_ACTION`      | Agent takes an action that can't be executed            | Small negative |
| `STOP_PENALTY`        | Agent stops                                             | Small negative |
| `DEADLOCK`            | Agent enters a deadlock state                           | Large negative |
| `COLLISION`           | Agent collides with another agent                       | Large negative |
| `SHORTEST_PATH_REWARD`| Agent moves along its shortest path toward target       | Small positive |
| `DESTINATION_REACHED` | Agent arrives at destination                            | Large positive |

Our wrapper provides the **full original reward** (sum of ALL terms including collision) to SB3 by default. The cost is a completely separate signal — there is no reason to hack the reward.

- **SB3 reward** (default, `use_original_reward=True`): sum of ALL terms including collision
- **SB3 reward** (if `use_original_reward=False`): sum of all terms except collision
- **`info["cost_collision"]`**: collision penalty magnitude (always tracked regardless of `cost_fn`)

The per-agent reward is **completely blind to other agents' delays**. A train is rewarded for its own progress; there is no "diff-waiting-time" global signal like in SUMO. This is what makes the externality-based cost functions work: any cost measuring harm to *other* agents automatically creates genuine reward-cost disagreement.

---

## 6. Agent States & Timetables

### 6.1 TrainState

| State               | Value | Meaning                                      |
|---------------------|-------|----------------------------------------------|
| `WAITING`           | 0     | Not yet ready to depart                      |
| `READY_TO_DEPART`   | 1     | Can enter the map                            |
| `MALFUNCTION_OFF_MAP` | 2   | Malfunctioned before entering                |
| `MOVING`            | 3     | Moving along the track                       |
| `STOPPED`           | 4     | Halted (by STOP_MOVING or blocked)           |
| `MALFUNCTION`       | 5     | Malfunctioned while on the map (blocks track) |
| `DONE`              | 6     | Reached destination                          |

### 6.2 Timetable

Each agent has a timetable with:
- `earliest_departure`: earliest step the agent can enter the map
- `latest_arrival`: latest step the agent should reach its target
- `get_travel_time_on_shortest_path(distance_map)`: shortest possible travel time
- `get_current_delay(elapsed_steps, distance_map)`: how far ahead/behind schedule the agent is

**Slack** = time buffer before the agent is late:
```
slack = latest_arrival − elapsed_steps − travel_time_remaining
```
Positive = ahead of schedule, negative = behind schedule.

---

## 7. Available Configs

### 7.1 Config Parameters

| Parameter                   | Description                                                      | Default |
|-----------------------------|------------------------------------------------------------------|---------|
| `n_agents`                  | Number of trains                                                 | varies  |
| `x_dim`, `y_dim`            | Grid dimensions                                                  | varies  |
| `n_cities`                  | Number of station clusters                                       | varies  |
| `max_rails_between_cities`  | Max parallel tracks connecting cities. **1 = tight (bottleneck)**; 2 = default | 2 |
| `max_rail_pairs_in_city`    | Max rail pairs within a city                                     | 2       |
| `max_depth`                 | Observation tree depth. Always 2 (252 base obs)                  | 2       |
| `predictor_max_depth`       | Lookahead depth for ShortestPathPredictor                        | varies  |
| `malfunction_interval`      | Mean steps between random train breakdowns. 10⁹ = disabled       | 10⁹     |

### 7.2 Registered Configs

| Config Name                | n_agents | Grid   | n_cities | max_rails | malf_interval | obs_dim | max_steps | Status |
|---------------------------|----------|--------|----------|-----------|---------------|---------|-----------|--------|
| `flatland-small-v0`       | 3        | 25×25  | 2        | 2         | disabled      | 256     | 212       | Baseline |
| `flatland-medium-v0`      | 5        | 35×35  | 3        | 2         | disabled      | 256     | 300       | Baseline |
| `flatland-large-v0`       | 10       | 50×50  | 4        | 2         | disabled      | 256     | 440       | Baseline |
| `flatland-xlarge-v0`      | 20       | 80×80  | 6        | 2         | disabled      | 256     | 720       | Baseline |
| `flatland-small-tight-v0` | 5        | 25×25  | 2        | **1**     | disabled      | 256     | 220       | ⚠️ See §9 |
| `flatland-medium-tight-v0`| 8        | 35×35  | 3        | **1**     | disabled      | 256     | 312       | ⚠️ See §9 |
| `flatland-small-malf-v0`  | 5        | 25×25  | 2        | **1**     | **40**        | 256     | 220       | ⚠️ See §9 |
| `flatland-medium-malf-v0` | 8        | 35×35  | 3        | **1**     | **40**        | 256     | 312       | ⚠️ See §9 |

All configs use `max_depth=2`. Obs dim is always **256** (252 tree + 4 slack features).

**⚠️ The tight/malf configs were designed for slack_priority and malfunction_detour cost functions, but the grid topologies they produce do not generate enough switches with conflict-free alternatives. See Section 9.**

### 7.3 Config Design Rationale

**Original configs** (`small-v0`, `medium-v0`, `large-v0`, `xlarge-v0`): Standard Flatland with `max_rails_between_cities=2`. Multiple parallel tracks between cities reduce contention — agents often have room to pass each other without conflict. Good for baseline PPO training but weak for Split-RL because conflicts are infrequent.

**Tight configs** (`small-tight-v0`, `medium-tight-v0`): `max_rails_between_cities=1`. Forces all trains between two cities onto a **single track corridor**, creating obligatory yield points at switches. More agents per grid than original (5 instead of 3 for small). This is where Split-RL becomes meaningful — trains regularly meet on shared single-track segments and must decide who yields. No malfunctions, so all conflicts are purely scheduling/routing.

**Malfunction configs** (`small-malf-v0`, `medium-malf-v0`): Tight configs with `malfunction_interval=40` (~2.5% chance per step per train of breaking down). Malfunctions block the occupied cell for several steps, creating unpredictable downstream congestion. Designed for the `malfunction_detour` cost function: agents approaching a blocked branch should reroute rather than queueing behind the broken train.

---

## 8. Cost Functions — What We Tried and Why It Failed

### 8.1 Legacy: `congestion` (FAILED)

```python
cost = 1 if agent moved into a branch with nearby other_agent / potential_conflict / opposite_dir traffic
```

**Why it failed**: Congestion cost measures harm **to** the acting agent (it's getting stuck), not harm the agent imposes on others. The reward already penalizes getting stuck (no progress → no `SHORTEST_PATH_REWARD`, step penalty accumulates). So cost aligns with reward — minimizing cost also maximizes reward. **No genuine tradeoff.**

### 8.2 Legacy: `danger` (FAILED)

```python
cost = 1 if ANY non-root tree branch shows nearby conflict/agents/opposite-dir/malfunction
```

**Why it failed**: Pure event detector, not a conflict detector. Fires on 30–50% of all steps regardless of what action the agent takes. The agent cannot control the cost because it cannot control where other agents are. "Label = cost is positive" is the same mistake as early SUMO attempts. **No action-contingent disagreement.**

### 8.3 Legacy: `collision` (FAILED)

```python
cost = collision penalty magnitude from Flatland rewards
```

**Why it failed**: Trivially negatively correlated with reward. Agent just learns not to collide — there's no interesting tradeoff. The cost doesn't capture externalities, it captures mistakes.

### 8.4 Root Cause

All three legacy approaches went **"obs → cost → hope for conflict"** instead of **"conflict → cost → obs"**. They confused representational capacity (the tree obs contains conflict features) with genuine objective conflict (reward wants X, cost wants Y, and these disagree at decision points).

---

## 9. Current Split-RL Scenarios — Status

### 9.0 — The Fundamental Problem

Both scenarios below assume the agent is at a **switch** where it has a conflicting branch (forward) and a **conflict-free alternative branch** (left or right) to reroute to. The cost fires when the agent chooses the conflicting branch; the label marks states where this choice exists.

**After fixing a dead-end detection bug (v3), we found this situation almost never occurs.** Flatland's procedural grid generation produces topologies where:
- Switches are rare (~5% of active steps)
- Most switches have dead-end L/R branches (data = -1.0 in the tree obs), not real through-routes
- Switches with forward conflict AND a conflict-free alternative branch: **0.2-0.7%** of active steps across all configs tested
- Label fires: **0-16 times per ~2500 active steps**, zero on many seeds

This applies to ALL grid configs we tested (max_rails=1, 2, and 3; 3-5 cities; 8-12 agents). The grid generator simply doesn't create enough mid-corridor passing sidings.

**The "170 cost events in 500 steps" and "+39% completion heuristic" reported earlier were entirely based on the dead-end bug** — the code was counting dead-end branches (all features = -1.0) as valid alternative routes.

### 9.1 — Slack-Priority Yield (`cost_fn="slack_priority"`) — DOES NOT FIRE

**The intended scenario**: Agent at a switch, forward branch has a head-on train with lower slack, a side branch is clear. Agent should reroute to let the urgent train pass.

**What reward wants**: Go forward — `SHORTEST_PATH_REWARD` fires for moving along the shortest path. The reward is per-agent and blind to other agents' deadlines.

**What cost would measure**: If the agent goes forward into a contested branch at a switch, delaying a lower-slack train, while a clear alternative exists → cost = 1. If the agent reroutes → cost = 0.

**Why it doesn't fire**: Flatland grids with `max_rails_between_cities=1` have single-track corridors. There IS only one route between cities. The L/R branches at switches are dead ends (city-internal stub tracks), not through-routes. The agent has no viable reroute. With `max_rails=2`, there are more switches but the forward-conflict + clear-alternative combination is still extremely rare (0.36% of steps).

**Pre-conditions for cost = 1** (all must hold):
1. Agent is at a switch (L or R branch exists with positive obs data)
2. Chosen branch has a conflict signal (`dist_other_agent`, `dist_potential_conflict`, or `num_agents_opposite_direction` within threshold)
3. At least one alternative branch EXISTS (positive data) and has NO conflict
4. A nearby conflicting agent has lower slack than us by at least `slack_margin` steps

**Pre-conditions for label = 1**: Same as above but checks forward branch specifically (not action-conditioned).

### 9.2 — Malfunction Detour (`cost_fn="malfunction_detour"`) — DOES NOT FIRE

**Same problem.** The label fires when forward branch has a malfunctioning agent AND an alternative branch is clear. After the dead-end fix: 3-6 events per ~1300 active steps (0.3%).

### 9.3 — What Would Need to Change

For either scenario to work, Flatland grids need **switches with two valid through-routes** — where one route has conflict and the other doesn't. Options:

1. **Custom grid generation**: Build grids with explicit passing sidings (not supported by `env_generator` out of the box)
2. **More cities**: More cities = more routing options. But `env_generator` creates corridors between pairs of cities, and with 4+ cities the grid gets large.
3. **Different environment**: Flatland's procedural generation isn't designed for this kind of split-RL scenario. The grid topology doesn't naturally produce "reroute vs. forward" decisions enough.
4. **Rethink the cost signal**: Instead of requiring a reroute alternative, measure the externality differently — e.g., cost = total delay imposed on other agents by this step, computed from the rail simulation. This removes the "must have alternative branch" requirement but is harder to implement.

---

## 10. How to Run Training

### 10.1 Command

```bash
python3.10 scripts/train_runner.py \
  --algo split_rl \
  --env_type flatland \
  --env_name flatland-medium-tight-v0 \
  --num_envs 4 \
  --total_timesteps 5000000 \
  --device cuda \
  --seed 0 \
  --env_kwargs '{"cost_fn": "slack_priority", "slack_margin": 3.0}' \
  --log_dir /path/to/logs
```

Key arguments:
- `--env_type flatland`: selects the Flatland pipeline in `train_runner.py`
- `--env_name`: any key from `FLATLAND_CONFIGS` (Section 7.2)
- `--env_kwargs`: JSON dict passed as `**kwargs` to `make_flatland_vec_env()`. Controls `cost_fn`, `slack_margin`, `use_original_reward`, etc.
- `--num_envs`: number of parallel Flatland instances (total VecEnv slots = `num_envs × n_agents`)

### 10.2 Scenario → Config → Cost Function

| What you want to test | `--env_name` | `--env_kwargs` |
|-----------------------|-------------|----------------|
| Slack-priority yield (small) | `flatland-small-tight-v0` | `'{"cost_fn": "slack_priority"}'` |
| Slack-priority yield (medium) | `flatland-medium-tight-v0` | `'{"cost_fn": "slack_priority"}'` |
| Malfunction detour (small) | `flatland-small-malf-v0` | `'{"cost_fn": "malfunction_detour"}'` |
| Malfunction detour (medium) | `flatland-medium-malf-v0` | `'{"cost_fn": "malfunction_detour"}'` |
| PPO baseline (no Split-RL) | `flatland-small-tight-v0` | (omit or any cost_fn — cost is ignored by PPO) |

### 10.3 Parameters Forwarded via `env_kwargs`

| Parameter             | Type  | Default           | Description                                     |
|-----------------------|-------|-------------------|-------------------------------------------------|
| `cost_fn`             | str   | `"slack_priority"` | Cost function name                              |
| `use_original_reward` | bool  | `True`            | Use full original reward (including collision)  |
| `conflict_dist_thresh`| float | `0.5`             | Normalized distance for conflict detection      |
| `agent_dist_thresh`   | float | `0.5`             | Normalized distance for other-agent detection   |
| `slack_margin`        | float | `3.0`             | Min slack gap (raw steps) to trigger label      |
| `collision_factor`    | float | `1.0`             | Collision penalty weight                        |
| `decompose_rewards`   | bool  | `True`            | Decompose rewards into dict terms               |

Any `FLATLAND_CONFIGS` parameter (e.g. `max_rails_between_cities`, `malfunction_interval`) can also be overridden via `env_kwargs`.

---

## 11. Metrics Logged

Per-episode metrics emitted in `info["episode"]` by `VecCostMonitor`:

| Key                  | Description                                                   |
|----------------------|---------------------------------------------------------------|
| `r`                  | Episode return (full original reward including collision)      |
| `l`                  | Episode length                                                |
| `c`                  | Episode cumulative cost                                       |
| `cost_collision`     | Episode cumulative collision cost                             |
| `cost_danger`        | Episode cumulative danger cost (legacy, always tracked)       |
| `original_r`         | r − c (informational)                                        |
| `normalized_score`   | Total original reward ÷ (max_steps × n_agents)               |
| `completion_rate`    | Fraction of agents that reached their destination             |

---

## 12. Key Design Decisions

1. **Switch-only costs**: Costs and labels fire only at switches, not on straight track. On straight track, STOP still blocks the rail — there is no alternative action that reduces externality. Yield only helps at switches where the agent can reroute.

2. **Action-conditioned cost, state-based label**: The cost depends on which branch the agent chose (action-conditioned). The label depends only on the state (is this a disagreement state?). This follows the Split-RL convention: the label marks *where* the objectives disagree, the cost measures *how much* the chosen action hurts others.

3. **Per-agent reward blindness**: Unlike SUMO (where `diff-waiting-time` sums all vehicles), Flatland reward is purely per-agent. This means ANY cost measuring harm to other agents automatically creates disagreement. The cost doesn't need to be subtle — the reward is structurally blind to externalities.

4. **Obs extension with slack features (always active)**: Even though the base tree observation already contains rich conflict signals, it does NOT contain timetable information (slack, deadlines). The 4 extra floats give the cost head the information it needs to distinguish "conflict with urgent train" from "conflict with non-urgent train." These features are always appended (not just for Split-RL) so that all algorithms see the same 256-dim observation.

5. **Unmodified reward**: The full original Flatland reward (including collision) is used for SB3 training. Cost is a completely separate signal — not carved out of the reward.

6. **Individual vs global termination**: Flatland terminates agents individually. Our wrapper batches termination so SB3 sees one episode end when ALL agents are done. This is critical for correct reward/cost statistics.
