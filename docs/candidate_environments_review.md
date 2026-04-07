# Candidate Environments for Split-RL: Detailed Review

**Date:** April 7, 2026
**Purpose:** Evaluate four candidate environments for Split-RL / GBRL integration, focusing on tree-friendliness, SB3 compatibility, and conflict-injection feasibility.

---

## Table of Contents

1. [Evaluation Criteria](#evaluation-criteria)
2. [CityLearn](#1-citylearn)
3. [OR-Gym](#2-or-gym)
4. [Overcooked AI](#3-overcooked-ai)
5. [MiniHack](#4-minihack)
6. [Comparative Summary](#comparative-summary)
7. [Recommendations](#recommendations)

---

## Evaluation Criteria

Each environment is scored on four axes (1–5 scale):

| Criterion | Description |
|-----------|-------------|
| **Tree-Friendliness** | Discrete actions, axis-aligned meaningful features, low-to-moderate obs dimensionality, features that split well on thresholds |
| **SB3 Compatibility** | Out-of-the-box Gymnasium API, existing SB3 wrappers, single-agent support, VecEnv friendliness |
| **Conflict Injection** | How naturally can we add a secondary cost signal that creates a genuine reward-vs-cost tradeoff (as done with SUMO) |
| **Implementation Effort** | Lines of code, external dependencies, integration complexity with the existing `gbrl_sb3` codebase |

For reference, the SUMO environment pattern requires:
- A **reward signal** (the primary objective)
- A **cost signal** ∈ [0, 1] (the secondary constraint, computed per step)
- A **label** ∈ {0, 1} (state-based, marking where reward and cost objectives disagree)
- Cost and label passed via `info['cost']` and `info['safety_label']`

---

## 1. CityLearn

**Source:** https://www.citylearn.net/
**Domain:** Smart building energy management (district-level battery/HVAC control)
**License:** MIT

### 1.1 Environment Description

CityLearn simulates a district of buildings, each equipped with energy storage (cooling, heating, DHW, electrical) and devices (heat pumps, heaters). The agent controls charge/discharge rates for storage devices across all buildings. The episode runs for one year at hourly resolution (8,760 timesteps).

### 1.2 Observation Space

| Category | Example Features | Count |
|----------|-----------------|-------|
| Calendar | month, day_type, hour, daylight_savings_status | 4 |
| Weather | outdoor_temp, humidity, solar irradiance (+ 6h/12h/24h predictions) | ~16 |
| District | carbon_intensity | 1 |
| Building (per building) | indoor_temp, storage SOCs, electricity consumption, pricing, demand, device efficiency, power_outage | ~20+ |

**Total obs dimension (centralized, 3 buildings):** ~60–80 features depending on schema config.

**Tree-friendliness of features:**
- **Excellent.** All features are scalar, continuous, physically meaningful, and axis-aligned. Temperature thresholds, SOC levels, pricing tiers, demand peaks — these are textbook decision-tree split candidates.
- Features like `hour`, `month`, `electricity_pricing`, `cooling_storage_soc` all have obvious threshold semantics.
- Observations can be min-max normalized to [0, 1] via built-in `NormalizedObservationWrapper`.

### 1.3 Action Space

**Continuous:** `Box([-1, 1], shape=(n_actions,))` where `n_actions` = number of controllable storage devices across all buildings (typically 2–6 per building).

For a 3-building centralized setup: `Box([-1, 1], shape=(6,))` to `Box([-1, 1], shape=(12,))`.

**Tree-friendliness:** ⚠️ **Continuous actions.** This is a significant drawback for GBRL. While GBRL supports continuous action PPO, the action space is inherently continuous (charge/discharge fractions). **Discretization is possible** (e.g., 5–11 bins per action dimension) but the multi-dimensional combinatorial explosion is a concern. With 6 actions × 5 bins = 15,625 joint actions — too large. Would need per-device independent discretization or factored action approach.

### 1.4 SB3 Compatibility

**Very good.** CityLearn provides:
- `StableBaselines3Wrapper` — official wrapper for SB3 compatibility
- `NormalizedObservationWrapper` — built-in obs normalization
- Centralized agent mode (`central_agent=True`) — flattens multi-building into single-agent
- Standard Gymnasium API (`step`, `reset`, `observation_space`, `action_space`)
- Tested with SB3 SAC in their official docs

**Caveats:**
- Requires `central_agent=True` for SB3 (no native multi-agent)
- Long episodes (8,760 steps) — may need careful GAE lambda tuning
- External data files (weather, building CSVs) bundled with package
- No GPU simulation — CPU-only, but lightweight

### 1.5 Conflict Injection Potential

**Outstanding — arguably the best candidate.** CityLearn has _built-in_ competing objectives:

| Reward Objective | Natural Cost Objective | Conflict Type |
|------------------|----------------------|---------------|
| Minimize electricity cost | Maintain thermal comfort (discomfort) | **Energy-efficiency vs comfort** |
| Minimize carbon emissions | Minimize peak demand | **Emissions vs grid stability** |
| Minimize total energy use | Maximize thermal resilience (during outages) | **Efficiency vs resilience** |
| Minimize electricity cost | Equitable comfort across buildings | **Cost vs fairness** |

**Concrete conflict scenarios:**

**S1: Comfort vs Cost (analogous to SUMO bus priority)**
- **Reward:** Negative electricity cost: `r_t = -electricity_cost_t`
- **Cost:** Comfort violation: `c_t = max(0, |T_indoor - T_setpoint| - T_band) / T_max_delta`
- **Conflict:** Cheapest action is to not run HVAC (discharge storage, avoid grid purchase during peak pricing). But occupants get uncomfortable. During high-price hours, reward says "don't consume" while cost says "maintain temperature."
- **Label:** `1` when `electricity_pricing > price_threshold AND |T_indoor - T_setpoint| > comfort_threshold` — the decision frontier where cost and reward disagree.

**S2: Peak Shaving vs Storage Preservation (analogous to SUMO convoy)**
- **Reward:** Minimize peak district demand (load factor)
- **Cost:** Battery degradation proxy: `c_t = |charge_rate| * cycle_count / max_cycles`
- **Conflict:** Aggressive peak shaving requires deep cycling batteries. Reward says "flatten the curve" while cost says "preserve battery health."
- **Label:** `1` when `district_demand > peak_threshold AND electrical_storage_soc < low_soc_threshold`

**S3: Grid Stability vs Self-Consumption (analogous to SUMO spillback — spatial conflict)**
- **Reward:** Maximize self-consumption of solar generation
- **Cost:** Grid ramping penalty: `c_t = |net_export_t - net_export_{t-1}| / max_ramp`
- **Conflict:** Maximizing solar self-consumption means storing excess midday and using it at night. But rapid transitions in net grid exchange (duck curve) destabilize the grid. Reward says "absorb all solar" while cost says "smooth the export profile."
- **Label:** `1` when `solar_generation > generation_threshold AND |ramp_rate| > ramp_threshold`

**Obs-based label feasibility:** All labels above are fully deterministic functions of observable features (pricing, SOC, temperature, solar generation). This is a clean fit for the Split-RL label philosophy.

### 1.6 Scoring

| Criterion | Score | Notes |
|-----------|-------|-------|
| Tree-Friendliness | **4/5** | Obs: excellent. Action: continuous (needs discretization or continuous GBRL) |
| SB3 Compatibility | **5/5** | Official SB3 wrapper, tested, Gymnasium-native |
| Conflict Injection | **5/5** | Built-in competing KPIs, natural cost functions, excellent label grounding |
| Implementation Effort | **4/5** | `pip install CityLearn`, wrapper exists, but long episodes and continuous actions need handling |
| **Overall** | **4.5/5** | |

---

## 2. OR-Gym

**Source:** https://github.com/hubbs5/or-gym
**Domain:** Operations research (knapsack, bin packing, supply chain, vehicle routing, portfolio optimization, TSP)
**License:** MIT

### 2.1 Environment Description

OR-Gym is a collection of ~15 classic OR problems wrapped as OpenAI Gym environments. The most relevant for Split-RL are:

| Environment | Description |
|-------------|-------------|
| `InvManagement-v0/v1` | Multi-echelon supply chain with backlogs/lost sales |
| `NetworkManagement-v0/v1` | Supply chain network with multiple products |
| `Knapsack-v0/v1/v2/v3` | Various knapsack formulations |
| `BinPacking-v0..v5` | Online bin packing |
| `PortfolioOpt-v0` | Multi-period portfolio allocation |
| `VehicleRouting-v0` | Pick-up and delivery with time windows |

### 2.2 Observation & Action Spaces

**InvManagement (most promising for Split-RL):**
- **Obs:** `Box(shape=(pipeline_length,))` where `pipeline_length = (num_stages - 1) * (max_lead_time + 1)`. For default 4-stage, lead times [3, 5, 10]: obs dim = 3 × 11 = 33. Features are inventory positions and pipeline inventories — integer-valued, physically meaningful.
- **Actions:** `Box(low=0, high=capacity, shape=(num_stages-1,))` — continuous reorder quantities for each echelon. Default: `Box(shape=(3,))` with capacity [100, 90, 80].

**Knapsack-v1 (binary knapsack):**
- **Obs:** `Box(shape=(200, 2))` — 200 items with (weight, value) pairs + current capacity.
- **Actions:** `Discrete(200)` — select which item to add.

**Tree-friendliness:**
- **InvManagement obs:** Good — inventory levels and pipeline positions are meaningful scalars with obvious threshold semantics (e.g., "reorder if inventory < safety stock").
- **InvManagement actions:** Continuous — same concern as CityLearn but lower-dimensional (3D). Discretization into bins (e.g., 0%, 25%, 50%, 75%, 100% of capacity) gives only 5³ = 125 actions — manageable.
- **Knapsack actions:** Discrete — excellent for trees.

### 2.3 SB3 Compatibility

**Moderate.** OR-Gym uses the **old** `gym` API (gym ≤ 0.21), NOT `gymnasium`.
- Last commit: **3 years ago** — the project is unmaintained.
- Uses `gym.Env`, not `gymnasium.Env` — requires `shimmy` compatibility layer or manual patching.
- No built-in SB3 wrapper.
- Examples use Ray/RLlib, not SB3.
- Some environments may have dtype issues with newer NumPy versions.

**Integration work required:**
1. Wrap with `gymnasium` compatibility shim
2. Potentially port the environment class to inherit from `gymnasium.Env`
3. Test all space definitions for SB3 compatibility
4. Handle episode termination semantics (old-style `done` bool vs new `terminated`/`truncated`)

### 2.4 Conflict Injection Potential

**Good for supply chain envs, limited for combinatorial envs.**

**InvManagement — Conflict Scenarios:**

**S1: Profit vs Service Level (backlog penalty vs holding cost)**
- **Reward:** Period profit: `r_t = sales_revenue - procurement_cost - holding_cost - backlog_cost`
- **Cost:** Service level violation: `c_t = unfulfilled_demand / total_demand` (fraction of unmet demand)
- **Conflict:** Holding minimal inventory maximizes profit (low holding cost). But stockouts spike when demand surges. Reward says "lean inventory" while cost says "buffer for demand uncertainty."
- **Label:** `1` when `inventory_position < safety_stock_threshold AND demand_forecast > mean_demand`

**S2: Cost Efficiency vs Supply Chain Resilience**
- **Reward:** Minimize total supply chain cost
- **Cost:** Supply concentration risk: `c_t = max(single_supplier_dependency) / total_supply`
- **Conflict:** Cheapest option may be single-sourcing. Cost penalizes single-point-of-failure.

**S3: Speed vs Holding Cost (express vs standard replenishment)**
- **Reward:** Minimize total logistics cost
- **Cost:** Customer wait time: `c_t = avg_backlog_age / max_acceptable_wait`
- **Conflict:** Longer lead times are cheaper but customer wait goes up.

**Knapsack — Limited conflict potential:**
- The knapsack problem is inherently single-objective (maximize value within weight). Adding a meaningful secondary constraint (e.g., "diversity of items") feels artificial.

**Obs-based label feasibility:** Inventory positions and pipeline states are fully observable — labels can be deterministic functions of state. This works well for Split-RL.

### 2.5 Scoring

| Criterion | Score | Notes |
|-----------|-------|-------|
| Tree-Friendliness | **3.5/5** | Obs: good axis-aligned features. Actions: mostly continuous (needs discretization). Knapsack: discrete. |
| SB3 Compatibility | **2/5** | Old gym API, unmaintained, no SB3 wrapper, requires porting effort |
| Conflict Injection | **3.5/5** | Supply chain envs have natural conflicts. Combinatorial envs (knapsack, TSP) are poor fits. |
| Implementation Effort | **2.5/5** | Need to port to gymnasium, handle API breaks, small community, 3 years stale |
| **Overall** | **2.9/5** | |

---

## 3. Overcooked AI

**Source:** https://github.com/HumanCompatibleAI/overcooked_ai
**Domain:** Cooperative multi-agent coordination (cooking game)
**License:** MIT

### 3.1 Environment Description

Overcooked-AI is a 2-player cooperative cooking game. Agents navigate a grid kitchen, pick up ingredients, place them in pots, wait for cooking, then deliver soups. The goal is to deliver as many soups as possible within a time horizon (typically 400 steps). Multiple kitchen layouts exist with different spatial constraints.

### 3.2 Observation Space

**Multiple featurization options:**

| Featurization | Shape | Description |
|---------------|-------|-------------|
| Lossless encoding | `(H, W, 26)` | Full grid state as multi-channel binary tensor (player positions, orientations, objects, pot states) |
| Custom featurize | `(62,)` to `(96,)` | Flattened vector with distances to objects, pot states, held items, etc. |

**Features in custom vector featurization:**
- Player position (x, y)
- Player orientation (4-directional one-hot)
- Held object type (one-hot)
- Nearest pot location, pot state (empty/cooking/ready, items in pot)
- Nearest ingredient locations
- Nearest serving location
- Other player info (same features)

**Tree-friendliness:**
- ⚠️ **Mixed.** The lossless encoding is a 3D tensor — terrible for trees (image-like). The custom vector featurization is better but contains encoded positions, orientations, and multi-hot object states. Many features are **spatial / relational** rather than axis-aligned semantic features.
- Distance features and Boolean flags are decent tree splits, but the core challenge is **positional reasoning** — "am I adjacent to the pot?" requires spatial understanding that trees handle poorly without engineered features.
- Would require custom featurization to extract tree-friendly features (distances, counts, Boolean states).

### 3.3 Action Space

**Discrete(6):** North, South, East, West, Interact, Stay.

**Tree-friendliness:** Excellent — small discrete action space, perfect for GBRL.

### 3.4 SB3 Compatibility

**Poor to Moderate.**
- Overcooked provides its own `Overcooked(gymnasium.Env)` wrapper — technically Gymnasium-compatible.
- **However**: It's fundamentally a **2-player** environment. The Gymnasium wrapper treats it as a single-agent env where the "action" is a tuple of both agents' actions.
- You either need to:
  1. Fix one agent's policy (e.g., always stay/random) and train the other — artificial, not meaningful
  2. Train both agents jointly — action space becomes `Discrete(6) × Discrete(6)` = 36 joint actions
  3. Self-play — requires custom training loop (the repo's DRL training code is **deprecated**)
- PettingZoo wrapper was **discontinued** (commented out in source code).
- The repo's BC and RL training pipelines are **explicitly deprecated** (see README: "NOTE + LOOKING FOR CONTRIBUTORS: DRL and BC implementations are now deprecated").
- No official SB3 integration.

### 3.5 Conflict Injection Potential

**Moderate — inherent coordination tension but hard to formalize as cost.**

The game is fully cooperative (shared reward), so there's no built-in conflict. Potential injected conflicts:

**S1: Speed vs Ingredient Quality**
- **Reward:** Soups delivered per episode
- **Cost:** Quality penalty: `c_t = 1 if soup delivered with wrong recipe`
- **Conflict:** Fast soups (any 3 ingredients) vs correct recipes (specific combinations). Could add recipe requirements where optimal speed conflicts with recipe adherence.
- **Problem:** Requires modifying the core MDP logic, which is complex and tightly coupled.

**S2: Individual Efficiency vs Coordination (for multi-agent)**
- **Reward:** Total soups delivered
- **Cost:** Collision/idle penalty: `c_t = 1 if agents block each other`
- **Conflict:** Greedy paths maximize individual throughput but cause traffic jams.
- **Problem:** The "conflict" is more about coordination failure than a meaningful policy tradeoff. Label definition is unclear — when does the cost objective "own" the gradient?

**S3: Throughput vs Dish Diversity**
- **Reward:** Total soups delivered
- **Cost:** Penalty for making same soup type repeatedly
- **Conflict:** Specialization (always make onion soup) vs variety (alternate recipes).
- **Problem:** Artificial — not grounded in a real domain tension.

**Obs-based label feasibility:** Possible but the spatial reasoning makes labels hard to define from raw observations. "Should cost own the gradient?" depends heavily on the full spatial configuration, which is hard to threshold.

### 3.6 Scoring

| Criterion | Score | Notes |
|-----------|-------|-------|
| Tree-Friendliness | **2.5/5** | Actions: excellent (Discrete(6)). Obs: spatial/relational, needs heavy feature engineering for trees |
| SB3 Compatibility | **1.5/5** | 2-player, deprecated RL code, no SB3 wrapper, PettingZoo discontinued |
| Conflict Injection | **2/5** | Natural conflicts are coordination-based (hard to formalize), custom conflicts feel artificial |
| Implementation Effort | **1.5/5** | Significant wrapper work, multi-agent handling, core MDP modifications for conflicts |
| **Overall** | **1.9/5** | |

---

## 4. MiniHack

**Source:** https://minihack.readthedocs.io/
**Domain:** Roguelike dungeon navigation and skill acquisition (built on NetHack/NLE)
**License:** Apache 2.0

### 4.1 Environment Description

MiniHack provides procedurally generated mini-environments based on the NetHack Learning Environment (NLE). Environments range from simple room navigation to complex multi-skill tasks requiring inventory management, combat, and puzzle-solving. Environment categories:

| Category | Examples | Action Size | Difficulty |
|----------|----------|-------------|------------|
| Navigation | Room, Corridor, MazeWalk, River, HideNSeek | 8–12 actions | Low–Medium |
| Skill Acquisition | Lava Crossing, Wand of Death, Quest | 75 actions | High |
| Ported | MiniGrid tasks, Boxoban, Sokoban | Varies | Medium |

### 4.2 Observation Space

MiniHack supports a **dictionary observation space** with many options:

| Key | Shape | Description |
|-----|-------|-------------|
| `glyphs` | `(21, 79)` | Glyph IDs (0–5991) for each map cell |
| `glyphs_crop` | `(N, N)` | Agent-centered NxN crop (default 9×9) |
| `chars` | `(21, 79)` | ASCII character codes |
| `chars_crop` | `(N, N)` | Cropped character view |
| `blstats` | `(27,)` | Player stats: position, HP, strength, level, gold, etc. |
| `message` | `(256,)` | UTF-8 encoded in-game message |
| `screen_descriptions` | `(21, 79, 80)` | Text descriptions per cell |
| `pixel` | `(H, W, 3)` | Visual rendering |

**Tree-friendliness:**
- ⚠️ **Poor for raw observations.** `glyphs` and `chars` are 2D integer matrices (essentially image-like). Glyph IDs range 0–5991 and encode entity types — not axis-aligned meaningful features.
- `blstats` (27-dim vector) is the **only** tree-friendly component: HP, max_HP, strength, dexterity, x_pos, y_pos, gold, experience level, hunger. These are classic threshold-splittable features.
- For tree-based methods, you'd need extensive **feature engineering**: flatten the crop, extract "distance to nearest monster", "number of adjacent walls", "items in inventory count", etc.
- The core observation format is designed for CNN/LSTM architectures, not decision trees.

### 4.3 Action Space

**Navigation tasks:** `Discrete(8–12)` — compass directions + kick/open/search.
**Skill tasks:** `Discrete(75)` — full NetHack command set.

**Tree-friendliness:**
- Navigation tasks: Excellent (small discrete).
- Skill tasks: Moderate (75 actions is large but still discrete).
- ⚠️ Some actions are **auto-regressive** — selecting "PUTON" requires follow-up selections from inventory menus. This is fundamentally incompatible with flat RL action selection.

### 4.4 SB3 Compatibility

**Poor.**
- MiniHack natively supports **TorchBeast** (IMPALA) and **RLlib** — NOT SB3.
- Dictionary observation space is not natively handled by SB3 `MlpPolicy`.
- Would need a custom wrapper to:
  1. Flatten/select observation keys (e.g., only use `blstats` + flattened `glyphs_crop`)
  2. Convert Dict space to Box space
  3. Handle the complex death/restart mechanics of NLE
- NLE (the backend) has specific **system-level dependencies**: requires building from source on Linux, depends on ncurses/bison/flex, and pins specific library versions.
- **Installation on cluster:** Requires `nle` which needs C compilation of NetHack. Non-trivial in containerized/cluster environments.

### 4.5 Conflict Injection Potential

**Moderate — some natural tensions exist in the roguelike domain.**

**S1: Exploration vs Self-Preservation (health management)**
- **Reward:** Sparse +1 for reaching the goal (staircase down)
- **Cost:** Health risk: `c_t = max(0, 1 - HP/max_HP)` (health fraction lost)
- **Conflict:** Fastest path to the goal goes through monsters (high reward). Safe path avoids combat (low cost). In corridor-battle tasks, the agent must decide: fight efficiently (risky, fast) or find detours (slow, safe).
- **Label:** `1` when `HP < HP_threshold AND monster_adjacent AND direct_path_to_goal_through_monster`
- **Quality:** Good! In MiniHack-CorridorBattle, the agent genuinely faces "fight or flee" at low HP.

**S2: Greed vs Speed (item collection vs goal completion)**
- **Reward:** +1 for reaching exit
- **Cost:** `c_t = valuable_items_visible_but_not_collected / total_visible_items`
- **Conflict:** Items (potions, scrolls) are useful but collecting them wastes turns. Reward (reach exit fast) vs cost (don't leave good items behind).
- **Label:** `1` when `valuable_item_adjacent AND exit_distance < threshold`

**S3: Resource Conservation vs Progress (consumable management)**
- **Reward:** Reach the goal
- **Cost:** Hunger/starvation risk: `c_t = hunger_level / max_hunger`
- **Conflict:** Moving burns food. Eating takes a turn. Reward says "move towards goal" while cost says "eat before you starve."
- **Label:** `1` when `hunger > critical_threshold AND food_in_inventory`

**Obs-based label feasibility:** ⚠️ The problem is that the relevant state features (monster adjacency, item locations, path distances) are NOT in `blstats`. They live in the `glyphs` matrix, which requires spatial parsing to extract into tree-friendly features. Labels that depend on "is a monster adjacent?" require extracting this from the glyph grid — doable but adds wrapper complexity.

### 4.6 Scoring

| Criterion | Score | Notes |
|-----------|-------|-------|
| Tree-Friendliness | **2/5** | Actions: good (discrete). Obs: mostly grid-based, only `blstats` (27-dim) is tree-friendly. Heavy feature engineering needed. |
| SB3 Compatibility | **1.5/5** | No SB3 support, Dict obs space, NLE system deps, complex installation |
| Conflict Injection | **3/5** | Roguelike domain has natural risk/reward tensions, but label extraction from grid obs is complex |
| Implementation Effort | **1.5/5** | NLE compilation, custom obs wrapper, feature engineering, SB3 Dict space handling |
| **Overall** | **2.0/5** | |

---

## Comparative Summary

| Criterion | CityLearn | OR-Gym | Overcooked AI | MiniHack |
|-----------|-----------|--------|---------------|----------|
| **Tree-Friendliness** | ⭐⭐⭐⭐ | ⭐⭐⭐½ | ⭐⭐½ | ⭐⭐ |
| **SB3 Compatibility** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐½ | ⭐½ |
| **Conflict Injection** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐½ | ⭐⭐ | ⭐⭐⭐ |
| **Implementation Effort** | ⭐⭐⭐⭐ | ⭐⭐½ | ⭐½ | ⭐½ |
| **Overall** | **4.5/5** | **2.9/5** | **1.9/5** | **2.0/5** |

### Quick Comparison: Action Spaces

| Environment | Action Space | Type | Tree-Friendly? |
|-------------|-------------|------|----------------|
| SUMO (reference) | `Discrete(5)` or `Discrete(8)` | Discrete | ✅ Excellent |
| CityLearn | `Box([-1,1], shape=(6-12,))` | Continuous | ⚠️ Needs discretization |
| OR-Gym InvMgmt | `Box(low=0, high=cap, shape=(3,))` | Continuous | ⚠️ Needs discretization |
| OR-Gym Knapsack | `Discrete(200)` | Discrete | ✅ Good (large) |
| Overcooked AI | `Discrete(6)` per agent | Discrete | ✅ Excellent |
| MiniHack Navigation | `Discrete(8-12)` | Discrete | ✅ Excellent |
| MiniHack Skills | `Discrete(75)` | Discrete | ⚠️ Large + autoregressive |

### Quick Comparison: Observation Spaces

| Environment | Obs Dim | Feature Type | Tree-Friendly? |
|-------------|---------|--------------|---------------- |
| SUMO (reference) | 18–33 + extensions | Normalized scalar (densities, queues, phases) | ✅ Excellent |
| CityLearn | 60–80 | Scalar (temps, SOCs, prices, demands) | ✅ Excellent |
| OR-Gym InvMgmt | 33 | Integer (inventory positions, pipeline) | ✅ Good |
| OR-Gym Knapsack | 200×2 + 1 | Integer (weights, values, capacity) | ⚠️ Matrix-like |
| Overcooked AI | 62–96 (vector) or (H,W,26) (tensor) | Mixed spatial/categorical | ⚠️ Needs engineering |
| MiniHack | 27 (blstats) + (9,9) grid | Integer stats + grid matrix | ❌ Mostly grid-based |

---

## Recommendations

### Tier 1 — **Strong Recommendation: CityLearn**

CityLearn is the clear winner across all criteria. The reasons:

1. **Natural multi-objective domain:** Energy cost, thermal comfort, carbon emissions, grid stability, and resilience are genuinely competing objectives. This is not an artificial injection — building operators face these tradeoffs daily.

2. **Excellent obs features for trees:** Temperature readings, SOC levels, electricity prices, solar generation — every feature has a natural threshold interpretation. A tree can learn "if SOC < 0.3 AND price < 0.05, charge" — exactly how building operators think.

3. **Clean SB3 integration:** Official wrapper, tested, `pip install CityLearn` and go.

4. **Label quality potential:** Labels can be defined as deterministic functions of observable features (price regimes × comfort violations), matching the SUMO label philosophy exactly.

5. **Research relevance:** Smart grid and demand response are active research areas where RL is gaining traction but safety/comfort constraints are paramount.

**Main challenges to address:**
- Continuous action space → either discretize or use continuous-action GBRL
- Long episodes (8,760 steps) → may need truncated episodes or periodic resets
- Per-building differences in schemas → standardize a reference schema for reproducibility

### Tier 2 — **Conditional Recommendation: OR-Gym (supply chain envs only)**

OR-Gym's `InvManagement` and `NetworkManagement` environments are decent candidates, but the implementation burden is significant:

- Must port from `gym` to `gymnasium` (the library is 3 years stale)
- Supply chain conflicts (service level vs cost, resilience vs efficiency) are natural and meaningful
- Obs features are tree-friendly (inventory levels, pipeline quantities)
- But the community is small and the codebase may have hidden bugs

**Recommended only if:** you want a second domain beyond CityLearn and are willing to invest in porting/maintaining the wrapper. Consider implementing a simpler custom inventory problem from scratch instead of depending on the stale OR-Gym package.

### Tier 3 — **Not Recommended: Overcooked AI and MiniHack**

Both environments have fundamental mismatches with the Split-RL + GBRL paradigm:

**Overcooked AI:**
- Multi-agent nature is the core challenge, not cost/reward tradeoffs
- RL training support is deprecated by the maintainers
- Conflicts are coordination-based, not constraint-based — poor fit for Split-RL's label-based gradient routing
- Spatial observations need heavy feature engineering for trees

**MiniHack:**
- Observations are grid/matrix-based — antithetical to tree methods
- NLE system dependencies make cluster deployment painful
- While roguelike risk/reward tensions are interesting, extracting label-relevant state from glyph grids adds enormous wrapper complexity
- The 75-action autoregressive action space (for skill tasks) is incompatible with flat policy heads

### Priority Action Plan

1. **Immediate:** Implement CityLearn wrapper (`env/citylearn.py`) with:
   - Discretized action space (5–11 bins per device)
   - 3 conflict scenarios (comfort-vs-cost, peak-vs-degradation, grid-vs-solar)
   - State-based label functions
   - Reference schema (3-building, 2023 challenge dataset)

2. **If second domain needed:** Port `InvManagement-v0` from OR-Gym to a standalone `env/supply_chain.py`:
   - Migrate to Gymnasium API
   - Add service-level cost + inventory-risk label
   - Discretize to 5 bins per echelon (125 joint actions for 3-echelon)

3. **Defer:** Overcooked and MiniHack — table for future consideration if the project needs non-tabular/spatial domains.
