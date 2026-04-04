# SUMO-RL Environment Reference

## 1. What Is This Environment?

A **4×4 grid of traffic intersections** where 16 RL agents each control one traffic light. All agents share a single policy (shared-policy parameter sharing). The simulator is [SUMO](https://sumo.dlr.de/) (Simulation of Urban MObility), wrapped via [sumo-rl](https://github.com/LucasAlegre/sumo-rl) and [PettingZoo](https://pettingzoo.farama.org/) into an SB3-compatible VecEnv.

**This is NOT a single intersection.** It's a full urban network with 16 coordinated signals, ~1500–2500 vehicles, and realistic traffic dynamics (car-following, lane-changing, yellow phases, queue spillback between intersections).

---

## 2. Available Networks

| Config ID | Network | Agents | Actions | Obs Dim | Lanes/Agent | Topology |
|---|---|---|---|---|---|---|
| `sumo-grid4x4-v0` | Synthetic grid | 16 | Discrete(8) | 33 | 12 (3×4 directions) | **Symmetric** — all directions equal (3 lanes each, same speed) |
| `sumo-arterial4x4-v0` | Synthetic arterial | 16 | Discrete(5) | 18 | 6 (2+2 ML, 1+1 side) | **Asymmetric** — arterial EW (2 lanes, 20 m/s) vs cross-street NS (1 lane, 11 m/s) |
| `sumo-cologne{1,3,8}-v0` | Real-world (Cologne) | 1/3/8 | varies | varies | varies | Real intersection topology |
| `sumo-ingolstadt{1,7,21}-v0` | Real-world (Ingolstadt) | 1/7/21 | varies | varies | varies | Real intersection topology |

**For SPLIT-RL experiments, we use `grid4x4` and `arterial4x4`** because they have uniform obs/action spaces across all 16 agents (required for shared-policy training).

### Network Geometry

**grid4x4** (1500m × 1500m):
```
D3 ── C3 ── B3 ── A3      Each intersection: 12 incoming lanes
│     │     │     │        (3 lanes × 4 directions)
D2 ── C2 ── B2 ── A2      All lanes: 273m long, 13.9 m/s speed limit
│     │     │     │        Demand: 1473 vehicles, 255 OD pairs (uniform)
D1 ── C1 ── B1 ── A1
│     │     │     │
D0 ── C0 ── B0 ── A0
```

**arterial4x4** (750m × 750m):
```
nt13 ═══ nt14 ═══ nt15 ═══ nt16    ═══ = arterial (2 lanes, 20 m/s)
 │        │        │        │       │  = cross-street (1 lane, 11 m/s)
nt9  ═══ nt10 ═══ nt11 ═══ nt12    
 │        │        │        │       Each intersection: 6 incoming lanes
nt5  ═══ nt6  ═══ nt7  ═══ nt8     (4 mainline + 2 side)
 │        │        │        │       Demand: 2484 vehicles, 8 OD pairs
nt1  ═══ nt2  ═══ nt3  ═══ nt4     (asymmetric — more EW than NS traffic)
```

---

## 3. What the Agent Controls (Action Space)

**The agent controls WHICH green phase is active at its intersection.**

Each action selects a green phase — a combination of signal states (Green/red) for all the links through the intersection. The agent does NOT control individual vehicles, routing, speed, or lane assignment.

### arterial4x4 — Discrete(5)

| Action | Green Phase | Serves | Direction |
|---|---|---|---|
| 0 | `GGgsrrGGgsrr` | Side-street lanes (NS through + left) | **SIDE** |
| 1 | `srrsrGsrrsrG` | Mainline lane 1 each direction (EW left turn) | **ML** |
| 2 | `srrGGrsrrGGr` | Mainline lane 0 each direction (EW through) | **ML** |
| 3 | `srrGGGsrrsrr` | Both mainline lanes, one direction (eastbound) | **ML** |
| 4 | `srrsrrsrrGGG` | Both mainline lanes, other direction (westbound) | **ML** |

**Key asymmetry:** Only action 0 serves side-streets. Actions 1–4 all serve mainline in different configurations. This is why throughput-maximizing policies naturally starve side-streets.

### grid4x4 — Discrete(8)

| Action | Serves | Direction |
|---|---|---|
| 0 | NS through (lanes 0,1,6,7) | **SIDE** |
| 1 | NS left turn (lanes 2,8) | **SIDE** |
| 2 | N through+left (lanes 0,1,2) | **SIDE** |
| 3 | S through+left (lanes 6,7,8) | **SIDE** |
| 4 | EW through (lanes 3,4,9,10) | **ML** |
| 5 | EW left turn (lanes 5,11) | **ML** |
| 6 | W through+left (lanes 9,10,11) | **ML** |
| 7 | E through+left (lanes 3,4,5) | **ML** |

**In grid4x4 the split is symmetric** — 4 actions for NS, 4 for EW. Direction assignment (ML vs SIDE) is arbitrary since all lanes have equal capacity.

### What Happens When the Agent Acts

```
t=0:   Agent selects action (green phase index)
       ├── IF same phase OR min_green+yellow not elapsed → NO-OP (keep current)
       └── IF different phase → Start YELLOW transition
t=0..2: Yellow phase (2 seconds)
t=2:   Switch to new green phase
t=0..5: SUMO simulates 5 seconds of traffic (delta_time=5)
t=5:   Agent acts again
```

**Timing constraints (enforced by sumo-rl, not by us):**
- `min_green = 5s` — cannot switch until 5s after last switch
- `yellow_time = 2s` — automatic yellow transition on every switch
- `delta_time = 5s` — agent acts every 5 sim-seconds = **720 steps per episode** (3600s / 5s)
- Guard: `time_since_last_change < yellow_time + min_green` → action ignored
- At defaults (5+2=7s with delta_time=5s), the earliest re-switch is 2 steps (10s) after a phase change

---

## 4. Observation Space

```
obs = [phase_one_hot | min_green | lane_densities | lane_queues]
         n_phases       1          n_lanes          n_lanes
```

All values are in `[0, 1]`.

### arterial4x4 — 18 dimensions

| Index | Feature | Description |
|---|---|---|
| `[0:5]` | `phase_one_hot` | Which of 5 green phases is active |
| `[5]` | `min_green` | 1 if `time_since_last_change ≥ min_green + yellow_time`, else 0 |
| `[6]` | `density[0]` = den_side_N | Side-street from north |
| `[7]` | `density[1]` = den_ML_E0 | Mainline eastbound lane 0 |
| `[8]` | `density[2]` = den_ML_E1 | Mainline eastbound lane 1 |
| `[9]` | `density[3]` = den_side_S | Side-street from south |
| `[10]` | `density[4]` = den_ML_W0 | Mainline westbound lane 0 |
| `[11]` | `density[5]` = den_ML_W1 | Mainline westbound lane 1 |
| `[12]` | `queue[0]` = q_side_N | Side-street from north |
| `[13]` | `queue[1]` = q_ML_E0 | Mainline eastbound lane 0 |
| `[14]` | `queue[2]` = q_ML_E1 | Mainline eastbound lane 1 |
| `[15]` | `queue[3]` = q_side_S | Side-street from south |
| `[16]` | `queue[4]` = q_ML_W0 | Mainline westbound lane 0 |
| `[17]` | `queue[5]` = q_ML_W1 | Mainline westbound lane 1 |

**Density** = `vehicles_on_lane / lane_capacity` (capacity = `lane_length / (MIN_GAP + avg_vehicle_length)`)
**Queue** = `halting_vehicles / lane_capacity` (halting = speed < 0.1 m/s)

### grid4x4 — 33 dimensions

Same layout: `[phase(8), min_green(1), density(12), queue(12)]` with 12 lanes (3 per direction × 4 directions).

---

## 5. Reward Functions

| Name | Formula | Used as |
|---|---|---|
| `diff-waiting-time` (default sumo-rl) | `-(total_wait_now - total_wait_prev)` | Positive when total waiting decreases |
| `throughput` | `-pressure = outgoing_vehicles - incoming_vehicles` | Positive when more vehicles flow out than in |
| `mainline` | `-(mainline_wait_now - mainline_wait_prev)` | Only counts EW (mainline) lanes |

**For SPLIT-RL, we use `throughput` as the honest reward** — it's what a transit authority actually wants to maximize (move vehicles through the network).

---

## 6. What We CAN Do (at Runtime via TraCI)

### Vehicles

| Capability | API | Notes |
|---|---|---|
| **Define new vehicle types** | `vehicletype.copy('type1', 'emergency')` then `.setVehicleClass()`, `.setMaxSpeed()`, `.setLength()`, `.setColor()` | Emergency, bus, truck — any SUMO vClass |
| **Inject vehicles** on any route | `route.add(routeID, edges)` then `vehicle.add(vehID, routeID, typeID=...)` | Can inject at any time during simulation |
| **Detect vehicle class per lane** | `lane.getLastStepVehicleIDs(lane)` then `vehicle.getVehicleClass(veh)` | Iterate vehicles to count emergency/bus/etc |
| **Color vehicles** | `vehicle.setColor(vehID, (r,g,b,a))` | For visualization |
| **Set vehicle speed** | `vehicle.setSpeed(vehID, speed)` | Override car-following model |
| **Reroute vehicles** | `vehicle.changeTarget(vehID, edgeID)` or `vehicle.setRoute(vehID, edges)` | Change destination mid-trip |
| **Query per vehicle**: position, speed, wait time, lane, route, type, class | Various `vehicle.get*()` | Full state of every vehicle |

### Lanes & Edges

| Capability | API | Notes |
|---|---|---|
| **Restrict lane access** (bus-only, etc.) | `lane.setAllowed(laneID, ['bus'])` or `lane.setDisallowed(laneID, ['passenger'])` | **Runtime lane restriction** |
| **Change speed limit** | `lane.setMaxSpeed(laneID, speed)` or `edge.setMaxSpeed(edgeID, speed)` | Per-lane or per-edge |
| **Query lane stats**: vehicle count, halting count, mean speed, occupancy | `lane.getLastStep*()` | Per-lane aggregate stats |
| **Query lane geometry** | `lane.getShape(laneID)` → `[(x1,y1), (x2,y2)]` | For visualization |

### Traffic Lights

| Capability | API | Notes |
|---|---|---|
| **Read current phase** | `trafficlight.getPhase(tlID)` | Already in obs (one-hot) |
| **Force phase** | `trafficlight.setPhase(tlID, phase)` | Bypass RL agent |
| **Change signal program** | `trafficlight.setProgramLogic(tlID, logic)` | Replace entire program |
| **Set individual link states** | `trafficlight.setRedYellowGreenState(tlID, state)` | Per-connection override |

### Simulation-Level

| Capability | API | Notes |
|---|---|---|
| **Current time** | `simulation.getTime()` | Simulation clock |
| **Emergency stop count** | `simulation.getEmergencyStoppingVehiclesNumber()` | Global failure event |
| **Teleport count** | `simulation.getStartingTeleportNumber()` | Deadlock resolution event |
| **Total vehicles** | `vehicle.getIDCount()` | Currently in network |

---

## 7. What We CANNOT Do

| Cannot | Why |
|---|---|
| **Add/remove lanes or roads** | Network geometry is fixed at load time (defined in `.net.xml`) |
| **Add new intersections** | Same — topology is static |
| **Control individual vehicle routing decisions** | Vehicles follow pre-defined routes. We can reroute them via TraCI but not control turn-by-turn decisions for ALL vehicles (too expensive) |
| **Change intersection geometry** (turn lanes, etc.) | Fixed in `.net.xml` |
| **Add pedestrians retroactively** | Needs walkingarea definitions in the network file |
| **Get per-vehicle-class lane counts directly** | No `lane.getLastStepEmergencyCount()` API — must iterate vehicle IDs |
| **Run multiple SUMO instances with libsumo** | `LIBSUMO_AS_TRACI=1` is single-instance only. Multiple envs need traci (TCP) |

---

## 8. Observation Extension

Our wrapper (`SumoRewardCostWrapper`) can **extend the observation** with additional features. The mechanism:

1. Override `observation_space()` to return a larger Box
2. In `_transform_obs()`, append new features to the raw sumo-rl observation
3. Features we can add per-agent (per intersection):

| Feature | Dims | Source | Example |
|---|---|---|---|
| `has_emergency[lane_i]` | n_lanes | Iterate vehicles on each lane, check vClass | Binary: is there an ambulance approaching? |
| `emergency_wait[lane_i]` | n_lanes | Sum wait time of emergency vehicles per lane | How long has the ambulance been waiting? |
| `has_bus[lane_i]` | n_lanes | Same, for vClass=bus | Binary: is there a bus approaching? |
| `bus_count[lane_i]` | n_lanes | Count of buses per lane | How many buses? |
| `bus_wait[lane_i]` | n_lanes | Sum wait time of buses per lane | Bus delay |
| `lane_restricted[lane_i]` | n_lanes | Check lane.getAllowed() | Binary: is this a bus lane? |
| `speed_limit[lane_i]` | n_lanes | `lane.getMaxSpeed()` / base_speed | Normalized speed limit |
| `time_of_day` | 1 | `simulation.getTime() / total_seconds` | For time-varying demand patterns |
| `steps_since_phase_change` | 1 | Track internally | How long current phase has been held |

These get appended after the standard obs: `[phase(5/8), min_green(1), density(n), queue(n), <NEW FEATURES>]`

---

## 9. What the Agent CANNOT See (by default)

These are NOT in the observation and could be added:

- Which vehicle types are on which lanes (emergency, bus, truck)
- Accumulated wait time per vehicle (only lane-aggregate via obs)
- What the DOWNSTREAM intersection is doing (no inter-agent communication)
- Exact vehicle positions within a lane (only aggregate density/queue)
- Route information (where vehicles are going)
- Whether a lane is restricted

This is exactly what makes constraint scenarios interesting — we add partial information to the obs and create a constraint that the agent must learn to satisfy.

---

## 10. Constraint Scenario Design Space

Given all of the above, here are the **scenario modification axes** available:

### A. Vehicle Injection

Inject special vehicle types at runtime to create constraint triggers:

| Vehicle Type | vClass | Properties | Constraint Story |
|---|---|---|---|
| Ambulance | `emergency` | Fast (30 m/s), long (7m), red color | Must get green immediately (preemption) |
| Bus | `bus` | Slow (15 m/s), long (12m), blue color | Must not wait more than τ seconds (TSP) |
| Truck | `truck` | Slow accel (1.5), long (15m) | Creates rolling bottlenecks on lanes |
| VIP convoy | `authority` | Medium speed, drives in platoon | Priority corridor constraint |

### B. Lane Modification

Modify lane properties at episode start or dynamically:

| Modification | API | Constraint Story |
|---|---|---|
| Bus-only lane | `lane.setAllowed(lane, ['bus'])` | One mainline lane is reserved → reduces effective capacity |
| Speed reduction zone | `lane.setMaxSpeed(lane, 5.0)` | Construction zone / school zone → congestion building |
| Lane closure | `lane.setDisallowed(lane, ['passenger', 'bus', 'truck'])` | Incident → all traffic rerouted |

### C. Demand Modification

Modify traffic demand during episode:

| Modification | How | Constraint Story |
|---|---|---|
| Rush-hour surge | Inject platoons at peak times | Sudden demand spike on one direction |
| Event traffic | All vehicles routed through one corridor | Stadium/concert — one-directional overload |
| Random incidents | Remove a route temporarily | Accidents → rerouting around blocked area |

### D. Observation Extension

Each scenario adds specific features to the obs that the tree can split on:

| Scenario | Added obs features | Label can split on |
|---|---|---|
| Emergency preemption | `has_emergency[6 lanes]` | Binary: ambulance present on non-served lane |
| Bus priority | `has_bus[6]`, `bus_wait[6]` | 2D: bus present AND wait > threshold |
| Speed zones | `speed_limit[6]` | Which lanes are slowed down |

---

## 11. Summary: What the Agent Does and Doesn't Control

```
┌─────────────────────────────────────────────────────┐
│                    THE AGENT                         │
│                                                      │
│  Controls:   Which green phase is active             │
│              (one discrete action per intersection)  │
│                                                      │
│  Sees:       Current phase (one-hot)                 │
│              Min-green flag (can I switch?)           │
│              Lane densities (how full?)               │
│              Lane queues (how many stopped?)          │
│              + Extended features (emergency, bus...)  │
│                                                      │
│  Cannot:     Route vehicles                          │
│              Control vehicle speed                   │
│              Change road layout                      │
│              Communicate with other intersections    │
│              See downstream conditions               │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                    WE CONTROL                        │
│             (scenario design / wrapper)              │
│                                                      │
│  Inject:     Emergency vehicles, buses, trucks       │
│  Modify:     Lane restrictions, speed limits         │
│  Extend:     Observation features                    │
│  Define:     Cost function + label                   │
│  Configure:  Injection rates, thresholds, timing     │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                    SUMO CONTROLS                     │
│                (simulator physics)                   │
│                                                      │
│  Vehicles:   Car-following (Krauss model)            │
│              Lane changing (LC2013)                   │
│              Yellow/red compliance                    │
│              Route following                         │
│  Network:    Queue spillback between intersections   │
│              Congestion propagation                  │
│              Emergency stops / teleports (deadlock)  │
└─────────────────────────────────────────────────────┘
```

---

## 12. Experiment Plan: Constrained SUMO Scenarios

### Design Principles

1. **Base reward is fixed**: `diff-waiting-time` (original SUMO-RL default). No reward hacking.
2. **Conflict comes from task constraints**, not from reward engineering.
3. **Each scenario adds minimal obs extensions** — only what the agent needs to see the constraint trigger.
4. **Label is state-based and obs-predictable** — a tree split can identify the constraint region exactly.
5. **Two networks × two scenarios = four experiments.**

```
                    grid4x4 (symmetric)       arterial4x4 (asymmetric)
                   ┌───────────────────┐     ┌────────────────────────┐
  Bus priority     │  G1: Random bus   │     │  A1: Random bus        │
                   │      routes       │     │      routes            │
                   └───────────────────┘     └────────────────────────┘
  Emergency        │  G2: Random emg   │     │  A2: Random emg        │
  preemption       │      routes       │     │      routes            │
                   └───────────────────┘     └────────────────────────┘
```

---

### Scenario 1: Bus Priority (Transit Signal Priority)

#### What changes in the simulation

- **Vehicle injection**: Buses (vClass=`bus`, length=12m, maxSpeed=15 m/s) are injected on **random routes** through the network at a controlled rate (~1 bus every 60–90s).
- **Routes are randomized each episode**: at episode start, pick random origin→destination pairs from the available boundary nodes and create routes. Buses travel through the full network — sometimes EW, sometimes NS, sometimes diagonal.
- **Both networks (A1, G1)**: Same injection logic. The conflict is NOT tied to a fixed direction. It comes from the fact that diff-waiting-time counts by vehicle count (10 cars > 1 bus), while the constraint says "serve the bus regardless of how many cars are waiting on other lanes."
- The constraint region **moves around** — sometimes the bus is on a mainline lane, sometimes on a side lane. The tree must learn to split on `has_bus[lane_i]` features directly, not memorize a fixed direction.

#### Observation extension

| Feature | Dims | Range | Description |
|---|---|---|---|
| `has_bus[lane_i]` | n_lanes (6 or 12) | {0, 1} | Binary: is there at least one bus on this incoming lane? |
| `bus_wait[lane_i]` | n_lanes (6 or 12) | [0, 1] | Max bus wait time on this lane, normalized by `T_bus` (capped at 1.0) |

Total obs extension: **+12 dims** (arterial) or **+24 dims** (grid).

#### Cost function

```
cost = max over all incoming lanes of:
         bus_wait[lane_i] / T_bus    (if bus present on lane_i)
       = 0                            (if no bus on any lane)
```

Continuous in [0, 1]. Measures how badly the bus delay cap is being violated.

- `T_bus` = tunable threshold (e.g., 30s). If any bus waits > 30s, cost ≈ 1.0.

#### Label

```
label = 1  IF  any bus is present on an incoming lane
              AND  that lane is NOT served by the current green phase
              AND  that bus's wait time > T_warn
         0  otherwise
```

- `T_warn` = warning threshold (e.g., 15s), strictly less than `T_bus`.
- This fires BEFORE the constraint is hard-violated — the "decision frontier" where the agent should consider switching.

#### Why this creates a state-localized conflict with diff-waiting-time

- **In label=0 states** (no bus, or bus just arrived, or bus already being served): diff-waiting-time gradient and constraint gradient are aligned or irrelevant. The agent should optimize traffic flow normally.
- **In label=1 states** (bus waiting on an unserved lane past T_warn): diff-waiting-time says "keep serving the direction with more queued cars" (which is typically the mainline/dominant direction — 10+ cars outweigh 1 bus in the waiting-time sum). But the constraint says "switch to serve the bus lane NOW."
- **The contradiction is structural**: diff-waiting-time is per-vehicle and weighted by count. One bus cannot outweigh 10 cars in the reward. But the constraint values the bus separately.
- **A shared gradient update blurs this**: the rare bus-urgent samples get averaged with the dominant car-flow samples. SPLIT-RL can route bus-urgent samples to the cost objective and car-flow samples to the reward objective.

#### What the tree can split on

The label is a deterministic function of:
1. `has_bus[lane_i]` (is there a bus?) — binary obs feature
2. `bus_wait[lane_i]` > T_warn — threshold on continuous obs feature  
3. Current phase doesn't serve that lane — from `phase_one_hot`

A depth-2 tree can identify this region exactly: first split on `has_bus`, second split on `bus_wait > threshold`.

#### Key parameters to tune

| Parameter | Role | Starting value | Tuning rationale |
|---|---|---|---|
| Bus injection rate | Controls how often constraint fires | 1 bus / 60s / route | Too rare = label fires <5%. Too frequent = always firing. |
| `T_bus` (cost threshold) | How long before cost = 1.0 | 30s | Should be reachable under random policy but avoidable by good policy |
| `T_warn` (label threshold) | When to begin cost-routing | 15s | Should fire earlier than T_bus to give the agent a chance |
| Bus routes | Where buses appear | Random origin→destination each episode | Conflict comes from count mismatch, not fixed direction |
| Buses per episode | How many buses total | ~40–60 (1 per 60–90s) | Enough to fire label ~15-25% of steps |

---

### Scenario 2: Emergency Preemption

#### What changes in the simulation

- **Vehicle injection**: Emergency vehicles (vClass=`emergency`, length=7m, maxSpeed=30 m/s, color=red) are injected on **random routes** through the network at a low rate (~1 every 90–120s).
- **Both networks (A2, G2)**: Emergencies appear on random routes (any direction, any intersection). The conflict comes from whichever direction the agent happens NOT to be serving when the emergency arrives. Since the agent is optimizing diff-waiting-time, it's usually serving the direction with the most queued cars — the emergency is likely on a different lane.

#### Observation extension

| Feature | Dims | Range | Description |
|---|---|---|---|
| `has_emergency[lane_i]` | n_lanes (6 or 12) | {0, 1} | Binary: is there an emergency vehicle on this incoming lane? |
| `emergency_wait[lane_i]` | n_lanes (6 or 12) | [0, 1] | Emergency vehicle wait time on this lane, normalized by `T_emg` (capped at 1.0) |

Total obs extension: **+12 dims** (arterial) or **+24 dims** (grid).

#### Cost function

```
cost = max over all incoming lanes of:
         emergency_wait[lane_i] / T_emg    (if emergency present on lane_i)
       = 0                                  (if no emergency on any lane)
```

Continuous in [0, 1]. Tight deadline — `T_emg` is short (e.g., 10s).

#### Label

```
label = 1  IF  any emergency vehicle is present on an incoming lane
              AND  that lane is NOT served by the current green phase
         0  otherwise
```

No wait threshold needed — emergency preemption is immediate. The label fires the instant an emergency vehicle arrives on an unserved lane.

#### Why this creates a state-localized conflict with diff-waiting-time

- **In label=0 states** (no emergency): optimize normally. No constraint active.
- **In label=1 states** (emergency on unserved lane): diff-waiting-time says "keep serving the direction with more waiting vehicles." The emergency is ONE vehicle. But the constraint says "give it green NOW, regardless of how many cars are waiting on the current direction."
- **This is the sharpest possible rare-event conflict.** The label=1 region is tiny (~5-10% of timesteps), extremely concentrated, and requires the exact opposite action from what reward says.
- **A shared policy averages away the emergency response.** The 90-95% of normal-state samples dominate gradient updates. SPLIT-RL can isolate the emergency samples and route them to a dedicated cost leaf.

#### What the tree can split on

Single binary feature: `has_emergency[lane_i] = 1` AND `phase_one_hot` doesn't serve that lane.

A depth-1 tree can identify this: split on `max(has_emergency) > 0`. The cleanest possible partition.

#### Difference from bus priority

| | Bus Priority | Emergency Preemption |
|---|---|---|
| Frequency | Moderate (~15-25% label rate) | Rare (~5-10% label rate) |
| Urgency | Gradual (wait builds toward T_bus) | Immediate (must serve NOW) |
| Partition shape | 2D: bus_present AND wait > threshold | 1D: emergency_present on unserved lane |
| Cost shape | Gradual ramp | Sharp spike |
| Tree depth needed | 2 (has_bus + wait threshold) | 1 (has_emergency) |

These test SPLIT-RL at two different operating points: moderate-frequency gradual conflict vs rare sharp conflict.

#### Key parameters to tune

| Parameter | Role | Starting value | Tuning rationale |
|---|---|---|---|
| Emergency injection rate | Controls label frequency | 1 / 90s | Target ~5-10% label rate. Too rare = too few training samples. |
| `T_emg` (cost deadline) | How quickly cost ramps to 1.0 | 10s (= 2 agent steps) | Must be tight — emergencies can't wait |
| Emergency routes | Where they appear | Random origin→destination each episode | Conflict comes from serving the wrong direction |
| Emergency duration | How long vehicle is in network | Natural (drives through and exits) | No tuning needed — SUMO handles it |

---

### Verification Plan (Before Training)

For each of the 4 scenarios (A1, A2, G1, G2), run the diagnostic under **random policy** for 3 episodes:

| Check | Criterion | Purpose |
|---|---|---|
| Label rate | 10–40% of agent-steps | Enough conflict to learn from, not so much it's always on |
| Cost fires when label=1 | >80% co-occurrence | Label correctly identifies high-cost states |
| Label obs-predictable | Decision tree (depth 3) accuracy > 0.95 | A tree CAN learn the partition |
| Action disagreement in label=1 | reward-best ≠ constraint-best in >50% of label=1 states | The conflict is real, not just cosmetic |
| Reward-cost correlation | Not strongly positive | Passing the cost doesn't also improve reward (or: the cost is NOT free to satisfy) |

**If any scenario fails these checks**, adjust injection rate / thresholds before committing to training.

---

### Implementation Order

```
1. Implement bus injection + obs extension in SumoRewardCostWrapper
2. Implement bus cost/label functions  
3. Run A1 diagnostic → verify conflict checks
4. Run G1 diagnostic → verify conflict checks
5. Implement emergency injection + obs extension
6. Implement emergency cost/label functions
7. Run A2 diagnostic → verify conflict checks
8. Run G2 diagnostic → verify conflict checks
9. If all pass → proceed to training experiments
```

Each step is a checkpoint. We do NOT move to the next until the current one passes verification.
