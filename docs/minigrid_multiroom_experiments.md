# MiniGrid MultiRoomCorridor — Split-AWR Experiment Log

## Environment

**MiniGrid-MultiRoomCorridor-v0**: 19×19 grid, 4 rooms + narrow corridor.

```
┌──────────────────┐
│   TOP ROOM       │
│     [GOAL]       │
│                  │
├───────[🚪G]──────┤  ← green locked door
│     CORRIDOR     │
├───────[ ]────────┤
│  LEFT  │  MID  │ RIGHT │
│ [🔑P]  │[AGENT]│ [📦G]  │
│        │ [🔵]→ │  🚪P   │
└────────┴───────┴───────┘
```

- **Agent** starts in middle bottom room
- **Blue ball** blocks entrance to left room
- **Purple key** in left room (unlocks right door)
- **Green box** in right room (contains green key)
- **Green locked door** at top of corridor (leads to goal)

**Solution sequence** (7 critical actions + navigation):
1. Pick up blue ball (unblock left entrance)
2. Drop ball somewhere
3. Navigate to left room, pick up purple key
4. Navigate to right door, unlock it (toggle)
5. Navigate to box, open box (toggle) → get green key
6. Navigate up corridor, unlock green door (toggle)
7. Navigate to goal

**Observation**: 7×7×3 egocentric partial view (same format in all subtask envs and full env). Direction scalar appended.

**Max steps**: 1444 (4 × 19 × 19)

---

## Expert Data

Three subtask expert policies trained via PPO in isolated 5×5 rooms, recorded 200 episodes each (BFS-optimal):

| Expert     | Transitions | Avg Ep Len | Nav Actions | Critical Actions |
|------------|-------------|------------|-------------|------------------|
| MoveBall   | 7,629       | 15.3       | 6,629 (87%) | 1,000 (13%)      |
| KeyDoor    | 10,962      | 21.9       | 9,462 (86%) | 1,500 (14%)      |
| BoxKey     | 10,247      | 20.5       | 8,247 (80%) | 2,000 (20%)      |
| **Total**  | **28,838**  |            | **24,338**  | **4,500**        |

Navigation = left/right/forward. Critical = pickup/drop/toggle.

The 7×7 egocentric views are observation-compatible with the full corridor environment (same walls, doors, keys, balls visible).

---

## Experiment Timeline

### 1. Initial Training (4-obj, dense distance shaping)

**Config**: Split-AWR, 4 objectives (AWR + moveball + keydoor + boxkey), `ent_coef=0.0`, dense distance-to-goal reward shaping, `max_depth=4` (default hyperparams).

**Result**: 200k+ steps, 9 training attempts, **0% success rate**. BC loss plateaued at ~0.67. Distance shaping was pulling agent toward corridor instead of completing subtasks.

**Conclusion**: Dense distance shaping is harmful — it provides gradient signal that conflicts with the sequential subtask structure.

---

### 2. Sparse Reward Only (4-obj)

**Config**: Stripped all shaping. Sparse reward only: `1.0 - 0.9*(step/max_steps)` on goal reach, else 0.

**Result**: BC loss dropped faster (0.663 vs 0.696 with shaping). Still 0% success after 158k steps, but training signal was cleaner.

**Conclusion**: Sparse reward is better than dense distance shaping for BC convergence, but no online episodes succeed → AWR can't learn.

---

### 3. BC-Only Evaluation (4-obj)

**Config**: Loaded trained 4-obj model (1000 trees), evaluated purely from BC-learned policy (no online training). Ran 20 evaluation episodes.

**Results**:
- BC accuracy: 84.5%, BC loss: 0.683
- Action distribution: **96.1% forward**, 1.5% left, 1.0% right, 0.9% pickup, 0.3% drop, 0.2% toggle
- Behavior: agent goes forward until hitting a wall, stays stuck
- Two modes: 7/20 episodes do ball+key pickup then stuck; 13/20 do nothing

**Root cause**: BC learned "go forward" as the dominant action (56% of all expert data is forward). The minority critical actions (pickup/drop/toggle = 16% of expert data) are drowned out in the shared policy gradient.

---

### 4. Five-Objective Label Split

**Insight**: With 4 objectives, all expert data (nav + critical) shares a single BC gradient per expert. The dominant forward action drowns out minority pick/drop/toggle actions.

**Fix**: Split expert data into 5 objectives:
- Obj 0: AWR (online data, label=0)
- Obj 1: MoveBall critical actions only (pickup/drop, label=1, 1,000 samples)
- Obj 2: KeyDoor critical actions only (pickup/drop/toggle, label=2, 1,500 samples)
- Obj 3: BoxKey critical actions only (pickup/drop/toggle, label=3, 2,000 samples)
- Obj 4: Shared navigation across all experts (left/right/forward, label=4, 24,338 samples)

**Implementation**: `_split_labels_by_action()` in `algos/split_awr.py` splits each expert's data during buffer prefill: navigation actions (0,1,2) → label 4, critical actions (3,4,5) → original expert label.

---

### 5. BC-Only Evaluation (5-obj)

**Config**: Trained 5-obj model (1000 trees) from expert data only, evaluated 20 episodes.

**Results**:
- **Critical action accuracy: 97.8%** (up from ~50% in 4-obj)
- Navigation accuracy: 82.0%
- Overall accuracy: 84.4%

**BUT runtime behavior was identical to 4-obj**: same room distribution, same action distribution, same stuck-in-middle pattern.

**Conclusion**: The 5-obj split massively improved critical action recognition (97.8% vs ~50%), but the bottleneck is NOT action classification — it's that the policy can't chain actions into successful trajectories. BC gets individual action predictions right but can't compose them into sequential behavior.

---

### 6. Discovery: `ent_coef=0.0` in ALL Previous Runs

**Critical finding**: The entropy coefficient was **0.0 in every single run**. This means:
- The entropy loss term was multiplied by zero: `loss = policy_loss + 0.0 * entropy_loss + vf_coef * value_loss`
- AWR had **no exploration incentive whatsoever**
- Online episodes just reinforced whatever the current (BC-dominated) policy did

---

### 7. Split-AWR with `ent_coef=0.01` (5-obj, sparse reward)

**Config**: First run with non-zero entropy. `ent_coef=0.01`, 5-obj, sparse reward, optimized hyperparams (`max_depth=6`, `policy_lr=0.00756`, `split_score_func=Cosine`, `batch_size=1024`, `beta=0.05`).

**Results at 86.6k steps** (17% of 500k):

| Step   | BC loss | Entropy loss | Policy loss | Success |
|--------|---------|-------------|-------------|---------|
| 14.4k  | 1.071   | -1.560      | 9.074       | 0%      |
| 28.9k  | 0.855   | -1.191      | 5.337       | 0%      |
| 43.3k  | 0.798   | -1.050      | 3.819       | 0%      |
| 57.8k  | 0.748   | -0.964      | 3.204       | 0%      |
| 72.2k  | 0.708   | -0.918      | 2.783       | 0%      |
| 86.6k  | 0.691   | -0.868      | 2.564       | 0%      |

All 60 episodes hit max_steps=1444 (timeout). Episode length = 1444 every time.

**Analysis of why entropy doesn't help at 0.01**:
- Entropy contribution to loss: `0.01 × 0.87 = 0.0087`
- Policy loss: 2.56
- Entropy is **0.34% of total loss** — negligible
- With all-zero rewards, AWR advantages ≈ 0, weights ≈ 1 → AWR just reinforces current behavior
- The entropy term can't overcome the BC + AWR reinforcement signal

**Conclusion**: `ent_coef=0.01` is far too small to drive meaningful exploration when the policy loss is 100× larger.

---

### 8. Analysis: Why the Agent Gets Stuck

The expert navigation data (24k samples) IS observation-compatible with the corridor env — same 7×7 views, same object types. The BC learns 82% navigation accuracy.

The problem is NOT missing data. The problem is:

1. **Compounding errors**: At 82% per-step accuracy over a 50+ step trajectory, probability of correct sequence = 0.82^50 ≈ 0.00003 (0.003%)
2. **No recovery signal**: With sparse reward, the agent gets zero reward for partial progress. AWR weights are all ~1.0 → just reinforces whatever it's doing.
3. **AWR degeneracy**: When all rewards are 0, `advantage = 0`, `exp(0/β) = 1`, so AWR = uniform-weighted policy gradient = reinforce current behavior. The online training loop is actively harmful — it locks in the BC-dominated forward-action behavior.

---

### 9. Current Run: `ent_coef=0.5` + Milestone Rewards (5-obj)

**Config changes**:
- `ent_coef=0.5` (50× increase from 0.01) — entropy now ~25% of loss, strong exploration pressure
- Milestone rewards: +0.1 for each subtask completion (ball picked, ball moved, purple key, right door opened, box opened, green key, top door opened)
- Full goal completion still gets big reward: `1.0 - 0.9*(step/max_steps)`
- Per-milestone TensorBoard tracking: `rollout/milestone_ball_picked`, `rollout/milestone_purple_key_picked`, etc.

**Rationale**:
- Milestone rewards give AWR non-uniform advantages → can distinguish good from bad trajectories
- High entropy coefficient ensures exploration diversity, prevents locking into forward-only
- Milestone tracking lets us see WHERE the agent gets stuck in the solution sequence

**Results at 173k steps** (35% of 500k):

| Milestone | 43k | 87k | 173k | Status |
|---|---|---|---|---|
| ball_picked | 80% | 92% | 97% | Near-mastered |
| ball_moved | 80% | 92% | 97% | Near-mastered |
| purple_key_picked | 55% | 68% | 79% | Steady improvement |
| right_door_opened | 0% | 2% | 2% | Bottleneck — plateau |
| box_opened | 0% | 0% | 1% | Just appeared |
| green_key_picked | 0% | 0% | 1% | Just appeared |
| top_door_opened | 0% | 0% | 0% | Not yet |
| goal_reached | 0% | 0% | 0% | Not yet |
| milestones_avg | 2.15 | 2.54 | 2.77 | Rising |

Training metrics: BC loss 0.643, entropy loss -0.883, policy loss 2.05, 2875 trees.

**Observations**:
1. BC successfully transfers subtasks 1-2 (ball pickup/move at 97%, key pickup at 79%)
2. The **bottleneck is right_door_opened (2%)**: the agent picks up the purple key but rarely navigates to the locked door and toggles it. This requires: carry key → navigate from left room → cross middle room → reach right wall → face door → toggle.
3. box_opened and green_key_picked appeared at 1% around 130k steps — these require right_door_opened first, so they're gated.
4. Milestones average is still climbing (2.77), meaning the agent is getting better at the early subtasks.
5. ep_rew_mean is climbing (0.19 → 0.28), confirming AWR now has non-degenerate advantages.

**Current bottleneck analysis**: The gap between purple_key_picked (79%) and right_door_opened (2%) suggests the agent knows HOW to pick up keys (from BC) but doesn't know how to NAVIGATE from the left room to the right door while carrying the key. This is a multi-room navigation + key-usage composition problem.

---

## Key Takeaways

1. **BC accuracy ≠ trajectory success**. 82% per-step accuracy does NOT translate to completing 50+ step sequences. Compounding errors make this exponentially harder.

2. **Objective splitting works for action classification**. The 5-obj split boosted critical action accuracy from ~50% to 97.8% — the multi-objective GBRL gradient routing is doing its job.

3. **AWR with zero rewards is degenerate**. When all episodes fail, advantages=0, AWR becomes uniform policy gradient that reinforces current (wrong) behavior. The entropy term is the only exploration driver.

4. **`ent_coef=0.0` was the hidden killer**. All runs before experiment #7 had no entropy at all. This was the biggest configuration error.

5. **Sparse milestone rewards break AWR degeneracy**. Even small per-subtask bonuses create non-zero advantages, giving AWR something to learn from. The agent doesn't need to solve the full task — just doing the first subtask (pick up ball) provides signal.

6. **Expert navigation data IS compatible**. The 7×7 egocentric views in subtask rooms are the same format as the corridor. The 24k navigation samples cover walls, doors, keys, balls. We don't need separate navigation training.

7. **Expert data must cover the out-of-view navigation gap**. The keydoor expert (max_size=12) had the door visible during 80% of key-to-door navigation, with 0% of episodes where the door was never visible. In the corridor, the door is NEVER visible from the left room. Increasing max_size to 30 gave 44% of episodes with 5+ blind navigation steps and improved right_door_opened from 2% to 6%.

---

### 10. Larger KeyDoor Expert Data (max_size=30)

**Root cause discovered**: The keydoor expert data (max_size=12) had the door visible 100% of the time during key-to-door navigation. In the corridor env, after picking up the purple key in the left room, the right door is completely invisible (9+ tiles away behind walls). The agent had **zero training examples** for "carrying key, door not visible, navigate to find it."

**Fix**: Increased `KeyDoorEnv.max_size` from 12 to 30, re-generated 500 expert episodes via BFS.

| Metric | Old (max_size=12) | New (max_size=30) |
|---|---|---|
| Total transitions | 10,962 | 17,113 |
| Avg ep length | 21.9 | 34.2 |
| Avg blind nav steps | 1.2 | **5.1** |
| Eps with ≥3 blind steps | 9% | **56%** |
| Eps with ≥5 blind steps | 6% | **44%** |

**Results at 173k steps** (comparison with exp #9):

| Milestone | Exp 9 (old keydoor) | Exp 10 (new keydoor) |
|---|---|---|
| ball_picked | 97% | 96% |
| purple_key_picked | 79% | 79% |
| **right_door_opened** | **2%** | **6%** (3×) |
| **box_opened** | **1%** | **3%** (3×) |
| top_door_opened | 0% | 0% |
| milestones_avg | 2.77 | **2.83** |

right_door_opened appeared earlier (57k vs 87k) and is 3× higher. Still the bottleneck but improving.

**Status**: Running to 500k steps.
