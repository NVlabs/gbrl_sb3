##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
"""
Wrappers around the standard Flatland RailEnv for use with SB3.

The Flatland environment is not modified — we only wrap it to:
  1. Extract reward/cost decomposition for constrained RL (SPLIT-RL).
  2. Compute a binary safety label from tree observation features.
  3. Convert the multi-agent env to an SB3-compatible VecEnv via the
     official Flatland → PettingZoo → SuperSuit pipeline.

Pipeline:  RailEnv (with FlattenedNormalizedTreeObsForRailEnv)
        → PettingzooFlatland  (Flatland's own PettingZoo adapter)
        → FlatlandRewardCostWrapper  (our thin PettingZoo wrapper)
        → SuperSuit pettingzoo_env_to_vec_env_v1 + concat_vec_envs_v1
        → SB3 VecEnv

Observations: FlattenedNormalizedTreeObsForRailEnv (Flatland built-in).
  - max_depth=2 → 252 floats, max_depth=3 → 1020 floats.
  - Uses ShortestPathPredictorForRailEnv for enriched obs.
  - Observations are always extended with 4 extra floats:
    [own_slack, min_conflicting_slack_L, min_conflicting_slack_F, min_conflicting_slack_R]
  - Total obs dim: 256 (depth 2).
Actions: Discrete(5) — DO_NOTHING, MOVE_LEFT, MOVE_FORWARD, MOVE_RIGHT, STOP_MOVING.
Rewards: per-agent scalar (configurable: decomposed or original).
Cost:    per-agent externality cost in info['cost'].
Label:   per-agent binary safety_label in info['safety_label'].

Split-RL Scenarios
------------------
  - "slack_priority": Cost fires when agent takes a contested switch branch
    that delays a lower-slack conflicting train. Label fires at switches where
    the reward-best action (forward/progress) differs from the cost-best
    action (yield/reroute to let urgent train pass).
  - "malfunction_detour": Cost fires when agent moves into a branch with
    downstream malfunctioning trains while an unblocked alternative branch
    exists. Creates spatial conflict: forward may be shortest path but feeds
    into a blocked area, hurting other trains stuck behind.
"""
from __future__ import annotations

import copy
import time
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv

from flatland.env_generation.env_generator import env_generator
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rewards import BaseDefaultRewards, DefaultPenalties, DefaultRewards
from flatland.envs.step_utils.state_machine import TrainState
from flatland.ml.observations.flatten_tree_observation_for_rail_env import (
    FlattenedNormalizedTreeObsForRailEnv,
)
from flatland.ml.pettingzoo.wrappers import PettingzooFlatland

import supersuit as ss
from stable_baselines3.common.vec_env import VecMonitor


# ──────────────────────────────────────────────────────────────────────────────
# Safety label from tree observation
# ──────────────────────────────────────────────────────────────────────────────

# Feature indices within each group of the flattened tree observation.
# Data segment [0, n_nodes*6): 6 distance features per node (DFS pre-order).
#   col 2 = dist_other_agent_encountered
#   col 3 = dist_potential_conflict
# Agent-data segment [n_nodes*7, n_nodes*12): 5 agent features per node.
#   col 1 = num_agents_opposite_direction
#   col 2 = num_agents_malfunctioning
# We only use these counts as danger evidence when they occur on a branch that
# already has a nearby conflict or nearby agent cue. This keeps the label local
# and avoids firing on broad subtree-level risk.
_DATA_COL_OTHER_AGENT = 2
_DATA_COL_POTENTIAL_CONFLICT = 3
_AGENT_COL_OPPOSITE_DIR = 1
_AGENT_COL_MALFUNCTIONING = 2


def compute_danger_label(
    obs: np.ndarray,
    n_nodes: int,
    conflict_dist_thresh: float = 0.5,
    agent_dist_thresh: float = 0.5,
) -> int:
    """
    Binary danger label from a flattened normalized tree observation.

        Returns 1 (danger) if ANY of the following hold:
            - A potential conflict is detected within *conflict_dist_thresh*
                on any non-root branch.
            - Another agent is encountered within *agent_dist_thresh*
                on any non-root branch.
            - A non-root branch reports opposite-direction traffic AND that same
                branch also has a nearby conflict or nearby other-agent cue.
            - A non-root branch reports a malfunctioning / blocked agent AND that
                same branch also has a nearby conflict or nearby other-agent cue.

    After Flatland's normalization the data features are divided by
    observation_radius (default 2) and clipped to [-1, 1].  A positive
    value means the feature was detected at that normalised distance;
    0.0 means not detected; -1.0 means missing branch.
    Agent-data features are simply clipped to [-1, 1].
    """
    # Reshape the grouped segments into (n_nodes, n_features) matrices
    data = obs[: n_nodes * 6].reshape(n_nodes, 6)
    agent_data = obs[n_nodes * 7 : n_nodes * 12].reshape(n_nodes, 5)

    # --- distance features (skip root node, index 0) ---
    child_data = data[1:]

    conflict = child_data[:, _DATA_COL_POTENTIAL_CONFLICT]
    nearby_conflict = (conflict > 0) & (conflict <= conflict_dist_thresh)
    if np.any(nearby_conflict):
        return 1

    other_agent = child_data[:, _DATA_COL_OTHER_AGENT]
    nearby_other_agent = (other_agent > 0) & (other_agent <= agent_dist_thresh)
    if np.any(nearby_other_agent):
        return 1

    # --- agent-count features (non-root branches only) ---
    child_agent_data = agent_data[1:]
    local_branch_hazard = nearby_conflict | nearby_other_agent

    if np.any((child_agent_data[:, _AGENT_COL_OPPOSITE_DIR] > 0) & local_branch_hazard):
        return 1

    if np.any((child_agent_data[:, _AGENT_COL_MALFUNCTIONING] > 0) & local_branch_hazard):
        return 1

    return 0


def compute_congestion_cost(
    obs: np.ndarray,
    n_nodes: int,
    action: int,
    conflict_dist_thresh: float = 0.5,
    agent_dist_thresh: float = 0.5,
) -> int:
    """
    Binary congestion cost: 1 if the agent moved into a contested direction.

    Returns 1 (congestion) if ALL of:
      - agent took a movement action (MOVE_LEFT=1, MOVE_FORWARD=2, MOVE_RIGHT=3)
      - the chosen direction's branch in the tree observation shows:
          * another agent within agent_dist_thresh, OR
          * a potential conflict within conflict_dist_thresh, OR
          * opposite-direction agents present

    Creates a structural conflict with progress reward:
      - Moving through busy junctions → reward ↑ (progress) + cost ↑ (congestion)
      - Stopping/yielding → reward ↓ (step penalty) + cost 0 (no congestion)
      - Agent cannot maximize progress without incurring congestion cost.

    Observation features are localized by direction: the tree has 3 main
    branches (left, forward, right), each with its own conflict/agent features.
    A tree-based cost head can route on the CHOSEN branch's features; an NN
    averages all branches through shared weights.

    Parameters
    ----------
    obs : np.ndarray
        Flattened normalized tree observation.
    n_nodes : int
        Number of nodes in the tree.
    action : int
        Action taken: 0=DO_NOTHING, 1=MOVE_LEFT, 2=MOVE_FORWARD, 3=MOVE_RIGHT, 4=STOP.
    conflict_dist_thresh : float
        Distance threshold for potential conflict detection.
    agent_dist_thresh : float
        Distance threshold for other-agent proximity.
    """
    # Only fire when the agent actively moves
    # DO_NOTHING=0, STOP_MOVING=4 → no congestion
    if action not in (1, 2, 3):
        return 0

    # Compute tree branching structure
    # Tree with branching factor 3 and max_depth d: n_nodes = (3^(d+1)-1)/2
    # Each branch subtree from root has (n_nodes - 1) / 3 nodes
    branch_size = (n_nodes - 1) // 3

    # Branch root indices (DFS pre-order): left=1, forward=1+branch_size, right=1+2*branch_size
    action_to_branch = {
        1: 1,                       # MOVE_LEFT → left branch
        2: 1 + branch_size,         # MOVE_FORWARD → forward branch
        3: 1 + 2 * branch_size,     # MOVE_RIGHT → right branch
    }
    branch_root = action_to_branch[action]

    # Extract features for the chosen branch subtree
    data = obs[: n_nodes * 6].reshape(n_nodes, 6)
    agent_data = obs[n_nodes * 7: n_nodes * 12].reshape(n_nodes, 5)

    # Check just the branch root node (immediate lookahead)
    node = branch_root
    if node >= n_nodes:
        return 0

    dist_other = data[node, _DATA_COL_OTHER_AGENT]
    dist_conflict = data[node, _DATA_COL_POTENTIAL_CONFLICT]
    n_opposite = agent_data[node, _AGENT_COL_OPPOSITE_DIR]

    if dist_other > 0 and dist_other <= agent_dist_thresh:
        return 1
    if dist_conflict > 0 and dist_conflict <= conflict_dist_thresh:
        return 1
    if n_opposite > 0:
        return 1

    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Externality-based cost/label for Split-RL scenarios
# ──────────────────────────────────────────────────────────────────────────────

def _branch_has_conflict(obs_node_data, obs_node_agent_data,
                         conflict_dist_thresh: float = 0.5,
                         agent_dist_thresh: float = 0.5) -> bool:
    """Check if a single tree node has conflict signals."""
    dist_other = obs_node_data[_DATA_COL_OTHER_AGENT]
    dist_conflict = obs_node_data[_DATA_COL_POTENTIAL_CONFLICT]
    n_opposite = obs_node_agent_data[_AGENT_COL_OPPOSITE_DIR]

    if dist_other > 0 and dist_other <= agent_dist_thresh:
        return True
    if dist_conflict > 0 and dist_conflict <= conflict_dist_thresh:
        return True
    if n_opposite > 0:
        return True
    return False


def _branch_has_malfunction(obs_node_agent_data) -> bool:
    """Check if a single tree node has malfunctioning agent signals."""
    return obs_node_agent_data[_AGENT_COL_MALFUNCTIONING] > 0


def _get_branch_indices(n_nodes: int):
    """Return (branch_size, {L: idx, F: idx, R: idx}) for depth-1 branch roots."""
    branch_size = (n_nodes - 1) // 3
    return branch_size, {
        "L": 1,
        "F": 1 + branch_size,
        "R": 1 + 2 * branch_size,
    }


def _is_at_switch(obs: np.ndarray, n_nodes: int) -> bool:
    """Check whether the agent is at a switch (L or R branch exists).

    A branch exists if its root node has any POSITIVE data value.
    In Flatland's normalized obs:
      - Valid branches have positive distance values (e.g. dist_own_target > 0).
      - Dead-end / missing branches have all features = -1.0.
      - 0.0 means "at this location" (valid).
    We check for any value > 0 to distinguish valid branches from dead ends.
    """
    _, branch_indices = _get_branch_indices(n_nodes)
    data = obs[: n_nodes * 6].reshape(n_nodes, 6)
    for key in ("L", "R"):
        idx = branch_indices[key]
        if idx < n_nodes:
            if np.any(data[idx] > 0):
                return True
    return False


def compute_slack_features(
    rail_env,
    agent_handle: int,
    obs: np.ndarray,
    n_nodes: int,
    elapsed_steps: int,
    conflict_dist_thresh: float = 0.5,
    agent_dist_thresh: float = 0.5,
    slack_margin: float = 3.0,
    proximity_dist: int = 8,
) -> dict:
    """
    Compute externality-based slack features for one agent.

    Returns dict with:
      own_slack: float — agent's current delay (positive = ahead of schedule)
      min_conflicting_slack: dict[str, float] — min slack of conflicting agents
          per branch (L/F/R), normalized to [-1, 1]
      cost: float — externality cost (0 or 1)
      label: int — binary disagreement label (0 or 1)
      at_switch: bool — whether agent is at a switch point
      branch_conflict: dict[str, bool] — whether each branch has conflict

    The cost fires when the agent takes a contested branch that delays a
    lower-slack conflicting train. The label fires at switches where the
    reward-best action (go forward) differs from the cost-best action (reroute).
    """
    agents = rail_env.agents
    distance_map = rail_env.distance_map
    agent = agents[agent_handle]

    result = {
        "own_slack": 0.0,
        "min_conflicting_slack": {"L": 1.0, "F": 1.0, "R": 1.0},
        "cost": 0.0,
        "label": 0,
        "at_switch": False,
        "branch_exists": {"L": False, "F": False, "R": False},
        "branch_conflict": {"L": False, "F": False, "R": False},
    }

    # Agent must be on the map and active
    if agent.position is None:
        return result
    state = agent.state_machine.state
    if state not in (TrainState.MOVING, TrainState.STOPPED):
        return result

    # Compute own slack
    own_slack = agent.get_current_delay(elapsed_steps, distance_map)
    # Normalize: divide by max_episode_steps proxy, clip to [-1, 1]
    max_steps = rail_env.max_episode_steps if hasattr(rail_env, 'max_episode_steps') else 200
    own_slack_norm = np.clip(own_slack / max(max_steps / 4, 1.0), -1.0, 1.0)
    result["own_slack"] = own_slack_norm

    # Check branch structure
    _, branch_indices = _get_branch_indices(n_nodes)
    data = obs[: n_nodes * 6].reshape(n_nodes, 6)
    agent_data = obs[n_nodes * 7: n_nodes * 12].reshape(n_nodes, 5)

    # Determine which branches exist and which have conflicts
    # A branch exists if any data value is positive (> 0).
    # Dead-end branches have all features = -1.0 in Flatland's encoding.
    branch_exists = {}
    for key in ("L", "F", "R"):
        idx = branch_indices[key]
        if idx < n_nodes and np.any(data[idx] > 0):
            branch_exists[key] = True
            result["branch_conflict"][key] = _branch_has_conflict(
                data[idx], agent_data[idx], conflict_dist_thresh, agent_dist_thresh
            )
        else:
            branch_exists[key] = False

    at_switch = branch_exists.get("L", False) or branch_exists.get("R", False)
    result["at_switch"] = at_switch
    result["branch_exists"] = dict(branch_exists)

    # Find nearby agents and assign their slacks to conflicting branches.
    # The tree obs tells us HOW MANY opposite-direction agents are on each branch
    # (agent_data col 1). We match by: for each conflicting branch, take the N
    # closest nearby agents (by Manhattan distance) where N = that branch's
    # num_agents_opposite_direction. This prevents dumping all agents into all branches.
    nearby_agents = []
    for j, other in enumerate(agents):
        if j == agent_handle:
            continue
        if other.position is None:
            continue
        other_state = other.state_machine.state
        if other_state not in (TrainState.MOVING, TrainState.STOPPED, TrainState.READY_TO_DEPART):
            continue
        dist = abs(agent.position[0] - other.position[0]) + abs(agent.position[1] - other.position[1])
        if dist > proximity_dist:
            continue
        other_slack = other.get_current_delay(elapsed_steps, distance_map)
        other_slack_norm = np.clip(other_slack / max(max_steps / 4, 1.0), -1.0, 1.0)
        nearby_agents.append((dist, other_slack_norm))

    # Sort by distance (closest first) for greedy matching
    nearby_agents.sort(key=lambda x: x[0])

    # Assign to each conflicting branch, respecting the count from tree obs
    agent_pool = list(nearby_agents)  # agents available for assignment
    for key in ("L", "F", "R"):
        if not result["branch_conflict"].get(key, False):
            continue
        idx = branch_indices[key]
        # Number of opposite-direction agents the tree obs reports on this branch
        n_opposite = max(0, int(agent_data[idx, _AGENT_COL_OPPOSITE_DIR]))
        if n_opposite == 0:
            # Branch has conflict but no opposite-dir agents — could be same-dir
            # or potential_conflict signal. Use closest 1 agent as fallback.
            n_opposite = 1
        # Take the N closest unassigned agents
        assigned = 0
        for ad_dist, ad_slack in agent_pool:
            if assigned >= n_opposite:
                break
            if ad_slack < result["min_conflicting_slack"][key]:
                result["min_conflicting_slack"][key] = ad_slack
            assigned += 1

    # Cost: fires when agent is at a switch, chosen branch has conflict,
    # and a conflicting agent on that branch has lower slack.
    # (Actual branch selection depends on action — computed in wrapper step())

    # Label: fires at switches where ALL of:
    #   1. Forward branch has conflict (head-on or potential conflict)
    #   2. A conflicting agent on forward has meaningfully lower slack
    #   3. At least one alternative branch EXISTS and is CONFLICT-FREE
    # Condition 3 ensures the agent actually has a viable reroute option.
    # Without it, the only cost-free action is STOP, which blocks the switch.
    has_conflict_free_alt = False
    for key in ("L", "R"):
        if branch_exists.get(key, False) and not result["branch_conflict"].get(key, False):
            has_conflict_free_alt = True
            break

    if at_switch and has_conflict_free_alt and result["branch_conflict"].get("F", False):
        fwd_min_slack = result["min_conflicting_slack"]["F"]
        if fwd_min_slack < own_slack_norm - (slack_margin / max(max_steps / 4, 1.0)):
            result["label"] = 1

    return result


def compute_malfunction_detour_features(
    obs: np.ndarray,
    n_nodes: int,
) -> dict:
    """
    Compute malfunction-based detour cost/label from tree observation.

    Returns dict with:
      cost: float — 1 if agent moves into branch with malfunctioning train
                    while an unblocked alternative exists
      label: int — binary disagreement label
      fwd_blocked: bool — forward branch has malfunctioning agent
      alt_clear: bool — at least one alternative branch has no malfunction

    The externality: joining a feeder route behind a malfunction hurts
    everyone else in that queue. The agent's own reward still prefers the
    short path forward.
    """
    result = {
        "cost": 0.0,
        "label": 0,
        "fwd_blocked": False,
        "alt_clear": False,
    }

    _, branch_indices = _get_branch_indices(n_nodes)
    data = obs[: n_nodes * 6].reshape(n_nodes, 6)
    agent_data = obs[n_nodes * 7: n_nodes * 12].reshape(n_nodes, 5)

    # Check which branches exist and have malfunctions
    # A branch exists if any data value is positive (dead ends have all -1).
    branch_exists = {}
    branch_malf = {}
    for key in ("L", "F", "R"):
        idx = branch_indices[key]
        if idx < n_nodes and np.any(data[idx] > 0):
            branch_exists[key] = True
            branch_malf[key] = _branch_has_malfunction(agent_data[idx])
        else:
            branch_exists[key] = False
            branch_malf[key] = False

    at_switch = branch_exists.get("L", False) or branch_exists.get("R", False)
    if not at_switch:
        return result

    # Forward blocked by malfunction?
    fwd_blocked = branch_malf.get("F", False)
    result["fwd_blocked"] = fwd_blocked

    if not fwd_blocked:
        return result

    # Is there an alternative branch without malfunction?
    alt_clear = False
    for key in ("L", "R"):
        if branch_exists.get(key, False) and not branch_malf.get(key, False):
            alt_clear = True
            break
    result["alt_clear"] = alt_clear

    if alt_clear:
        result["cost"] = 1.0
        result["label"] = 1

    return result


# ──────────────────────────────────────────────────────────────────────────────
# PettingZoo wrapper: reward/cost split + label
# ──────────────────────────────────────────────────────────────────────────────

class FlatlandRewardCostWrapper(ParallelEnv):
    """
    Thin PettingZoo ParallelEnv wrapper that intercepts Flatland's
    BaseDefaultRewards dict-valued rewards and converts them to:

      reward = sum of non-collision terms  (scalar float)
      info['cost']            = configurable cost signal
      info['cost_collision']  = collision penalty >= 0 (always tracked)
      info['original_reward'] = full scalar sum of ALL terms
      info['safety_label']    = binary disagreement label from scenario
      info['reward_terms']    = raw dict of all penalty terms

    Also casts observations to float32 (Flatland's obs builder declares
    float64, which is unnecessary memory/throughput overhead for SB3).

    Tracks per-episode cumulative cost, original reward, and reward
    terms, emitted in info at episode end.

    Cost functions (selected via ``cost_fn``):
      - "slack_priority": Externality-based. Cost fires when agent takes a
        contested branch at a switch that delays a lower-slack conflicting
        train. Obs extended with [own_slack, min_conflicting_slack_L/F/R].
        Label fires at switches where forward has conflict and conflicting
        agent has meaningfully lower slack — true objective disagreement.
      - "malfunction_detour": Externality-based. Cost fires when agent
        moves into a branch with downstream malfunctioning trains while an
        unblocked alternative branch exists at a switch.
      - "congestion" (legacy): 1 if the agent moved into a contested direction.
        NOT recommended: cost aligns with reward (both punish congestion).
      - "danger" (legacy): 1 if observation shows nearby conflict/agents.
        NOT recommended: event detector, not conflict detector.
      - "collision" (legacy): collision penalty magnitude.
        NOT recommended: negatively correlated with non-collision reward.

    Parameters
    ----------
    env : ParallelEnv
        PettingZoo parallel env from PettingzooFlatland.
    cost_fn : str
        Cost function to use.
    use_original_reward : bool
        If True (default), SB3 optimises the full original reward (all terms
        including collision).  If False, SB3 optimises only
        the non-collision terms.
    conflict_dist_thresh : float
        Normalised distance threshold for conflict detection.
    agent_dist_thresh : float
        Normalised distance threshold for other-agent proximity.
    slack_margin : float
        Minimum slack gap (in raw steps) to trigger slack_priority label.
    """

    # Valid cost function names
    _VALID_COST_FNS = ("slack_priority", "malfunction_detour", "congestion", "danger", "collision")

    def __init__(
        self,
        env: ParallelEnv,
        cost_fn: str = "slack_priority",
        use_original_reward: bool = True,
        conflict_dist_thresh: float = 0.5,
        agent_dist_thresh: float = 0.5,
        n_agents: int = 1,
        max_episode_steps: int = 200,
        slack_margin: float = 3.0,
    ):
        if cost_fn not in self._VALID_COST_FNS:
            raise ValueError(
                f"Unknown cost_fn '{cost_fn}'. Use one of {self._VALID_COST_FNS}."
            )
        self._env = env
        self.possible_agents = env.possible_agents
        self.agents = env.agents
        self.metadata = env.metadata
        self.render_mode = getattr(env, "render_mode", None)

        self._cost_fn = cost_fn
        self._use_original_reward = use_original_reward
        self._conflict_dist_thresh = conflict_dist_thresh
        self._agent_dist_thresh = agent_dist_thresh
        self._n_agents = n_agents
        self._max_episode_steps = max_episode_steps
        self._slack_margin = slack_margin

        # Access the raw RailEnv for agent timetable / distance_map queries.
        # PettingZooParallelEnvWrapper stores it as _wrap.
        self._rail_env = getattr(env, "_wrap", None)

        # Base observation spaces from Flatland (float32 cast)
        base_obs_spaces = {
            agent: gym.spaces.Box(
                low=space.low.astype(np.float32),
                high=space.high.astype(np.float32),
                shape=space.shape,
                dtype=np.float32,
            )
            for agent, space in env.observation_spaces.items()
        }

        # Always extend observation space with slack features (4 extra floats)
        # so all algorithms (PPO, Split-RL, etc.) see the same obs dim.
        self._extend_obs = True
        self._n_extra_features = 4
        self.observation_spaces = {}
        for agent, space in base_obs_spaces.items():
            new_shape = (space.shape[0] + self._n_extra_features,)
            self.observation_spaces[agent] = gym.spaces.Box(
                low=-1.0, high=1.0, shape=new_shape, dtype=np.float32,
            )

        self.action_spaces = env.action_spaces

        # Infer n_nodes from base observation shape (obs_dim = 12 * n_nodes)
        sample_space = next(iter(env.observation_spaces.values()))
        self._n_nodes = sample_space.shape[0] // 12

        # Stored observations for pre-step label computation
        self._prev_obs: Dict[str, np.ndarray] = {}
        # Stored slack features for cost computation in step()
        self._prev_slack_features: Dict[str, dict] = {}
        self._prev_malf_features: Dict[str, dict] = {}

        # Step counter for timetable queries
        self._elapsed_steps = 0

        # Episode accumulators
        self._episode_costs: Dict = {}
        self._episode_original_rewards: Dict = {}
        self._episode_reward_terms: Dict = {}
        self._agents_completed: set = set()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self._env.action_space(agent)

    def _cast_obs(self, obs: Dict) -> Dict:
        """Cast all observations to float32."""
        return {
            agent: np.asarray(o, dtype=np.float32)
            for agent, o in obs.items()
        }

    def _extend_obs_with_slack(self, obs: Dict) -> Dict:
        """Append slack features [own_slack, min_slack_L, min_slack_F, min_slack_R] to obs."""
        if not self._extend_obs or self._rail_env is None:
            return obs
        extended = {}
        for agent, o in obs.items():
            # Always compute slack features for obs extension
            feats = compute_slack_features(
                self._rail_env, agent, o, self._n_nodes,
                self._elapsed_steps,
                self._conflict_dist_thresh, self._agent_dist_thresh,
                self._slack_margin,
            )
            self._prev_slack_features[agent] = feats
            extra = np.array([
                feats["own_slack"],
                feats["min_conflicting_slack"]["L"],
                feats["min_conflicting_slack"]["F"],
                feats["min_conflicting_slack"]["R"],
            ], dtype=np.float32)

            # Scenario-specific pre-computations stored for step()
            if self._cost_fn == "malfunction_detour":
                malf_feats = compute_malfunction_detour_features(o, self._n_nodes)
                self._prev_malf_features[agent] = malf_feats

            extended[agent] = np.concatenate([o, extra])
        return extended

    def _empty_reward_terms(self) -> Dict[str, float]:
        return {p.value: 0.0 for p in DefaultPenalties}

    def reset(self, seed=None, options=None):
        obs, infos = self._env.reset(seed=seed, options=options)
        self.agents = self._env.agents
        obs = self._cast_obs(obs)
        self._elapsed_steps = 0

        # Compute slack/malf features and extend observations
        self._prev_slack_features = {}
        self._prev_malf_features = {}
        if self._extend_obs:
            obs = self._extend_obs_with_slack(obs)

        # Store observations for pre-step label computation
        self._prev_obs = {agent: obs[agent].copy() for agent in obs}

        # Reset episode accumulators
        self._episode_costs = {agent: 0.0 for agent in self.possible_agents}
        self._episode_original_rewards = {agent: 0.0 for agent in self.possible_agents}
        self._episode_reward_terms = {
            agent: self._empty_reward_terms() for agent in self.possible_agents
        }
        self._agents_completed = set()

        for agent in self.agents:
            infos[agent] = dict(infos[agent])
            infos[agent]["cost"] = 0.0
            infos[agent]["original_reward"] = 0.0
            # Label from scenario-specific logic
            if self._cost_fn == "slack_priority":
                infos[agent]["safety_label"] = self._prev_slack_features.get(agent, {}).get("label", 0)
            elif self._cost_fn == "malfunction_detour":
                infos[agent]["safety_label"] = self._prev_malf_features.get(agent, {}).get("label", 0)
            else:
                infos[agent]["safety_label"] = compute_danger_label(
                    self._prev_obs[agent][:self._n_nodes * 12] if self._extend_obs else self._prev_obs[agent],
                    self._n_nodes,
                    self._conflict_dist_thresh, self._agent_dist_thresh,
                )
        return obs, infos

    def step(self, actions):
        # Compute labels/costs from PRE-STEP observations and features
        pre_step_labels = {}
        pre_step_costs = {}
        pre_step_danger = {}
        pre_step_congestion = {}

        action_to_branch_key = {1: "L", 2: "F", 3: "R"}

        for agent in actions:
            base_obs = self._prev_obs.get(agent)
            if base_obs is None:
                pre_step_labels[agent] = 0
                pre_step_costs[agent] = 0.0
                pre_step_danger[agent] = 0.0
                pre_step_congestion[agent] = 0.0
                continue

            # Base obs for legacy functions (strip slack extension if present)
            obs_for_legacy = base_obs[:self._n_nodes * 12] if self._extend_obs else base_obs

            pre_step_danger[agent] = float(compute_danger_label(
                obs_for_legacy, self._n_nodes,
                self._conflict_dist_thresh, self._agent_dist_thresh,
            ))
            pre_step_congestion[agent] = float(compute_congestion_cost(
                obs_for_legacy, self._n_nodes, actions[agent],
                self._conflict_dist_thresh, self._agent_dist_thresh,
            ))

            if self._cost_fn == "slack_priority":
                slack_feats = self._prev_slack_features.get(agent, {})
                label = slack_feats.get("label", 0)
                # Cost fires when agent moves into contested branch where
                # conflicting agent has lower slack AND a conflict-free
                # alternative branch exists (so the agent could have rerouted).
                cost = 0.0
                chosen_branch = action_to_branch_key.get(actions[agent])
                branch_conflict = slack_feats.get("branch_conflict", {})
                branch_exists = slack_feats.get("branch_exists", {})
                if chosen_branch and slack_feats.get("at_switch", False):
                    # Check that a conflict-free alternative exists
                    has_alt = any(
                        branch_exists.get(k, False) and not branch_conflict.get(k, False)
                        for k in ("L", "F", "R")
                        if k != chosen_branch
                    )
                    if has_alt and branch_conflict.get(chosen_branch, False):
                        own_slack = slack_feats.get("own_slack", 0.0)
                        branch_min_slack = slack_feats.get("min_conflicting_slack", {}).get(chosen_branch, 1.0)
                        max_steps = self._max_episode_steps
                        margin_norm = self._slack_margin / max(max_steps / 4, 1.0)
                        if branch_min_slack < own_slack - margin_norm:
                            cost = 1.0
                pre_step_labels[agent] = label
                pre_step_costs[agent] = cost

            elif self._cost_fn == "malfunction_detour":
                malf_feats = self._prev_malf_features.get(agent, {})
                label = malf_feats.get("label", 0)
                # Cost fires when agent moves forward into malfunction-blocked
                # branch while alt is clear
                cost = 0.0
                if actions[agent] == 2 and malf_feats.get("fwd_blocked", False) and malf_feats.get("alt_clear", False):
                    cost = 1.0
                pre_step_labels[agent] = label
                pre_step_costs[agent] = cost

            elif self._cost_fn == "congestion":
                pre_step_labels[agent] = int(pre_step_danger[agent])
                pre_step_costs[agent] = pre_step_congestion[agent]
            elif self._cost_fn == "danger":
                pre_step_labels[agent] = int(pre_step_danger[agent])
                pre_step_costs[agent] = pre_step_danger[agent]
            else:  # collision — cost assigned after step from rewards
                pre_step_labels[agent] = int(pre_step_danger[agent])
                pre_step_costs[agent] = 0.0  # filled below from reward dict

        obs, rewards, terminations, truncations, infos = self._env.step(actions)
        self.agents = self._env.agents
        obs = self._cast_obs(obs)
        self._elapsed_steps += 1

        # Compute new slack/malf features and extend observations
        if self._extend_obs:
            obs = self._extend_obs_with_slack(obs)

        # Flatland terminates agents individually as they reach their
        # destinations, but the episode continues until ALL agents are done.
        # SuperSuit's MarkovVectorEnv propagates per-agent done=True to the
        # VecMonitor, which wrongly counts each as a separate episode ending,
        # flooding stats with spurious 1-step "episodes" and causing reward
        # oscillation.  Fix: only report termination/truncation when the full
        # episode is over (all agents done).
        all_term = all(terminations.get(a, False) for a in self.possible_agents)
        all_trunc = all(truncations.get(a, False) for a in self.possible_agents)
        env_done = all_term or all_trunc

        # Track which agents completed (reached destination = terminated,
        # not just truncated by time limit).  Must check BEFORE override.
        for a in self.possible_agents:
            if terminations.get(a, False) and not truncations.get(a, False):
                self._agents_completed.add(a)

        # Track which agents are individually done (for reward/cost bookkeeping)
        agent_done = {
            a: terminations.get(a, False) or truncations.get(a, False)
            for a in self.possible_agents
        }

        # Override: only signal done to SB3/VecMonitor when full episode ends
        if not env_done:
            terminations = {a: False for a in terminations}
            truncations = {a: False for a in truncations}

        # Update stored observations for next step's label computation
        self._prev_obs = {agent: obs[agent].copy() for agent in obs}

        scalar_rewards = {}
        for agent in rewards:
            raw = rewards[agent]
            info = dict(infos[agent])

            if isinstance(raw, dict):
                collision_cost = max(0.0, -float(raw.get(DefaultPenalties.COLLISION.value, 0)))
                non_collision_reward = sum(
                    float(v) for k, v in raw.items()
                    if k != DefaultPenalties.COLLISION.value
                )
                original_reward = sum(float(v) for v in raw.values())

                if self._use_original_reward:
                    scalar_rewards[agent] = original_reward
                else:
                    scalar_rewards[agent] = non_collision_reward

                # Cost signal
                if self._cost_fn == "collision":
                    info["cost"] = collision_cost
                else:
                    info["cost"] = pre_step_costs.get(agent, 0.0)

                info["cost_collision"] = collision_cost
                info["cost_danger"] = pre_step_danger.get(agent, 0.0)
                info["cost_congestion"] = pre_step_congestion.get(agent, 0.0)
                info["original_reward"] = original_reward
                info["reward_terms"] = dict(raw)

                # Accumulate episode stats (using the selected cost)
                self._episode_costs[agent] = self._episode_costs.get(agent, 0.0) + info["cost"]
                self._episode_original_rewards[agent] = (
                    self._episode_original_rewards.get(agent, 0.0) + original_reward
                )
                for k, v in raw.items():
                    self._episode_reward_terms.setdefault(agent, self._empty_reward_terms())
                    self._episode_reward_terms[agent][k] = (
                        self._episode_reward_terms[agent].get(k, 0.0) + float(v)
                    )
            else:
                scalar_rewards[agent] = float(raw)
                info["cost"] = pre_step_costs.get(agent, 0.0)
                info["cost_collision"] = 0.0
                info["cost_danger"] = pre_step_danger.get(agent, 0.0)
                info["cost_congestion"] = pre_step_congestion.get(agent, 0.0)
                info["original_reward"] = float(raw)

            info["safety_label"] = pre_step_labels.get(agent, 0)

            # Emit episode-level stats when the full episode ends
            if env_done:
                info["episode_cost"] = self._episode_costs.get(agent, 0.0)
                info["episode_original_reward"] = self._episode_original_rewards.get(agent, 0.0)
                info["episode_reward_terms"] = dict(
                    self._episode_reward_terms.get(agent, self._empty_reward_terms())
                )
                total_original_rew = sum(self._episode_original_rewards.values())
                info["normalized_score"] = (
                    total_original_rew / (self._max_episode_steps * self._n_agents)
                )
                info["completion_rate"] = (
                    len(self._agents_completed) / self._n_agents
                )

            infos[agent] = info

        return obs, scalar_rewards, terminations, truncations, infos

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    def state(self):
        return self._env.state()


# ──────────────────────────────────────────────────────────────────────────────
# Predefined Flatland configurations
# ──────────────────────────────────────────────────────────────────────────────

FLATLAND_CONFIGS = {
    # Original configs (default max_rails_between_cities=2)
    "flatland-small-v0": {
        "n_agents": 3, "x_dim": 25, "y_dim": 25, "n_cities": 2,
        "max_depth": 2, "predictor_max_depth": 30,
    },
    "flatland-medium-v0": {
        "n_agents": 5, "x_dim": 35, "y_dim": 35, "n_cities": 3,
        "max_depth": 2, "predictor_max_depth": 50,
    },
    "flatland-large-v0": {
        "n_agents": 10, "x_dim": 50, "y_dim": 50, "n_cities": 4,
        "max_depth": 2, "predictor_max_depth": 50,
    },
    "flatland-xlarge-v0": {
        "n_agents": 20, "x_dim": 80, "y_dim": 80, "n_cities": 6,
        "max_depth": 2, "predictor_max_depth": 50,
    },
    # Tight configs (max_rails_between_cities=1 → more resource contention)
    "flatland-small-tight-v0": {
        "n_agents": 5, "x_dim": 25, "y_dim": 25, "n_cities": 2,
        "max_depth": 2, "predictor_max_depth": 30,
        "max_rails_between_cities": 1,
    },
    "flatland-medium-tight-v0": {
        "n_agents": 8, "x_dim": 35, "y_dim": 35, "n_cities": 3,
        "max_depth": 2, "predictor_max_depth": 50,
        "max_rails_between_cities": 1,
    },
    # Malfunction configs (tight + malfunctions enabled)
    "flatland-small-malf-v0": {
        "n_agents": 5, "x_dim": 25, "y_dim": 25, "n_cities": 2,
        "max_depth": 2, "predictor_max_depth": 30,
        "max_rails_between_cities": 1,
        "malfunction_interval": 40,
    },
    "flatland-medium-malf-v0": {
        "n_agents": 8, "x_dim": 35, "y_dim": 35, "n_cities": 3,
        "max_depth": 2, "predictor_max_depth": 50,
        "max_rails_between_cities": 1,
        "malfunction_interval": 40,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Factory: create a standard Flatland env → VecEnv for SB3
# ──────────────────────────────────────────────────────────────────────────────

def make_flatland_raw_env(
    n_agents: int = 3,
    x_dim: int = 25,
    y_dim: int = 25,
    n_cities: int = 2,
    max_depth: int = 2,
    predictor_max_depth: int = 30,
    max_rails_between_cities: int = 2,
    max_rail_pairs_in_city: int = 2,
    malfunction_interval: int = 10**9,
    collision_factor: float = 1.0,
    decompose_rewards: bool = True,
    seed: Optional[int] = None,
    **env_generator_kwargs,
):
    """
    Create a standard Flatland RailEnv using env_generator with
    FlattenedNormalizedTreeObsForRailEnv observations.

    Returns the raw RailEnv (unmodified Flatland environment).

    Parameters
    ----------
    malfunction_interval : int
        Mean steps between malfunctions. Default 10^9 (disabled).
        Set to e.g. 40 for ~2.5% malfunction chance per step.
    """
    obs_builder = FlattenedNormalizedTreeObsForRailEnv(
        max_depth=max_depth,
        predictor=ShortestPathPredictorForRailEnv(max_depth=predictor_max_depth),
    )

    if decompose_rewards:
        rewards = BaseDefaultRewards(collision_factor=collision_factor)
    else:
        rewards = DefaultRewards(collision_factor=collision_factor)

    raw_env, _, _ = env_generator(
        n_agents=n_agents,
        x_dim=x_dim,
        y_dim=y_dim,
        n_cities=n_cities,
        max_rails_between_cities=max_rails_between_cities,
        max_rail_pairs_in_city=max_rail_pairs_in_city,
        malfunction_interval=malfunction_interval,
        seed=seed,
        obs_builder_object=obs_builder,
        rewards=rewards,
        **env_generator_kwargs,
    )

    # Store max_episode_steps on the env for slack normalization
    raw_env.max_episode_steps = 4 * (x_dim + y_dim + n_agents)

    return raw_env


def make_flatland_vec_env(
    env_name: str = "flatland-small-v0",
    n_envs: int = 1,
    seed: Optional[int] = None,
    decompose_rewards: bool = True,
    collision_factor: float = 1.0,
    cost_fn: str = "slack_priority",
    use_original_reward: bool = True,
    conflict_dist_thresh: float = 0.5,
    agent_dist_thresh: float = 0.5,
    slack_margin: float = 3.0,
    **override_kwargs,
):
    """
    Build a Flatland SB3 VecEnv using the official pipeline:
      RailEnv → PettingZoo → RewardCostWrapper → SuperSuit VecEnv.

    Each agent in each Flatland env gets its own VecEnv slot, so with
    n_agents=3, n_envs=2 you get num_envs=6 in the VecEnv.

    Parameters
    ----------
    env_name : str
        Key from FLATLAND_CONFIGS, or ignored if all params supplied in kwargs.
    n_envs : int
        Number of parallel Flatland environments to concatenate.
    seed : int or None
        Random seed for the RailEnv.
    decompose_rewards : bool
        Use BaseDefaultRewards (dict→scalar split) for reward/cost.
    collision_factor : float
        Collision penalty weight.
    cost_fn : str
        Cost function: "slack_priority" (default), "malfunction_detour",
        "congestion", "danger", or "collision".
    use_original_reward : bool
        If True (default), SB3 optimises the full original reward (all terms).
        If False, SB3 optimises only non-collision terms.
    conflict_dist_thresh : float
        Normalised distance threshold for the danger label.
    agent_dist_thresh : float
        Normalised distance threshold for other-agent proximity label.
    slack_margin : float
        Minimum slack gap (in raw steps) to trigger slack_priority label.
    **override_kwargs
        Override any FLATLAND_CONFIGS or env_generator parameter.

    Returns
    -------
    SB3-compatible VecEnv
    """
    config = FLATLAND_CONFIGS.get(env_name, {}).copy()
    config.update(override_kwargs)
    config["decompose_rewards"] = decompose_rewards
    config["collision_factor"] = collision_factor
    if seed is not None:
        config["seed"] = seed

    raw_env = make_flatland_raw_env(**config)

    # Flatland → PettingZoo
    pz_wrapper = PettingzooFlatland(raw_env)
    par_env = pz_wrapper.parallel_env()

    # Reward/cost decomposition + label wrapper
    n_agents = config.get("n_agents", 3)
    x_dim = config.get("x_dim", 25)
    y_dim = config.get("y_dim", 25)
    max_episode_steps = 4 * (x_dim + y_dim + n_agents)

    par_env = FlatlandRewardCostWrapper(
        par_env,
        cost_fn=cost_fn,
        use_original_reward=use_original_reward,
        conflict_dist_thresh=conflict_dist_thresh,
        agent_dist_thresh=agent_dist_thresh,
        n_agents=n_agents,
        max_episode_steps=max_episode_steps,
        slack_margin=slack_margin,
    )

    # PettingZoo → SB3 VecEnv
    vec_env = ss.pettingzoo_env_to_vec_env_v1(par_env)
    vec_env = ss.concat_vec_envs_v1(
        vec_env, n_envs, num_cpus=1, base_class="stable_baselines3"
    )

    # ConcatVecEnv lacks .seed() which SB3's set_random_seed() calls
    # via VecEnvWrapper.seed() → self.venv.seed().  Patch the inner venv.
    inner = getattr(vec_env, "venv", vec_env)
    if not hasattr(inner, "seed"):
        inner.seed = lambda s=None: None

    # Add VecCostMonitor so SB3 logs ep_rew_mean / ep_len_mean / ep_cost_mean
    vec_env = VecCostMonitor(vec_env)

    return vec_env


class VecCostMonitor(VecMonitor):
    """VecMonitor subclass that also tracks per-episode cost.

    Accumulates ``info["cost"]`` each step and stores the total in
    ``episode_info["c"]`` so that ``split_rl.py`` can log
    ``rollout/ep_cost_mean`` in safety mode.
    Also tracks cost_collision and cost_danger separately for monitoring.
    """

    def __init__(self, venv, filename=None, info_keywords=()):
        super().__init__(venv, filename=filename, info_keywords=info_keywords)
        self.episode_costs = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_cost_collision = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_cost_danger = np.zeros(self.num_envs, dtype=np.float64)

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.episode_costs[:] = 0.0
        self.episode_cost_collision[:] = 0.0
        self.episode_cost_danger[:] = 0.0
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.episode_returns += rewards
        self.episode_lengths += 1
        for i in range(self.num_envs):
            self.episode_costs[i] += infos[i].get("cost", 0.0)
            self.episode_cost_collision[i] += infos[i].get("cost_collision", 0.0)
            self.episode_cost_danger[i] += infos[i].get("cost_danger", 0.0)
        new_infos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                ep_rew = self.episode_returns[i]
                ep_cost = self.episode_costs[i]
                episode_info = {
                    "r": ep_rew,
                    "l": self.episode_lengths[i],
                    "c": ep_cost,
                    "cost_collision": float(self.episode_cost_collision[i]),
                    "cost_danger": float(self.episode_cost_danger[i]),
                    "original_r": ep_rew - ep_cost,
                    "t": round(time.time() - self.t_start, 6),
                }
                # Flatland-specific metrics
                if "normalized_score" in info:
                    episode_info["normalized_score"] = info["normalized_score"]
                if "completion_rate" in info:
                    episode_info["completion_rate"] = info["completion_rate"]
                for key in self.info_keywords:
                    episode_info[key] = info[key]
                info["episode"] = episode_info
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                self.episode_costs[i] = 0.0
                self.episode_cost_collision[i] = 0.0
                self.episode_cost_danger[i] = 0.0
                if self.results_writer:
                    self.results_writer.write_row(episode_info)
                new_infos[i] = info
        return obs, rewards, dones, new_infos
