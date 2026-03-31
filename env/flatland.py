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
Actions: Discrete(5) — DO_NOTHING, MOVE_LEFT, MOVE_FORWARD, MOVE_RIGHT, STOP_MOVING.
Rewards: per-agent scalar (configurable: decomposed or original).
Cost:    per-agent collision penalty in info['cost'].
Label:   per-agent binary safety_label in info['safety_label'].
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
# PettingZoo wrapper: reward/cost split + label
# ──────────────────────────────────────────────────────────────────────────────

class FlatlandRewardCostWrapper(ParallelEnv):
    """
    Thin PettingZoo ParallelEnv wrapper that intercepts Flatland's
    BaseDefaultRewards dict-valued rewards and converts them to:

      reward = sum of non-collision terms  (scalar float)
      info['cost']            = configurable cost signal (default: danger label)
      info['cost_collision']  = collision penalty >= 0 (always tracked)
      info['original_reward'] = full scalar sum of ALL terms
      info['safety_label']    = binary danger label from tree obs
      info['reward_terms']    = raw dict of all penalty terms

    Also casts observations to float32 (Flatland's obs builder declares
    float64, which is unnecessary memory/throughput overhead for SB3).

    Tracks per-episode cumulative cost, original reward, and reward
    terms, emitted in info at episode end.

    Cost functions (selected via ``cost_fn``):
      - "congestion" (default): 1 if the agent moved into a contested direction
        (opposite-direction agents or potential conflict on chosen branch).
        Creates structural conflict: moving through busy areas = progress +
        blocking/congestion. The tree cost head can route on the CHOSEN branch's
        features while the reward head routes on other features.
      - "danger": 1 if the agent's observation indicates nearby
        conflict or other agents. Positively correlated with productive
        navigation — NOT a structural conflict.
      - "collision": legacy cost = collision penalty magnitude. NOT recommended
        for SPLIT-RL: negatively correlated with non-collision reward (r ~ -0.38).

    Parameters
    ----------
    env : ParallelEnv
        PettingZoo parallel env from PettingzooFlatland.
    cost_fn : str
        Cost function: "danger" (observation-based, default) or "collision" (legacy).
    use_original_reward : bool
        If True, SB3 optimises the full original reward (all terms
        including collision).  If False (default), SB3 optimises only
        the non-collision terms.  Either way, info always contains
        both cost and original_reward for monitoring.
    conflict_dist_thresh : float
        Normalised distance threshold for the conflict-detected and
        other-agent-encountered features in the label function.
    agent_dist_thresh : float
        Normalised distance threshold for other-agent proximity.
    """

    def __init__(
        self,
        env: ParallelEnv,
        cost_fn: str = "congestion",
        use_original_reward: bool = False,
        conflict_dist_thresh: float = 0.5,
        agent_dist_thresh: float = 0.5,
        n_agents: int = 1,
        max_episode_steps: int = 200,
    ):
        if cost_fn not in ("danger", "collision", "congestion"):
            raise ValueError(
                f"Unknown cost_fn '{cost_fn}'. Use 'congestion', 'danger', or 'collision'."
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

        # Override observation spaces to float32
        self.observation_spaces = {
            agent: gym.spaces.Box(
                low=space.low.astype(np.float32),
                high=space.high.astype(np.float32),
                shape=space.shape,
                dtype=np.float32,
            )
            for agent, space in env.observation_spaces.items()
        }
        self.action_spaces = env.action_spaces

        # Infer n_nodes from observation shape (obs_dim = 12 * n_nodes)
        sample_space = next(iter(self.observation_spaces.values()))
        self._n_nodes = sample_space.shape[0] // 12

        # Stored observations for pre-step label computation
        self._prev_obs: Dict[str, np.ndarray] = {}

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

    def _empty_reward_terms(self) -> Dict[str, float]:
        return {p.value: 0.0 for p in DefaultPenalties}

    def reset(self, seed=None, options=None):
        obs, infos = self._env.reset(seed=seed, options=options)
        self.agents = self._env.agents
        obs = self._cast_obs(obs)

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
            infos[agent]["safety_label"] = compute_danger_label(
                obs[agent], self._n_nodes,
                self._conflict_dist_thresh, self._agent_dist_thresh,
            )
        return obs, infos

    def step(self, actions):
        # Compute labels from PRE-STEP observations (current state)
        pre_step_labels = {}
        pre_step_congestion = {}
        for agent in actions:
            if agent in self._prev_obs:
                pre_step_labels[agent] = compute_danger_label(
                    self._prev_obs[agent], self._n_nodes,
                    self._conflict_dist_thresh, self._agent_dist_thresh,
                )
                pre_step_congestion[agent] = compute_congestion_cost(
                    self._prev_obs[agent], self._n_nodes,
                    actions[agent],
                    self._conflict_dist_thresh, self._agent_dist_thresh,
                )
            else:
                pre_step_labels[agent] = 0
                pre_step_congestion[agent] = 0

        obs, rewards, terminations, truncations, infos = self._env.step(actions)
        self.agents = self._env.agents
        obs = self._cast_obs(obs)

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

                # Cost signal depends on cost_fn
                danger_label = float(pre_step_labels.get(agent, 0))
                congestion_label = float(pre_step_congestion.get(agent, 0))
                if self._cost_fn == "congestion":
                    info["cost"] = congestion_label
                elif self._cost_fn == "danger":
                    info["cost"] = danger_label
                else:  # "collision"
                    info["cost"] = collision_cost
                info["cost_collision"] = collision_cost
                info["cost_danger"] = danger_label
                info["cost_congestion"] = congestion_label
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
                info["cost"] = 0.0
                info["cost_collision"] = 0.0
                info["cost_danger"] = 0.0
                info["cost_congestion"] = 0.0
                info["original_reward"] = float(raw)

            info["safety_label"] = pre_step_labels.get(agent, 0)

            # Emit episode-level stats when the full episode ends
            if env_done:
                info["episode_cost"] = self._episode_costs.get(agent, 0.0)
                info["episode_original_reward"] = self._episode_original_rewards.get(agent, 0.0)
                info["episode_reward_terms"] = dict(
                    self._episode_reward_terms.get(agent, self._empty_reward_terms())
                )
                # Flatland normalized score:
                #   sum(all agents' original rewards) / (max_ep_steps * n_agents)
                # Perfect (no penalties) = 0, worst ≈ -1
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
        "max_depth": 3, "predictor_max_depth": 50,
    },
    "flatland-xlarge-v0": {
        "n_agents": 20, "x_dim": 80, "y_dim": 80, "n_cities": 6,
        "max_depth": 3, "predictor_max_depth": 50,
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
    return raw_env


def make_flatland_vec_env(
    env_name: str = "flatland-small-v0",
    n_envs: int = 1,
    seed: Optional[int] = None,
    decompose_rewards: bool = True,
    collision_factor: float = 1.0,
    cost_fn: str = "congestion",
    use_original_reward: bool = False,
    conflict_dist_thresh: float = 0.5,
    agent_dist_thresh: float = 0.5,
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
    use_original_reward : bool
        If True, SB3 optimises the full original reward (all terms).
        If False, SB3 optimises only non-collision terms.
    conflict_dist_thresh : float
        Normalised distance threshold for the danger label.
    agent_dist_thresh : float
        Normalised distance threshold for other-agent proximity label.
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
