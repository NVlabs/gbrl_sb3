##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
"""
Wrappers around SUMO-RL for use with SB3 via shared-policy parameter sharing.

SUMO-RL exposes a PettingZoo ParallelEnv where each traffic signal is an agent
with its own observation/action space. For shared-policy training (single SB3
policy controlling all agents), we convert to a VecEnv where each agent is a
separate "environment slot" for shared-policy parameter sharing.

Pipeline:  sumo_rl.parallel_env(...)
        → SumoRewardCostWrapper  (reward/cost split via native SUMO events)
        → SuperSuit pettingzoo_env_to_vec_env_v1 + concat_vec_envs_v1
        → VecCostMonitor
        → SB3 VecEnv

Observations: [phase_one_hot, min_green, lane_densities, lane_queues]
  - Dimension varies per intersection topology (different #lanes).
  - For shared policy, all intersections must have the SAME obs/action dims.
    SUMO-RL RESCO benchmarks satisfy this for grid networks (grid4x4, arterial4x4).
Actions: Discrete(n_phases) — select next green phase.
Rewards: per-agent scalar — negative pressure (default) or diff-waiting-time.
Cost:    fairness (default), or legacy queue/wait/saturation/conflict.
Label:   cost-advantage based (computed post-rollout in SPLIT-RL), with
         event-proximal fallback from wrapper.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv
from stable_baselines3.common.vec_env import VecMonitor

import supersuit as ss


# ──────────────────────────────────────────────────────────────────────────────
# Cost function registry
# ──────────────────────────────────────────────────────────────────────────────

COST_FN_REGISTRY: Dict[str, str] = {
    "service_gap": "Per-agent: 1 when any side-street lane has demand but hasn't been served for > tau_service steps",
    "directional": "Per-agent: max(0, side-street waiting increase) — service deficit constraint",
    "starvation": "Per-agent: 1 if max halting vehicles on non-priority lanes > starvation_threshold (CC-like conflict with directional reward)",
    "conflict": "Per-agent: 1 if fairness OR phase-churn fires",
    "fairness": "Per-agent: 1 if max_wait > tau_abs AND max_wait/(mean_wait+eps) > rho",
    "phase_churn": "Per-agent: 1 if phase changed this step",
    "combined": "Per-agent: sum of all legacy costs (queue + wait + saturation)",
    "queue_overflow": "Per-agent: 1 if total queued vehicles > queue_threshold",
    "max_wait": "Per-agent: 1 if max accumulated wait on any lane > wait_threshold",
    "lane_saturation": "Per-agent: 1 if any lane queue ratio > saturation_threshold",
    "emergency": "Global: 1 if emergency stops occur (+ teleport_weight if teleports)",
    "side_deficit": "Per-agent: continuous sum_l q_l * max(0, g_l - tau) — severity-weighted side-street neglect",
    "side_queue": "Per-agent: max queue ratio on side-street lanes — purely observation-based, no hidden state",
    "bus_priority": "Per-agent: max(bus_wait/T_bus) across unserved lanes with buses — volume-driven priority conflict, obs-extended with has_bus+bus_count+bus_wait",
    "convoy_priority": "Per-agent: 1 if convoy is being split (lead passed, tail stuck on unserved lane) — platoon integrity constraint, obs-extended",
    "spillback": "Per-agent: max downstream occupancy on currently served lanes — road works / spillback protection, obs-extended",
    "premium_priority": "Per-agent: steep cost when premium vehicle waits on unserved lane past T_premium — value-asymmetry priority, obs-extended",
}


# ──────────────────────────────────────────────────────────────────────────────
# PettingZoo wrapper: reward/cost split with configurable cost functions
# ──────────────────────────────────────────────────────────────────────────────

class SumoRewardCostWrapper(ParallelEnv):
    """
    PettingZoo ParallelEnv wrapper around sumo_rl.parallel_env that:

      reward = throughput (vehicles on outgoing lanes) or original sumo-rl reward
      info['cost']            = configurable constraint violation cost
      info['original_reward'] = original sumo-rl reward (e.g. diff-waiting-time)
      info['safety_label']    = event-proximal fallback label (1 if cost > 0
                                within last ``label_horizon`` steps).
                                The preferred label is cost-advantage based,
                                computed post-rollout in SPLIT-RL.

    Cost functions (selected via ``cost_fn``):
      - "conflict": 1 if fairness OR phase-churn fires (default — genuinely conflicts with throughput)
      - "fairness": 1 if max_wait > tau_abs AND max_wait/(mean_wait+eps) > rho
      - "phase_churn": 1 if agent switched phase this step
      - "combined": legacy sum of queue + wait + saturation (correlated with diff-wait reward)
      - "queue_overflow": per-agent, 1 if queued vehicles > queue_threshold
      - "max_wait": per-agent, 1 if max lane wait > wait_threshold seconds
      - "lane_saturation": per-agent, 1 if any lane queue ratio > saturation_threshold
      - "emergency": global, from native TraCI failure events

    The "fairness" cost + throughput reward creates a genuine trade-off:
    throughput wants maximum vehicle flow (greedy phase holding), but fairness
    penalizes lane starvation when one lane's max wait dwarfs the mean.
    NOTE: "conflict" (fairness OR churn) is NOT recommended — churn is
    negatively correlated with throughput, masking the fairness signal.

    Parameters
    ----------
    env : ParallelEnv
        PettingZoo parallel env from sumo_rl.parallel_env().
    override_reward : str or None
        If "throughput", replace sumo-rl reward with outgoing vehicle count.
        None keeps the original sumo-rl reward (e.g., diff-waiting-time).
    cost_fn : str
        Cost function to use. See COST_FN_REGISTRY.
    fairness_tau : float
        Absolute floor for fairness cost: max_wait must exceed this (seconds).
    fairness_rho : float
        Relative ratio for fairness: max_wait / (mean_wait + eps) must exceed this.
    queue_threshold : int
        For legacy cost_fn="queue_overflow": max allowed queued vehicles.
    wait_threshold : float
        For legacy cost_fn="max_wait": max allowed accumulated wait (seconds).
    saturation_threshold : float
        For legacy cost_fn="lane_saturation": max allowed lane queue ratio (0-1).
    teleport_weight : float
        For cost_fn="emergency": multiplier for teleport events.
    label_horizon : int
        Number of past steps to look back for the event-proximal fallback label.
    """

    def __init__(
        self,
        env: ParallelEnv,
        override_reward: Optional[str] = None,
        cost_fn: str = "conflict",
        starvation_threshold: int = 3,
        fairness_tau: float = 120.0,
        fairness_rho: float = 3.0,
        queue_threshold: int = 10,
        wait_threshold: float = 200.0,
        saturation_threshold: float = 0.8,
        teleport_weight: float = 1.0,
        label_horizon: int = 3,
        use_categorical_phase: bool = False,
        mainline_direction: str = "auto",
        tau_service: int = 6,
        side_queue_cap: float = 0.7,
        deficit_kappa: float = 5.0,
        # Bus / convoy injection parameters
        bus_injection_interval: float = 25.0,
        bus_count_per_injection: int = 1,
        bus_cost_threshold: float = 30.0,
        bus_warn_threshold: float = 10.0,
        convoy_injection_interval: float = 80.0,
        convoy_size_min: int = 10,
        convoy_size_max: int = 12,
        convoy_headway: float = 2.0,
        convoy_cost_threshold: float = 20.0,
        convoy_warn_threshold: float = 8.0,
        # Premium vehicle injection parameters
        premium_injection_interval: float = 40.0,
        premium_cost_threshold: float = 15.0,
        premium_warn_threshold: float = 5.0,
        premium_pressure_beta: float = 1.5,
        # Spillback / road works parameters
        roadwork_interval_mean: float = 300.0,
        roadwork_duration_mean: float = 400.0,
        roadwork_speed: float = 0.3,
        spillback_occ_threshold: float = 0.5,
        # Scenario event probability (0.0 = events every episode, 0.25 = 25% clean)
        clean_episode_prob: float = 0.0,
    ):
        if cost_fn not in COST_FN_REGISTRY:
            raise ValueError(
                f"Unknown cost_fn '{cost_fn}'. Available: {list(COST_FN_REGISTRY.keys())}"
            )
        if override_reward is not None and override_reward not in ("throughput", "directional", "ns_wait", "arterial", "mainline"):
            raise ValueError(
                f"Unknown override_reward '{override_reward}'. Use 'mainline', 'arterial', 'throughput', 'directional', 'ns_wait', or None."
            )
        self._env = env
        self.possible_agents = env.possible_agents
        self.agents = list(env.possible_agents)
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)

        self._override_reward = override_reward
        self._cost_fn = cost_fn
        self._fairness_tau = fairness_tau
        self._fairness_rho = fairness_rho
        self._queue_threshold = queue_threshold
        self._wait_threshold = wait_threshold
        self._saturation_threshold = saturation_threshold
        self._teleport_weight = teleport_weight
        self._starvation_threshold = starvation_threshold
        self._label_horizon = label_horizon
        self._use_categorical_phase = use_categorical_phase
        self._mainline_direction = mainline_direction
        self._tau_service = tau_service
        self._side_queue_cap = side_queue_cap
        self._deficit_kappa = deficit_kappa

        # Bus / convoy injection parameters
        self._bus_injection_interval = bus_injection_interval
        self._bus_count_per_injection = bus_count_per_injection
        self._bus_cost_threshold = bus_cost_threshold    # T_bus
        self._bus_warn_threshold = bus_warn_threshold    # T_warn
        self._convoy_injection_interval = convoy_injection_interval
        self._convoy_size_min = convoy_size_min
        self._convoy_size_max = convoy_size_max
        self._convoy_headway = convoy_headway
        self._convoy_cost_threshold = convoy_cost_threshold  # T_convoy
        self._convoy_warn_threshold = convoy_warn_threshold  # T_warn_convoy

        # Premium vehicle injection parameters
        self._premium_injection_interval = premium_injection_interval
        self._premium_cost_threshold = premium_cost_threshold   # T_premium
        self._premium_warn_threshold = premium_warn_threshold   # T_warn_premium
        self._premium_pressure_beta = premium_pressure_beta      # beta for label pressure ratio

        # Spillback / road works parameters
        self._roadwork_interval_mean = roadwork_interval_mean
        self._roadwork_duration_mean = roadwork_duration_mean
        self._roadwork_speed = roadwork_speed            # near-zero speed on blocked edge
        self._spillback_occ_threshold = spillback_occ_threshold  # T_occ for label

        # Clean episode probability (shared across all event-based scenarios)
        self._clean_episode_prob = clean_episode_prob
        self._is_clean_episode = False  # set at each reset()

        # Bus / convoy runtime state (populated in reset)
        self._boundary_edges: List[str] = []       # edges with dead-end nodes
        self._next_bus_time: float = 0.0           # sim-time for next bus injection
        self._next_convoy_time: float = 0.0        # sim-time for next convoy injection
        self._next_premium_time: float = 0.0       # sim-time for next premium injection
        self._bus_vtype_created: bool = False
        self._convoy_vtype_created: bool = False
        self._premium_vtype_created: bool = False
        self._bus_counter: int = 0
        self._convoy_counter: int = 0
        self._premium_counter: int = 0
        # Per-agent per-lane bus/convoy/premium features (updated each step)
        self._has_bus: Dict[str, np.ndarray] = {}
        self._bus_wait: Dict[str, np.ndarray] = {}
        self._bus_count: Dict[str, np.ndarray] = {}  # number of buses per lane
        self._has_convoy: Dict[str, np.ndarray] = {}
        self._convoy_count: Dict[str, np.ndarray] = {}
        self._convoy_wait: Dict[str, np.ndarray] = {}
        self._convoy_progress: Dict[str, np.ndarray] = {}  # fraction of convoy past stop line
        # Track which convoy IDs belong to which platoon for split detection
        self._active_convoys: Dict[int, List[str]] = {}  # convoy_id → [veh_id, ...]
        # Per-agent tracking: which convoy vehicle IDs have ever been seen
        # on this agent's incoming lanes.  Used for correct per-intersection
        # progress: progress = (ever_seen − still_on_incoming) / ever_seen
        self._convoy_seen_by_agent: Dict[str, Dict[int, Set[str]]] = {}  # agent → convoy_id → {vid, …}
        # Premium vehicle features
        self._has_premium: Dict[str, np.ndarray] = {}
        self._premium_wait: Dict[str, np.ndarray] = {}

        # Spillback / road works runtime state
        self._lane_to_downstream_edges: Dict[str, Dict[int, List[str]]] = {}  # agent → lane_idx → [out_edge, ...]
        self._downstream_occ: Dict[str, np.ndarray] = {}  # per-agent per-lane downstream occupancy
        self._active_roadworks: List[Dict] = []  # [{edge, original_speed, clear_time}, ...]
        self._internal_edges: List[str] = []  # candidate edges for road works
        self._next_roadwork_time: float = 0.0
        self._roadwork_counter: int = 0

        # Number of green phases per agent (for obs transformation)
        self._n_green_phases: Optional[int] = None

        # Per-agent lane split for directional reward/starvation cost
        # Populated in reset() once TraCI is available
        self._priority_lanes: Dict[str, List[str]] = {}
        self._non_priority_lanes: Dict[str, List[str]] = {}

        # Per-agent mainline/side-street lane indices (into ts.lanes)
        # Populated in reset() via _classify_mainline_side()
        self._mainline_lane_indices: Dict[str, List[int]] = {}
        self._side_lane_indices: Dict[str, List[int]] = {}
        # Previous-step accumulated waiting for diff computation
        self._prev_mainline_wait: Dict[str, float] = {}
        self._prev_side_wait: Dict[str, float] = {}
        # Per-agent consecutive steps without side-street service
        self._side_consecutive_unserved: Dict[str, int] = {}

        # Phase → served lane indices mapping (populated in reset)
        # phase_to_lanes[agent_id][green_phase_idx] = [lane_idx, ...]
        self._phase_to_lanes: Dict[str, Dict[int, List[int]]] = {}
        # Per-agent, per-side-lane steps since that lane was last served
        # steps_since_served[agent_id][lane_idx] = int
        self._steps_since_served: Dict[str, Dict[int, int]] = {}

        # Cache original observation/action spaces
        raw_obs_spaces = {
            agent: env.observation_space(agent) for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: env.action_space(agent) for agent in self.possible_agents
        }

        # Validate uniform action spaces (phase selection must match)
        sample_act = next(iter(self.action_spaces.values()))
        for agent in self.possible_agents:
            assert self.action_spaces[agent].n == sample_act.n, (
                f"Agent {agent} action space {self.action_spaces[agent].n} "
                f"!= {sample_act.n}. Shared policy requires uniform action spaces."
            )

        # Pad observation spaces to max dim for real-world networks with
        # non-uniform intersection topologies (different #lanes → different obs dims).
        obs_dims = [raw_obs_spaces[a].shape[0] for a in self.possible_agents]
        self._max_obs_dim = max(obs_dims)
        self._needs_obs_padding = not all(d == self._max_obs_dim for d in obs_dims)
        self._agent_obs_dims = {a: raw_obs_spaces[a].shape[0] for a in self.possible_agents}

        if self._use_categorical_phase:
            # Detect n_phases from action space for later VecEnv-level transformation
            sample_agent = self.possible_agents[0]
            self._n_green_phases = self.action_spaces[sample_agent].n

        # Bus/convoy obs extension: compute n_lanes from obs layout
        # obs = [phase_one_hot(n_phases), min_green(1), density(n_lanes), queue(n_lanes)]
        # n_lanes = (obs_dim - n_phases - 1) / 2
        self._obs_extension_dim = 0
        self._raw_max_obs_dim = self._max_obs_dim  # pre-extension dim for padding
        if self._cost_fn == "bus_priority":
            n_phases_act = next(iter(self.action_spaces.values())).n
            sample_obs_dim = next(iter(raw_obs_spaces.values())).shape[0]
            n_lanes = (sample_obs_dim - n_phases_act - 1) // 2
            self._obs_extension_dim = 3 * n_lanes  # has_bus + bus_count + bus_wait per lane
            self._max_obs_dim += self._obs_extension_dim
        elif self._cost_fn == "convoy_priority":
            n_phases_act = next(iter(self.action_spaces.values())).n
            sample_obs_dim = next(iter(raw_obs_spaces.values())).shape[0]
            n_lanes = (sample_obs_dim - n_phases_act - 1) // 2
            self._obs_extension_dim = 3 * n_lanes  # has_convoy + convoy_count + convoy_progress
            self._max_obs_dim += self._obs_extension_dim
        elif self._cost_fn == "spillback":
            n_phases_act = next(iter(self.action_spaces.values())).n
            sample_obs_dim = next(iter(raw_obs_spaces.values())).shape[0]
            n_lanes = (sample_obs_dim - n_phases_act - 1) // 2
            self._obs_extension_dim = 1 * n_lanes  # downstream_occ per lane
            self._max_obs_dim += self._obs_extension_dim
        elif self._cost_fn == "premium_priority":
            n_phases_act = next(iter(self.action_spaces.values())).n
            sample_obs_dim = next(iter(raw_obs_spaces.values())).shape[0]
            n_lanes = (sample_obs_dim - n_phases_act - 1) // 2
            self._obs_extension_dim = 2 * n_lanes  # has_premium + premium_wait per lane
            self._max_obs_dim += self._obs_extension_dim

        if self._needs_obs_padding or self._obs_extension_dim > 0:
            padded_space = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self._max_obs_dim,), dtype=np.float32,
            )
            self.observation_spaces = {a: padded_space for a in self.possible_agents}
        else:
            self.observation_spaces = raw_obs_spaces

        # TraCI connection (set after first reset)
        self._sumo = None
        self._sumo_env = None  # underlying SumoEnvironment

        # Accumulated failure events across sub-steps (for cost_fn="emergency")
        self._accumulated_emergency_stops = 0
        self._accumulated_teleports = 0

        # Per-agent ring buffer for event-proximal labels
        self._failure_history: Dict[str, List[bool]] = {
            a: [] for a in self.possible_agents
        }

        # Per-agent phase tracking for churn cost
        self._last_green_phase: Dict[str, Optional[int]] = {
            a: None for a in self.possible_agents
        }

        # Episode accumulators
        self._episode_costs: Dict = {}
        self._episode_original_rewards: Dict = {}
        self._episode_max_queued: Dict[str, int] = {}
        self._episode_max_wait: Dict[str, float] = {}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _transform_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Zero-pad observations for non-uniform networks.

        Pads to self._raw_max_obs_dim (pre-extension). Bus/convoy
        features are appended separately by _extend_obs().

        Categorical phase transformation is handled at the VecEnv level
        (VecCategoricalPhase wrapper) because SuperSuit cannot concatenate
        object arrays.
        """
        if not self._needs_obs_padding:
            return obs
        padded = {}
        for agent, ob in obs.items():
            ob = np.asarray(ob, dtype=np.float32)
            if ob.shape[0] < self._raw_max_obs_dim:
                ob = np.pad(ob, (0, self._raw_max_obs_dim - ob.shape[0]))
            padded[agent] = ob
        return padded

    def _get_sumo_env(self):
        """Get the underlying SumoEnvironment instance."""
        if self._sumo_env is None:
            env = self._env.unwrapped
            if hasattr(env, "env"):
                env = env.env
            self._sumo_env = env
        return self._sumo_env

    def _get_traci(self):
        """Get TraCI connection from the underlying SumoEnvironment."""
        if self._sumo is None:
            sumo_env = self._get_sumo_env()
            self._sumo = getattr(sumo_env, "sumo", None)
        return self._sumo

    def _get_traffic_signal(self, agent_id: str):
        """Get the TrafficSignal object for a specific agent."""
        sumo_env = self._get_sumo_env()
        if sumo_env is not None and hasattr(sumo_env, 'traffic_signals'):
            return sumo_env.traffic_signals.get(agent_id)
        return None

    def _patch_sumo_step(self):
        """Monkey-patch SumoEnvironment._sumo_step to accumulate failure events.

        Only needed for cost_fn="emergency". For other cost functions we query
        TrafficSignal objects directly after the step.
        """
        if self._cost_fn != "emergency":
            return

        sumo_env = self._get_sumo_env()
        if sumo_env is None or not hasattr(sumo_env, '_sumo_step'):
            return
        if hasattr(sumo_env, '_original_sumo_step'):
            return  # already patched

        import types
        wrapper = self  # capture reference

        original_step = sumo_env._sumo_step

        def patched_sumo_step(self_env):
            original_step()
            conn = getattr(self_env, 'sumo', None)
            if conn is not None:
                wrapper._accumulated_emergency_stops += conn.simulation.getEmergencyStoppingVehiclesNumber()
                wrapper._accumulated_teleports += conn.simulation.getStartingTeleportNumber()

        sumo_env._original_sumo_step = original_step
        sumo_env._sumo_step = types.MethodType(patched_sumo_step, sumo_env)

    # ── Lane direction classification ────────────────────────────────────

    def _classify_lanes_by_direction(self):
        """Classify each agent's incoming lanes as NS or EW using geometry.

        Uses TraCI junction positions to determine whether each incoming
        lane comes from a North/South direction or East/West direction.
        Populates self._ns_lane_indices and self._ew_lane_indices with
        indices into ts.lanes (the incoming lane list).
        """
        sumo = self._get_traci()
        if sumo is None:
            return

        self._ns_lane_indices.clear()
        self._ew_lane_indices.clear()
        self._prev_ns_wait.clear()
        self._prev_ew_wait.clear()

        for agent_id in self.possible_agents:
            ts = self._get_traffic_signal(agent_id)
            if ts is None:
                continue

            agent_pos = sumo.junction.getPosition(agent_id)
            ns_idx = []
            ew_idx = []

            for i, lane in enumerate(ts.lanes):
                edge = '_'.join(lane.split('_')[:-1])
                src_node = sumo.edge.getFromJunction(edge)
                src_pos = sumo.junction.getPosition(src_node)

                dx = src_pos[0] - agent_pos[0]
                dy = src_pos[1] - agent_pos[1]

                if abs(dy) >= abs(dx):
                    ns_idx.append(i)  # North or South
                else:
                    ew_idx.append(i)  # East or West

            self._ns_lane_indices[agent_id] = ns_idx
            self._ew_lane_indices[agent_id] = ew_idx
            self._prev_ns_wait[agent_id] = 0.0
            self._prev_ew_wait[agent_id] = 0.0

    def _classify_mainline_side(self):
        """Classify lanes as mainline vs side-street by traffic volume proxy.

        For asymmetric networks (arterial4x4): the direction with more
        incoming lanes is the mainline (higher capacity = higher demand).
        For symmetric networks (grid4x4): picks EW as mainline arbitrarily
        (both directions have equal lane counts, so the split is symmetric).

        Also builds the phase→served-lanes mapping needed for service-gap
        tracking, and initialises per-side-lane steps_since_served counters.
        """
        sumo = self._get_traci()
        if sumo is None:
            return

        self._mainline_lane_indices.clear()
        self._side_lane_indices.clear()
        self._prev_mainline_wait.clear()
        self._prev_side_wait.clear()
        self._phase_to_lanes.clear()
        self._steps_since_served.clear()
        self._side_consecutive_unserved.clear()

        for agent_id in self.possible_agents:
            ts = self._get_traffic_signal(agent_id)
            if ts is None:
                continue

            agent_pos = sumo.junction.getPosition(agent_id)

            # Classify each lane as NS or EW
            ns_idx = []
            ew_idx = []
            for i, lane in enumerate(ts.lanes):
                edge = '_'.join(lane.split('_')[:-1])
                src_node = sumo.edge.getFromJunction(edge)
                src_pos = sumo.junction.getPosition(src_node)
                dx = src_pos[0] - agent_pos[0]
                dy = src_pos[1] - agent_pos[1]
                if abs(dy) >= abs(dx):
                    ns_idx.append(i)
                else:
                    ew_idx.append(i)

            # Mainline = direction with more lanes (or EW if tied)
            if len(ew_idx) >= len(ns_idx):
                self._mainline_lane_indices[agent_id] = ew_idx
                self._side_lane_indices[agent_id] = ns_idx
            else:
                self._mainline_lane_indices[agent_id] = ns_idx
                self._side_lane_indices[agent_id] = ew_idx

            self._prev_mainline_wait[agent_id] = 0.0
            self._prev_side_wait[agent_id] = 0.0
            self._side_consecutive_unserved[agent_id] = 0

            # Build phase → served lane indices mapping
            connections = sumo.trafficlight.getControlledLinks(agent_id)
            logic = sumo.trafficlight.getAllProgramLogics(agent_id)[0]
            phase_map = {}
            green_idx = 0
            for phase in logic.phases:
                if 'G' not in phase.state and 'g' not in phase.state:
                    continue
                served = set()
                for ci, c in enumerate(phase.state):
                    if c in ('G', 'g') and ci < len(connections):
                        for (in_lane, _, _) in connections[ci]:
                            if in_lane in ts.lanes:
                                served.add(ts.lanes.index(in_lane))
                phase_map[green_idx] = sorted(served)
                green_idx += 1
            self._phase_to_lanes[agent_id] = phase_map

            # Init steps_since_served for each side-street lane
            side_idx = self._side_lane_indices[agent_id]
            self._steps_since_served[agent_id] = {li: 0 for li in side_idx}

    # ── Lane → downstream edge mapping (for spillback) ───────────────

    def _build_lane_downstream_mapping(self):
        """Build mapping: agent → lane_idx → list of downstream edge IDs.

        Uses getControlledLinks which returns (in_lane, out_lane, via_lane)
        per connection. The out_lane's parent edge is the downstream edge
        that vehicles from in_lane travel to.
        """
        sumo = self._get_traci()
        if sumo is None:
            return

        self._lane_to_downstream_edges.clear()

        for agent_id in self.possible_agents:
            ts = self._get_traffic_signal(agent_id)
            if ts is None:
                continue

            connections = sumo.trafficlight.getControlledLinks(agent_id)
            lane_downstream: Dict[int, set] = {i: set() for i in range(len(ts.lanes))}

            for conn_group in connections:
                for (in_lane, out_lane, _via) in conn_group:
                    if in_lane in ts.lanes:
                        lane_idx = ts.lanes.index(in_lane)
                        # out_lane format: "edgeID_laneIndex" → extract edge
                        out_edge = '_'.join(out_lane.split('_')[:-1])
                        if out_edge and not out_edge.startswith(':'):
                            lane_downstream[lane_idx].add(out_edge)

            self._lane_to_downstream_edges[agent_id] = {
                k: list(v) for k, v in lane_downstream.items()
            }

    def _discover_internal_edges(self):
        """Find internal (non-boundary) edges suitable for road works.

        Road works should be placed on edges between two intersections,
        not on boundary edges where vehicles enter/exit the network.
        Only considers edges that appear as downstream targets for at
        least one controlled lane (so blocking them has cost impact).
        """
        sumo = self._get_traci()
        if sumo is None:
            return

        # Collect all downstream edges referenced by any agent
        downstream_set = set()
        for agent_mapping in self._lane_to_downstream_edges.values():
            for edges in agent_mapping.values():
                downstream_set.update(edges)

        # Internal = downstream edges that are NOT boundary edges
        all_edges = sumo.edge.getIDList()
        normal_edges = [e for e in all_edges if not e.startswith(':')]

        from collections import Counter
        node_edge_count = Counter()
        for e in normal_edges:
            node_edge_count[sumo.edge.getFromJunction(e)] += 1
            node_edge_count[sumo.edge.getToJunction(e)] += 1
        boundary_nodes = {n for n, c in node_edge_count.items() if c <= 2}

        self._internal_edges = [
            e for e in downstream_set
            if (sumo.edge.getFromJunction(e) not in boundary_nodes
                and sumo.edge.getToJunction(e) not in boundary_nodes)
        ]
        # Fallback: if network is too small, use all downstream edges
        if len(self._internal_edges) < 2:
            self._internal_edges = list(downstream_set)

    # ── Road works injection / clearing (for spillback) ──────────────

    def _inject_roadwork(self, sim_time: float):
        """Possibly inject a new road works event (reduce edge speed)."""
        if sim_time < self._next_roadwork_time:
            return
        if not self._internal_edges:
            return

        sumo = self._get_traci()
        if sumo is None:
            return

        rng = np.random.RandomState(int(sim_time * 1000) % (2**31))

        # Pick a random internal edge not already blocked
        blocked_edges = {rw["edge"] for rw in self._active_roadworks}
        candidates = [e for e in self._internal_edges if e not in blocked_edges]
        if not candidates:
            # All internal edges blocked, skip
            self._next_roadwork_time = sim_time + 60.0
            return

        edge = rng.choice(candidates)

        # Save original speed and reduce to near-zero
        try:
            original_speed = sumo.edge.getMaxSpeed(edge)
            sumo.edge.setMaxSpeed(edge, self._roadwork_speed)
        except Exception:
            self._next_roadwork_time = sim_time + 60.0
            return

        # Random duration: mean ± 40% jitter
        duration = self._roadwork_duration_mean * (0.6 + 0.8 * rng.random())
        clear_time = sim_time + duration

        self._active_roadworks.append({
            "edge": edge,
            "original_speed": original_speed,
            "clear_time": clear_time,
        })
        self._roadwork_counter += 1

        # Schedule next road work: maybe another, maybe not (randomness)
        # 70% chance of another road work after interval, 30% skip
        if rng.random() < 0.7:
            interval = self._roadwork_interval_mean * (0.6 + 0.8 * rng.random())
            self._next_roadwork_time = sim_time + interval
        else:
            # Skip this cycle — no new road work for a while
            self._next_roadwork_time = sim_time + self._roadwork_interval_mean * 2.0

    def _clear_roadworks(self, sim_time: float):
        """Clear expired road works, restoring original edge speeds."""
        sumo = self._get_traci()
        if sumo is None:
            return

        still_active = []
        for rw in self._active_roadworks:
            if sim_time >= rw["clear_time"]:
                try:
                    sumo.edge.setMaxSpeed(rw["edge"], rw["original_speed"])
                except Exception:
                    pass
            else:
                still_active.append(rw)
        self._active_roadworks = still_active

    # ── Downstream occupancy computation (for spillback) ─────────────

    def _compute_downstream_occupancy(self):
        """Compute max downstream edge occupancy per incoming lane per agent.

        For each agent's incoming lane, looks up which downstream edges
        that lane feeds into, queries their occupancy via TraCI, and
        stores the max across all downstream edges for that lane.
        """
        sumo = self._get_traci()
        if sumo is None:
            return

        for agent_id in self.possible_agents:
            ts = self._get_traffic_signal(agent_id)
            if ts is None:
                continue

            n_lanes = len(ts.lanes)
            ds_occ = np.zeros(n_lanes, dtype=np.float32)
            lane_map = self._lane_to_downstream_edges.get(agent_id, {})

            for i in range(n_lanes):
                edges = lane_map.get(i, [])
                if not edges:
                    continue
                max_occ = 0.0
                for edge in edges:
                    try:
                        occ = sumo.edge.getLastStepOccupancy(edge)
                        max_occ = max(max_occ, occ / 100.0 if occ > 1.0 else occ)
                    except Exception:
                        continue
                ds_occ[i] = max_occ

            self._downstream_occ[agent_id] = ds_occ

    # ── Bus / Convoy vehicle injection ─────────────────────────────────

    def _discover_boundary_edges(self):
        """Find edges connected to boundary (dead-end) nodes for route generation.

        Boundary nodes are those with only one connected edge — the entry/exit
        points of the network. We collect edges that START at boundary nodes
        (suitable as route origins) and edges that END at boundary nodes
        (suitable as route destinations).
        """
        sumo = self._get_traci()
        if sumo is None:
            return

        all_edges = sumo.edge.getIDList()
        # Filter out internal edges (start with ':')
        normal_edges = [e for e in all_edges if not e.startswith(':')]

        # Count edge connections per node
        from collections import Counter
        node_edge_count = Counter()
        for e in normal_edges:
            node_edge_count[sumo.edge.getFromJunction(e)] += 1
            node_edge_count[sumo.edge.getToJunction(e)] += 1

        # Boundary nodes: those with few connections (typically 1-2)
        # For grid/arterial networks, boundary nodes have exactly 1 edge
        # Use threshold of 2 to also catch corner nodes
        boundary_nodes = {n for n, c in node_edge_count.items() if c <= 2}

        # Collect edges starting from boundary nodes (for origins)
        self._origin_edges = [e for e in normal_edges
                              if sumo.edge.getFromJunction(e) in boundary_nodes]
        # Collect edges ending at boundary nodes (for destinations)
        self._dest_edges = [e for e in normal_edges
                            if sumo.edge.getToJunction(e) in boundary_nodes]

        # Fallback: if we didn't find enough, use all normal edges
        if len(self._origin_edges) < 2:
            self._origin_edges = normal_edges
        if len(self._dest_edges) < 2:
            self._dest_edges = normal_edges

    def _select_stress_target(self, rng) -> Optional[List[str]]:
        """Pick a high-conflict incoming lane and return a route through it.

        Selects an incoming lane that:
          - belongs to a non-current (unserved) phase,
          - sits at a junction where the currently served phase has high
            competing pressure relative to the unserved phase.

        Returns a list of edge IDs forming a valid SUMO route that
        traverses the chosen stress edge, or None if no suitable target.
        """
        sumo = self._get_traci()
        if sumo is None or not self._phase_to_lanes:
            return None

        # Score every unserved lane across all agents
        candidates: List[Tuple[float, str, int, str]] = []  # (score, agent_id, lane_idx, edge)
        for agent_id in self.possible_agents:
            ts = self._get_traffic_signal(agent_id)
            if ts is None:
                continue
            phase = ts.green_phase
            phase_map = self._phase_to_lanes.get(agent_id, {})
            served_lanes = set(phase_map.get(phase, []))
            if not served_lanes:
                continue

            lane_queues = ts.get_lanes_queue()
            lane_densities = ts.get_lanes_density()

            def _pressure(lane_indices):
                p = 0.0
                for li in lane_indices:
                    if li < len(lane_queues):
                        p += lane_queues[li] + lane_densities[li]
                return p

            served_pressure = _pressure(served_lanes)

            for ph, lanes in phase_map.items():
                if ph == phase:
                    continue
                unserved_pressure = _pressure(lanes)
                # Higher score = served phase has much more traffic than
                # the unserved phase → switching away is costly for reward,
                # but needed for cost.  That is the conflict.
                score = served_pressure - unserved_pressure
                if score <= 0:
                    continue  # unserved phase already has more traffic — no conflict
                for li in lanes:
                    if li not in served_lanes:
                        lane_name = ts.lanes[li]
                        edge = '_'.join(lane_name.split('_')[:-1])
                        candidates.append((score, agent_id, li, edge))

        if not candidates:
            return None

        # Weighted random selection: higher-scored lanes more likely
        scores = np.array([c[0] for c in candidates], dtype=np.float64)
        scores = scores / (scores.sum() + 1e-9)
        idx = rng.choice(len(candidates), p=scores)
        _, _, _, target_edge = candidates[idx]

        # Build a route that traverses target_edge.
        # Strategy: find the node upstream of target_edge, then find a
        # boundary origin edge whose shortest path to a boundary dest
        # passes through target_edge.
        try:
            target_from = sumo.edge.getFromJunction(target_edge)
            target_to = sumo.edge.getToJunction(target_edge)
        except Exception:
            return None

        # Try to find a boundary entry edge leading to target_from
        # (one hop upstream: any boundary edge ending at target_from)
        upstream_boundary = [
            e for e in self._origin_edges
            if sumo.edge.getToJunction(e) == target_from
        ]
        # If no direct boundary feeder, look one more hop up
        if not upstream_boundary:
            all_edges = sumo.edge.getIDList()
            feeder_edges = [
                e for e in all_edges
                if not e.startswith(':') and sumo.edge.getToJunction(e) == target_from
            ]
            for fe in feeder_edges:
                fe_from = sumo.edge.getFromJunction(fe)
                for oe in self._origin_edges:
                    if sumo.edge.getToJunction(oe) == fe_from:
                        upstream_boundary.append(oe)

        # Find boundary exit edges reachable from target_to
        downstream_boundary = [
            e for e in self._dest_edges
            if sumo.edge.getFromJunction(e) == target_to
        ]
        if not downstream_boundary:
            all_edges = sumo.edge.getIDList()
            out_edges = [
                e for e in all_edges
                if not e.startswith(':') and sumo.edge.getFromJunction(e) == target_to
            ]
            for oe in out_edges:
                oe_to = sumo.edge.getToJunction(oe)
                for de in self._dest_edges:
                    if sumo.edge.getFromJunction(de) == oe_to:
                        downstream_boundary.append(de)

        # Try (origin, dest) combos until findRoute gives a path through target_edge
        rng.shuffle(upstream_boundary)
        rng.shuffle(downstream_boundary)
        # Also mix in some random boundary edges as fallback
        fallback_origins = list(self._origin_edges)
        rng.shuffle(fallback_origins)
        fallback_dests = list(self._dest_edges)
        rng.shuffle(fallback_dests)

        origins_to_try = upstream_boundary[:4] + fallback_origins[:4]
        dests_to_try = downstream_boundary[:4] + fallback_dests[:4]

        for origin in origins_to_try:
            for dest in dests_to_try:
                if origin == dest:
                    continue
                try:
                    stage = sumo.simulation.findRoute(origin, dest)
                    if hasattr(stage, 'edges') and target_edge in stage.edges:
                        return list(stage.edges)
                except Exception:
                    continue

        # Last resort: two-leg route (origin→target, target→dest) via findRoute
        for origin in origins_to_try[:3]:
            try:
                leg1 = sumo.simulation.findRoute(origin, target_edge)
                if not hasattr(leg1, 'edges') or not leg1.edges:
                    continue
            except Exception:
                continue
            for dest in dests_to_try[:3]:
                try:
                    leg2 = sumo.simulation.findRoute(target_edge, dest)
                    if hasattr(leg2, 'edges') and leg2.edges:
                        # Merge: leg1 edges + leg2 edges (skip duplicate target_edge)
                        route = list(leg1.edges)
                        for e in leg2.edges:
                            if e != route[-1]:
                                route.append(e)
                        return route
                except Exception:
                    continue

        return None

    def _create_bus_vtype(self):
        """Create a 'priority_bus' vehicle type via TraCI."""
        sumo = self._get_traci()
        if sumo is None or self._bus_vtype_created:
            return
        try:
            sumo.vehicletype.copy("DEFAULT_VEHTYPE", "priority_bus")
            sumo.vehicletype.setLength("priority_bus", 12.0)
            sumo.vehicletype.setMaxSpeed("priority_bus", 11.1)  # ~40 km/h
            sumo.vehicletype.setColor("priority_bus", (0, 128, 255, 255))  # blue
            sumo.vehicletype.setVehicleClass("priority_bus", "bus")
            self._bus_vtype_created = True
        except Exception:
            pass  # vtype might already exist from previous episode

    def _create_convoy_vtype(self):
        """Create a 'convoy_vehicle' vehicle type via TraCI.

        Convoy vehicles are regular-sized cars with a distinctive color.
        They drive at normal speed and queue normally at red lights.
        The constraint comes from the GROUP needing sustained green.
        """
        sumo = self._get_traci()
        if sumo is None or self._convoy_vtype_created:
            return
        try:
            sumo.vehicletype.copy("DEFAULT_VEHTYPE", "convoy_vehicle")
            sumo.vehicletype.setLength("convoy_vehicle", 5.0)
            sumo.vehicletype.setMaxSpeed("convoy_vehicle", 13.9)  # ~50 km/h
            sumo.vehicletype.setColor("convoy_vehicle", (0, 200, 0, 255))  # green
            self._convoy_vtype_created = True
        except Exception:
            pass

    def _create_premium_vtype(self):
        """Create a 'premium_vehicle' vehicle type via TraCI.

        Premium vehicles are normal-sized cars at normal speed. They look
        identical to regular cars in the traffic simulation — same length,
        same acceleration, same max speed. The only difference is the tag.
        The cost function treats them as high-priority.
        """
        sumo = self._get_traci()
        if sumo is None or self._premium_vtype_created:
            return
        try:
            sumo.vehicletype.copy("DEFAULT_VEHTYPE", "premium_vehicle")
            sumo.vehicletype.setLength("premium_vehicle", 5.0)
            sumo.vehicletype.setMaxSpeed("premium_vehicle", 13.9)  # same as default
            sumo.vehicletype.setColor("premium_vehicle", (255, 215, 0, 255))  # gold
            self._premium_vtype_created = True
        except Exception:
            pass

    def _inject_vehicles(self, sim_time: float):
        """Inject bus / convoy / premium vehicles based on current simulation time.

        Called each agent step (every delta_time seconds). Checks if it's
        time to inject a new bus, convoy, or premium vehicle and does so via TraCI.
        """
        sumo = self._get_traci()
        if sumo is None:
            return
        rng = np.random.RandomState(int(sim_time * 1000) % (2**31))

        # Bus injection (multiple vehicles per event for volume pressure)
        if self._cost_fn == "bus_priority" and sim_time >= self._next_bus_time:
            for _b in range(self._bus_count_per_injection):
                origin = rng.choice(self._origin_edges)
                dest = rng.choice(self._dest_edges)
                attempts = 0
                while dest == origin and attempts < 10:
                    dest = rng.choice(self._dest_edges)
                    attempts += 1
                if dest != origin:
                    bus_id = f"bus_{self._bus_counter}"
                    route_id = f"bus_route_{self._bus_counter}"
                    try:
                        sumo.route.add(route_id, [origin, dest])
                        sumo.vehicle.add(bus_id, route_id, typeID="priority_bus")
                        self._bus_counter += 1
                    except Exception:
                        pass  # route might be invalid (no path), skip
            # Schedule next bus injection with some randomness
            self._next_bus_time = sim_time + self._bus_injection_interval * (0.8 + 0.4 * rng.random())

        # Convoy injection — targeted at a stress lane for sharp conflict
        if self._cost_fn == "convoy_priority" and sim_time >= self._next_convoy_time:
            route_edges = self._select_stress_target(rng)
            # Fallback to random OD via findRoute if stress targeting fails
            if route_edges is None:
                origins = list(self._origin_edges)
                dests = list(self._dest_edges)
                rng.shuffle(origins)
                rng.shuffle(dests)
                for _o in origins[:6]:
                    for _d in dests[:6]:
                        if _o == _d:
                            continue
                        try:
                            stage = sumo.simulation.findRoute(_o, _d)
                            if hasattr(stage, 'edges') and len(stage.edges) >= 2:
                                route_edges = list(stage.edges)
                                break
                        except Exception:
                            continue
                    if route_edges is not None:
                        break
            if route_edges is not None and len(route_edges) >= 2:
                platoon_size = rng.randint(self._convoy_size_min, self._convoy_size_max + 1)
                route_id = f"convoy_route_{self._convoy_counter}"
                convoy_id = self._convoy_counter
                try:
                    sumo.route.add(route_id, route_edges)
                except Exception:
                    platoon_size = 0  # route invalid, skip
                veh_ids = []
                for v in range(platoon_size):
                    vid = f"convoy_{convoy_id}_v{v}"
                    try:
                        depart = str(sim_time + v * self._convoy_headway)
                        sumo.vehicle.add(vid, route_id, typeID="convoy_vehicle",
                                         depart=depart)
                        veh_ids.append(vid)
                    except Exception:
                        pass
                if veh_ids:
                    self._active_convoys[convoy_id] = veh_ids
                self._convoy_counter += 1
            self._next_convoy_time = sim_time + self._convoy_injection_interval * (0.8 + 0.4 * rng.random())

        # Premium injection — targeted at a stress lane for sharp conflict
        if self._cost_fn == "premium_priority" and sim_time >= self._next_premium_time:
            route_edges = self._select_stress_target(rng)
            # Fallback to random OD via findRoute if stress targeting fails
            if route_edges is None:
                origins = list(self._origin_edges)
                dests = list(self._dest_edges)
                rng.shuffle(origins)
                rng.shuffle(dests)
                for _o in origins[:6]:
                    for _d in dests[:6]:
                        if _o == _d:
                            continue
                        try:
                            stage = sumo.simulation.findRoute(_o, _d)
                            if hasattr(stage, 'edges') and len(stage.edges) >= 2:
                                route_edges = list(stage.edges)
                                break
                        except Exception:
                            continue
                    if route_edges is not None:
                        break
            if route_edges is not None and len(route_edges) >= 2:
                prem_id = f"premium_{self._premium_counter}"
                route_id = f"premium_route_{self._premium_counter}"
                try:
                    sumo.route.add(route_id, route_edges)
                    sumo.vehicle.add(prem_id, route_id, typeID="premium_vehicle")
                    self._premium_counter += 1
                except Exception:
                    pass
            self._next_premium_time = sim_time + self._premium_injection_interval * (0.8 + 0.4 * rng.random())

    def _detect_special_vehicles(self):
        """Detect bus/convoy/premium vehicles on each agent's incoming lanes.

        Updates per-agent per-lane feature arrays used for obs extension,
        cost computation, and label.
        """
        sumo = self._get_traci()
        if sumo is None:
            return

        need_bus = self._cost_fn == "bus_priority"
        need_convoy = self._cost_fn == "convoy_priority"
        need_premium = self._cost_fn == "premium_priority"

        for agent_id in self.possible_agents:
            ts = self._get_traffic_signal(agent_id)
            if ts is None:
                continue

            n_lanes = len(ts.lanes)
            has_bus = np.zeros(n_lanes, dtype=np.float32)
            bus_wait = np.zeros(n_lanes, dtype=np.float32)
            bus_count = np.zeros(n_lanes, dtype=np.float32)
            has_convoy = np.zeros(n_lanes, dtype=np.float32)
            convoy_count = np.zeros(n_lanes, dtype=np.float32)
            convoy_progress = np.zeros(n_lanes, dtype=np.float32)
            has_premium = np.zeros(n_lanes, dtype=np.float32)
            premium_wait = np.zeros(n_lanes, dtype=np.float32)

            # Track which convoy vehicles are on THIS agent's incoming lanes
            convoy_on_incoming: Dict[int, int] = {}   # convoy_id → count on incoming
            convoy_on_incoming_vids: Dict[int, Set[str]] = {}  # convoy_id → set of vids on incoming
            convoy_lane_map: Dict[int, int] = {}       # convoy_id → lane_idx (first seen)

            for i, lane in enumerate(ts.lanes):
                try:
                    veh_ids = sumo.lane.getLastStepVehicleIDs(lane)
                except Exception:
                    continue
                for vid in veh_ids:
                    try:
                        vtype = sumo.vehicle.getTypeID(vid)
                        vwait = sumo.vehicle.getWaitingTime(vid)
                    except Exception:
                        continue
                    if need_bus and vtype == "priority_bus":
                        has_bus[i] = 1.0
                        bus_count[i] += 1.0
                        bus_wait[i] = max(bus_wait[i], vwait)
                    if need_convoy and vtype == "convoy_vehicle":
                        has_convoy[i] = 1.0
                        convoy_count[i] += 1.0
                        # Parse convoy_id from vid: "convoy_{id}_v{n}"
                        try:
                            cid = int(vid.split('_')[1])
                            convoy_on_incoming[cid] = convoy_on_incoming.get(cid, 0) + 1
                            convoy_on_incoming_vids.setdefault(cid, set()).add(vid)
                            if cid not in convoy_lane_map:
                                convoy_lane_map[cid] = i
                        except (IndexError, ValueError):
                            pass
                    if need_premium and vtype == "premium_vehicle":
                        has_premium[i] = 1.0
                        premium_wait[i] = max(premium_wait[i], vwait)

            # Normalize and store bus features
            if need_bus:
                bus_wait = np.clip(bus_wait / self._bus_cost_threshold, 0.0, 1.0)
                bus_count = np.clip(bus_count / 5.0, 0.0, 1.0)  # normalize by max expected
                self._has_bus[agent_id] = has_bus
                self._bus_wait[agent_id] = bus_wait
                self._bus_count[agent_id] = bus_count

            # Compute convoy progress PER AGENT using persistent tracking.
            # progress = (ever_seen_on_incoming − still_on_incoming) / ever_seen
            # Only counts vehicles that THIS agent has actually observed on
            # its own incoming lanes.  No cross-agent leakage.
            if need_convoy:
                agent_history = self._convoy_seen_by_agent.setdefault(agent_id, {})

                for cid, lane_idx in convoy_lane_map.items():
                    current_vids = convoy_on_incoming_vids.get(cid, set())
                    # Register newly-seen vehicles
                    seen = agent_history.setdefault(cid, set())
                    seen.update(current_vids)
                    # Progress: fraction of ever-seen that are no longer on incoming
                    if len(seen) > 0:
                        still_here = len(current_vids & seen)
                        passed = len(seen) - still_here
                        progress = passed / len(seen)
                        convoy_progress[lane_idx] = max(
                            convoy_progress[lane_idx], progress
                        )

                convoy_count_norm = np.clip(convoy_count / self._convoy_size_max, 0.0, 1.0)
                self._has_convoy[agent_id] = has_convoy
                self._convoy_count[agent_id] = convoy_count_norm
                self._convoy_progress[agent_id] = convoy_progress

            # Store premium features
            if need_premium:
                premium_wait = np.clip(premium_wait / self._premium_cost_threshold, 0.0, 1.0)
                self._has_premium[agent_id] = has_premium
                self._premium_wait[agent_id] = premium_wait

    def _extend_obs_spaces(self):
        """No-op: obs spaces are pre-extended in __init__."""
        pass

    def _extend_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Append bus/convoy/premium/spillback features to observation vectors."""
        if self._obs_extension_dim == 0:
            return obs

        extended = {}
        for agent, ob in obs.items():
            ob = np.asarray(ob, dtype=np.float32)

            if self._cost_fn == "bus_priority":
                n_lanes = self._obs_extension_dim // 3
                hb = self._has_bus.get(agent, np.zeros(n_lanes, dtype=np.float32))
                bc = self._bus_count.get(agent, np.zeros(n_lanes, dtype=np.float32))
                bw = self._bus_wait.get(agent, np.zeros(n_lanes, dtype=np.float32))
                ob = np.concatenate([ob, hb, bc, bw])
            elif self._cost_fn == "convoy_priority":
                n_lanes = self._obs_extension_dim // 3
                hc = self._has_convoy.get(agent, np.zeros(n_lanes, dtype=np.float32))
                cc = self._convoy_count.get(agent, np.zeros(n_lanes, dtype=np.float32))
                cp = self._convoy_progress.get(agent, np.zeros(n_lanes, dtype=np.float32))
                ob = np.concatenate([ob, hc, cc, cp])
            elif self._cost_fn == "spillback":
                ds = self._downstream_occ.get(agent, np.zeros(self._obs_extension_dim, dtype=np.float32))
                ob = np.concatenate([ob, ds])
            elif self._cost_fn == "premium_priority":
                n_lanes = self._obs_extension_dim // 2
                hp = self._has_premium.get(agent, np.zeros(n_lanes, dtype=np.float32))
                pw = self._premium_wait.get(agent, np.zeros(n_lanes, dtype=np.float32))
                ob = np.concatenate([ob, hp, pw])

            extended[agent] = ob
        return extended

    # ── Bus priority cost/label ───────────────────────────────────────────

    def _compute_bus_priority_cost(self, agent_id: str) -> float:
        """Max bus wait ratio across *unserved* incoming lanes.

        cost = max over unserved lanes of: bus_wait[lane] / T_bus
             = 0  if no bus, or all buses are on served (green) lanes

        Continuous in [0, 1]. Ramps as bus waits longer on a red lane.
        """
        bus_wait = self._bus_wait.get(agent_id)
        has_bus = self._has_bus.get(agent_id)
        if bus_wait is None or has_bus is None:
            return 0.0
        if not np.any(has_bus > 0):
            return 0.0

        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0.0
        phase = ts.green_phase
        served_lanes = set(self._phase_to_lanes.get(agent_id, {}).get(phase, []))

        max_cost = 0.0
        for i in range(len(has_bus)):
            if has_bus[i] > 0 and i not in served_lanes:
                max_cost = max(max_cost, float(bus_wait[i]))
        return max_cost

    def _bus_priority_label(self, agent_id: str) -> int:
        """Label = 1 when a bus is waiting on an unserved lane.

        This fires on the same condition as cost > 0: a bus exists on
        a lane that the current green phase does NOT serve.

        Why this IS the conflict: the reward chose this phase because
        it serves more vehicles (cars clear faster than buses). The
        cost wants the bus served. The bus being on an unserved lane
        means the agent's reward-optimal action is already hurting cost.
        No additional threshold needed — the conflict exists the moment
        the bus is on an unserved lane.
        """
        has_bus = self._has_bus.get(agent_id)
        if has_bus is None:
            return 0
        if not np.any(has_bus > 0):
            return 0

        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0

        phase = ts.green_phase
        served_lanes = set(self._phase_to_lanes.get(agent_id, {}).get(phase, []))

        for i in range(len(has_bus)):
            if has_bus[i] > 0 and i not in served_lanes:
                return 1
        return 0

    # ── Convoy priority cost/label ───────────────────────────────────────

    def _compute_convoy_priority_cost(self, agent_id: str) -> float:
        """Convoy split-detection cost: fires ONLY during active split.

        cost = progress when 0 < progress < 1 and convoy is on an
        unserved lane (some vehicles passed, rest still stuck).

        Aligned with _convoy_priority_label: both fire in exactly the
        same conflict window. No pre-split "risk" cost.
        """
        has_convoy = self._has_convoy.get(agent_id)
        convoy_progress = self._convoy_progress.get(agent_id)
        if has_convoy is None or convoy_progress is None:
            return 0.0
        if not np.any(has_convoy > 0):
            return 0.0

        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0.0
        phase = ts.green_phase
        served_lanes = set(self._phase_to_lanes.get(agent_id, {}).get(phase, []))

        max_cost = 0.0
        for i in range(len(has_convoy)):
            if has_convoy[i] > 0 and i not in served_lanes:
                progress = convoy_progress[i]
                if 0 < progress < 1.0:
                    # Active split: some passed, rest stuck on unserved lane.
                    # Cost = progress (higher = more split = worse).
                    max_cost = max(max_cost, float(progress))
        return min(max_cost, 1.0)

    def _convoy_priority_label(self, agent_id: str) -> int:
        """State partition for convoy priority: g(s) → {0, 1}.

        Routes gradients to the cost head when a convoy is actively
        splitting (0 < progress < 1 on an unserved lane).

        This partitions the state space into the region where the
        cost objective (keep the convoy together) should dominate
        vs where the reward objective (minimise total waiting) is
        sufficient on its own.
        """
        has_convoy = self._has_convoy.get(agent_id)
        convoy_progress = self._convoy_progress.get(agent_id)
        if has_convoy is None or convoy_progress is None:
            return 0

        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0

        phase = ts.green_phase
        served_lanes = set(self._phase_to_lanes.get(agent_id, {}).get(phase, []))

        for i in range(len(has_convoy)):
            if has_convoy[i] > 0 and i not in served_lanes:
                # Active split: some passed, some still on incoming lane
                if 0 < convoy_progress[i] < 1.0:
                    return 1
        return 0

    # ── Spillback cost/label ─────────────────────────────────────────────

    def _compute_spillback_cost(self, agent_id: str) -> float:
        """Max downstream occupancy across currently *served* incoming lanes.

        c_t = max_{l in served(phase_t)} downstream_occ(l)

        Continuous in [0, 1]. Cost is high when the agent is actively
        sending vehicles into a blocked downstream direction.
        Cost = 0 if downstream is clear, regardless of road works.
        """
        ds_occ = self._downstream_occ.get(agent_id)
        if ds_occ is None:
            return 0.0

        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0.0
        phase = ts.green_phase
        served_lanes = self._phase_to_lanes.get(agent_id, {}).get(phase, [])
        if not served_lanes:
            return 0.0

        max_cost = 0.0
        for i in served_lanes:
            if i < len(ds_occ):
                max_cost = max(max_cost, float(ds_occ[i]))
        return max_cost

    def _spillback_label(self, agent_id: str) -> int:
        """Label = 1 when serving into a blocked downstream.

        y_t = 1[max_{l in served(phase_t)} downstream_occ(l) > T_occ]

        Pure function of observation state: phase one-hot (in obs) and
        downstream_occ features (in obs extension). No action dependence.
        """
        ds_occ = self._downstream_occ.get(agent_id)
        if ds_occ is None:
            return 0

        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0
        phase = ts.green_phase
        served_lanes = self._phase_to_lanes.get(agent_id, {}).get(phase, [])

        for i in served_lanes:
            if i < len(ds_occ) and ds_occ[i] > self._spillback_occ_threshold:
                return 1
        return 0

    # ── Premium priority cost/label ──────────────────────────────────────

    def _compute_premium_priority_cost(self, agent_id: str) -> float:
        """Steep cost when a premium vehicle waits on an unserved lane.

        cost = max over unserved lanes of: (premium_wait[lane])^2

        Squared to create a steep ramp: the cost accelerates as the premium
        vehicle waits longer. This makes the agent pay a rapidly increasing
        price for not serving the premium vehicle quickly.

        The premium vehicle is a normal-sized car — the reward treats it as
        1 vehicle among many. But the cost treats it as high-priority.
        This is pure value-asymmetry conflict.
        """
        has_premium = self._has_premium.get(agent_id)
        premium_wait = self._premium_wait.get(agent_id)
        if has_premium is None or premium_wait is None:
            return 0.0
        if not np.any(has_premium > 0):
            return 0.0

        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0.0
        phase = ts.green_phase
        served_lanes = set(self._phase_to_lanes.get(agent_id, {}).get(phase, []))

        max_cost = 0.0
        for i in range(len(has_premium)):
            if has_premium[i] > 0 and i not in served_lanes:
                # Squared cost: ramps steeply as normalized wait increases
                max_cost = max(max_cost, float(premium_wait[i]) ** 2)
        return min(max_cost, 1.0)

    def _premium_priority_label(self, agent_id: str) -> int:
        """State partition for premium priority: g(s) → {0, 1}.

        Labels the region of state space where the cost head should own
        gradient updates. Fires when ALL of:
          1. Premium vehicle on an unserved lane (prerequisite).
          2. min_green satisfied — agent CAN switch phase.
          3. premium_wait > τ_warn (normalised) — premium has waited
             long enough to be material.
          4. current_phase_pressure > β · premium_phase_pressure —
             the phase the agent is running has more traffic pressure
             than the phase that would serve the premium lane.

        Condition 4 compares phase-vs-phase (not phase-vs-lane) for
        a fair comparison regardless of how many lanes each phase serves.
        """
        has_premium = self._has_premium.get(agent_id)
        premium_wait = self._premium_wait.get(agent_id)
        if has_premium is None or premium_wait is None:
            return 0
        if not np.any(has_premium > 0):
            return 0

        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0

        # Condition 2: min_green satisfied (agent can actually switch)
        if ts.time_since_last_phase_change < ts.min_green + ts.yellow_time:
            return 0

        phase = ts.green_phase
        phase_map = self._phase_to_lanes.get(agent_id, {})
        served_lanes = set(phase_map.get(phase, []))

        # Normalised warn threshold (premium_wait is already / cost_threshold)
        warn_norm = self._premium_warn_threshold / max(self._premium_cost_threshold, 1e-6)

        lane_queues = ts.get_lanes_queue()
        lane_densities = ts.get_lanes_density()

        def _phase_pressure(lane_indices):
            p = 0.0
            for li in lane_indices:
                if li < len(lane_queues):
                    p += lane_queues[li] + lane_densities[li]
            return p

        # Current phase pressure
        served_pressure = _phase_pressure(served_lanes)

        for i in range(len(has_premium)):
            if has_premium[i] > 0 and i not in served_lanes:
                # Condition 3: premium has waited enough
                if premium_wait[i] <= warn_norm:
                    continue
                # Condition 4: find the best phase that serves this lane
                # (max pressure among all candidate phases, conservative).
                best_prem_pressure = None
                for ph, lanes in phase_map.items():
                    if i in lanes and ph != phase:
                        pp = _phase_pressure(lanes)
                        if best_prem_pressure is None or pp > best_prem_pressure:
                            best_prem_pressure = pp
                if best_prem_pressure is None:
                    continue  # no phase serves this lane
                if served_pressure > self._premium_pressure_beta * max(best_prem_pressure, 1e-6):
                    return 1
        return 0

    def _update_service_tracking(self, agent_id: str):
        """Update steps_since_served counters based on current green phase.

        Call AFTER each env.step(). Checks which lanes the current green
        phase serves, resets their counter to 0, increments all others.
        """
        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return

        phase = ts.green_phase
        served_lanes = set(self._phase_to_lanes.get(agent_id, {}).get(phase, []))
        counters = self._steps_since_served.get(agent_id, {})
        side_idx = self._side_lane_indices.get(agent_id, [])

        any_side_served = False
        for li in side_idx:
            if li in served_lanes:
                counters[li] = 0
                any_side_served = True
            else:
                counters[li] = counters.get(li, 0) + 1

        # Track consecutive steps with NO side-street service at all
        if any_side_served:
            self._side_consecutive_unserved[agent_id] = 0
        else:
            self._side_consecutive_unserved[agent_id] = \
                self._side_consecutive_unserved.get(agent_id, 0) + 1

    def _compute_mainline_reward(self, agent_id: str) -> float:
        """Diff-waiting-time on mainline incoming lanes only.

        r_t = (W_main(t-1) - W_main(t)) / 100

        Positive when mainline waiting decreases (mainline served).
        Negative when mainline waiting increases (mainline starved).
        """
        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0.0
        ml_idx = self._mainline_lane_indices.get(agent_id, [])
        if not ml_idx:
            return 0.0
        wait_per_lane = ts.get_accumulated_waiting_time_per_lane()
        ml_wait = sum(wait_per_lane[i] for i in ml_idx) / 100.0
        prev = self._prev_mainline_wait.get(agent_id, 0.0)
        self._prev_mainline_wait[agent_id] = ml_wait
        return -(ml_wait - prev)

    def _compute_side_deficit_cost(self, agent_id: str) -> float:
        """Continuous side-street service-deficit cost, bounded to [0, 1].

        raw_deficit = sum_{l in side} q_l(t) * max(0, g_l(t) - tau)
        c_t = min(raw_deficit / kappa, 1.0)

        where q_l is queued vehicles and g_l is steps since lane l was
        last served.  Cost is continuous (not binary) but bounded:
          - 0 when all side-streets are served or have no demand
          - Ramps linearly to 1 as neglect severity grows
          - Saturates at 1 when raw deficit >= kappa

        This keeps the cost head numerically stable (bounded targets)
        while preserving the continuous severity signal that makes a
        cheap green flick only partially relieve the cost.
        """
        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0.0

        side_idx = self._side_lane_indices.get(agent_id, [])
        if not side_idx:
            return 0.0

        counters = self._steps_since_served.get(agent_id, {})
        lane_queues = ts.get_lanes_queue()

        deficit = 0.0
        for li in side_idx:
            q = lane_queues[li] if li < len(lane_queues) else 0
            gap = counters.get(li, 0)
            excess = max(0, gap - self._tau_service)
            deficit += q * excess
        return min(deficit / self._deficit_kappa, 1.0)

    def _side_deficit_label(self, agent_id: str) -> int:
        """State-based label for side_deficit cost.

        y_t = 1[ sum_l q_l * max(0, g_l - tau) > kappa ]

        Identical quantity to the cost, thresholded.  Cleanly separates
        low-deficit states (reward update) from urgent-neglect states
        (cost update) for SPLIT-RL gradient routing.
        """
        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0

        side_idx = self._side_lane_indices.get(agent_id, [])
        if not side_idx:
            return 0

        counters = self._steps_since_served.get(agent_id, {})
        lane_queues = ts.get_lanes_queue()

        deficit = 0.0
        for li in side_idx:
            q = lane_queues[li] if li < len(lane_queues) else 0
            gap = counters.get(li, 0)
            excess = max(0, gap - self._tau_service)
            deficit += q * excess
        return 1 if deficit > self._deficit_kappa else 0

    def _compute_side_queue_cost(self, agent_id: str) -> float:
        """Max queue ratio on side-street lanes — purely observation-based.

        c_t = max_{l in side} queue_ratio_l(t)

        Returns the worst (highest) side-street queue ratio ∈ [0, 1].
        This is a continuous cost that uses only observation features
        (lane queue ratios are part of the SUMO-RL obs vector).

        Structural conflict with mainline reward:
          - Holding mainline green → mainline waiting drops (reward ↑)
            but side-street queues grow unchecked (cost ↑)
          - Switching to serve side-streets → mainline waiting grows
            (reward ↓) but side-street queues drain (cost ↓)
          - Green time is finite: cannot serve both simultaneously

        Why purely obs-based matters:
          - The cost value head V^C(s) can learn this function exactly,
            since max(side queue ratios) is a deterministic function of
            the observation vector
          - The label is also obs-based → the tree can split on the
            side-street queue features → SPLIT-RL gradient routing is
            meaningful, not random
        """
        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0.0

        side_idx = self._side_lane_indices.get(agent_id, [])
        if not side_idx:
            return 0.0

        lane_queues = ts.get_lanes_queue()
        side_queues = [lane_queues[i] for i in side_idx if i < len(lane_queues)]
        return max(side_queues) if side_queues else 0.0

    def _side_queue_label(self, agent_id: str) -> int:
        """Observation-based label for side_queue cost.

        y_t = 1[ max_{l in side} queue_ratio_l > side_queue_cap ]

        Fires when the worst side-street queue ratio exceeds the
        threshold.  This is a deterministic function of the current
        observation — the tree can predict it exactly from the
        side-street queue features in the obs vector.

        The label partitions the state space into:
          - label=0: side-streets are manageable → optimize reward freely
          - label=1: side-streets are overloaded → cost gradient must
            steer the policy toward serving them
        """
        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0

        side_idx = self._side_lane_indices.get(agent_id, [])
        if not side_idx:
            return 0

        lane_queues = ts.get_lanes_queue()
        side_queues = [lane_queues[i] for i in side_idx if i < len(lane_queues)]
        if not side_queues:
            return 0
        return 1 if max(side_queues) > self._side_queue_cap else 0

    def _compute_service_gap_cost(self, agent_id: str) -> float:
        """Binary service-gap cost: 1 when side-street guarantee is violated.

        c_t = 1[ max_{l in side} steps_since_served_l > tau  AND  q_l > 0 ]

        Fires ONLY when a side-street lane has actual demand (queue > 0)
        AND hasn't been given green for > tau_service consecutive steps.
        This prevents false triggers on empty side-streets.

        The conflict with mainline reward is structural:
          - Holding mainline green → mainline reward ↑
          - But side-street steps_since_served grows → cost fires
          - Must periodically serve side-streets → mainline reward ↓
        """
        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0.0

        side_idx = self._side_lane_indices.get(agent_id, [])
        if not side_idx:
            return 0.0

        counters = self._steps_since_served.get(agent_id, {})
        lane_queues = ts.get_lanes_queue()

        for li in side_idx:
            steps_unserved = counters.get(li, 0)
            has_demand = lane_queues[li] > 0 if li < len(lane_queues) else False
            if steps_unserved > self._tau_service and has_demand:
                return 1.0
        return 0.0

    # ── Per-agent cost functions ──────────────────────────────────────────

    def _compute_agent_cost_queue_overflow(self, agent_id: str) -> float:
        """1 if total queued vehicles at this intersection > queue_threshold."""
        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0.0
        queued = ts.get_total_queued()
        return 1.0 if queued > self._queue_threshold else 0.0

    def _compute_agent_cost_max_wait(self, agent_id: str) -> float:
        """1 if max accumulated waiting time on any lane > wait_threshold."""
        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0.0
        wait_per_lane = ts.get_accumulated_waiting_time_per_lane()
        if not wait_per_lane:
            return 0.0
        return 1.0 if max(wait_per_lane) > self._wait_threshold else 0.0

    def _compute_agent_cost_lane_saturation(self, agent_id: str) -> float:
        """1 if any lane's queue ratio > saturation_threshold."""
        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0.0
        lane_queues = ts.get_lanes_queue()
        if not lane_queues:
            return 0.0
        return 1.0 if max(lane_queues) > self._saturation_threshold else 0.0

    def _compute_agent_cost_fairness(self, agent_id: str) -> float:
        """1 if one lane is being starved relative to others.

        Fires when max_wait > tau_abs AND max_wait / (mean_wait + eps) > rho.
        The absolute floor prevents spurious triggers when all waits are tiny.
        """
        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0.0
        wait_per_lane = ts.get_accumulated_waiting_time_per_lane()
        if not wait_per_lane or len(wait_per_lane) < 2:
            return 0.0
        max_w = max(wait_per_lane)
        mean_w = sum(wait_per_lane) / len(wait_per_lane)
        if max_w > self._fairness_tau and max_w / (mean_w + 1e-6) > self._fairness_rho:
            return 1.0
        return 0.0

    def _compute_agent_cost_phase_churn(self, agent_id: str) -> float:
        """1 if the agent switched phase this step.

        Uses sumo-rl's TrafficSignal.green_phase compared to last known phase.
        Every switch costs yellow time = lost throughput capacity, creating
        real tension with a throughput reward.
        """
        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0.0
        current_phase = ts.green_phase
        last_phase = self._last_green_phase.get(agent_id)
        # Update tracked phase
        self._last_green_phase[agent_id] = current_phase
        if last_phase is None:
            return 0.0  # first step, no switch
        return 1.0 if current_phase != last_phase else 0.0

    def _compute_agent_throughput(self, agent_id: str) -> float:
        """Negative pressure as throughput proxy.

        pressure = #vehicles_incoming - #vehicles_outgoing.
        Lower pressure (more negative) = vehicles flowing through = good.
        We return -pressure so that HIGHER = BETTER (standard reward direction).

        Unlike the raw out-lane vehicle count (getLastStepVehicleNumber),
        pressure is a *differential* quantity: it measures the NET flow
        through the intersection, not cumulative lane occupancy.  This
        avoids the pathological case where congested outgoing lanes
        produce artificially high "throughput" rewards.
        """
        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0.0
        return float(-ts.get_pressure())

    def _compute_agent_directional_reward(self, agent_id: str) -> float:
        """Number of moving vehicles on priority lanes only.

        Creates a CC-like structural conflict with starvation cost:
        - Hold green for priority direction -> vehicles flow -> reward UP
          but non-priority lanes starve -> cost UP
        - Give green to non-priority -> reward DOWN but cost DOWN
        - Agent CANNOT optimize both simultaneously.

        moving = total_vehicles - halting_vehicles on priority lanes.
        Always >= 0.  Higher = more throughput on the priority direction.
        """
        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0.0
        priority = self._priority_lanes.get(agent_id, [])
        if not priority:
            return 0.0
        moving = sum(
            ts.sumo.lane.getLastStepVehicleNumber(l)
            - ts.sumo.lane.getLastStepHaltingNumber(l)
            for l in priority
        )
        return float(moving)

    def _compute_agent_cost_starvation(self, agent_id: str) -> float:
        """1 if max halting vehicles on any non-priority lane > threshold.

        CC-latency analogue: non-priority lanes accumulate stopped vehicles
        when the agent holds green for its priority direction.
        """
        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0.0
        non_priority = self._non_priority_lanes.get(agent_id, [])
        if not non_priority:
            return 0.0
        halting = [ts.sumo.lane.getLastStepHaltingNumber(l) for l in non_priority]
        return 1.0 if max(halting) > self._starvation_threshold else 0.0

    def _compute_directional_ns_reward(self, agent_id: str) -> float:
        """Diff-waiting-time on mainline incoming lanes.

        Returns -(mainline_wait_now - mainline_wait_prev), so positive
        when mainline waiting decreases (= mainline is being served).
        Named 'ns_wait' / 'arterial' in override_reward for compat.
        """
        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0.0
        ml_idx = self._mainline_lane_indices.get(agent_id, [])
        if not ml_idx:
            return 0.0
        wait_per_lane = ts.get_accumulated_waiting_time_per_lane()
        ml_wait = sum(wait_per_lane[i] for i in ml_idx) / 100.0
        prev = self._prev_mainline_wait.get(agent_id, 0.0)
        self._prev_mainline_wait[agent_id] = ml_wait
        return -(ml_wait - prev)

    def _compute_directional_ew_cost(self, agent_id: str) -> float:
        """Side-street service deficit: max(0, side waiting increase).

        Returns max(0, side_wait_now - side_wait_prev) / 100.
        Positive when side-street waiting grows (service deficit),
        zero when side-street is being served (no deficit).

        Constrained RL framing:
          Reward = mainline delay reduction (maximize)
          Cost   = side-street service deficit (minimize, subject to budget)
          Green time is finite: serving mainline starves side-streets.
        """
        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0.0
        side_idx = self._side_lane_indices.get(agent_id, [])
        if not side_idx:
            return 0.0
        wait_per_lane = ts.get_accumulated_waiting_time_per_lane()
        side_wait = sum(wait_per_lane[i] for i in side_idx) / 100.0
        prev = self._prev_side_wait.get(agent_id, 0.0)
        self._prev_side_wait[agent_id] = side_wait
        return max(0.0, side_wait - prev)

    def _compute_agent_cost(self, agent_id: str) -> float:
        """Dispatch to the selected cost function."""
        if self._cost_fn == "service_gap":
            return self._compute_service_gap_cost(agent_id)
        elif self._cost_fn == "directional":
            return self._compute_directional_ew_cost(agent_id)
        elif self._cost_fn == "conflict":
            # Binary OR: 1 if fairness or phase-churn fires
            c_fair = self._compute_agent_cost_fairness(agent_id)
            c_churn = self._compute_agent_cost_phase_churn(agent_id)
            return 1.0 if (c_fair > 0 or c_churn > 0) else 0.0
        elif self._cost_fn == "fairness":
            return self._compute_agent_cost_fairness(agent_id)
        elif self._cost_fn == "phase_churn":
            return self._compute_agent_cost_phase_churn(agent_id)
        elif self._cost_fn == "combined":
            return (self._compute_agent_cost_queue_overflow(agent_id) +
                    self._compute_agent_cost_max_wait(agent_id) +
                    self._compute_agent_cost_lane_saturation(agent_id))
        elif self._cost_fn == "queue_overflow":
            return self._compute_agent_cost_queue_overflow(agent_id)
        elif self._cost_fn == "max_wait":
            return self._compute_agent_cost_max_wait(agent_id)
        elif self._cost_fn == "lane_saturation":
            return self._compute_agent_cost_lane_saturation(agent_id)
        elif self._cost_fn == "emergency":
            return self._compute_emergency_cost()
        elif self._cost_fn == "side_queue":
            return self._compute_side_queue_cost(agent_id)
        return 0.0

    def _compute_emergency_cost(self) -> float:
        """Global cost from native SUMO failure events."""
        events = self._query_failure_events()
        cost = 0.0
        if events["emergency_stops"] > 0:
            cost += 1.0
        if events["teleports"] > 0:
            cost += self._teleport_weight
        return cost

    def _query_failure_events(self) -> Dict[str, int]:
        """Return accumulated failure events since last query, then reset."""
        events = {
            "emergency_stops": self._accumulated_emergency_stops,
            "teleports": self._accumulated_teleports,
        }
        self._accumulated_emergency_stops = 0
        self._accumulated_teleports = 0
        return events

    # ── Per-agent metrics (always computed, for logging) ──────────────────

    def _get_agent_metrics(self, agent_id: str) -> Dict[str, float]:
        """Get per-agent traffic metrics for logging (independent of cost_fn)."""
        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return {"queued": 0, "max_wait": 0.0, "max_lane_queue": 0.0,
                    "avg_speed": 1.0, "pressure": 0}
        queued = ts.get_total_queued()
        wait_per_lane = ts.get_accumulated_waiting_time_per_lane()
        max_wait = max(wait_per_lane) if wait_per_lane else 0.0
        lane_queues = ts.get_lanes_queue()
        max_lane_queue = max(lane_queues) if lane_queues else 0.0
        avg_speed = ts.get_average_speed()
        pressure = ts.get_pressure()
        return {
            "queued": queued,
            "max_wait": max_wait,
            "max_lane_queue": max_lane_queue,
            "avg_speed": avg_speed,
            "pressure": pressure,
        }

    def _update_failure_history(self, agent_id: str, has_failure: bool):
        """Update per-agent ring buffer for event-proximal label."""
        history = self._failure_history[agent_id]
        history.append(has_failure)
        if len(history) > self._label_horizon:
            history.pop(0)

    def _event_proximal_label(self, agent_id: str) -> int:
        """Fallback label: 1 if any cost fired in last label_horizon steps for this agent."""
        return int(any(self._failure_history[agent_id]))

    def _directional_label(self, agent_id: str) -> int:
        """Sharp event label: 1 when side-street service guarantee is violated.

        Fires when ANY of these events occur:
          1. Side-street not served for ``tau_service`` consecutive steps
             (no-service timeout — a vehicle has been waiting > tau*delta_time).
          2. Average side-street queue ratio exceeds ``side_queue_cap``
             (queue overflow — approaching gridlock on side-street).

        These are sharp, operationally meaningful events — not dense
        continuous thresholds.  The label tells the tree: 'at this state,
        the cost gradient should shape the leaf because the side-street
        guarantee is being violated.'
        """
        # Event 1: no service for tau_service consecutive steps
        if self._side_consecutive_unserved.get(agent_id, 0) >= self._tau_service:
            return 1

        # Event 2: side-street queue above cap
        ts = self._get_traffic_signal(agent_id)
        if ts is not None:
            side_idx = self._side_lane_indices.get(agent_id, [])
            if side_idx:
                lane_queues = ts.get_lanes_queue()
                avg_side_q = sum(lane_queues[i] for i in side_idx) / len(side_idx)
                if avg_side_q > self._side_queue_cap:
                    return 1

        return 0

    def reset(self, seed=None, options=None):
        self._traci_dead = False  # clear crash flag for new episode

        # sumo_rl's reset() calls close() on the old TraCI connection.
        # If TraCI is already dead (e.g. from a crash in the previous
        # episode), close() will throw. Catch that and force a fresh start.
        try:
            obs, infos = self._env.reset(seed=seed, options=options)
        except Exception:
            # Kill any lingering SUMO process and retry
            try:
                self._env.close()
            except Exception:
                pass
            obs, infos = self._env.reset(seed=seed, options=options)

        self.agents = list(self._env.agents) if hasattr(self._env, 'agents') else list(self.possible_agents)

        # Re-acquire TraCI handle (new simulation instance after reset)
        self._sumo = None
        self._sumo_env = None
        self._accumulated_emergency_stops = 0
        self._accumulated_teleports = 0
        self._failure_history = {a: [] for a in self.possible_agents}
        self._last_green_phase = {a: None for a in self.possible_agents}

        # Patch _sumo_step for emergency cost (only if needed)
        self._patch_sumo_step()

        # Compute per-agent lane splits for directional reward / starvation cost
        if self._override_reward == "directional" or self._cost_fn == "starvation":
            self._priority_lanes.clear()
            self._non_priority_lanes.clear()
            for agent_id in self.possible_agents:
                ts = self._get_traffic_signal(agent_id)
                if ts is not None:
                    n = len(ts.in_lanes)
                    self._priority_lanes[agent_id] = list(ts.in_lanes[:n // 2])
                    self._non_priority_lanes[agent_id] = list(ts.in_lanes[n // 2:])

        # Classify NS vs EW lanes for directional reward/cost split
        if self._override_reward in ("ns_wait", "arterial", "mainline") or self._cost_fn in ("directional", "service_gap", "side_deficit", "side_queue"):
            self._classify_mainline_side()

        # Clean episode coin flip: with probability clean_episode_prob,
        # no scenario events occur this episode (obs extensions stay zero,
        # cost/label stay zero). Same train/test distribution for all methods.
        self._is_clean_episode = (
            self._clean_episode_prob > 0
            and np.random.random() < self._clean_episode_prob
        )

        # Bus / convoy / premium injection setup
        if self._cost_fn in ("bus_priority", "convoy_priority", "premium_priority"):
            # Build phase→lane mapping (needed for label: "is lane served?")
            if not self._phase_to_lanes:
                self._classify_mainline_side()  # also builds phase_to_lanes
            self._discover_boundary_edges()
            self._bus_vtype_created = False
            self._convoy_vtype_created = False
            self._premium_vtype_created = False
            self._bus_counter = 0
            self._convoy_counter = 0
            self._premium_counter = 0
            self._active_convoys = {}  # reset convoy tracking
            self._convoy_seen_by_agent = {}  # reset per-agent convoy history
            if self._is_clean_episode:
                # Push injection past episode horizon → no events
                self._next_bus_time = 1e9
                self._next_convoy_time = 1e9
                self._next_premium_time = 1e9
            else:
                self._next_bus_time = self._bus_injection_interval * 0.5
                self._next_convoy_time = self._convoy_injection_interval * 0.5
                self._next_premium_time = self._premium_injection_interval * 0.5
            if self._cost_fn == "bus_priority":
                self._create_bus_vtype()
            elif self._cost_fn == "convoy_priority":
                self._create_convoy_vtype()
            elif self._cost_fn == "premium_priority":
                self._create_premium_vtype()
            # Extend obs spaces on first reset
            self._extend_obs_spaces()
            # Init per-agent feature arrays
            for agent_id in self.possible_agents:
                ts = self._get_traffic_signal(agent_id)
                if ts is not None:
                    n_lanes = len(ts.lanes)
                    self._has_bus[agent_id] = np.zeros(n_lanes, dtype=np.float32)
                    self._bus_wait[agent_id] = np.zeros(n_lanes, dtype=np.float32)
                    self._bus_count[agent_id] = np.zeros(n_lanes, dtype=np.float32)
                    self._has_convoy[agent_id] = np.zeros(n_lanes, dtype=np.float32)
                    self._convoy_count[agent_id] = np.zeros(n_lanes, dtype=np.float32)
                    self._convoy_progress[agent_id] = np.zeros(n_lanes, dtype=np.float32)
                    self._has_premium[agent_id] = np.zeros(n_lanes, dtype=np.float32)
                    self._premium_wait[agent_id] = np.zeros(n_lanes, dtype=np.float32)

        # Spillback / road works setup
        if self._cost_fn == "spillback":
            if not self._phase_to_lanes:
                self._classify_mainline_side()  # also builds phase_to_lanes
            self._build_lane_downstream_mapping()
            self._discover_internal_edges()
            self._active_roadworks = []
            self._roadwork_counter = 0
            if self._is_clean_episode:
                self._next_roadwork_time = 1e9  # no road works this episode
            else:
                self._next_roadwork_time = self._roadwork_interval_mean * 0.3
            self._extend_obs_spaces()
            # Init per-agent downstream occupancy arrays
            for agent_id in self.possible_agents:
                ts = self._get_traffic_signal(agent_id)
                if ts is not None:
                    n_lanes = len(ts.lanes)
                    self._downstream_occ[agent_id] = np.zeros(n_lanes, dtype=np.float32)

        # Reset episode accumulators
        self._episode_costs = {agent: 0.0 for agent in self.possible_agents}
        self._episode_original_rewards = {agent: 0.0 for agent in self.possible_agents}
        self._episode_max_queued = {a: 0 for a in self.possible_agents}
        self._episode_max_wait = {a: 0.0 for a in self.possible_agents}

        new_infos = {}
        for agent in self.agents:
            info = dict(infos.get(agent, {}))
            info["cost"] = 0.0
            info["original_reward"] = 0.0
            info["safety_label"] = 0
            new_infos[agent] = info

        obs = self._transform_obs(obs)
        obs = self._extend_obs(obs)
        return obs, new_infos

    def _force_terminate_episode(self, reason: str):
        """Return safe dummy outputs when TraCI/SUMO connection is dead."""
        import warnings
        warnings.warn(f"SUMO/TraCI crashed, forcing episode termination: {reason}")
        self._traci_dead = True
        obs = {a: np.zeros(self.observation_spaces[a].shape, dtype=np.float32)
               for a in self.possible_agents}
        rewards = {a: 0.0 for a in self.possible_agents}
        terminations = {a: True for a in self.possible_agents}
        truncations = {a: True for a in self.possible_agents}
        infos = {}
        for a in self.possible_agents:
            infos[a] = {
                "cost": 0.0, "original_reward": 0.0, "safety_label": 0,
                "queued": 0, "max_wait": 0.0, "max_lane_queue": 0,
                "avg_speed": 0.0, "pressure": 0.0,
                "episode_cost": self._episode_costs.get(a, 0.0),
                "episode_original_reward": self._episode_original_rewards.get(a, 0.0),
                "episode_max_queued": self._episode_max_queued.get(a, 0),
                "episode_max_wait": self._episode_max_wait.get(a, 0.0),
            }
        self.agents = []
        obs = self._transform_obs(obs)
        obs = self._extend_obs(obs)
        return obs, rewards, terminations, truncations, infos

    def step(self, actions):
        # If a previous step already killed the TraCI connection, keep
        # returning terminal observations until the env is reset.
        if getattr(self, '_traci_dead', False):
            return self._force_terminate_episode("TraCI connection already dead")

        try:
            obs, rewards, terminations, truncations, infos = self._env.step(actions)
        except Exception as e:
            return self._force_terminate_episode(str(e))

        self.agents = list(self._env.agents) if hasattr(self._env, 'agents') else list(self.possible_agents)

        # ── Post-step processing (reward/cost/metrics) also uses TraCI ──
        # Wrap in try/except so a mid-step SUMO crash doesn't kill training.
        try:
            return self._process_step(obs, rewards, terminations, truncations, infos)
        except Exception as e:
            return self._force_terminate_episode(str(e))

    def _process_step(self, obs, rewards, terminations, truncations, infos):
        """Compute reward overrides, costs, and per-agent metrics.

        Separated from step() so the outer try/except can catch TraCI
        failures in any of the downstream calls.

        label = g(s_{t+1}): describes the state the agent is NOW in.
        cost  = c(s_{t+1}): cost of the resulting state.
        """
        # Inject new scenario events + detect from resulting state s_{t+1}.
        # Both injection and detection happen post-step so label/cost
        # reflect the current state the agent just transitioned into.
        if self._cost_fn in ("bus_priority", "convoy_priority", "premium_priority"):
            sumo = self._get_traci()
            if sumo is not None:
                sim_time = sumo.simulation.getTime()
                self._inject_vehicles(sim_time)
                self._detect_special_vehicles()

        if self._cost_fn == "spillback":
            sumo = self._get_traci()
            if sumo is not None:
                sim_time = sumo.simulation.getTime()
                self._clear_roadworks(sim_time)
                self._inject_roadwork(sim_time)
                self._compute_downstream_occupancy()

        # For emergency cost_fn, compute once (global)
        emergency_cost = None
        if self._cost_fn == "emergency":
            emergency_cost = self._compute_emergency_cost()

        new_rewards = {}
        new_infos = {}
        for agent in rewards:
            original_reward = float(rewards[agent])
            info = dict(infos.get(agent, {}))

            # ── Reward: throughput/directional override or original ──
            if self._override_reward == "throughput":
                reward = self._compute_agent_throughput(agent)
            elif self._override_reward == "directional":
                reward = self._compute_agent_directional_reward(agent)
            elif self._override_reward in ("ns_wait", "arterial"):
                reward = self._compute_directional_ns_reward(agent)
            elif self._override_reward == "mainline":
                reward = self._compute_mainline_reward(agent)
            else:
                reward = original_reward

            # ── Compute the training cost signal ──
            if self._cost_fn == "side_queue":
                cost = self._compute_side_queue_cost(agent)
            elif self._cost_fn == "side_deficit":
                self._update_service_tracking(agent)
                cost = self._compute_side_deficit_cost(agent)
            elif self._cost_fn == "service_gap":
                self._update_service_tracking(agent)
                cost = self._compute_service_gap_cost(agent)
            elif self._cost_fn == "directional":
                cost = self._compute_directional_ew_cost(agent)
            elif self._cost_fn == "starvation":
                cost = self._compute_agent_cost_starvation(agent)
            elif self._cost_fn == "conflict":
                c_fair = self._compute_agent_cost_fairness(agent)
                c_churn = self._compute_agent_cost_phase_churn(agent)
                cost = 1.0 if (c_fair > 0 or c_churn > 0) else 0.0
            elif self._cost_fn == "fairness":
                cost = self._compute_agent_cost_fairness(agent)
            elif self._cost_fn == "phase_churn":
                cost = self._compute_agent_cost_phase_churn(agent)
            elif self._cost_fn == "combined":
                cost = (self._compute_agent_cost_queue_overflow(agent) +
                        self._compute_agent_cost_max_wait(agent) +
                        self._compute_agent_cost_lane_saturation(agent))
            elif self._cost_fn == "queue_overflow":
                cost = self._compute_agent_cost_queue_overflow(agent)
            elif self._cost_fn == "max_wait":
                cost = self._compute_agent_cost_max_wait(agent)
            elif self._cost_fn == "lane_saturation":
                cost = self._compute_agent_cost_lane_saturation(agent)
            elif self._cost_fn == "emergency":
                cost = emergency_cost
            elif self._cost_fn == "bus_priority":
                cost = self._compute_bus_priority_cost(agent)
            elif self._cost_fn == "convoy_priority":
                cost = self._compute_convoy_priority_cost(agent)
            elif self._cost_fn == "spillback":
                cost = self._compute_spillback_cost(agent)
            elif self._cost_fn == "premium_priority":
                cost = self._compute_premium_priority_cost(agent)
            else:
                cost = 0.0

            # Per-agent metrics (always logged, regardless of cost_fn)
            metrics = self._get_agent_metrics(agent)

            # Update side-street service tracking for directional label
            if self._cost_fn == "directional":
                if cost > 0:
                    self._side_consecutive_unserved[agent] = (
                        self._side_consecutive_unserved.get(agent, 0) + 1
                    )
                else:
                    self._side_consecutive_unserved[agent] = 0
            # service_gap tracking already called above in cost dispatch

            # Update per-agent label history
            has_cost = cost > 0
            self._update_failure_history(agent, has_cost)

            # Label: state-based for side_queue/side_deficit/directional/bus/convoy/premium, else event-proximal
            if self._cost_fn == "side_queue":
                label = self._side_queue_label(agent)
            elif self._cost_fn == "side_deficit":
                label = self._side_deficit_label(agent)
            elif self._cost_fn == "directional":
                label = self._directional_label(agent)
            elif self._cost_fn == "bus_priority":
                label = self._bus_priority_label(agent)
            elif self._cost_fn == "convoy_priority":
                label = self._convoy_priority_label(agent)
            elif self._cost_fn == "spillback":
                label = self._spillback_label(agent)
            elif self._cost_fn == "premium_priority":
                label = self._premium_priority_label(agent)
            else:
                label = self._event_proximal_label(agent)

            new_rewards[agent] = reward
            info["cost"] = cost
            info["original_reward"] = original_reward
            info["safety_label"] = label
            # Per-agent traffic metrics
            info["queued"] = metrics["queued"]
            info["max_wait"] = metrics["max_wait"]
            info["max_lane_queue"] = metrics["max_lane_queue"]
            info["avg_speed"] = metrics["avg_speed"]
            info["pressure"] = metrics["pressure"]

            # Track per-agent episode stats
            self._episode_costs[agent] = self._episode_costs.get(agent, 0.0) + cost
            self._episode_original_rewards[agent] = (
                self._episode_original_rewards.get(agent, 0.0) + original_reward
            )
            self._episode_max_queued[agent] = max(
                self._episode_max_queued.get(agent, 0), metrics["queued"]
            )
            self._episode_max_wait[agent] = max(
                self._episode_max_wait.get(agent, 0.0), metrics["max_wait"]
            )

            # Emit episode-level stats when agent terminates
            if terminations.get(agent, False) or truncations.get(agent, False):
                info["episode_cost"] = self._episode_costs.get(agent, 0.0)
                info["episode_original_reward"] = self._episode_original_rewards.get(agent, 0.0)
                info["episode_max_queued"] = self._episode_max_queued.get(agent, 0)
                info["episode_max_wait"] = self._episode_max_wait.get(agent, 0.0)

            new_infos[agent] = info

        obs = self._transform_obs(obs)
        obs = self._extend_obs(obs)
        return obs, new_rewards, terminations, truncations, new_infos

    def render(self):
        return self._env.render()

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass  # TraCI connection already dead

    def state(self):
        return self._env.state()


# ──────────────────────────────────────────────────────────────────────────────
# Predefined SUMO-RL configurations (RESCO benchmarks)
# ──────────────────────────────────────────────────────────────────────────────

def _find_sumo_rl_nets():
    """Locate the sumo_rl nets directory."""
    try:
        import sumo_rl
        import os
        pkg_dir = os.path.dirname(sumo_rl.__file__)
        nets_dir = os.path.join(pkg_dir, "nets")
        if os.path.isdir(nets_dir):
            return nets_dir
    except ImportError:
        pass
    return None


def _resco_path(scenario: str, filename: str) -> str:
    nets = _find_sumo_rl_nets()
    if nets is None:
        raise ImportError(
            "sumo_rl is not installed. Install with: pip install sumo-rl"
        )
    import os
    path = os.path.join(nets, "RESCO", scenario, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"RESCO file not found: {path}")
    return path


# RESCO benchmark configurations
# Each has uniform obs/action spaces across all intersections
SUMO_CONFIGS = {
    # --- Synthetic grids (uniform topology) ---
    "sumo-grid4x4-v0": {
        "net_file_fn": lambda: _resco_path("grid4x4", "grid4x4.net.xml"),
        "route_file_fn": lambda: _resco_path("grid4x4", "grid4x4_1.rou.xml"),
        "num_seconds": 3600,
        "delta_time": 5,
        "yellow_time": 2,
        "min_green": 5,
        "max_green": 50,
        "reward_fn": "diff-waiting-time",
        "n_agents": 16,  # 4x4 grid
    },
    "sumo-arterial4x4-v0": {
        "net_file_fn": lambda: _resco_path("arterial4x4", "arterial4x4.net.xml"),
        "route_file_fn": lambda: _resco_path("arterial4x4", "arterial4x4_1.rou.xml"),
        "num_seconds": 3600,
        "delta_time": 5,
        "yellow_time": 2,
        "min_green": 5,
        "max_green": 50,
        "reward_fn": "diff-waiting-time",
        "n_agents": 16,
    },
    # --- Real-world networks (RESCO) ---
    "sumo-ingolstadt1-v0": {
        "net_file_fn": lambda: _resco_path("ingolstadt1", "ingolstadt1.net.xml"),
        "route_file_fn": lambda: _resco_path("ingolstadt1", "ingolstadt1.rou.xml"),
        "num_seconds": 3600,
        "delta_time": 5,
        "yellow_time": 2,
        "min_green": 5,
        "max_green": 50,
        "reward_fn": "diff-waiting-time",
        "n_agents": 1,
    },
    "sumo-cologne1-v0": {
        "net_file_fn": lambda: _resco_path("cologne1", "cologne1.net.xml"),
        "route_file_fn": lambda: _resco_path("cologne1", "cologne1.rou.xml"),
        "num_seconds": 3600,
        "delta_time": 5,
        "yellow_time": 2,
        "min_green": 5,
        "max_green": 50,
        "reward_fn": "diff-waiting-time",
        "n_agents": 1,
    },
    "sumo-cologne3-v0": {
        "net_file_fn": lambda: _resco_path("cologne3", "cologne3.net.xml"),
        "route_file_fn": lambda: _resco_path("cologne3", "cologne3.rou.xml"),
        "num_seconds": 3600,
        "delta_time": 5,
        "yellow_time": 2,
        "min_green": 5,
        "max_green": 50,
        "reward_fn": "diff-waiting-time",
        "n_agents": 3,
    },
    "sumo-cologne8-v0": {
        "net_file_fn": lambda: _resco_path("cologne8", "cologne8.net.xml"),
        "route_file_fn": lambda: _resco_path("cologne8", "cologne8.rou.xml"),
        "num_seconds": 3600,
        "delta_time": 5,
        "yellow_time": 2,
        "min_green": 5,
        "max_green": 50,
        "reward_fn": "diff-waiting-time",
        "n_agents": 8,
    },
    "sumo-ingolstadt7-v0": {
        "net_file_fn": lambda: _resco_path("ingolstadt7", "ingolstadt7.net.xml"),
        "route_file_fn": lambda: _resco_path("ingolstadt7", "ingolstadt7.rou.xml"),
        "num_seconds": 3600,
        "delta_time": 5,
        "yellow_time": 2,
        "min_green": 5,
        "max_green": 50,
        "reward_fn": "diff-waiting-time",
        "n_agents": 7,
    },
    "sumo-ingolstadt21-v0": {
        "net_file_fn": lambda: _resco_path("ingolstadt21", "ingolstadt21.net.xml"),
        "route_file_fn": lambda: _resco_path("ingolstadt21", "ingolstadt21.rou.xml"),
        "num_seconds": 3600,
        "delta_time": 5,
        "yellow_time": 2,
        "min_green": 5,
        "max_green": 50,
        "reward_fn": "diff-waiting-time",
        "n_agents": 21,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# VecEnv construction
# ──────────────────────────────────────────────────────────────────────────────

def make_sumo_raw_env(
    net_file: str,
    route_file: str,
    num_seconds: int = 3600,
    delta_time: int = 5,
    yellow_time: int = 2,
    min_green: int = 5,
    max_green: int = 50,
    reward_fn: str = "diff-waiting-time",
    sumo_seed: Union[str, int] = "random",
    use_gui: bool = False,
    add_system_info: bool = True,
    add_per_agent_info: bool = True,
):
    """
    Create a SUMO-RL PettingZoo parallel env.

    Returns the raw PettingZoo parallel env (unmodified SUMO-RL).
    """
    import sumo_rl

    env = sumo_rl.parallel_env(
        net_file=net_file,
        route_file=route_file,
        num_seconds=num_seconds,
        delta_time=delta_time,
        yellow_time=yellow_time,
        min_green=min_green,
        max_green=max_green,
        reward_fn=reward_fn,
        sumo_seed=sumo_seed,
        use_gui=use_gui,
        add_system_info=add_system_info,
        add_per_agent_info=add_per_agent_info,
    )
    return env


def make_sumo_vec_env(
    env_name: str = "sumo-grid4x4-v0",
    n_envs: int = 1,
    seed: Optional[int] = None,
    override_reward: Optional[str] = None,
    cost_fn: str = "convoy_priority",
    starvation_threshold: int = 3,
    fairness_tau: float = 60.0,
    fairness_rho: float = 3.0,
    queue_threshold: int = 10,
    wait_threshold: float = 200.0,
    saturation_threshold: float = 0.8,
    teleport_weight: float = 1.0,
    label_horizon: int = 3,
    use_categorical_phase: bool = False,
    mainline_direction: str = "auto",
    tau_service: int = 6,
    side_queue_cap: float = 0.7,
    deficit_kappa: float = 5.0,
    bus_injection_interval: float = 25.0,
    bus_count_per_injection: int = 1,
    bus_cost_threshold: float = 30.0,
    bus_warn_threshold: float = 10.0,
    convoy_injection_interval: float = 80.0,
    convoy_size_min: int = 10,
    convoy_size_max: int = 12,
    convoy_headway: float = 2.0,
    convoy_cost_threshold: float = 20.0,
    convoy_warn_threshold: float = 8.0,
    premium_injection_interval: float = 40.0,
    premium_cost_threshold: float = 15.0,
    premium_warn_threshold: float = 5.0,
    premium_pressure_beta: float = 1.5,
    roadwork_interval_mean: float = 300.0,
    roadwork_duration_mean: float = 400.0,
    roadwork_speed: float = 0.3,
    spillback_occ_threshold: float = 0.5,
    clean_episode_prob: float = 0.0,
    use_gui: bool = False,
    **override_kwargs,
):
    """
    Build a SUMO-RL SB3 VecEnv using shared-policy parameter sharing.

    Pipeline: sumo_rl.parallel_env → SumoRewardCostWrapper → SuperSuit VecEnv → VecCostMonitor

    Each traffic signal agent gets its own VecEnv slot, so with
    n_agents=16 (grid4x4) and n_envs=2 you get num_envs=32 in the VecEnv.

    Parameters
    ----------
    env_name : str
        Key from SUMO_CONFIGS.
    n_envs : int
        Number of parallel SUMO simulations to concatenate.
        NOTE: with LIBSUMO_AS_TRACI=1, only n_envs=1 is supported.
    seed : int or None
        Random seed for SUMO.
    override_reward : str or None
        If "mainline", reward = diff-waiting-time on mainline lanes only (default).
        If "arterial" or "ns_wait", reward = diff-waiting-time on NS lanes only.
        If "directional", reward = moving vehicles on priority lanes.
        If "throughput", replace sumo-rl reward with negative pressure.
        None keeps the original sumo-rl reward (e.g., diff-waiting-time).
    cost_fn : str
        Cost function. See COST_FN_REGISTRY. Default "side_deficit".
    starvation_threshold : int
        For cost_fn="starvation": max halting vehicles per non-priority lane. Default 3.
    fairness_tau : float
        Absolute floor for fairness cost (seconds). Default 60.
    fairness_rho : float
        Relative ratio threshold for fairness. Default 3.0.
    queue_threshold : int
        For cost_fn="queue_overflow": max allowed queued vehicles per intersection.
    wait_threshold : float
        For cost_fn="max_wait": max allowed accumulated wait (seconds) on any lane.
    saturation_threshold : float
        For cost_fn="lane_saturation": max allowed lane queue ratio (0-1).
    teleport_weight : float
        For cost_fn="emergency": multiplier for teleport events.
    label_horizon : int
        Number of past steps for event-proximal fallback label.
    deficit_kappa : float
        For cost_fn="side_deficit": label threshold. Label=1 when deficit > kappa. Default 5.0.
    use_gui : bool
        Whether to render SUMO GUI.
    **override_kwargs
        Override any SUMO_CONFIGS parameter.

    Returns
    -------
    SB3-compatible VecEnv
    """
    if env_name not in SUMO_CONFIGS:
        raise ValueError(
            f"Unknown env_name '{env_name}'. Available: {list(SUMO_CONFIGS.keys())}"
        )

    config = SUMO_CONFIGS[env_name].copy()
    config.update(override_kwargs)

    # Resolve lazy file paths
    net_file = config.pop("net_file_fn")()
    route_file = config.pop("route_file_fn")()
    n_agents = config.pop("n_agents")

    sumo_seed = seed if seed is not None else "random"

    par_env = make_sumo_raw_env(
        net_file=net_file,
        route_file=route_file,
        num_seconds=config.get("num_seconds", 3600),
        delta_time=config.get("delta_time", 5),
        yellow_time=config.get("yellow_time", 2),
        min_green=config.get("min_green", 5),
        max_green=config.get("max_green", 50),
        reward_fn=config.get("reward_fn", "diff-waiting-time"),
        sumo_seed=sumo_seed,
        use_gui=use_gui,
    )

    # Reward/cost wrapper (configurable constraint-based cost)
    par_env = SumoRewardCostWrapper(
        par_env,
        override_reward=override_reward,
        cost_fn=cost_fn,
        starvation_threshold=starvation_threshold,
        fairness_tau=fairness_tau,
        fairness_rho=fairness_rho,
        queue_threshold=queue_threshold,
        wait_threshold=wait_threshold,
        saturation_threshold=saturation_threshold,
        teleport_weight=teleport_weight,
        label_horizon=label_horizon,
        use_categorical_phase=use_categorical_phase,
        mainline_direction=mainline_direction,
        tau_service=tau_service,
        side_queue_cap=side_queue_cap,
        deficit_kappa=deficit_kappa,
        bus_injection_interval=bus_injection_interval,
        bus_count_per_injection=bus_count_per_injection,
        bus_cost_threshold=bus_cost_threshold,
        bus_warn_threshold=bus_warn_threshold,
        convoy_injection_interval=convoy_injection_interval,
        convoy_size_min=convoy_size_min,
        convoy_size_max=convoy_size_max,
        convoy_headway=convoy_headway,
        convoy_cost_threshold=convoy_cost_threshold,
        convoy_warn_threshold=convoy_warn_threshold,
        premium_injection_interval=premium_injection_interval,
        premium_cost_threshold=premium_cost_threshold,
        premium_warn_threshold=premium_warn_threshold,
        premium_pressure_beta=premium_pressure_beta,
        roadwork_interval_mean=roadwork_interval_mean,
        roadwork_duration_mean=roadwork_duration_mean,
        roadwork_speed=roadwork_speed,
        spillback_occ_threshold=spillback_occ_threshold,
        clean_episode_prob=clean_episode_prob,
    )

    # PettingZoo → SB3 VecEnv via SuperSuit
    vec_env = ss.pettingzoo_env_to_vec_env_v1(par_env)
    vec_env = ss.concat_vec_envs_v1(
        vec_env, n_envs, num_cpus=1, base_class="stable_baselines3"
    )

    # Patch missing .seed() for SB3 compatibility
    inner = getattr(vec_env, "venv", vec_env)
    if not hasattr(inner, "seed"):
        inner.seed = lambda s=None: None

    # Flag for GBRL mixed-mode detection
    if use_categorical_phase:
        n_phases = config.get("n_agents", None)
        # Get n_phases from action space
        n_phases = vec_env.action_space.n  # Discrete action = n_green_phases
        vec_env = VecCategoricalPhase(vec_env, n_phases=n_phases)

    # Add VecCostMonitor for episode tracking
    vec_env = VecCostMonitor(vec_env)

    return vec_env


class VecCategoricalPhase(VecMonitor.__bases__[0] if hasattr(VecMonitor, '__bases__') else object):
    """VecEnv wrapper that replaces one-hot phase with a categorical string.

    Converts float32 obs of shape (n_envs, n_phases + 1 + 2*n_lanes) to
    object obs of shape (n_envs, 1 + 1 + 2*n_lanes) where column 0 is a
    categorical string "p0"..."p7" and the rest are float64.

    Must be placed AFTER SuperSuit (which requires float arrays) and BEFORE
    any GBRL-facing wrapper (VecCostMonitor, training loop).
    """

    def __init__(self, venv, n_phases: int):
        self.venv = venv
        self._n_phases = n_phases
        self.is_mixed = True
        self.is_categorical = True

        # Proxy attributes from inner venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        if hasattr(venv, 'reward_range'):
            self.reward_range = venv.reward_range
        if hasattr(venv, 'metadata'):
            self.metadata = venv.metadata

    def _convert_obs(self, obs: np.ndarray) -> np.ndarray:
        """Convert float obs batch to mixed object array with categorical phase."""
        n_envs, obs_dim = obs.shape
        n_phases = self._n_phases
        new_dim = obs_dim - n_phases + 1  # replace n_phases cols with 1 col
        mixed = np.empty((n_envs, new_dim), dtype=object)
        # Phase: argmax of one-hot → categorical string
        phase_ids = np.argmax(obs[:, :n_phases], axis=1)
        for i in range(n_envs):
            mixed[i, 0] = f"p{phase_ids[i]}"
        # Remaining features: min_green + density + queue
        mixed[:, 1:] = obs[:, n_phases:].astype(np.float64)
        return mixed

    def reset(self, **kwargs):
        obs = self.venv.reset(**kwargs)
        return self._convert_obs(obs)

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return self._convert_obs(obs), rewards, dones, infos

    def step(self, actions):
        obs, rewards, dones, infos = self.venv.step(actions)
        return self._convert_obs(obs), rewards, dones, infos

    def close(self):
        self.venv.close()

    def seed(self, seed=None):
        if hasattr(self.venv, 'seed'):
            return self.venv.seed(seed)

    def render(self, **kwargs):
        return self.venv.render(**kwargs)

    def __getattr__(self, name):
        return getattr(self.venv, name)


class VecCostMonitor(VecMonitor):
    """VecMonitor subclass that also tracks per-episode cost and traffic metrics.

    Accumulates info["cost"] each step and stores total in episode_info["c"].
    Also tracks per-agent queue/wait violations and peak metrics.
    """

    def __init__(self, venv, filename=None, info_keywords=()):
        super().__init__(venv, filename=filename, info_keywords=info_keywords)
        # Propagate mixed/categorical flags from inner env for GBRL detection
        if getattr(venv, 'is_mixed', False):
            self.is_mixed = True
        if getattr(venv, 'is_categorical', False):
            self.is_categorical = True
        self.episode_costs = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_original_rewards = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_max_queued = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_max_wait = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_scalarizations = np.zeros(self.num_envs, dtype=np.float64)
        self.cost_limit = 0.1

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.episode_costs[:] = 0.0
        self.episode_original_rewards[:] = 0.0
        self.episode_max_queued[:] = 0
        self.episode_max_wait[:] = 0.0
        self.episode_scalarizations[:] = 0.0
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.episode_returns += rewards
        self.episode_lengths += 1
        for i in range(self.num_envs):
            self.episode_costs[i] += infos[i].get("cost", 0.0)
            self.episode_original_rewards[i] += infos[i].get("original_reward", 0.0)
            # Track per-step traffic metrics
            queued = infos[i].get("queued", 0)
            max_wait = infos[i].get("max_wait", 0.0)
            if queued > self.episode_max_queued[i]:
                self.episode_max_queued[i] = queued
            if max_wait > self.episode_max_wait[i]:
                self.episode_max_wait[i] = max_wait
            # Scalarization: reward when under cost budget, else penalised cost
            cost_i = infos[i].get("cost", 0.0)
            rew_i = float(rewards[i])
            if cost_i <= self.cost_limit:
                self.episode_scalarizations[i] += rew_i
            else:
                self.episode_scalarizations[i] += -cost_i - 1.5
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
                    "original_r": float(self.episode_original_rewards[i]),
                    "max_queued": int(self.episode_max_queued[i]),
                    "max_wait": float(self.episode_max_wait[i]),
                    "s": float(self.episode_scalarizations[i]),
                    "t": round(time.time() - self.t_start, 6),
                }
                for key in self.info_keywords:
                    episode_info[key] = info[key]
                info["episode"] = episode_info
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                self.episode_costs[i] = 0.0
                self.episode_original_rewards[i] = 0.0
                self.episode_max_queued[i] = 0
                self.episode_max_wait[i] = 0.0
                self.episode_scalarizations[i] = 0.0
                if self.results_writer:
                    self.results_writer.write_row(episode_info)
                new_infos[i] = info
        return obs, rewards, dones, new_infos
