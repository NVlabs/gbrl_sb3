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
    "bus_priority": "Per-agent: 1 if any bus local wait >= T_cost — binary violation, label fires at T_warn (anticipatory), fixed-corridor injection, obs-extended",
    "convoy_priority": "Per-agent: 1 if convoy is actively being split (0 < progress < 1) on unserved lane — binary violation, label fires anticipatorily (regime-based, no decision-mask), obs-extended",
    "premium_priority": "Per-agent: 1 if premium vehicle route_delay >= T_cost — binary violation, route-level delay tracking, label fires at T_warn (anticipatory), obs-extended",
}


# ──────────────────────────────────────────────────────────────────────────────
# PettingZoo wrapper: reward/cost split with configurable cost functions
# ──────────────────────────────────────────────────────────────────────────────

class SumoRewardCostWrapper(ParallelEnv):
    """
    PettingZoo ParallelEnv wrapper around sumo_rl.parallel_env that:

      reward = pressure (or override_reward) — the primary objective
      info['cost']            = binary constraint violation cost (0/1)
      info['original_reward'] = original sumo-rl reward (e.g. diff-waiting-time)
      info['safety_label']    = anticipatory danger-zone label g(s) ∈ {0, 1}
                                Pure function of state observations. Fires
                                BEFORE cost (T_warn < T_cost) to give SPLIT-RL
                                an anticipatory routing advantage.

    Primary benchmark scenarios (exogenous event injection):
      - "premium_priority": binary cost when premium vehicle waits past T_cost.
      - "convoy_priority":  binary cost when convoy is actively split.
      - "bus_priority":     binary cost when bus waits past T_cost.

    Labels are ownership rules that partition the state space:
      cost objective owns when a priority entity is urgent, on a red
      lane, and the controller can act.  Convoy active splits own
      unconditionally.  No pressure gates, no topology dependence.

    Event generation is exogenous: static conflict routes precomputed at
    reset from network topology (side-street approaches), selected via
    persistent episode RNG. All methods see identical event distributions.

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
        bus_injection_interval: float = 30.0,
        bus_count_per_injection: int = 1,
        bus_cost_threshold: float = 15.0,
        bus_warn_threshold: float = 6.0,
        convoy_injection_interval: float = 80.0,
        convoy_size_min: int = 10,
        convoy_size_max: int = 12,
        convoy_headway: float = 2.0,
        # Premium vehicle injection parameters
        premium_injection_interval: float = 15.0,
        premium_cost_threshold: float = 8.0,
        premium_warn_threshold: float = 3.0,
        premium_count_per_injection: int = 1,
        premium_headway: float = 2.0,
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
        if mainline_direction not in ("ns", "ew", "auto"):
            raise ValueError(
                f"Unknown mainline_direction '{mainline_direction}'. Use 'ns', 'ew', or 'auto'."
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

        # Bus / convoy / premium injection parameters
        self._bus_injection_interval = bus_injection_interval
        self._bus_count_per_injection = bus_count_per_injection
        self._bus_cost_threshold = bus_cost_threshold    # T_bus
        self._bus_warn_threshold = bus_warn_threshold    # T_warn
        self._convoy_injection_interval = convoy_injection_interval
        self._convoy_size_min = convoy_size_min
        self._convoy_size_max = convoy_size_max
        self._convoy_headway = convoy_headway

        # Premium vehicle injection parameters
        self._premium_injection_interval = premium_injection_interval
        self._premium_cost_threshold = premium_cost_threshold   # T_premium
        self._premium_warn_threshold = premium_warn_threshold   # T_warn_premium
        self._premium_count_per_injection = premium_count_per_injection
        self._premium_headway = premium_headway

        # Clean episode probability (shared across all event-based scenarios)
        self._clean_episode_prob = clean_episode_prob
        self._is_clean_episode = False  # set at each reset()
        # Persistent episode-level RNG for all scenario randomness.
        # Created fresh each reset() from the episode seed.
        self._scenario_rng: Optional[np.random.RandomState] = None
        self._episode_count: int = 0

        # Precomputed conflict-route pool (built at reset from network metadata)
        self._conflict_routes: List[List[str]] = []
        self._bus_routes: List[List[str]] = []     # deterministic side-direction routes for buses
        self._bus_route_cursor: int = 0            # round-robin index into _bus_routes
        self._side_edges: Set[str] = set()

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
        # Live sets of injected special vehicle IDs (for fast detection)
        self._active_bus_ids: Set[str] = set()
        self._active_premium_ids: Set[str] = set()
        # Per-agent per-lane bus/convoy/premium features (updated each step)
        self._has_bus: Dict[str, np.ndarray] = {}
        self._bus_wait: Dict[str, np.ndarray] = {}
        self._bus_count: Dict[str, np.ndarray] = {}  # number of buses per lane
        self._has_convoy: Dict[str, np.ndarray] = {}
        self._convoy_count: Dict[str, np.ndarray] = {}  # raw visible count per lane
        self._convoy_seen_frac: Dict[str, np.ndarray] = {}  # seen/total_size per lane (obs-exposed)
        self._convoy_wait: Dict[str, np.ndarray] = {}
        self._convoy_progress: Dict[str, np.ndarray] = {}  # fraction of convoy past stop line
        # Track which convoy IDs belong to which platoon for split detection
        self._active_convoys: Dict[int, List[str]] = {}  # convoy_id → [veh_id, ...]
        self._convoy_info: Dict[int, dict] = {}  # convoy_id → {total_size, vids}
        # Per-agent tracking: which convoy vehicle IDs have ever been seen
        # on this agent's incoming lanes.  Used for correct per-intersection
        # progress: progress = (ever_seen − still_on_incoming) / ever_seen
        self._convoy_seen_by_agent: Dict[str, Dict[int, Set[str]]] = {}  # agent → convoy_id → {vid, …}
        # Premium vehicle features
        self._has_premium: Dict[str, np.ndarray] = {}
        self._premium_wait: Dict[str, np.ndarray] = {}
        # Route-level tracking: vid → {depart_time, expected_time}
        # expected_time is free-flow travel time from origin to current position
        self._premium_vehicle_info: Dict[str, Dict] = {}  # vid → {depart_time, route_edges}

        # Debug mode: when True, extra keys are added to info dict each step
        self._debug: bool = False
        self._debug_step_counter: int = 0

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
        raw_action_spaces = {
            agent: env.action_space(agent) for agent in self.possible_agents
        }

        # Action space handling: uniform by default, opt-in heterogeneous.
        action_dims = [raw_action_spaces[a].n for a in self.possible_agents]
        self._max_n_actions = max(action_dims)
        self._agent_n_actions = {a: raw_action_spaces[a].n for a in self.possible_agents}
        self._heterogeneous_actions = not all(d == self._max_n_actions for d in action_dims)

        if self._heterogeneous_actions:
            import warnings
            warnings.warn(
                f"Heterogeneous action spaces detected "
                f"(min={min(action_dims)}, max={max(action_dims)}). "
                f"Out-of-range actions will be clipped to action 0. "
                f"For clean benchmarks, use networks with uniform phase counts.",
                stacklevel=2,
            )

        self.action_spaces = {
            agent: gym.spaces.Discrete(self._max_n_actions)
            for agent in self.possible_agents
        }

        # Pad observation spaces to max dim for real-world networks with
        # non-uniform intersection topologies (different #lanes → different obs dims).
        obs_dims = [raw_obs_spaces[a].shape[0] for a in self.possible_agents]
        self._max_obs_dim = max(obs_dims)
        self._needs_obs_padding = not all(d == self._max_obs_dim for d in obs_dims)
        self._agent_obs_dims = {a: raw_obs_spaces[a].shape[0] for a in self.possible_agents}

        if self._use_categorical_phase:
            self._n_green_phases = self._max_n_actions

        # Bus/convoy obs extension: compute max_n_lanes from obs layout
        # obs = [phase_one_hot(n_phases), min_green(1), density(n_lanes), queue(n_lanes)]
        # n_lanes = (obs_dim - n_phases_for_that_agent - 1) / 2
        # Use max_n_lanes across ALL agents for uniform obs extension size.
        self._obs_extension_dim = 0
        self._raw_max_obs_dim = self._max_obs_dim  # pre-extension dim for padding
        if self._cost_fn in ("bus_priority", "convoy_priority", "premium_priority"):
            max_n_lanes = max(
                (raw_obs_spaces[a].shape[0] - self._agent_n_actions[a] - 1) // 2
                for a in self.possible_agents
            )
            self._ext_max_n_lanes = max_n_lanes
            if self._cost_fn == "bus_priority":
                self._obs_extension_dim = 3 * max_n_lanes  # has_bus + bus_count + bus_wait
            elif self._cost_fn == "convoy_priority":
                self._obs_extension_dim = 3 * max_n_lanes  # has_convoy + convoy_seen_frac + convoy_progress
            elif self._cost_fn == "premium_priority":
                self._obs_extension_dim = 2 * max_n_lanes  # has_premium + premium_wait
            self._max_obs_dim += self._obs_extension_dim

        if self._needs_obs_padding or self._obs_extension_dim > 0:
            padded_space = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self._max_obs_dim,), dtype=np.float32,
            )
            self.observation_spaces = {a: padded_space for a in self.possible_agents}
        else:
            self.observation_spaces = raw_obs_spaces

        # Per-step TraCI query cache — avoids redundant IPC for the same
        # agent+method across cost / label / metrics within one step.
        # Cleared at top of _process_step(); keyed by (agent_id, method_name).
        self._step_cache: Dict[tuple, Any] = {}

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

    def _ts_query(self, agent_id: str, method_name: str):
        """Cached TrafficSignal query — avoids redundant TraCI calls within one step.

        Each (agent_id, method_name) pair is queried at most once per
        env step.  Cache is cleared at the start of _process_step().
        """
        key = (agent_id, method_name)
        result = self._step_cache.get(key)
        if result is not None:
            return result
        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return None
        result = getattr(ts, method_name)()
        self._step_cache[key] = result
        return result

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

    def _classify_mainline_side(self):
        """Classify lanes as mainline vs side-street per agent.

        Uses ``self._mainline_direction`` to decide which direction is
        mainline:

        - ``"ns"``: force North-South as mainline for all agents.
        - ``"ew"``: force East-West as mainline for all agents.
        - ``"auto"``: direction with more incoming lanes is mainline;
          ties broken by lane count (EW wins ties).

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

            # Assign mainline/side based on mainline_direction setting
            md = self._mainline_direction
            if md == "ns":
                self._mainline_lane_indices[agent_id] = ns_idx
                self._side_lane_indices[agent_id] = ew_idx
            elif md == "ew":
                self._mainline_lane_indices[agent_id] = ew_idx
                self._side_lane_indices[agent_id] = ns_idx
            else:
                # "auto": direction with more lanes is mainline (EW wins ties)
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

    def _select_random_route(self, rng) -> Optional[List[str]]:
        """Pick a random valid route through the network.

        Fallback when no conflict routes are available.  Exogenous: uses
        ONLY network structure + RNG seed, not the agent's current phase
        or queue state.

        Returns a list of edge IDs forming a valid SUMO route, or
        None if no route can be found.
        """
        sumo = self._get_traci()
        if sumo is None:
            return None

        origins = list(self._origin_edges)
        dests = list(self._dest_edges)
        rng.shuffle(origins)
        rng.shuffle(dests)

        import os as _os
        _stderr_fd = _os.dup(2)
        try:
            _devnull = _os.open(_os.devnull, _os.O_WRONLY)
            _os.dup2(_devnull, 2)
            _os.close(_devnull)
            _suppress = True
        except OSError:
            _suppress = False

        result = None
        for origin in origins[:8]:
            for dest in dests[:8]:
                if origin == dest:
                    continue
                try:
                    stage = sumo.simulation.findRoute(origin, dest)
                    if hasattr(stage, 'edges') and len(stage.edges) >= 2:
                        result = list(stage.edges)
                        break
                except Exception:
                    continue
            if result is not None:
                break

        if _suppress:
            _os.dup2(_stderr_fd, 2)
        _os.close(_stderr_fd)
        return result

    def _precompute_conflict_routes(self):
        """Build a pool of routes through side-street approaches.

        Computed once at reset from static network metadata:
        - _classify_mainline_side() → side-street lane indices per agent
        - Side-street lanes → incoming side-street edges
        - Enumerate boundary→boundary routes via findRoute
        - Keep routes with highest side-street edge fraction

        These routes force special vehicles through minority approaches,
        creating structural conflict with the pressure reward (which
        optimises mainline throughput).  The routes are:
        - Static: determined by network topology, not live state
        - Exogenous: no dependence on current policy / queues / phases
        - Conflict-inducing: side-street traversal opposes pressure
        """
        sumo = self._get_traci()
        if sumo is None:
            self._conflict_routes = []
            return

        # Collect all side-street incoming-edge IDs across intersections
        side_edges = set()
        for agent_id in self.possible_agents:
            ts = self._get_traffic_signal(agent_id)
            if ts is None:
                continue
            for i in self._side_lane_indices.get(agent_id, []):
                if i < len(ts.lanes):
                    edge = '_'.join(ts.lanes[i].split('_')[:-1])
                    side_edges.add(edge)

        self._side_edges = side_edges
        if not side_edges:
            self._conflict_routes = []
            return

        # Enumerate candidate routes, score by side-edge fraction.
        # Suppress stderr during findRoute: SUMO C++ prints "No connection"
        # warnings for unreachable pairs, which we handle gracefully.
        import os as _os
        _stderr_fd = _os.dup(2)
        try:
            _devnull = _os.open(_os.devnull, _os.O_WRONLY)
            _os.dup2(_devnull, 2)
            _os.close(_devnull)
            _suppress = True
        except OSError:
            _suppress = False

        candidates = []
        seen = set()
        for origin in self._origin_edges:
            for dest in self._dest_edges:
                if origin == dest:
                    continue
                try:
                    stage = sumo.simulation.findRoute(origin, dest)
                    if not hasattr(stage, 'edges') or len(stage.edges) < 2:
                        continue
                    route = tuple(stage.edges)
                    if route in seen:
                        continue
                    seen.add(route)
                    n_side = len(side_edges.intersection(route))
                    if n_side > 0:
                        score = n_side / len(route)
                        candidates.append((score, list(route)))
                except Exception:
                    continue

        # Restore stderr
        if _suppress:
            _os.dup2(_stderr_fd, 2)
        _os.close(_stderr_fd)

        # Sort by side-edge fraction (descending), keep top routes
        candidates.sort(key=lambda x: x[0], reverse=True)
        self._conflict_routes = [r for _, r in candidates[:50]]

    def _precompute_bus_routes(self):
        """Build a small fixed set of bus-corridor routes.

        Bus corridors are the top 2-4 conflict routes ranked by
        side-edge fraction.  These are used round-robin so that
        buses always arrive on the SAME designated lanes — a
        recurring structured service obligation.

        Falls back to conflict routes if fewer than 2 candidates.
        """
        if not self._conflict_routes:
            self._bus_routes = []
            return
        # Take top 4 routes with most side-street edges (already sorted)
        self._bus_routes = self._conflict_routes[:4]
        self._bus_route_cursor = 0

    def _select_bus_route(self) -> Optional[List[str]]:
        """Pick next bus corridor route (round-robin).

        Deterministic cycling through the fixed bus-corridor pool.
        Every bus uses the same small set of lanes.
        """
        if not self._bus_routes:
            return self._select_conflict_route()
        route = list(self._bus_routes[self._bus_route_cursor % len(self._bus_routes)])
        self._bus_route_cursor += 1
        return route

    def _precompute_premium_routes(self):
        """Build a small fixed set of premium-corridor routes.

        Premium corridors are conflict routes that are DIFFERENT from
        bus corridors, creating a distinct recurring obligation.  Uses
        the next 2-4 conflict routes after the bus pool (or the full
        pool if bus routes are empty).  Round-robin injection ensures
        premium vehicles appear on the same designated lanes repeatedly,
        strengthening the local gradient-conflict signal.

        Falls back to conflict routes if fewer than 2 candidates.
        """
        if not self._conflict_routes:
            self._premium_routes = []
            return
        # Use routes that are NOT in the bus pool to avoid overlap
        bus_set = {tuple(r) for r in getattr(self, '_bus_routes', [])}
        candidates = [r for r in self._conflict_routes if tuple(r) not in bus_set]
        if len(candidates) < 2:
            candidates = self._conflict_routes
        self._premium_routes = candidates[:4]
        self._premium_route_cursor = 0

    def _select_premium_route(self) -> Optional[List[str]]:
        """Pick next premium corridor route (round-robin).

        Deterministic cycling through fixed premium-corridor pool.
        Every premium vehicle uses the same small set of lanes.
        """
        if not self._premium_routes:
            return self._select_conflict_route()
        route = list(self._premium_routes[self._premium_route_cursor % len(self._premium_routes)])
        self._premium_route_cursor += 1
        return route

    def _select_conflict_route(self) -> Optional[List[str]]:
        """Pick a route from the precomputed conflict-route pool.

        Static and conflict-inducing: routes traverse side-street
        approaches.  Selection is exogenous (episode RNG).
        Falls back to random route if no conflict routes available.
        """
        if not self._conflict_routes:
            return self._select_random_route(self._scenario_rng)
        idx = self._scenario_rng.randint(0, len(self._conflict_routes))
        return list(self._conflict_routes[idx])

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
        rng = self._scenario_rng

        # Bus injection — fixed corridor routes (round-robin)
        if self._cost_fn == "bus_priority" and sim_time >= self._next_bus_time:
            for _b in range(self._bus_count_per_injection):
                route_edges = self._select_bus_route()
                if route_edges is not None and len(route_edges) >= 2:
                    bus_id = f"bus_{self._bus_counter}"
                    route_id = f"bus_route_{self._bus_counter}"
                    try:
                        sumo.route.add(route_id, route_edges)
                        sumo.vehicle.add(bus_id, route_id, typeID="priority_bus")
                        self._active_bus_ids.add(bus_id)
                        self._bus_counter += 1
                    except Exception:
                        pass
            # Schedule next bus injection with some randomness
            self._next_bus_time = sim_time + self._bus_injection_interval * (0.8 + 0.4 * rng.random())

        # Convoy injection — static conflict route (side-street approach)
        if self._cost_fn == "convoy_priority" and sim_time >= self._next_convoy_time:
            route_edges = self._select_conflict_route()
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
                    self._convoy_info[convoy_id] = {
                        "total_size": len(veh_ids),
                        "vids": set(veh_ids),
                    }
                self._convoy_counter += 1
            self._next_convoy_time = sim_time + self._convoy_injection_interval * (0.8 + 0.4 * rng.random())

        # Premium injection — single VIP on fixed corridor (round-robin)
        if self._cost_fn == "premium_priority" and sim_time >= self._next_premium_time:
            route_edges = self._select_premium_route()
            if route_edges is not None and len(route_edges) >= 2:
                route_id = f"premium_route_{self._premium_counter}"
                try:
                    sumo.route.add(route_id, route_edges)
                    # Precompute cumulative free-flow prefix times per edge
                    # prefix_times[i] = free-flow time to reach START of edge i
                    prefix_times = [0.0]
                    for edge_id in route_edges:
                        try:
                            e_len = sumo.lane.getLength(edge_id + "_0")
                            e_spd = sumo.lane.getMaxSpeed(edge_id + "_0")
                            prefix_times.append(prefix_times[-1] + e_len / max(e_spd, 1.0))
                        except Exception:
                            prefix_times.append(prefix_times[-1])
                    # Build edge→index lookup for fast position mapping
                    edge_to_idx = {e: i for i, e in enumerate(route_edges)}
                    for k in range(self._premium_count_per_injection):
                        prem_id = f"premium_{self._premium_counter}_{k}"
                        depart_t = sim_time + k * self._premium_headway
                        depart = str(depart_t)
                        sumo.vehicle.add(prem_id, route_id, typeID="premium_vehicle", depart=depart)
                        self._active_premium_ids.add(prem_id)
                        # Record route-level info for delay tracking
                        self._premium_vehicle_info[prem_id] = {
                            "depart_time": depart_t,
                            "route_edges": list(route_edges),
                            "prefix_times": prefix_times,
                            "edge_to_idx": edge_to_idx,
                        }
                    self._premium_counter += 1
                except Exception:
                    pass
            self._next_premium_time = sim_time + self._premium_injection_interval * (0.8 + 0.4 * rng.random())

    def _detect_special_vehicles(self):
        """Detect bus/convoy/premium vehicles on each agent's incoming lanes.

        Optimised: instead of scanning ALL vehicles on ALL lanes, we query
        only the known injected special vehicle IDs (tracked in
        _active_bus_ids, _active_convoys, _active_premium_ids) and look
        up their lane via a single TraCI call per vehicle.  This reduces
        TraCI calls from O(agents × lanes × vehicles) to O(special_vehicles).
        """
        sumo = self._get_traci()
        if sumo is None:
            return

        need_bus = self._cost_fn == "bus_priority"
        need_convoy = self._cost_fn == "convoy_priority"
        need_premium = self._cost_fn == "premium_priority"

        # Reset per-agent per-lane arrays (reuse existing allocations)
        for agent_id in self.possible_agents:
            if need_bus:
                self._has_bus[agent_id].fill(0)
                self._bus_wait[agent_id].fill(0)
                self._bus_count[agent_id].fill(0)
            if need_convoy:
                self._has_convoy[agent_id].fill(0)
                self._convoy_count[agent_id].fill(0)
                self._convoy_seen_frac[agent_id].fill(0)
                self._convoy_progress[agent_id].fill(0)
            if need_premium:
                self._has_premium[agent_id].fill(0)
                self._premium_wait[agent_id].fill(0)

        # Per-agent convoy tracking for this step
        # convoy_on_incoming[agent_id][convoy_id] = set of vids on incoming
        convoy_on_incoming: Dict[str, Dict[int, Set[str]]] = {} if need_convoy else {}
        convoy_lane_map: Dict[str, Dict[int, int]] = {} if need_convoy else {}  # agent → {cid → lane_idx}

        # ── Query only known special vehicles ──
        # Get the set of vehicles currently in the simulation ONCE to avoid
        # querying departed vehicles (which causes noisy libsumo stderr).
        live_vehicles = set(sumo.vehicle.getIDList())
        departed = set()

        if need_bus:
            for vid in list(self._active_bus_ids):
                if vid not in live_vehicles:
                    if vid in self._seen_bus_ids:
                        departed.add(vid)  # was in sim, now gone
                    continue
                self._seen_bus_ids.add(vid)
                lane = sumo.vehicle.getLaneID(vid)
                vwait = sumo.vehicle.getWaitingTime(vid)
                mapping = self._lane_to_agent.get(lane)
                if mapping is None:
                    continue
                agent_id, lane_idx = mapping
                self._has_bus[agent_id][lane_idx] = 1.0
                self._bus_count[agent_id][lane_idx] += 1.0
                self._bus_wait[agent_id][lane_idx] = max(
                    self._bus_wait[agent_id][lane_idx], vwait)
            self._active_bus_ids -= departed

        if need_convoy:
            all_convoy_vids = set()
            for cid, vids in self._active_convoys.items():
                all_convoy_vids.update(vids)
            departed_convoy = set()
            for vid in all_convoy_vids:
                if vid not in live_vehicles:
                    if vid in self._seen_convoy_vids:
                        departed_convoy.add(vid)  # was in sim, now gone
                    continue
                self._seen_convoy_vids.add(vid)
                lane = sumo.vehicle.getLaneID(vid)
                mapping = self._lane_to_agent.get(lane)
                if mapping is None:
                    continue
                agent_id, lane_idx = mapping
                self._has_convoy[agent_id][lane_idx] = 1.0
                self._convoy_count[agent_id][lane_idx] += 1.0
                # Parse convoy_id from vid: "convoy_{id}_v{n}"
                try:
                    cid = int(vid.split('_')[1])
                except (IndexError, ValueError):
                    continue
                agent_convoys = convoy_on_incoming.setdefault(agent_id, {})
                agent_convoys.setdefault(cid, set()).add(vid)
                agent_lanes = convoy_lane_map.setdefault(agent_id, {})
                if cid not in agent_lanes:
                    agent_lanes[cid] = lane_idx
            # Prune departed convoy vehicles from _active_convoys
            if departed_convoy:
                for cid in list(self._active_convoys):
                    self._active_convoys[cid] = [
                        v for v in self._active_convoys[cid] if v not in departed_convoy]
                    if not self._active_convoys[cid]:
                        del self._active_convoys[cid]

        if need_premium:
            departed.clear()
            sim_time = sumo.simulation.getTime()
            for vid in list(self._active_premium_ids):
                if vid not in live_vehicles:
                    if vid in self._seen_premium_ids:
                        departed.add(vid)  # was in sim, now gone
                    continue
                self._seen_premium_ids.add(vid)
                lane = sumo.vehicle.getLaneID(vid)
                mapping = self._lane_to_agent.get(lane)
                if mapping is None:
                    continue
                agent_id, lane_idx = mapping
                self._has_premium[agent_id][lane_idx] = 1.0
                # Route-level delay: elapsed - prefix_free_flow to current edge
                # This gives real-time delay at the vehicle's current position,
                # not clipped-to-zero until the full route time elapses.
                vinfo = self._premium_vehicle_info.get(vid)
                if vinfo is not None:
                    elapsed = sim_time - vinfo["depart_time"]
                    # Find current edge from lane ID
                    current_edge = '_'.join(lane.split('_')[:-1])
                    edge_idx = vinfo["edge_to_idx"].get(current_edge)
                    if edge_idx is not None:
                        # prefix_times[edge_idx] = free-flow to reach START of this edge
                        # prefix_times[edge_idx+1] = free-flow to reach END of this edge
                        # Use prefix to start of current edge as expected elapsed
                        expected_elapsed = vinfo["prefix_times"][edge_idx]
                        # Add within-edge progress: lanePosition / edge_speed
                        try:
                            lane_pos = sumo.vehicle.getLanePosition(vid)
                            edge_spd = sumo.lane.getMaxSpeed(lane)
                            expected_elapsed += lane_pos / max(edge_spd, 1.0)
                        except Exception:
                            pass
                        route_delay = max(0.0, elapsed - expected_elapsed)
                    else:
                        # Vehicle on internal/junction edge — use last known
                        route_delay = max(0.0, elapsed - vinfo["prefix_times"][-1])
                else:
                    route_delay = 0.0
                self._premium_wait[agent_id][lane_idx] = max(
                    self._premium_wait[agent_id][lane_idx], route_delay)
            self._active_premium_ids -= departed
            # Clean up info for departed vehicles
            for vid in departed:
                self._premium_vehicle_info.pop(vid, None)

        # ── Normalize and store ──
        if need_bus:
            for agent_id in self.possible_agents:
                self._bus_wait[agent_id] = np.clip(
                    self._bus_wait[agent_id] / self._bus_cost_threshold, 0.0, 1.0)
                self._bus_count[agent_id] = np.clip(
                    self._bus_count[agent_id] / 5.0, 0.0, 1.0)

        if need_convoy:
            for agent_id in self.possible_agents:
                # Convoy progress PER AGENT using persistent tracking
                agent_history = self._convoy_seen_by_agent.setdefault(agent_id, {})
                agent_cmap = convoy_lane_map.get(agent_id, {})
                agent_con = convoy_on_incoming.get(agent_id, {})

                for cid, lane_idx in agent_cmap.items():
                    current_vids = agent_con.get(cid, set())
                    seen = agent_history.setdefault(cid, set())
                    seen.update(current_vids)
                    # Use actual total_size from convoy metadata for progress
                    cinfo = self._convoy_info.get(cid)
                    total = cinfo["total_size"] if cinfo else len(seen)
                    if total > 0:
                        still_here = len(current_vids & seen)
                        passed = len(seen) - still_here
                        progress = passed / total
                        self._convoy_progress[agent_id][lane_idx] = max(
                            self._convoy_progress[agent_id][lane_idx], progress)

                self._convoy_count[agent_id] = np.clip(
                    self._convoy_count[agent_id] / self._convoy_size_max, 0.0, 1.0)

                # Compute convoy_seen_frac: seen / total_size per lane
                # This is the obs-exposed feature the label reads from.
                seen_frac = np.zeros_like(self._convoy_count[agent_id])
                for cid, lane_idx in agent_cmap.items():
                    cinfo = self._convoy_info.get(cid)
                    total = cinfo["total_size"] if cinfo else self._convoy_size_max
                    seen_count = len(agent_history.get(cid, set()))
                    if total > 0:
                        seen_frac[lane_idx] = max(seen_frac[lane_idx], seen_count / total)
                self._convoy_seen_frac[agent_id] = np.clip(seen_frac, 0.0, 1.0)

            # Persist for label lookups
            self._convoy_lane_map = convoy_lane_map

        if need_premium:
            for agent_id in self.possible_agents:
                self._premium_wait[agent_id] = np.clip(
                    self._premium_wait[agent_id] / self._premium_cost_threshold, 0.0, 1.0)

    def _extend_obs_spaces(self):
        """No-op: obs spaces are pre-extended in __init__."""
        pass

    def _extend_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Append bus/convoy/premium features to observation vectors.

        Per-agent arrays are zero-padded to ``_ext_max_n_lanes`` so that
        all agents produce the same extension size, even on non-uniform networks.
        """
        if self._obs_extension_dim == 0:
            return obs

        M = self._ext_max_n_lanes  # uniform pad target

        def _pad(arr, target_len):
            """Zero-pad a per-agent array to target_len."""
            if len(arr) >= target_len:
                return arr[:target_len]
            return np.concatenate([arr, np.zeros(target_len - len(arr), dtype=np.float32)])

        extended = {}
        for agent, ob in obs.items():
            ob = np.asarray(ob, dtype=np.float32)

            if self._cost_fn == "bus_priority":
                hb = _pad(self._has_bus.get(agent, np.zeros(0, dtype=np.float32)), M)
                bc = _pad(self._bus_count.get(agent, np.zeros(0, dtype=np.float32)), M)
                bw = _pad(self._bus_wait.get(agent, np.zeros(0, dtype=np.float32)), M)
                ob = np.concatenate([ob, hb, bc, bw])
            elif self._cost_fn == "convoy_priority":
                hc = _pad(self._has_convoy.get(agent, np.zeros(0, dtype=np.float32)), M)
                sf = _pad(self._convoy_seen_frac.get(agent, np.zeros(0, dtype=np.float32)), M)
                cp = _pad(self._convoy_progress.get(agent, np.zeros(0, dtype=np.float32)), M)
                ob = np.concatenate([ob, hc, sf, cp])
            elif self._cost_fn == "premium_priority":
                hp = _pad(self._has_premium.get(agent, np.zeros(0, dtype=np.float32)), M)
                pw = _pad(self._premium_wait.get(agent, np.zeros(0, dtype=np.float32)), M)
                ob = np.concatenate([ob, hp, pw])

            extended[agent] = ob
        return extended

    # ── Bus priority cost/label ───────────────────────────────────────────

    def _compute_bus_priority_cost(self, agent_id: str) -> float:
        """Binary violation cost: 1 when a bus is still overdue.

        cost = 1  if any bus has bus_wait >= 1.0 (normalised by T_cost)
             = 0  otherwise

        Phase-independent: cost stays active as long as the entity is
        overdue, even if its lane is currently green.  The vehicle may
        still be stuck in queue — switching to green is necessary but
        not sufficient.
        """
        bus_wait = self._bus_wait.get(agent_id)
        has_bus = self._has_bus.get(agent_id)
        if bus_wait is None or has_bus is None:
            return 0.0
        return 1.0 if np.any((has_bus > 0) & (bus_wait >= 1.0 - 1e-6)) else 0.0

    def _bus_priority_label(self, agent_id: str) -> int:
        """Narrow intervention label for bus priority.

        Cost owns only during the pre-violation window:
          warn_norm <= bus_wait < 1.0

        Below warn_norm: too early, reward head should own.
        At/above 1.0: violation already fired, post-mortem — reward head.

        Gates: can_switch, not served.
        """
        has_bus = self._has_bus.get(agent_id)
        bus_wait = self._bus_wait.get(agent_id)
        if has_bus is None or bus_wait is None:
            return 0
        if not np.any(has_bus > 0):
            return 0

        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0
        if ts.time_since_last_phase_change < ts.min_green + ts.yellow_time:
            return 0

        phase = ts.green_phase
        served_lanes = set(self._phase_to_lanes.get(agent_id, {}).get(phase, []))
        warn_norm = self._bus_warn_threshold / max(self._bus_cost_threshold, 1e-6)

        for i in range(len(has_bus)):
            if has_bus[i] > 0 and i not in served_lanes and warn_norm <= bus_wait[i] < 1.0:
                return 1
        return 0

    # ── Convoy priority cost/label ───────────────────────────────────────

    def _compute_convoy_priority_cost(self, agent_id: str) -> float:
        """Binary violation cost: 1 when a convoy is actively being split.

        cost = 1  if any convoy on an unserved lane has 0 < progress < 1
                  (some vehicles passed, rest still stuck)
             = 0  otherwise

        Binary (0/1). No gradient direction for CUP/CPO.
        Split-RL gets anticipatory routing via _convoy_priority_label.
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

        for i in range(len(has_convoy)):
            if has_convoy[i] > 0 and i not in served_lanes:
                if 0 < convoy_progress[i] < 1.0:
                    return 1.0
        return 0.0

    def _convoy_priority_label(self, agent_id: str) -> int:
        """Narrow intervention label for convoy priority.

        Cost owns when the agent can act (can_switch gate) and:

          A. Pre-split (unserved only): convoy arriving on a red lane,
             progress == 0, seen_frac >= 0.33.  Decision frontier —
             switching now prevents the split.

          B. Early split (served or unserved): 0 < progress < 0.5.
             Convoy is being split but less than half lost.  If the
             convoy lane is already green, cost still owns because the
             agent must hold green (not switch away).  If unserved,
             cost owns because the agent should switch to serve it.

        Both branches require can_switch.  Branch A also requires
        unserved; Branch B does not.
        """
        has_convoy = self._has_convoy.get(agent_id)
        convoy_progress = self._convoy_progress.get(agent_id)
        if has_convoy is None or convoy_progress is None:
            return 0

        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0
        if ts.time_since_last_phase_change < ts.min_green + ts.yellow_time:
            return 0

        phase = ts.green_phase
        served_lanes = set(self._phase_to_lanes.get(agent_id, {}).get(phase, []))
        seen_frac = self._convoy_seen_frac.get(agent_id)

        for i in range(len(has_convoy)):
            if has_convoy[i] > 0:
                prog = convoy_progress[i]
                sf = float(seen_frac[i]) if seen_frac is not None and i < len(seen_frac) else 0.0
                # Branch A: pre-split, unserved only
                if i not in served_lanes and prog == 0.0 and sf >= 0.33:
                    return 1
                # Branch B: early split, served or unserved (hold-green ownership)
                if 0.0 < prog < 0.5:
                    return 1
        return 0

    # ── Premium priority cost/label ──────────────────────────────────────

    def _compute_premium_priority_cost(self, agent_id: str) -> float:
        """Binary violation cost: 1 when a premium vehicle's route delay exceeds budget.

        cost = 1  if any premium has route_delay / T_cost >= 1.0
             = 0  otherwise

        Route-level: tracks cumulative delay across the vehicle's entire
        journey, not just local waiting at this intersection.  A premium
        vehicle that was delayed upstream carries that delay forward.
        """
        has_premium = self._has_premium.get(agent_id)
        premium_wait = self._premium_wait.get(agent_id)
        if has_premium is None or premium_wait is None:
            return 0.0
        return 1.0 if np.any((has_premium > 0) & (premium_wait >= 1.0 - 1e-6)) else 0.0

    def _premium_priority_label(self, agent_id: str) -> int:
        """Narrow intervention label for premium priority.

        Cost owns when any premium vehicle is on a red lane with
        0 < wait < 1.0 (pre-violation window).

        No urgency threshold — premium T_cost=8s with 7s control
        latency leaves no room for a warning gate.  The moment a
        premium appears on a red lane IS the decision frontier.

        Upper bound < 1.0: violation already fired, post-mortem.

        Gates: can_switch, not served.
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
        if ts.time_since_last_phase_change < ts.min_green + ts.yellow_time:
            return 0

        phase = ts.green_phase
        served_lanes = set(self._phase_to_lanes.get(agent_id, {}).get(phase, []))

        for i in range(len(has_premium)):
            if has_premium[i] > 0 and i not in served_lanes and 0.0 < premium_wait[i] < 1.0:
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
        ml_idx = self._mainline_lane_indices.get(agent_id, [])
        if not ml_idx:
            return 0.0
        wait_per_lane = self._ts_query(agent_id, 'get_accumulated_waiting_time_per_lane')
        if not wait_per_lane:
            return 0.0
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
        side_idx = self._side_lane_indices.get(agent_id, [])
        if not side_idx:
            return 0.0

        counters = self._steps_since_served.get(agent_id, {})
        lane_queues = self._ts_query(agent_id, 'get_lanes_queue') or []

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
        side_idx = self._side_lane_indices.get(agent_id, [])
        if not side_idx:
            return 0

        counters = self._steps_since_served.get(agent_id, {})
        lane_queues = self._ts_query(agent_id, 'get_lanes_queue') or []

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
        side_idx = self._side_lane_indices.get(agent_id, [])
        if not side_idx:
            return 0.0

        lane_queues = self._ts_query(agent_id, 'get_lanes_queue') or []
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
        side_idx = self._side_lane_indices.get(agent_id, [])
        if not side_idx:
            return 0

        lane_queues = self._ts_query(agent_id, 'get_lanes_queue') or []
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
        side_idx = self._side_lane_indices.get(agent_id, [])
        if not side_idx:
            return 0.0

        counters = self._steps_since_served.get(agent_id, {})
        lane_queues = self._ts_query(agent_id, 'get_lanes_queue') or []

        for li in side_idx:
            steps_unserved = counters.get(li, 0)
            has_demand = lane_queues[li] > 0 if li < len(lane_queues) else False
            if steps_unserved > self._tau_service and has_demand:
                return 1.0
        return 0.0

    # ── Per-agent cost functions ──────────────────────────────────────────

    def _compute_agent_cost_queue_overflow(self, agent_id: str) -> float:
        """1 if total queued vehicles at this intersection > queue_threshold."""
        queued = self._ts_query(agent_id, 'get_total_queued')
        if queued is None:
            return 0.0
        return 1.0 if queued > self._queue_threshold else 0.0

    def _compute_agent_cost_max_wait(self, agent_id: str) -> float:
        """1 if max accumulated waiting time on any lane > wait_threshold."""
        wait_per_lane = self._ts_query(agent_id, 'get_accumulated_waiting_time_per_lane')
        if not wait_per_lane:
            return 0.0
        return 1.0 if max(wait_per_lane) > self._wait_threshold else 0.0

    def _compute_agent_cost_lane_saturation(self, agent_id: str) -> float:
        """1 if any lane's queue ratio > saturation_threshold."""
        lane_queues = self._ts_query(agent_id, 'get_lanes_queue')
        if not lane_queues:
            return 0.0
        return 1.0 if max(lane_queues) > self._saturation_threshold else 0.0

    def _compute_agent_cost_fairness(self, agent_id: str) -> float:
        """1 if one lane is being starved relative to others.

        Fires when max_wait > tau_abs AND max_wait / (mean_wait + eps) > rho.
        The absolute floor prevents spurious triggers when all waits are tiny.
        """
        wait_per_lane = self._ts_query(agent_id, 'get_accumulated_waiting_time_per_lane')
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
        return float(-self._ts_query(agent_id, 'get_pressure'))

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
        ml_idx = self._mainline_lane_indices.get(agent_id, [])
        if not ml_idx:
            return 0.0
        wait_per_lane = self._ts_query(agent_id, 'get_accumulated_waiting_time_per_lane')
        if not wait_per_lane:
            return 0.0
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
        side_idx = self._side_lane_indices.get(agent_id, [])
        if not side_idx:
            return 0.0
        wait_per_lane = self._ts_query(agent_id, 'get_accumulated_waiting_time_per_lane')
        if not wait_per_lane:
            return 0.0
        side_wait = sum(wait_per_lane[i] for i in side_idx) / 100.0
        prev = self._prev_side_wait.get(agent_id, 0.0)
        self._prev_side_wait[agent_id] = side_wait
        return max(0.0, side_wait - prev)

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
        """Get per-agent traffic metrics for logging (independent of cost_fn).

        Uses _ts_query() so that queries already made by cost/label
        functions in the same step are served from cache.
        """
        queued = self._ts_query(agent_id, 'get_total_queued')
        if queued is None:
            return {"queued": 0, "max_wait": 0.0, "max_lane_queue": 0.0,
                    "avg_speed": 1.0, "pressure": 0}
        wait_per_lane = self._ts_query(agent_id, 'get_accumulated_waiting_time_per_lane')
        max_wait = max(wait_per_lane) if wait_per_lane else 0.0
        lane_queues = self._ts_query(agent_id, 'get_lanes_queue')
        max_lane_queue = max(lane_queues) if lane_queues else 0.0
        avg_speed = self._ts_query(agent_id, 'get_average_speed')
        pressure = self._ts_query(agent_id, 'get_pressure')
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
        side_idx = self._side_lane_indices.get(agent_id, [])
        if side_idx:
            lane_queues = self._ts_query(agent_id, 'get_lanes_queue')
            if lane_queues:
                avg_side_q = sum(lane_queues[i] for i in side_idx) / len(side_idx)
                if avg_side_q > self._side_queue_cap:
                    return 1

        return 0

    def reset(self, seed=None, options=None):
        self._traci_dead = False  # clear crash flag for new episode

        # sumo_rl's reset() calls close() on the old TraCI connection.
        # If TraCI is already dead (e.g. from a crash in the previous
        # episode), close() will throw. Catch that and force a fresh start.
        # We retry up to 3 times, fully cleaning TraCI + libsumo each time.
        last_reset_err = None
        for _reset_attempt in range(3):
            try:
                obs, infos = self._env.reset(seed=seed, options=options)
                last_reset_err = None
                break
            except Exception as e:
                last_reset_err = e
                import warnings
                warnings.warn(
                    f"SumoEnvWrapper.reset attempt {_reset_attempt+1}/3 failed: {e}"
                )
                # Kill any lingering SUMO process and TraCI state
                try:
                    self._env.close()
                except Exception:
                    pass
                # Force-clear TraCI's global connection registry
                try:
                    import traci
                    if hasattr(traci, '_connections'):
                        traci._connections.pop("default", None)
                except Exception:
                    pass
                # Force-close libsumo backend (clears corrupted process state)
                try:
                    import libsumo
                    libsumo.close()
                except Exception:
                    pass
                import time
                time.sleep(1)
        if last_reset_err is not None:
            raise last_reset_err

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

        # ── Episode-level RNG for all scenario randomness ──
        # Seeded from the episode seed (reproducible) or episode counter
        # (deterministic but varying). Used for injection timing, route
        # selection, and clean-episode coin flip.
        self._episode_count += 1
        ep_seed = seed if seed is not None else self._episode_count
        self._scenario_rng = np.random.RandomState(ep_seed % (2**31))

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
            and self._scenario_rng.random() < self._clean_episode_prob
        )

        # Bus / convoy / premium injection setup
        if self._cost_fn in ("bus_priority", "convoy_priority", "premium_priority"):
            # Build phase→lane mapping (needed for label: "is lane served?")
            # Rebuild every episode: reset creates a fresh SUMO instance
            self._classify_mainline_side()  # also builds phase_to_lanes
            self._discover_boundary_edges()
            self._precompute_conflict_routes()
            if self._cost_fn == "bus_priority":
                self._precompute_bus_routes()
            if self._cost_fn == "premium_priority":
                self._precompute_premium_routes()
            self._bus_vtype_created = False
            self._convoy_vtype_created = False
            self._premium_vtype_created = False
            self._bus_counter = 0
            self._convoy_counter = 0
            self._premium_counter = 0
            self._active_convoys = {}  # reset convoy tracking
            self._convoy_info = {}  # reset convoy metadata
            self._convoy_lane_map = {}  # reset lane→convoy mapping
            self._active_bus_ids = set()
            self._active_premium_ids = set()
            # Track which vehicles have been seen in sim at least once.
            # Vehicles not yet spawned (future depart time) won't be in
            # getIDList() but must NOT be pruned from active tracking.
            self._seen_bus_ids = set()
            self._seen_convoy_vids = set()
            self._seen_premium_ids = set()
            self._premium_vehicle_info = {}  # reset route-level tracking
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
                    self._convoy_seen_frac[agent_id] = np.zeros(n_lanes, dtype=np.float32)
                    self._convoy_progress[agent_id] = np.zeros(n_lanes, dtype=np.float32)
                    self._has_premium[agent_id] = np.zeros(n_lanes, dtype=np.float32)
                    self._premium_wait[agent_id] = np.zeros(n_lanes, dtype=np.float32)
            # Build reverse lookup: lane_name → (agent_id, lane_index)
            self._lane_to_agent = {}
            for agent_id in self.possible_agents:
                ts = self._get_traffic_signal(agent_id)
                if ts is not None:
                    for i, lane in enumerate(ts.lanes):
                        self._lane_to_agent[lane] = (agent_id, i)

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

        # Clip actions for agents with fewer green phases than max_n_actions.
        # Out-of-range actions map to action 0 (safe default); no modulo
        # aliasing, so policy outputs have unambiguous semantics.
        if self._heterogeneous_actions:
            actions = {
                agent: int(act) if int(act) < self._agent_n_actions[agent] else 0
                for agent, act in actions.items()
            }

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

        Uses _ts_query() for all TrafficSignal queries to avoid redundant
        TraCI calls within a single step (cost + label + metrics often
        query the same data).

        label = g(s_{t+1}): describes the state the agent is NOW in.
        cost  = c(s_{t+1}): cost of the resulting state.
        """
        # Clear per-step TraCI query cache
        self._step_cache.clear()

        # Inject new scenario events + detect from resulting state s_{t+1}.
        # Both injection and detection happen post-step so label/cost
        # reflect the current state the agent just transitioned into.
        if self._cost_fn in ("bus_priority", "convoy_priority", "premium_priority"):
            sumo = self._get_traci()
            if sumo is not None:
                sim_time = sumo.simulation.getTime()
                self._inject_vehicles(sim_time)
                self._detect_special_vehicles()

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
            elif self._cost_fn == "premium_priority":
                label = self._premium_priority_label(agent)
            else:
                label = self._event_proximal_label(agent)

            new_rewards[agent] = reward
            info["cost"] = cost
            info["original_reward"] = original_reward
            info["raw_reward"] = reward
            info["safety_label"] = label
            # Per-agent traffic metrics
            info["queued"] = metrics["queued"]
            info["max_wait"] = metrics["max_wait"]
            info["max_lane_queue"] = metrics["max_lane_queue"]
            info["avg_speed"] = metrics["avg_speed"]
            info["pressure"] = metrics["pressure"]

            # ── Debug info (only when _debug=True) ──
            if self._debug:
                ts = self._get_traffic_signal(agent)
                if ts is not None:
                    can_switch = int(ts.time_since_last_phase_change >= ts.min_green + ts.yellow_time)
                    info["dbg_phase"] = ts.green_phase
                    info["dbg_can_switch"] = can_switch
                    info["dbg_time_in_phase"] = ts.time_since_last_phase_change
                    info["dbg_sim_time"] = self._get_traci().simulation.getTime() if self._get_traci() else 0.0
                else:
                    info["dbg_phase"] = -1
                    info["dbg_can_switch"] = 0
                    info["dbg_time_in_phase"] = 0.0
                    info["dbg_sim_time"] = 0.0
                info["dbg_step"] = self._debug_step_counter
                # Scenario-specific features
                if self._cost_fn == "convoy_priority":
                    hc = self._has_convoy.get(agent, np.zeros(0))
                    cp = self._convoy_progress.get(agent, np.zeros(0))
                    cc = self._convoy_count.get(agent, np.zeros(0))
                    info["dbg_has_convoy"] = int(np.any(hc > 0))
                    info["dbg_convoy_progress_max"] = float(cp.max()) if len(cp) > 0 else 0.0
                    info["dbg_convoy_count_max"] = float(cc.max()) if len(cc) > 0 else 0.0
                    info["dbg_n_active_convoys"] = len(self._active_convoys)
                elif self._cost_fn == "bus_priority":
                    hb = self._has_bus.get(agent, np.zeros(0))
                    bw = self._bus_wait.get(agent, np.zeros(0))
                    info["dbg_has_bus"] = int(np.any(hb > 0))
                    info["dbg_bus_wait_max"] = float(bw.max()) if len(bw) > 0 else 0.0
                elif self._cost_fn == "premium_priority":
                    hp = self._has_premium.get(agent, np.zeros(0))
                    pw = self._premium_wait.get(agent, np.zeros(0))
                    info["dbg_has_premium"] = int(np.any(hp > 0))
                    info["dbg_premium_wait_max"] = float(pw.max()) if len(pw) > 0 else 0.0

            self._debug_step_counter += 1

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


def _sumo_rl_path(*parts) -> str:
    """Resolve a path relative to sumo_rl/nets/."""
    nets = _find_sumo_rl_nets()
    if nets is None:
        raise ImportError("sumo_rl is not installed.")
    import os
    path = os.path.join(nets, *parts)
    if not os.path.exists(path):
        raise FileNotFoundError(f"SUMO-RL file not found: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Per-env × per-scenario injection defaults
# ──────────────────────────────────────────────────────────────────────────────
# Scenario parameters tuned per environment so that each intersection sees
# ~10-30% label rate under a random policy.  Single-intersection envs need
# slower injection (every entity passes through); multi-agent networks need
# faster injection (entities only touch a fraction of intersections).
#
# Keys: (env_name, cost_fn) → dict of scenario hyperparams.
# Any key not listed uses the global defaults in make_sumo_vec_env / __init__.
# Caller can still override via env_kwargs — these are just defaults.

SCENARIO_DEFAULTS: Dict[Tuple[str, str], dict] = {
    # ── arterial4x4 (16 agents, asymmetric, EW mainline) ────────────
    ("sumo-arterial4x4-v0", "bus_priority"): {
        "bus_injection_interval": 20.0,
        "bus_cost_threshold": 15.0,
        "bus_warn_threshold": 6.0,
    },
    ("sumo-arterial4x4-v0", "convoy_priority"): {
        "convoy_injection_interval": 80.0,
        "convoy_size_min": 10,
        "convoy_size_max": 12,
        "convoy_headway": 2.0,
    },
    ("sumo-arterial4x4-v0", "premium_priority"): {
        "premium_injection_interval": 10.0,
        "premium_cost_threshold": 8.0,
        "premium_warn_threshold": 3.0,
    },
    # ── grid4x4 (16 agents, symmetric, balanced traffic) ────────────
    ("sumo-grid4x4-v0", "bus_priority"): {
        "bus_injection_interval": 15.0,
        "bus_cost_threshold": 15.0,
        "bus_warn_threshold": 6.0,
    },
    ("sumo-grid4x4-v0", "convoy_priority"): {
        "convoy_injection_interval": 80.0,
        "convoy_size_min": 10,
        "convoy_size_max": 12,
        "convoy_headway": 2.0,
    },
    ("sumo-grid4x4-v0", "premium_priority"): {
        "premium_injection_interval": 8.0,
        "premium_cost_threshold": 8.0,
        "premium_warn_threshold": 3.0,
    },
    # ── single-vhvh (1 agent, all entities pass through) ────────────
    ("sumo-single-vhvh-v0", "bus_priority"): {
        "bus_injection_interval": 90.0,
        "bus_cost_threshold": 15.0,
        "bus_warn_threshold": 6.0,
    },
    ("sumo-single-vhvh-v0", "convoy_priority"): {
        "convoy_injection_interval": 300.0,
        "convoy_size_min": 10,
        "convoy_size_max": 12,
        "convoy_headway": 2.0,
    },
    ("sumo-single-vhvh-v0", "premium_priority"): {
        "premium_injection_interval": 60.0,
        "premium_cost_threshold": 8.0,
        "premium_warn_threshold": 3.0,
    },
}


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
        "reward_fn": "pressure",
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
        "reward_fn": "pressure",
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
        "reward_fn": "pressure",
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
        "reward_fn": "pressure",
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
        "reward_fn": "pressure",
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
        "reward_fn": "pressure",
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
        "reward_fn": "pressure",
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
        "reward_fn": "pressure",
        "n_agents": 21,
    },
    # --- Single intersection (1 agent, asymmetric traffic) ---
    # Reward = pressure (#out - #in).  External costs (convoy, bus, premium)
    # inject special vehicles on side-street (low-demand) approaches
    # via precomputed conflict routes.  Serving them to reduce cost pulls green
    # away from the mainline and worsens pressure.
    #
    # 4-way intersection: NS ~1000 veh/hr (major), EW ~250 veh/hr (minor).
    # Use mainline_direction="ns" to correctly identify NS as mainline.
    "sumo-single-vhvh-v0": {
        "net_file_fn": lambda: _sumo_rl_path("2way-single-intersection", "single-intersection.net.xml"),
        "route_file_fn": lambda: _sumo_rl_path("2way-single-intersection", "single-intersection-vhvh.rou.xml"),
        "num_seconds": 3600,
        "delta_time": 5,
        "yellow_time": 2,
        "min_green": 5,
        "max_green": 50,
        "reward_fn": "pressure",
        "n_agents": 1,
    },
    # 2-way intersection: WE prob=0.5 (major), NS prob=0.2 (minor).
    "sumo-single-v0": {
        "net_file_fn": lambda: _sumo_rl_path("single-intersection", "single-intersection.net.xml"),
        "route_file_fn": lambda: _sumo_rl_path("single-intersection", "single-intersection.rou.xml"),
        "num_seconds": 3600,
        "delta_time": 5,
        "yellow_time": 2,
        "min_green": 5,
        "max_green": 50,
        "reward_fn": "pressure",
        "n_agents": 1,
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
    add_system_info: bool = False,
    add_per_agent_info: bool = False,
):
    """
    Create a SUMO-RL PettingZoo parallel env.

    Returns the raw PettingZoo parallel env (unmodified SUMO-RL).
    """
    import sumo_rl

    # Retry env construction — libsumo can fail transiently with
    # "A network was not yet constructed" if a previous trial left
    # stale state in the process.
    last_err = None
    for _attempt in range(3):
        try:
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
        except Exception as e:
            last_err = e
            import warnings, time
            warnings.warn(f"make_sumo_raw_env attempt {_attempt+1}/3 failed: {e}")
            # Force-clear any stale TraCI / libsumo state
            try:
                import traci
                traci.close()
            except Exception:
                pass
            try:
                import traci
                if hasattr(traci, '_connections'):
                    traci._connections.pop("default", None)
            except Exception:
                pass
            try:
                import libsumo
                libsumo.close()
            except Exception:
                pass
            time.sleep(1)
    raise last_err  # all retries exhausted


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
    clean_episode_prob: float = 0.0,
    use_gui: bool = False,
    debug: bool = False,
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
        Cost function. See COST_FN_REGISTRY. Default "convoy_priority".
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

    # ── Apply per-env scenario defaults ──
    # SCENARIO_DEFAULTS[(env_name, cost_fn)] provides tuned injection params.
    # Global fallbacks for envs not in the table.
    # Caller can override any param via override_kwargs / env_kwargs.
    _GLOBAL_SCENARIO_FALLBACKS = {
        "bus_injection_interval": 30.0,
        "bus_count_per_injection": 1,
        "bus_cost_threshold": 15.0,
        "bus_warn_threshold": 6.0,
        "convoy_injection_interval": 80.0,
        "convoy_size_min": 10,
        "convoy_size_max": 12,
        "convoy_headway": 2.0,
        "premium_injection_interval": 15.0,
        "premium_cost_threshold": 8.0,
        "premium_warn_threshold": 3.0,
        "premium_count_per_injection": 1,
        "premium_headway": 2.0,
    }
    # Layer: global fallbacks < SCENARIO_DEFAULTS < caller overrides
    scenario_params = dict(_GLOBAL_SCENARIO_FALLBACKS)
    scenario_params.update(SCENARIO_DEFAULTS.get((env_name, cost_fn), {}))
    # Caller overrides from env_kwargs (passed through override_kwargs)
    _scenario_keys = set(_GLOBAL_SCENARIO_FALLBACKS.keys()) | {"clean_episode_prob"}
    for k in list(config.keys()):
        if k in _scenario_keys:
            scenario_params[k] = config.pop(k)
    scenario_params.setdefault("clean_episode_prob", clean_episode_prob)

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
        bus_injection_interval=scenario_params["bus_injection_interval"],
        bus_count_per_injection=scenario_params["bus_count_per_injection"],
        bus_cost_threshold=scenario_params["bus_cost_threshold"],
        bus_warn_threshold=scenario_params["bus_warn_threshold"],
        convoy_injection_interval=scenario_params["convoy_injection_interval"],
        convoy_size_min=scenario_params["convoy_size_min"],
        convoy_size_max=scenario_params["convoy_size_max"],
        convoy_headway=scenario_params["convoy_headway"],
        premium_injection_interval=scenario_params["premium_injection_interval"],
        premium_cost_threshold=scenario_params["premium_cost_threshold"],
        premium_warn_threshold=scenario_params["premium_warn_threshold"],
        premium_count_per_injection=scenario_params["premium_count_per_injection"],
        premium_headway=scenario_params["premium_headway"],
        clean_episode_prob=scenario_params["clean_episode_prob"],
    )

    if debug:
        par_env._debug = True

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
        if par_env._heterogeneous_actions:
            raise ValueError(
                "use_categorical_phase=True is incompatible with heterogeneous "
                "action spaces. Phase one-hot layout differs per agent, so "
                "argmax-based categorical conversion would corrupt observations. "
                "Either use a uniform-phase network or disable categorical phase."
            )
        n_phases = vec_env.action_space.n  # Discrete action = n_green_phases
        vec_env = VecCategoricalPhase(vec_env, n_phases=n_phases)

    # Add VecCostMonitor for episode tracking
    vec_env = VecCostMonitor(vec_env)

    return vec_env


class VecCategoricalPhase(VecMonitor.__bases__[0] if hasattr(VecMonitor, '__bases__') else object):
    """VecEnv wrapper that replaces one-hot phase with a categorical phase ID.

    Converts float32 obs of shape (n_envs, n_phases + 1 + 2*n_lanes) to
    object obs of shape (n_envs, 1 + 1 + 2*n_lanes) where column 0 is
    a categorical string "p0".."p{n-1}" and the rest are float64.

    GBRL auto-detects the string column as categorical (equality splits)
    and the float columns as numerical (threshold splits) via its
    per-element dtype inspection in process_array().

    observation_space shape is correct (new_dim); dtype is approximate
    because gym.spaces.Box cannot represent mixed str/float columns.
    This is standard for mixed-type RL wrappers. The is_categorical flag
    ensures downstream algos use CategoricalReplayBuffer (dtype=object),
    bypassing Box dtype checks.

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
        self.action_space = venv.action_space

        # Build observation space with correct shape (n_phases one-hot → 1 col)
        # dtype=float64 is approximate; actual column 0 is a string.
        inner_space = venv.observation_space
        new_dim = inner_space.shape[0] - n_phases + 1
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_dim,), dtype=np.float64
        )
        if hasattr(venv, 'reward_range'):
            self.reward_range = venv.reward_range
        if hasattr(venv, 'metadata'):
            self.metadata = venv.metadata

    def _convert_obs(self, obs: np.ndarray) -> np.ndarray:
        """Convert float obs batch: one-hot phase → categorical string + floats.

        Returns dtype=object array so GBRL's process_array() sees:
          col 0: string "p3" → categorical → equality splits
          col 1+: float64    → numerical   → threshold splits
        """
        n_envs, obs_dim = obs.shape
        n_phases = self._n_phases
        new_dim = obs_dim - n_phases + 1
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
        # SuperSuit may return integer dones (0/1) instead of boolean.
        # VecNormalize uses boolean indexing (returns[dones] = 0) which
        # breaks with integer dones when num_envs=1 (index 1 out of bounds).
        dones = np.asarray(dones, dtype=bool)
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
