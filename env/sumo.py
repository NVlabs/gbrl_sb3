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
separate "environment slot" — identical to the Flatland wrapper approach.

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
from typing import Any, Callable, Dict, List, Optional, Union

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

        if self._needs_obs_padding:
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

        Categorical phase transformation is handled at the VecEnv level
        (VecCategoricalPhase wrapper) because SuperSuit cannot concatenate
        object arrays.
        """
        if not self._needs_obs_padding:
            return obs
        padded = {}
        for agent, ob in obs.items():
            ob = np.asarray(ob, dtype=np.float32)
            if ob.shape[0] < self._max_obs_dim:
                ob = np.pad(ob, (0, self._max_obs_dim - ob.shape[0]))
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
        if self._override_reward in ("ns_wait", "arterial", "mainline") or self._cost_fn in ("directional", "service_gap"):
            self._classify_mainline_side()

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

        return self._transform_obs(obs), new_infos

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
        return self._transform_obs(obs), rewards, terminations, truncations, infos

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
        """
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
            if self._cost_fn == "service_gap":
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

            # Label: directional (state-based) or fallback (event-proximal)
            if self._cost_fn == "directional":
                label = self._directional_label(agent)
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

        return self._transform_obs(obs), new_rewards, terminations, truncations, new_infos

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
    override_reward: Optional[str] = "mainline",
    cost_fn: str = "service_gap",
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
        Cost function. See COST_FN_REGISTRY. Default "service_gap".
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
