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
Rewards: per-agent scalar — throughput (default) or diff-waiting-time.
Cost:    fairness + phase-churn (default "conflict"), or legacy queue/wait/saturation.
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
    "conflict": "Per-agent: 1 if fairness OR phase-churn fires (genuinely conflicts with throughput)",
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

    The "conflict" cost + throughput reward creates a genuine trade-off:
    throughput wants maximum vehicle flow, but fairness penalizes lane starvation
    and phase-churn penalizes excessive signal switching.

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
        fairness_tau: float = 120.0,
        fairness_rho: float = 3.0,
        queue_threshold: int = 10,
        wait_threshold: float = 200.0,
        saturation_threshold: float = 0.8,
        teleport_weight: float = 1.0,
        label_horizon: int = 3,
    ):
        if cost_fn not in COST_FN_REGISTRY:
            raise ValueError(
                f"Unknown cost_fn '{cost_fn}'. Available: {list(COST_FN_REGISTRY.keys())}"
            )
        if override_reward is not None and override_reward != "throughput":
            raise ValueError(
                f"Unknown override_reward '{override_reward}'. Use 'throughput' or None."
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
        self._label_horizon = label_horizon

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
        # Per-agent metric accumulators for logging
        self._episode_queue_violations: Dict[str, int] = {}
        self._episode_wait_violations: Dict[str, int] = {}
        self._episode_max_queued: Dict[str, int] = {}
        self._episode_max_wait: Dict[str, float] = {}
        self._episode_cost_queue: Dict[str, float] = {}
        self._episode_cost_wait: Dict[str, float] = {}
        self._episode_cost_saturation: Dict[str, float] = {}
        self._episode_cost_fairness: Dict[str, float] = {}
        self._episode_cost_churn: Dict[str, float] = {}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _pad_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Zero-pad observations to max_obs_dim for non-uniform networks."""
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
        """Vehicle count on outgoing lanes — proxy for throughput.

        Counts vehicles that have cleared through the intersection.
        This is a pure productivity signal: higher = more cars moved.
        """
        ts = self._get_traffic_signal(agent_id)
        if ts is None:
            return 0.0
        return float(sum(
            ts.sumo.lane.getLastStepVehicleNumber(lane) for lane in ts.out_lanes
        ))

    def _compute_agent_cost(self, agent_id: str) -> float:
        """Dispatch to the selected cost function."""
        if self._cost_fn == "conflict":
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

    def reset(self, seed=None, options=None):
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

        # Reset episode accumulators
        self._episode_costs = {agent: 0.0 for agent in self.possible_agents}
        self._episode_original_rewards = {agent: 0.0 for agent in self.possible_agents}
        self._episode_cost_queue = {a: 0.0 for a in self.possible_agents}
        self._episode_cost_wait = {a: 0.0 for a in self.possible_agents}
        self._episode_cost_saturation = {a: 0.0 for a in self.possible_agents}
        self._episode_cost_fairness = {a: 0.0 for a in self.possible_agents}
        self._episode_cost_churn = {a: 0.0 for a in self.possible_agents}
        self._episode_queue_violations = {a: 0 for a in self.possible_agents}
        self._episode_wait_violations = {a: 0 for a in self.possible_agents}
        self._episode_max_queued = {a: 0 for a in self.possible_agents}
        self._episode_max_wait = {a: 0.0 for a in self.possible_agents}

        new_infos = {}
        for agent in self.agents:
            info = dict(infos.get(agent, {}))
            info["cost"] = 0.0
            info["original_reward"] = 0.0
            info["safety_label"] = 0
            new_infos[agent] = info

        return self._pad_obs(obs), new_infos

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self._env.step(actions)
        self.agents = list(self._env.agents) if hasattr(self._env, 'agents') else list(self.possible_agents)

        # For emergency cost_fn, compute once (global)
        emergency_cost = None
        if self._cost_fn == "emergency":
            emergency_cost = self._compute_emergency_cost()

        new_rewards = {}
        new_infos = {}
        for agent in rewards:
            original_reward = float(rewards[agent])
            info = dict(infos.get(agent, {}))

            # ── Reward: throughput override or original ──
            if self._override_reward == "throughput":
                reward = self._compute_agent_throughput(agent)
            else:
                reward = original_reward

            # ── Always compute ALL cost primitives for monitoring ──
            # Legacy costs
            c_queue = self._compute_agent_cost_queue_overflow(agent)
            c_wait = self._compute_agent_cost_max_wait(agent)
            c_sat = self._compute_agent_cost_lane_saturation(agent)
            # New conflict costs
            c_fair = self._compute_agent_cost_fairness(agent)
            c_churn = self._compute_agent_cost_phase_churn(agent)

            # ── Determine the training cost signal ──
            if self._cost_fn == "conflict":
                cost = 1.0 if (c_fair > 0 or c_churn > 0) else 0.0
            elif self._cost_fn == "fairness":
                cost = c_fair
            elif self._cost_fn == "phase_churn":
                cost = c_churn
            elif self._cost_fn == "combined":
                cost = c_queue + c_wait + c_sat
            elif self._cost_fn == "queue_overflow":
                cost = c_queue
            elif self._cost_fn == "max_wait":
                cost = c_wait
            elif self._cost_fn == "lane_saturation":
                cost = c_sat
            elif self._cost_fn == "emergency":
                cost = emergency_cost
            else:
                cost = 0.0

            # Per-agent metrics (always logged, regardless of cost_fn)
            metrics = self._get_agent_metrics(agent)

            # Update per-agent label history
            has_cost = cost > 0
            self._update_failure_history(agent, has_cost)
            fallback_label = self._event_proximal_label(agent)

            new_rewards[agent] = reward
            info["cost"] = cost
            info["original_reward"] = original_reward
            info["safety_label"] = fallback_label
            # All cost components (always logged for monitoring)
            info["cost_queue"] = c_queue
            info["cost_wait"] = c_wait
            info["cost_saturation"] = c_sat
            info["cost_fairness"] = c_fair
            info["cost_churn"] = c_churn
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
            # Per-component episode cost totals
            self._episode_cost_queue[agent] = (
                self._episode_cost_queue.get(agent, 0.0) + c_queue
            )
            self._episode_cost_wait[agent] = (
                self._episode_cost_wait.get(agent, 0.0) + c_wait
            )
            self._episode_cost_saturation[agent] = (
                self._episode_cost_saturation.get(agent, 0.0) + c_sat
            )
            self._episode_cost_fairness[agent] = (
                self._episode_cost_fairness.get(agent, 0.0) + c_fair
            )
            self._episode_cost_churn[agent] = (
                self._episode_cost_churn.get(agent, 0.0) + c_churn
            )
            if metrics["queued"] > self._queue_threshold:
                self._episode_queue_violations[agent] = (
                    self._episode_queue_violations.get(agent, 0) + 1
                )
            if metrics["max_wait"] > self._wait_threshold:
                self._episode_wait_violations[agent] = (
                    self._episode_wait_violations.get(agent, 0) + 1
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
                info["episode_cost_queue"] = self._episode_cost_queue.get(agent, 0.0)
                info["episode_cost_wait"] = self._episode_cost_wait.get(agent, 0.0)
                info["episode_cost_saturation"] = self._episode_cost_saturation.get(agent, 0.0)
                info["episode_cost_fairness"] = self._episode_cost_fairness.get(agent, 0.0)
                info["episode_cost_churn"] = self._episode_cost_churn.get(agent, 0.0)
                info["episode_queue_violations"] = self._episode_queue_violations.get(agent, 0)
                info["episode_wait_violations"] = self._episode_wait_violations.get(agent, 0)
                info["episode_max_queued"] = self._episode_max_queued.get(agent, 0)
                info["episode_max_wait"] = self._episode_max_wait.get(agent, 0.0)

            new_infos[agent] = info

        return self._pad_obs(obs), new_rewards, terminations, truncations, new_infos

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

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
    override_reward: Optional[str] = "throughput",
    cost_fn: str = "conflict",
    fairness_tau: float = 120.0,
    fairness_rho: float = 3.0,
    queue_threshold: int = 10,
    wait_threshold: float = 200.0,
    saturation_threshold: float = 0.8,
    teleport_weight: float = 1.0,
    label_horizon: int = 3,
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
        If "throughput", replace sumo-rl reward with outgoing vehicle count.
        None keeps the original sumo-rl reward (e.g., diff-waiting-time).
    cost_fn : str
        Cost function. See COST_FN_REGISTRY. Default "conflict" (fairness OR churn).
    fairness_tau : float
        Absolute floor for fairness cost (seconds). Default 120.
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
        fairness_tau=fairness_tau,
        fairness_rho=fairness_rho,
        queue_threshold=queue_threshold,
        wait_threshold=wait_threshold,
        saturation_threshold=saturation_threshold,
        teleport_weight=teleport_weight,
        label_horizon=label_horizon,
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

    # Add VecCostMonitor for episode tracking
    vec_env = VecCostMonitor(vec_env)

    return vec_env


class VecCostMonitor(VecMonitor):
    """VecMonitor subclass that also tracks per-episode cost and traffic metrics.

    Accumulates info["cost"] each step and stores total in episode_info["c"].
    Also tracks per-agent queue/wait violations and peak metrics.
    """

    def __init__(self, venv, filename=None, info_keywords=()):
        super().__init__(venv, filename=filename, info_keywords=info_keywords)
        self.episode_costs = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_cost_queue = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_cost_wait = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_cost_saturation = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_cost_fairness = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_cost_churn = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_queue_violations = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_wait_violations = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_max_queued = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_max_wait = np.zeros(self.num_envs, dtype=np.float64)

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.episode_costs[:] = 0.0
        self.episode_cost_queue[:] = 0.0
        self.episode_cost_wait[:] = 0.0
        self.episode_cost_saturation[:] = 0.0
        self.episode_cost_fairness[:] = 0.0
        self.episode_cost_churn[:] = 0.0
        self.episode_queue_violations[:] = 0
        self.episode_wait_violations[:] = 0
        self.episode_max_queued[:] = 0
        self.episode_max_wait[:] = 0.0
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.episode_returns += rewards
        self.episode_lengths += 1
        for i in range(self.num_envs):
            self.episode_costs[i] += infos[i].get("cost", 0.0)
            self.episode_cost_queue[i] += infos[i].get("cost_queue", 0.0)
            self.episode_cost_wait[i] += infos[i].get("cost_wait", 0.0)
            self.episode_cost_saturation[i] += infos[i].get("cost_saturation", 0.0)
            self.episode_cost_fairness[i] += infos[i].get("cost_fairness", 0.0)
            self.episode_cost_churn[i] += infos[i].get("cost_churn", 0.0)
            # Track per-step metrics
            queued = infos[i].get("queued", 0)
            max_wait = infos[i].get("max_wait", 0.0)
            if queued > self.episode_max_queued[i]:
                self.episode_max_queued[i] = queued
            if max_wait > self.episode_max_wait[i]:
                self.episode_max_wait[i] = max_wait
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
                    "original_r": ep_rew,
                    "cost_queue": float(self.episode_cost_queue[i]),
                    "cost_wait": float(self.episode_cost_wait[i]),
                    "cost_saturation": float(self.episode_cost_saturation[i]),
                    "cost_fairness": float(self.episode_cost_fairness[i]),
                    "cost_churn": float(self.episode_cost_churn[i]),
                    "max_queued": int(self.episode_max_queued[i]),
                    "max_wait": float(self.episode_max_wait[i]),
                    "t": round(time.time() - self.t_start, 6),
                }
                for key in self.info_keywords:
                    episode_info[key] = info[key]
                info["episode"] = episode_info
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                self.episode_costs[i] = 0.0
                self.episode_cost_queue[i] = 0.0
                self.episode_cost_wait[i] = 0.0
                self.episode_cost_saturation[i] = 0.0
                self.episode_cost_fairness[i] = 0.0
                self.episode_cost_churn[i] = 0.0
                self.episode_queue_violations[i] = 0
                self.episode_wait_violations[i] = 0
                self.episode_max_queued[i] = 0
                self.episode_max_wait[i] = 0.0
                if self.results_writer:
                    self.results_writer.write_row(episode_info)
                new_infos[i] = info
        return obs, rewards, dones, new_infos
