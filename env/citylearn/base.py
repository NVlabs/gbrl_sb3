##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
"""Base CityLearn environment setup and shared wrapper utilities.

Provides
--------
- make_citylearn_inner_env : CityLearnEnv → NormalizedObs → SB3Wrapper
- CityLearnBaseWrapper     : gym.Wrapper with explicit electricity-cost
                             reward, raw-state helpers, and abstract
                             _compute_cost / _compute_label for subclasses.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from utils.helpers import make_cost_vec_env

# ---------------------------------------------------------------------------
# Default schema
# ---------------------------------------------------------------------------
DEFAULT_SCHEMA = "citylearn_challenge_2023_phase_2_local_evaluation"


# -----------------------------------------------------------------------
# Inner env factory (shared across all scenarios)
# -----------------------------------------------------------------------

def make_citylearn_inner_env(
    schema: str = DEFAULT_SCHEMA,
    episode_time_steps: Optional[int] = None,
) -> gym.Env:
    """Create the inner CityLearn stack before any CMDP wrapper.

    Stack:  CityLearnEnv  →  NormalizedObservationWrapper
                           →  StableBaselines3Wrapper
    """
    from citylearn.citylearn import CityLearnEnv
    from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper

    cl_kwargs: Dict[str, Any] = {
        "schema": schema,
        "central_agent": True,
    }
    if episode_time_steps is not None:
        cl_kwargs["episode_time_steps"] = episode_time_steps

    env = CityLearnEnv(**cl_kwargs)
    env = NormalizedObservationWrapper(env)
    env = StableBaselines3Wrapper(env)
    return env


# -----------------------------------------------------------------------
# Base wrapper
# -----------------------------------------------------------------------

class CityLearnBaseWrapper(gym.Wrapper):
    """Base wrapper with explicit reward and raw-state helpers.

    Reward
    ------
    Replaces the native CityLearn reward (negative net consumption) with
    **negative price-weighted electricity cost**:

        reward_t  =  -(1/N) Σ_i  consumption_i(t) × price_i(t)

    Higher reward ⟹ lower spending.  The native reward is preserved in
    ``info["original_reward"]`` for comparison.

    Subclass contract
    -----------------
    Subclasses **must** implement:

    * ``_compute_cost(obs) -> float``   (∈ [0, 1])
    * ``_compute_label(obs) -> int``    (∈ {0, 1})

    and **may** override ``_scenario_init()`` for one-time setup that
    needs the live env (called once, lazily, on the first ``step``).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._base_initialized = False
        self._n_buildings = 0

    # ------------------------------------------------------------------
    # Lazy init (deferred until env is alive)
    # ------------------------------------------------------------------
    def _ensure_base_init(self):
        if self._base_initialized:
            return
        cl = self.unwrapped
        if hasattr(cl, "buildings"):
            self._n_buildings = len(cl.buildings)
        self._scenario_init()
        self._base_initialized = True

    def _scenario_init(self):
        """Override in subclasses for one-time setup (e.g. price threshold)."""

    # ------------------------------------------------------------------
    # Raw building-state helpers
    # ------------------------------------------------------------------
    def _get_buildings(self):
        """Buildings list from the unwrapped CityLearnEnv."""
        return self.unwrapped.buildings

    def _get_time_step(self) -> int:
        """Current time_step (already incremented after step)."""
        return self.unwrapped.time_step

    @staticmethod
    def _latest(series):
        """Last element of a live-appended array (temp, setpoint, consumption)."""
        if series is None:
            return None
        if hasattr(series, "__len__"):
            return float(series[-1]) if len(series) > 0 else None
        return float(series)

    def _at_timestep(self, series):
        """Element at ``time_step - 1`` of a pre-allocated array (pricing, SOC)."""
        ts = self._get_time_step() - 1
        if hasattr(series, "__len__") and 0 <= ts < len(series):
            return float(series[ts])
        return None

    def _max_current_price(self, buildings) -> Optional[float]:
        """Max electricity price at the current timestep across buildings."""
        max_p = None
        for b in buildings:
            if hasattr(b, "pricing") and b.pricing is not None:
                p = self._at_timestep(b.pricing.electricity_pricing)
                if p is not None and (max_p is None or p > max_p):
                    max_p = p
        return max_p

    # ------------------------------------------------------------------
    # Explicit electricity-cost reward
    # ------------------------------------------------------------------
    def _compute_electricity_cost_reward(self) -> float:
        """Negative price-weighted electricity consumption per building.

        ``reward = -(1/N) Σ_i consumption_i × price_i``

        Positive consumption = grid import (cost).
        Negative consumption = grid export (revenue).

        Note: ``net_electricity_consumption`` has a trailing zero placeholder,
        so the current step's value is at index ``time_step - 1`` (same
        as pricing and SOC arrays), not at ``[-1]``.
        """
        self._ensure_base_init()
        total_cost = 0.0
        n_priced = 0
        for b in self._get_buildings():
            consumption = self._at_timestep(b.net_electricity_consumption)
            price = None
            if hasattr(b, "pricing") and b.pricing is not None:
                price = self._at_timestep(b.pricing.electricity_pricing)
            if consumption is not None and price is not None:
                total_cost += consumption * price
                n_priced += 1
        if n_priced > 0:
            return -total_cost / n_priced
        return 0.0

    # ------------------------------------------------------------------
    # Abstract cost / label interface
    # ------------------------------------------------------------------
    def _compute_cost(self, obs: np.ndarray) -> float:
        raise NotImplementedError

    def _compute_label(self, obs: np.ndarray) -> int:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info) if info else {}
        info["cost"] = 0.0
        info["original_reward"] = 0.0
        info["safety_label"] = 0
        return obs, info

    def step(self, action):
        obs, native_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info) if info else {}

        reward = self._compute_electricity_cost_reward()
        info["original_reward"] = float(native_reward)
        info["cost"] = self._compute_cost(obs)
        info["safety_label"] = self._compute_label(obs)

        return obs, float(reward), terminated, truncated, info
