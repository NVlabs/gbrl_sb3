##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
"""CityLearn integration for constrained Split-RL experiments.

Scenario: Electricity Cost vs Thermal Comfort (CMDP)
  - reward = negative electricity cost (minimize spending)
  - cost   = thermal comfort violation (temperature drift from setpoint)
  - label  = 1 when high price AND usable storage exists
             (genuine dispatch decision: discharge stored energy vs buy from grid)

Note: In CityLearn, comfort violations and peak prices are structurally
anti-correlated (HVAC load keeps temps near setpoint during peak hours).
The genuine decision frontier is storage dispatch, not direct comfort vs cost.

CityLearn is used in centralized mode (central_agent=True) with continuous
Box actions via its official SB3 wrapper stack.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from utils.helpers import make_cost_vec_env

# ---------------------------------------------------------------------------
# Default schema – CityLearn 2023 Phase 2 (3 residential buildings, 1 year)
# ---------------------------------------------------------------------------
DEFAULT_SCHEMA = "citylearn_challenge_2023_phase_2_local_evaluation"

# ---------------------------------------------------------------------------
# Observation name indices (populated at runtime from the wrapped env)
# ---------------------------------------------------------------------------

# Comfort thresholds
DEFAULT_COMFORT_BAND = 1.0       # °C band around setpoint before cost fires
DEFAULT_COMFORT_MAX_DELTA = 5.0  # °C max delta for normalizing cost to [0, 1]

# Label thresholds
DEFAULT_PRICE_QUANTILE = 0.70    # price above this quantile → "high price"
DEFAULT_SOC_USABLE_THRESH = 0.15 # SOC above this → storage has usable capacity


class CityLearnCostWrapper(gym.Wrapper):
    """Adds cost and safety_label to a CityLearn SB3-wrapped environment.

    The underlying env is already wrapped with:
      NormalizedObservationWrapper → StableBaselines3Wrapper
    so obs is a 1-D float32 array and reward is a scalar float.

    We intercept step/reset to inject:
      info['cost']          – thermal comfort violation ∈ [0, 1]
      info['safety_label']  – binary label where cost & reward disagree
      info['original_reward'] – copy of the original reward
    """

    def __init__(
        self,
        env: gym.Env,
        comfort_band: float = DEFAULT_COMFORT_BAND,
        comfort_max_delta: float = DEFAULT_COMFORT_MAX_DELTA,
        price_quantile: float = DEFAULT_PRICE_QUANTILE,
        soc_usable_thresh: float = DEFAULT_SOC_USABLE_THRESH,
    ):
        super().__init__(env)
        self._comfort_band = comfort_band
        self._comfort_max_delta = comfort_max_delta
        self._price_quantile = price_quantile
        self._soc_usable_thresh = soc_usable_thresh

        # Price threshold (in raw currency units) computed once from schema data
        self._price_threshold: Optional[float] = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------
    def _lazy_init(self):
        if self._initialized:
            return
        # Pre-compute price threshold from historical electricity pricing
        cl_env = self.unwrapped
        if hasattr(cl_env, "buildings") and cl_env.buildings:
            all_prices = []
            for b in cl_env.buildings:
                if hasattr(b, "pricing") and b.pricing is not None:
                    ep = b.pricing.electricity_pricing
                    if hasattr(ep, "__len__") and len(ep) > 0:
                        all_prices.extend(ep)
                    elif ep is not None:
                        all_prices.append(ep)
            if all_prices:
                self._price_threshold = float(np.quantile(all_prices, self._price_quantile))
        self._initialized = True

    # ------------------------------------------------------------------
    # Raw building state access
    # ------------------------------------------------------------------
    def _get_buildings(self):
        """Return the list of buildings from the unwrapped CityLearnEnv."""
        return self.unwrapped.buildings

    def _get_time_step(self) -> int:
        """Return the current time_step from the unwrapped env.

        After env.step(), time_step has already been incremented, so the
        state we just observed corresponds to time_step - 1.
        """
        return self.unwrapped.time_step

    @staticmethod
    def _latest(series):
        """Get the latest value from a live-appended array (temp, setpoint)."""
        if hasattr(series, "__len__") and len(series) > 0:
            return float(series[-1])
        return float(series) if series is not None else None

    def _at_timestep(self, series):
        """Get value at current timestep from a pre-allocated array (pricing, SOC)."""
        ts = self._get_time_step() - 1  # already incremented after step
        if hasattr(series, "__len__") and 0 <= ts < len(series):
            return float(series[ts])
        return None

    # ------------------------------------------------------------------
    # Cost computation: thermal comfort violation (from raw °C data)
    # ------------------------------------------------------------------
    def _compute_cost(self, obs: np.ndarray) -> float:
        """Comfort cost from raw building temperatures.

        cost = max over buildings of:
            clamp(|T_indoor - T_setpoint| - band, 0, max_delta) / max_delta

        Returns float ∈ [0, 1].
        """
        self._lazy_init()
        max_violation = 0.0
        for b in self._get_buildings():
            t_indoor = self._latest(b.indoor_dry_bulb_temperature)
            t_setpoint = self._latest(b.indoor_dry_bulb_temperature_cooling_set_point)
            if t_indoor is not None and t_setpoint is not None:
                delta = abs(t_indoor - t_setpoint)
                violation = max(0.0, delta - self._comfort_band) / self._comfort_max_delta
                max_violation = max(max_violation, violation)
        return float(np.clip(max_violation, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Label computation: genuine control choice frontier (from raw data)
    # ------------------------------------------------------------------
    def _compute_label(self, obs: np.ndarray) -> int:
        """Label = 1 when the agent faces a genuine cost–reward dispatch choice:
        1) Electricity price is high (above quantile of historical prices)
        2) There is usable storage (at least one SOC > threshold)

        In CityLearn, comfort violations and high prices are structurally
        anti-correlated (peak HVAC load keeps temps near setpoint during
        peak price hours). The genuine decision frontier is: discharge
        stored energy (save money, deplete buffer) vs buy from grid
        (expensive, preserve buffer). This requires high price AND
        available storage.
        """
        self._lazy_init()
        buildings = self._get_buildings()
        if not buildings:
            return 0

        # Condition 1: high electricity price (pre-allocated array, use timestep)
        if self._price_threshold is None:
            return 0
        price = self._at_timestep(buildings[0].pricing.electricity_pricing) \
            if hasattr(buildings[0], "pricing") and buildings[0].pricing is not None else None
        if price is None or price <= self._price_threshold:
            return 0

        # Condition 2: usable storage capacity (pre-allocated arrays, use timestep)
        has_usable = False
        for b in buildings:
            if b.electrical_storage is not None:
                soc = self._at_timestep(b.electrical_storage.soc)
                if soc is not None and soc > self._soc_usable_thresh:
                    has_usable = True
                    break
            if b.dhw_storage is not None:
                soc = self._at_timestep(b.dhw_storage.soc)
                if soc is not None and soc > self._soc_usable_thresh:
                    has_usable = True
                    break
        if not has_usable:
            # If no storage at all, label every high-price step
            has_any_storage = any(
                b.electrical_storage is not None or b.dhw_storage is not None
                for b in buildings
            )
            if has_any_storage:
                return 0

        return 1

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info) if info else {}
        info["cost"] = 0.0
        info["original_reward"] = 0.0
        info["safety_label"] = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info) if info else {}
        info["original_reward"] = float(reward)
        info["cost"] = self._compute_cost(obs)
        info["safety_label"] = self._compute_label(obs)
        return obs, float(reward), terminated, truncated, info


# -----------------------------------------------------------------------
# Factory functions
# -----------------------------------------------------------------------

def make_citylearn_raw_env(
    env_name: str = DEFAULT_SCHEMA,
    episode_time_steps: Optional[int] = None,
    comfort_band: float = DEFAULT_COMFORT_BAND,
    comfort_max_delta: float = DEFAULT_COMFORT_MAX_DELTA,
    price_quantile: float = DEFAULT_PRICE_QUANTILE,
    soc_usable_thresh: float = DEFAULT_SOC_USABLE_THRESH,
    **kwargs,
) -> gym.Env:
    """Create a single CityLearn env wrapped for Split-RL.

    Wrapping order:
      CityLearnEnv → NormalizedObservationWrapper → StableBaselines3Wrapper
                   → CityLearnCostWrapper
    """
    from citylearn.citylearn import CityLearnEnv
    from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper

    cl_kwargs: Dict[str, Any] = {
        "schema": env_name,
        "central_agent": True,
    }
    if episode_time_steps is not None:
        cl_kwargs["episode_time_steps"] = episode_time_steps

    cl_env = CityLearnEnv(**cl_kwargs)
    cl_env = NormalizedObservationWrapper(cl_env)
    cl_env = StableBaselines3Wrapper(cl_env)
    env = CityLearnCostWrapper(
        cl_env,
        comfort_band=comfort_band,
        comfort_max_delta=comfort_max_delta,
        price_quantile=price_quantile,
        soc_usable_thresh=soc_usable_thresh,
    )
    return env


def make_citylearn_vec_env(
    env_name: str = DEFAULT_SCHEMA,
    n_envs: int = 1,
    seed: Optional[int] = None,
    episode_time_steps: Optional[int] = None,
    comfort_band: float = DEFAULT_COMFORT_BAND,
    comfort_max_delta: float = DEFAULT_COMFORT_MAX_DELTA,
    price_quantile: float = DEFAULT_PRICE_QUANTILE,
    soc_usable_thresh: float = DEFAULT_SOC_USABLE_THRESH,
    **kwargs,
):
    """Build a VecEnv of CityLearn envs with cost/label tracking."""
    env_kwargs = {
        "env_name": env_name,
        "episode_time_steps": episode_time_steps,
        "comfort_band": comfort_band,
        "comfort_max_delta": comfort_max_delta,
        "price_quantile": price_quantile,
        "soc_usable_thresh": soc_usable_thresh,
    }
    return make_cost_vec_env(
        make_citylearn_raw_env,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
    )
