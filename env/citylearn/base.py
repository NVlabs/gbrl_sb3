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

import os
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from utils.helpers import make_cost_vec_env

# ---------------------------------------------------------------------------
# Default schema
# ---------------------------------------------------------------------------
# Resolve to local dataset directory bundled in the repo so CityLearn never
# needs to download anything from GitHub at runtime.
_DATASETS_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                             "datasets", "citylearn")

DEFAULT_SCHEMA = "citylearn_challenge_2023_phase_2_local_evaluation"


def _resolve_schema(schema: str) -> str:
    """Return a local schema.json path if the dataset is bundled, else pass through."""
    local = os.path.join(_DATASETS_DIR, schema, "schema.json")
    if os.path.isfile(local):
        return os.path.realpath(local)
    return schema


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
        "schema": _resolve_schema(schema),
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
    Replaces the native CityLearn reward with **negative price-weighted
    net electricity cash-flow**:

        reward_t  =  -(1/N) Σ_i  consumption_i(t) × price_i(t)

    Import (consumption > 0) incurs cost; export (consumption < 0) earns
    revenue.  Higher reward ⟹ better net cash-flow.  The native reward
    is preserved in ``info["original_reward"]`` for comparison.

    Label timing
    ------------
    ``safety_label`` is computed **before** ``env.step(action)`` from
    the state the agent observed when it chose the action.  This makes
    the label a *pre-decision* frontier indicator, matching the ``obs``
    stored in the rollout buffer.

    Subclass contract
    -----------------
    Subclasses **must** implement:

    * ``_compute_cost(obs) -> float``   (∈ [0, 1])
    * ``_compute_label(obs) -> int``    (∈ {0, 1})

    and **may** override ``_scenario_init()`` for one-time setup that
    needs the live env (called once, lazily, on the first ``step``).

    """

    _N_AGGREGATE_FEATURES = 8

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._base_initialized = False
        self._n_buildings = 0

        # Normalization bounds (set in _init_aggregate_features)
        self._max_district_nsl: float = 1.0
        self._max_district_solar: float = 1.0
        self._max_net_base_load: float = 1.0

        # # Extend observation space with aggregate features (disabled)
        # low = np.append(self.observation_space.low,
        #                 np.zeros(self._N_AGGREGATE_FEATURES, dtype=np.float32))
        # high = np.append(self.observation_space.high,
        #                  np.ones(self._N_AGGREGATE_FEATURES, dtype=np.float32))
        # self.observation_space = gym.spaces.Box(
        #     low=low, high=high, dtype=self.observation_space.dtype,
        # )

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
        # self._init_aggregate_features()  # disabled
        self._base_initialized = True

    def _scenario_init(self):
        """Override in subclasses for one-time setup (e.g. price threshold)."""

    # ------------------------------------------------------------------
    # Cross-building aggregate features
    # ------------------------------------------------------------------
    def _init_aggregate_features(self):
        """Compute normalization bounds from full-horizon schedules."""
        buildings = self._get_buildings()
        if not buildings:
            return
        n_hours = None
        for b in buildings:
            es = getattr(b, "energy_simulation", None)
            nsl = getattr(es, "_non_shiftable_load", None)
            if nsl is not None and hasattr(nsl, "__len__") and len(nsl) > 0:
                n_hours = len(nsl) if n_hours is None else min(n_hours, len(nsl))
        if n_hours is None or n_hours == 0:
            return
        district_nsl = np.zeros(n_hours)
        district_solar = np.zeros(n_hours)
        for b in buildings:
            es = getattr(b, "energy_simulation", None)
            nsl = getattr(es, "_non_shiftable_load", None)
            if nsl is not None and hasattr(nsl, "__len__") and len(nsl) >= n_hours:
                district_nsl += np.asarray(nsl[:n_hours], dtype=float)
            sg = getattr(es, "_solar_generation", None)
            if sg is not None and hasattr(sg, "__len__") and len(sg) >= n_hours:
                district_solar += np.asarray(sg[:n_hours], dtype=float)
        self._max_district_nsl = max(float(np.max(district_nsl)), 1e-6)
        self._max_district_solar = max(float(np.max(district_solar)), 1e-6)
        self._max_net_base_load = max(float(np.max(
            np.maximum(district_nsl - district_solar, 0.0))), 1e-6)

    def _compute_aggregate_features(self) -> np.ndarray:
        """8 cross-building summary statistics, all normalized to [0, 1].

        0. district_nsl        Σ NSL_i / max
        1. district_solar      Σ solar_i / max
        2. net_base_load       max(Σ NSL − Σ solar, 0) / max
        3. mean_elec_soc       mean(SOC_i)
        4. min_elec_soc        min(SOC_i)
        5. max_elec_soc        max(SOC_i)
        6. mean_dhw_soc        mean(DHW_SOC_i)
        7. mean_price          mean(price_i) / max_price  (approx)
        """
        self._ensure_base_init()
        buildings = self._get_buildings()
        acc = self._at_pre_step

        d_nsl = 0.0
        d_solar = 0.0
        for b in buildings:
            es = getattr(b, "energy_simulation", None)
            nsl = getattr(es, "_non_shiftable_load", None)
            if nsl is not None:
                v = acc(nsl)
                if v is not None:
                    d_nsl += v
            sg = getattr(es, "_solar_generation", None)
            if sg is not None:
                v = acc(sg)
                if v is not None:
                    d_solar += v

        e_socs = []
        for b in buildings:
            if b.electrical_storage is not None:
                soc = acc(b.electrical_storage.soc)
                if soc is not None:
                    e_socs.append(float(soc))
        mean_e = float(np.mean(e_socs)) if e_socs else 0.5
        min_e = float(min(e_socs)) if e_socs else 0.5
        max_e = float(max(e_socs)) if e_socs else 0.5

        dhw_socs = []
        for b in buildings:
            dhw = getattr(b, "dhw_storage", None)
            if dhw is not None and hasattr(dhw, "soc"):
                soc = acc(dhw.soc)
                if soc is not None:
                    dhw_socs.append(float(soc))
        mean_dhw = float(np.mean(dhw_socs)) if dhw_socs else 0.5

        prices = []
        for b in buildings:
            if hasattr(b, "pricing") and b.pricing is not None:
                p = acc(b.pricing.electricity_pricing)
                if p is not None:
                    prices.append(p)
        mean_price = float(np.mean(prices)) if prices else 0.0
        # Normalize price by a rough upper bound (max observed ~ 0.06)
        mean_price_norm = min(1.0, mean_price / 0.06) if mean_price > 0 else 0.0

        return np.array([
            min(1.0, d_nsl / self._max_district_nsl),
            min(1.0, d_solar / self._max_district_solar),
            min(1.0, max(d_nsl - d_solar, 0.0) / self._max_net_base_load),
            mean_e, min_e, max_e,
            mean_dhw,
            mean_price_norm,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Raw building-state helpers
    # ------------------------------------------------------------------
    def _get_buildings(self):
        """Buildings list from the unwrapped CityLearnEnv."""
        return self.unwrapped.buildings

    def _get_time_step(self) -> int:
        """CityLearn's internal time_step counter."""
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
        """Element at ``time_step - 1`` of a pre-allocated array (pricing, SOC).

        Correct for **post-step** reads (reward, cost) where CityLearn has
        already incremented ``time_step``.
        """
        ts = self._get_time_step() - 1
        if hasattr(series, "__len__") and 0 <= ts < len(series):
            return float(series[ts])
        return None

    def _at_pre_step(self, series):
        """Element at ``time_step`` of a pre-allocated array.

        Correct for **pre-step** reads (label) where CityLearn has NOT
        yet incremented ``time_step``.  Reads the state the agent is
        currently observing.
        """
        ts = self._get_time_step()
        if hasattr(series, "__len__") and 0 <= ts < len(series):
            return float(series[ts])
        return None

    def _max_current_price(self, buildings) -> Optional[float]:
        """Max electricity price at the current timestep across buildings.

        Uses post-step accessor (for reward/cost computation).
        """
        max_p = None
        for b in buildings:
            if hasattr(b, "pricing") and b.pricing is not None:
                p = self._at_timestep(b.pricing.electricity_pricing)
                if p is not None and (max_p is None or p > max_p):
                    max_p = p
        return max_p

    def _max_pre_step_price(self, buildings) -> Optional[float]:
        """Max electricity price using pre-step accessor (for label computation)."""
        max_p = None
        for b in buildings:
            if hasattr(b, "pricing") and b.pricing is not None:
                p = self._at_pre_step(b.pricing.electricity_pricing)
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
        # obs = np.append(obs, self._compute_aggregate_features())  # disabled
        return obs, info

    def step(self, action):
        # ---- pre-step: label from the state the agent OBSERVED --------
        # The label is computed BEFORE env.step(action) so it reflects
        # the same state the agent used to choose the action.  Pre-step
        # accessors (_at_pre_step, _max_pre_step_price) read pre-allocated
        # arrays at index time_step (not yet incremented); _latest reads
        # the tail of live-appended series.
        self._ensure_base_init()
        pre_label = self._compute_label(None)

        # ---- step ----------------------------------------------------
        obs, native_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info) if info else {}

        # ---- post-step: reward & cost from the transition outcome ----
        reward = self._compute_electricity_cost_reward()
        info["original_reward"] = float(native_reward)
        info["cost"] = self._compute_cost(obs)
        info["safety_label"] = pre_label

        # obs = np.append(obs, self._compute_aggregate_features())  # disabled
        return obs, float(reward), terminated, truncated, info


# -----------------------------------------------------------------------
# Pairwise feature wrapper (applied AFTER scenario wrappers)
# -----------------------------------------------------------------------

class RandomPairwiseFeatureWrapper(gym.ObservationWrapper):
    """Append random pairwise linear and product features.

    This is a representation aid for tree-based policies:

    * **Linear pair features** ``0.5 * (x_i ± x_j)`` approximate sparse
      oblique splits.  Trees can only do axis-aligned splits; these
      expose differences and sums between observation dimensions.
    * **Product pair features** ``x_i * x_j`` approximate second-order
      interactions.  The CityLearn reward is literally
      ``price × consumption``; trees cannot represent products
      without exponentially many splits.

    All features are deterministic transforms of the existing observation
    and contain no new environment information.  Neural networks can
    already learn both classes of features in their early layers.

    This wrapper should be placed **after** the scenario wrapper so that
    the scenario-specific scalar feature (e.g. ``steps_until_peak``) is
    included in the pairwise pool.

    Parameters
    ----------
    env : gym.Env
        Environment with the final observation (including scenario feature).
    n_linear : int
        Number of random pairwise linear features.
    n_product : int
        Number of random pairwise product features.
    seed : int
        RNG seed for reproducible pair selection.
    """

    def __init__(
        self,
        env: gym.Env,
        n_linear: int = 64,
        n_product: int = 64,
        seed: int = 20240517,
    ):
        super().__init__(env)

        self.n_linear = n_linear
        self.n_product = n_product

        obs_dim = int(np.prod(env.observation_space.shape))
        rng = np.random.RandomState(seed)

        # Random index pairs for linear features (x_i ± x_j)
        self.linear_i = rng.randint(0, obs_dim, size=n_linear)
        self.linear_j = rng.randint(0, obs_dim, size=n_linear)
        self.linear_sign = rng.choice(
            [-1.0, 1.0], size=n_linear,
        ).astype(np.float32)

        # Random index pairs for product features (x_i * x_j)
        self.prod_i = rng.randint(0, obs_dim, size=n_product)
        self.prod_j = rng.randint(0, obs_dim, size=n_product)

        n_extra = n_linear + n_product
        low = np.append(
            env.observation_space.low,
            -np.ones(n_extra, dtype=np.float32),
        )
        high = np.append(
            env.observation_space.high,
            np.ones(n_extra, dtype=np.float32),
        )
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32,
        )

    def observation(self, obs):
        x = np.asarray(obs, dtype=np.float32).ravel()
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

        # Center normalized obs so products and differences are balanced
        xc = 2.0 * x - 1.0

        # Sparse oblique pairwise linear features
        lin = 0.5 * (xc[self.linear_i] + self.linear_sign * xc[self.linear_j])

        # Pairwise interaction (product) features
        prod = xc[self.prod_i] * xc[self.prod_j]

        z = np.concatenate([lin, prod]).astype(np.float32)
        return np.concatenate([x, z]).astype(np.float32)
