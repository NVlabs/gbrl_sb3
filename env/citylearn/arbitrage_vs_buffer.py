##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
"""Scenario B — Electricity Arbitrage vs Storage Buffer / Resilience (CMDP).

reward  =  negative price-weighted electricity consumption  (minimize spending)
cost    =  storage buffer depletion: fires when mean battery SOC drops below
           a safety level, penalising aggressive discharge that leaves
           buildings without a resilience buffer.
label   =  1 when cost should own the sample:
           (a) SOC already below safety (recovery), or
           (b) high price + near-safety frontier (imminent crossing risk).
           label=0 elsewhere → reward objective owns the sample.

No storage at all → label = 0, cost = 0  (no dispatch choice exists).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from env.citylearn.base import (
    CityLearnBaseWrapper,
    DEFAULT_SCHEMA,
    make_citylearn_inner_env,
)
from utils.helpers import make_cost_vec_env

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PRICE_QUANTILE = 0.70
DEFAULT_SOC_USABLE_THRESH = 0.15  # label condition: SOC above this
DEFAULT_SOC_SAFETY_LEVEL = 0.30   # cost fires when SOC drops below this
DEFAULT_SOC_FRONTIER_BAND = 0.05  # label fires when mean SOC is within
                                   # [safety, safety + band]


# -----------------------------------------------------------------------
# Wrapper
# -----------------------------------------------------------------------

class ArbitrageVsBufferWrapper(CityLearnBaseWrapper):
    """CMDP: minimise electricity cost subject to buffer-resilience constraint.

    Cost — storage buffer depletion
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    When the mean electrical-storage SOC across buildings drops below
    ``soc_safety_level``, cost rises linearly to 1.0 at SOC = 0:

        cost  =  clamp((safety − mean_soc) / safety,  0,  1)

    If no building has electrical storage, cost is always 0.

    Label — objective routing (label=1 → cost, label=0 → reward)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Three zones:

    1. **Comfortably above safety** (SOC > safety + band) → label=0.
       Reward owns these states; no risk of cost activation.
    2. **Near-safety frontier** (safety ≤ SOC ≤ safety + band, AND
       high price + usable storage) → label=1.
       Reward is tempted to discharge but doing so risks crossing below
       safety. Cost objective should govern.
    3. **Below safety** (SOC < safety, with usable storage) → label=1.
       Cost is already active; cost objective should drive recovery.

    No storage at all → label = 0  (no dispatch choice exists).
    """

    def __init__(
        self,
        env,
        price_quantile: float = DEFAULT_PRICE_QUANTILE,
        soc_usable_thresh: float = DEFAULT_SOC_USABLE_THRESH,
        soc_safety_level: float = DEFAULT_SOC_SAFETY_LEVEL,
        soc_frontier_band: float = DEFAULT_SOC_FRONTIER_BAND,
    ):
        super().__init__(env)
        self._price_quantile = price_quantile
        self._soc_usable_thresh = soc_usable_thresh
        self._soc_safety_level = soc_safety_level
        self._soc_frontier_band = soc_frontier_band
        self._price_threshold: Optional[float] = None

    # ------------------------------------------------------------------
    # One-time init
    # ------------------------------------------------------------------
    def _scenario_init(self):
        cl_env = self.unwrapped
        if not (hasattr(cl_env, "buildings") and cl_env.buildings):
            return
        all_prices = []
        for b in cl_env.buildings:
            if hasattr(b, "pricing") and b.pricing is not None:
                ep = b.pricing.electricity_pricing
                if hasattr(ep, "__len__") and len(ep) > 0:
                    all_prices.extend(ep)
        if all_prices:
            self._price_threshold = float(
                np.quantile(all_prices, self._price_quantile)
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _mean_electrical_soc(self) -> Optional[float]:
        """Mean electrical-storage SOC across buildings that have batteries.

        Always uses ``_at_timestep`` which reads ``soc[ts-1]``.  This is
        correct for both post-step (cost) and pre-step (label) reads
        because SOC arrays are written during each step, so the
        last-written value is always at index ``time_step - 1``.
        """
        soc_values = []
        for b in self._get_buildings():
            if b.electrical_storage is not None:
                soc = self._at_timestep(b.electrical_storage.soc)
                if soc is not None:
                    soc_values.append(soc)
        return float(np.mean(soc_values)) if soc_values else None

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------
    def _compute_cost(self, obs) -> float:
        """Buffer depletion cost ∈ [0, 1].

        Fires when mean battery SOC is below the safety level.
        """
        self._ensure_base_init()
        mean_soc = self._mean_electrical_soc()
        if mean_soc is None:
            return 0.0
        if mean_soc >= self._soc_safety_level:
            return 0.0
        return float(min(
            1.0,
            (self._soc_safety_level - mean_soc) / self._soc_safety_level,
        ))

    # ------------------------------------------------------------------
    # Label
    # ------------------------------------------------------------------
    def _compute_label(self, obs) -> int:
        self._ensure_base_init()
        buildings = self._get_buildings()
        if not buildings:
            return 0

        mean_soc = self._mean_electrical_soc()
        if mean_soc is None:
            return 0

        # Usable storage exists (no storage → no dispatch choice → 0)
        has_usable = False
        for b in buildings:
            if b.electrical_storage is not None:
                soc = self._at_timestep(b.electrical_storage.soc)
                if soc is not None and soc > self._soc_usable_thresh:
                    has_usable = True
                    break
        if not has_usable:
            return 0

        # Already below safety → cost owns recovery
        if mean_soc < self._soc_safety_level:
            return 1

        # Frontier: high price tempts discharge near safety boundary
        if self._price_threshold is None:
            return 0
        max_price = self._max_pre_step_price(buildings)
        if max_price is None or max_price <= self._price_threshold:
            return 0

        frontier_upper = self._soc_safety_level + self._soc_frontier_band
        if mean_soc <= frontier_upper:
            return 1

        return 0


# -----------------------------------------------------------------------
# Factory helpers
# -----------------------------------------------------------------------

def make_arbitrage_vs_buffer_env(
    env_name: str = DEFAULT_SCHEMA,
    episode_time_steps: Optional[int] = None,
    price_quantile: float = DEFAULT_PRICE_QUANTILE,
    soc_usable_thresh: float = DEFAULT_SOC_USABLE_THRESH,
    soc_safety_level: float = DEFAULT_SOC_SAFETY_LEVEL,
    soc_frontier_band: float = DEFAULT_SOC_FRONTIER_BAND,
    **kwargs,
):
    """Single raw CityLearn env with Arbitrage-vs-Buffer CMDP wrapper."""
    inner = make_citylearn_inner_env(
        schema=env_name, episode_time_steps=episode_time_steps,
    )
    return ArbitrageVsBufferWrapper(
        inner,
        price_quantile=price_quantile,
        soc_usable_thresh=soc_usable_thresh,
        soc_safety_level=soc_safety_level,
        soc_frontier_band=soc_frontier_band,
    )


def make_arbitrage_vs_buffer_vec_env(
    env_name: str = DEFAULT_SCHEMA,
    n_envs: int = 1,
    seed: Optional[int] = None,
    episode_time_steps: Optional[int] = None,
    price_quantile: float = DEFAULT_PRICE_QUANTILE,
    soc_usable_thresh: float = DEFAULT_SOC_USABLE_THRESH,
    soc_safety_level: float = DEFAULT_SOC_SAFETY_LEVEL,
    soc_frontier_band: float = DEFAULT_SOC_FRONTIER_BAND,
    **kwargs,
):
    """VecEnv of CityLearn envs with Arbitrage-vs-Buffer CMDP wrapper."""
    env_kwargs = {
        "env_name": env_name,
        "episode_time_steps": episode_time_steps,
        "price_quantile": price_quantile,
        "soc_usable_thresh": soc_usable_thresh,
        "soc_safety_level": soc_safety_level,
        "soc_frontier_band": soc_frontier_band,
    }
    return make_cost_vec_env(
        make_arbitrage_vs_buffer_env,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
    )
