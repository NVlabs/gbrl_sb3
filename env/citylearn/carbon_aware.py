##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
"""Scenario D — Electricity Cost vs Carbon Emissions (CMDP).

reward  =  negative price-weighted net electricity cash-flow  (maximize savings)
cost    =  dirty-grid import penalty: fires when the agent imports
           electricity during high carbon-intensity periods, penalising
           cheap-but-dirty grid usage.
label   =  1 when  (1) carbon intensity is above a high threshold,
           (2) electricity price is below a moderate threshold (the
           disagreement: grid import is cheap but dirty),  (3) the
           district is currently importing (NEC > 0),  and  (4) usable
           electrical storage exists (discharge avoids grid import).
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
DEFAULT_CARBON_QUANTILE = 0.70   # carbon above this quantile → "dirty grid"
DEFAULT_PRICE_QUANTILE = 0.50    # price below this → "cheap but maybe dirty"
DEFAULT_SOC_USABLE_THRESH = 0.15 # SOC above this → storage can discharge


# -----------------------------------------------------------------------
# Wrapper
# -----------------------------------------------------------------------

class CarbonAwareWrapper(CityLearnBaseWrapper):
    """CMDP: minimise electricity cost subject to carbon constraint.

    Cost — dirty-grid import penalty
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    At each step, for each building importing from the grid (NEC > 0)
    while carbon intensity exceeds the ``carbon_quantile`` threshold:

        building_cost_i  =  (ci − threshold) / (ci_max − threshold)

    Buildings that export (NEC ≤ 0) or face low carbon contribute 0.
    Cost is the mean across buildings.

    Label — cheap-but-dirty dispatch frontier
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    All four conditions must hold simultaneously:

    1. Current carbon intensity is above the ``carbon_quantile``-th
       percentile — the grid is dirty right now.
    2. Current max electricity price is **below** the ``price_quantile``-th
       percentile — the grid is cheap right now.  This is the actual
       disagreement: reward says "import" but cost says "don't."
    3. District-total NEC > 0 — there is actual import pressure
       (if the district is exporting, there is no dirty-import decision).
    4. At least one building has usable electrical storage
       (SOC > ``soc_usable_thresh``), giving the agent a discharge lever
       to avoid grid import.

    No storage → label = 0.
    """

    def __init__(
        self,
        env,
        carbon_quantile: float = DEFAULT_CARBON_QUANTILE,
        price_quantile: float = DEFAULT_PRICE_QUANTILE,
        soc_usable_thresh: float = DEFAULT_SOC_USABLE_THRESH,
    ):
        super().__init__(env)
        self._carbon_quantile = carbon_quantile
        self._price_quantile = price_quantile
        self._soc_usable_thresh = soc_usable_thresh

        self._carbon_threshold: Optional[float] = None
        self._price_threshold: Optional[float] = None
        self._carbon_min: Optional[float] = None
        self._carbon_max: Optional[float] = None

    # ------------------------------------------------------------------
    # One-time init
    # ------------------------------------------------------------------
    def _scenario_init(self):
        cl_env = self.unwrapped
        if not (hasattr(cl_env, "buildings") and cl_env.buildings):
            return

        # Carbon intensity threshold and max
        all_carbon = []
        for b in cl_env.buildings:
            if hasattr(b, "carbon_intensity") and b.carbon_intensity is not None:
                ci = b.carbon_intensity.carbon_intensity
                if hasattr(ci, "__len__") and len(ci) > 0:
                    all_carbon.extend(ci)
        if all_carbon:
            self._carbon_threshold = float(
                np.quantile(all_carbon, self._carbon_quantile)
            )
            self._carbon_min = float(np.min(all_carbon))
            self._carbon_max = float(np.max(all_carbon))

        # Price threshold
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
    def _current_carbon_intensity(self, building) -> Optional[float]:
        """Carbon intensity at the current timestep for a building."""
        if not (hasattr(building, "carbon_intensity")
                and building.carbon_intensity is not None):
            return None
        return self._at_timestep(building.carbon_intensity.carbon_intensity)

    def _max_current_carbon(self, buildings) -> Optional[float]:
        """Max carbon intensity at current timestep across buildings."""
        max_ci = None
        for b in buildings:
            ci = self._current_carbon_intensity(b)
            if ci is not None and (max_ci is None or ci > max_ci):
                max_ci = ci
        return max_ci

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------
    def _compute_cost(self, obs) -> float:
        """Carbon cost ∈ [0, 1] — fires only during dirty-grid periods.

        For each building that is importing from the grid (NEC > 0) at
        a time when carbon intensity exceeds the ``carbon_quantile``
        threshold:

            building_cost  =  (ci − threshold) / (ci_max − threshold)

        If carbon is at or below the threshold, or the building is
        exporting, that building contributes 0.

        This mirrors the arbitrage cost structure: cost = 0 in the safe
        region, cost rises linearly in the violation zone.
        """
        self._ensure_base_init()
        buildings = self._get_buildings()
        if (not buildings
                or self._carbon_threshold is None
                or self._carbon_max is None
                or self._carbon_max <= self._carbon_threshold):
            return 0.0

        violation_range = self._carbon_max - self._carbon_threshold
        total = 0.0
        n = 0
        for b in buildings:
            nec = self._at_timestep(b.net_electricity_consumption)
            ci = self._current_carbon_intensity(b)
            if nec is not None and ci is not None:
                n += 1
                if nec > 0 and ci > self._carbon_threshold:
                    total += (ci - self._carbon_threshold) / violation_range
        if n == 0:
            return 0.0

        return float(min(1.0, total / n))

    # ------------------------------------------------------------------
    # Label
    # ------------------------------------------------------------------
    def _compute_label(self, obs) -> int:
        self._ensure_base_init()
        buildings = self._get_buildings()
        if not buildings:
            return 0

        # 1) High carbon intensity
        if self._carbon_threshold is None:
            return 0
        max_carbon = self._max_current_carbon(buildings)
        if max_carbon is None or max_carbon <= self._carbon_threshold:
            return 0

        # 2) Low electricity price (the disagreement)
        if self._price_threshold is None:
            return 0
        max_price = self._max_current_price(buildings)
        if max_price is None or max_price > self._price_threshold:
            return 0

        # 3) District is currently importing (no import = no dirty decision)
        total_nec = 0.0
        n = 0
        for b in buildings:
            nec = self._at_timestep(b.net_electricity_consumption)
            if nec is not None:
                total_nec += nec
                n += 1
        if n == 0 or total_nec <= 0:
            return 0

        # 4) Usable storage exists
        for b in buildings:
            if b.electrical_storage is not None:
                soc = self._at_timestep(b.electrical_storage.soc)
                if soc is not None and soc > self._soc_usable_thresh:
                    return 1

        return 0


# -----------------------------------------------------------------------
# Factory helpers
# -----------------------------------------------------------------------

def make_carbon_aware_env(
    env_name: str = DEFAULT_SCHEMA,
    episode_time_steps: Optional[int] = None,
    carbon_quantile: float = DEFAULT_CARBON_QUANTILE,
    price_quantile: float = DEFAULT_PRICE_QUANTILE,
    soc_usable_thresh: float = DEFAULT_SOC_USABLE_THRESH,
    **kwargs,
):
    """Single raw CityLearn env with Carbon-Aware CMDP wrapper."""
    inner = make_citylearn_inner_env(
        schema=env_name, episode_time_steps=episode_time_steps,
    )
    return CarbonAwareWrapper(
        inner,
        carbon_quantile=carbon_quantile,
        price_quantile=price_quantile,
        soc_usable_thresh=soc_usable_thresh,
    )


def make_carbon_aware_vec_env(
    env_name: str = DEFAULT_SCHEMA,
    n_envs: int = 1,
    seed: Optional[int] = None,
    episode_time_steps: Optional[int] = None,
    carbon_quantile: float = DEFAULT_CARBON_QUANTILE,
    price_quantile: float = DEFAULT_PRICE_QUANTILE,
    soc_usable_thresh: float = DEFAULT_SOC_USABLE_THRESH,
    **kwargs,
):
    """VecEnv of CityLearn envs with Carbon-Aware CMDP wrapper."""
    env_kwargs = {
        "env_name": env_name,
        "episode_time_steps": episode_time_steps,
        "carbon_quantile": carbon_quantile,
        "price_quantile": price_quantile,
        "soc_usable_thresh": soc_usable_thresh,
    }
    return make_cost_vec_env(
        make_carbon_aware_env,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
    )
