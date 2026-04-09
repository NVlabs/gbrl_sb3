##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
"""Scenario A — Electricity Cost vs Thermal Comfort (CMDP).

reward  =  negative price-weighted electricity consumption  (minimize spending)
cost    =  thermal discomfort: mean deviation outside the two-sided comfort
           band  [heating_sp − band,  cooling_sp + band]  across buildings
label   =  1 when cost should own the sample:
           (a) any building is already violating comfort bounds with
           usable storage (recovery), or
           (b) high price + low thermal headroom + usable storage
           (pre-violation frontier).
           label=0 elsewhere → reward objective owns the sample.

The label explicitly excludes already-violating states (where the conflict
has already materialised) and DHW storage (hot water is not indoor thermal
comfort).

Active actions in this schema: [dhw_storage, electrical_storage, cooling_device]
per building.  The cooling_device action directly controls the heat pump and
is comfort-relevant.  Electrical storage dispatch affects electricity cost.
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

# Comfort cost
DEFAULT_COMFORT_BAND = 1.0       # °C tolerance around each setpoint
DEFAULT_COMFORT_MAX_DELTA = 5.0  # °C beyond band that maps to cost = 1.0

# Label
DEFAULT_PRICE_QUANTILE = 0.70    # price above this quantile → "high price"
DEFAULT_HEADROOM_THRESH = 0.5    # °C: headroom below this → "low thermal slack"
DEFAULT_SOC_USABLE_THRESH = 0.15 # SOC above this → storage has usable capacity


# -----------------------------------------------------------------------
# Wrapper
# -----------------------------------------------------------------------

class CostVsComfortWrapper(CityLearnBaseWrapper):
    """CMDP: minimise electricity cost subject to comfort constraint.

    Cost — mean thermal discomfort across buildings
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For each building the comfort zone is the two-sided band

        [heating_sp − band,   cooling_sp + band]

    where *band* is ``comfort_band`` (default 1 °C).  If only a cooling
    setpoint exists for a given building, only the upper bound is enforced.

    Violation for building *i*:

        v_i  =  max(0, T_in − upper) + max(0, lower − T_in)

    Cost  =  mean_i  clamp(v_i / max_delta,  0,  1)

    Label — objective routing (label=1 → cost, label=0 → reward)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Two zones route to cost:

    1. **Already violating** — any building has headroom ≤ 0 (outside
       comfort band) AND has usable electrical storage → label=1.
       Cost should drive recovery.
    2. **Pre-violation frontier** — high price + a building is still
       comfortable but has headroom < ``headroom_thresh`` + usable
       electrical storage → label=1.

    No electrical storage at all → label = 0  (no dispatch choice exists).
    """

    def __init__(
        self,
        env,
        comfort_band: float = DEFAULT_COMFORT_BAND,
        comfort_max_delta: float = DEFAULT_COMFORT_MAX_DELTA,
        price_quantile: float = DEFAULT_PRICE_QUANTILE,
        headroom_thresh: float = DEFAULT_HEADROOM_THRESH,
        soc_usable_thresh: float = DEFAULT_SOC_USABLE_THRESH,
    ):
        super().__init__(env)
        self._comfort_band = comfort_band
        self._comfort_max_delta = comfort_max_delta
        self._price_quantile = price_quantile
        self._headroom_thresh = headroom_thresh
        self._soc_usable_thresh = soc_usable_thresh
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
    # Per-building comfort helpers
    # ------------------------------------------------------------------
    def _comfort_bounds(self, building):
        """Return (lower_bound, upper_bound) of the comfort zone (°C)."""
        cooling_sp = self._latest(
            building.indoor_dry_bulb_temperature_cooling_set_point
        )
        heating_sp = self._latest(
            building.indoor_dry_bulb_temperature_heating_set_point
        ) if hasattr(building, "indoor_dry_bulb_temperature_heating_set_point") else None

        upper = (cooling_sp + self._comfort_band) if cooling_sp is not None else None
        lower = (heating_sp - self._comfort_band) if heating_sp is not None else None
        return lower, upper

    def _building_discomfort(self, building) -> float:
        """Normalised violation for one building ∈ [0, 1]."""
        t_in = self._latest(building.indoor_dry_bulb_temperature)
        if t_in is None:
            return 0.0
        lower, upper = self._comfort_bounds(building)
        violation = 0.0
        if upper is not None:
            violation += max(0.0, t_in - upper)
        if lower is not None:
            violation += max(0.0, lower - t_in)
        return min(violation / self._comfort_max_delta, 1.0)

    def _building_headroom(self, building) -> float:
        """Min distance from indoor temp to nearest comfort boundary (°C).

        Positive → inside comfort zone.  Negative → already violating.
        Returns the raw signed distance (not clipped to 0).
        """
        t_in = self._latest(building.indoor_dry_bulb_temperature)
        if t_in is None:
            return float("inf")
        lower, upper = self._comfort_bounds(building)
        headroom = float("inf")
        if upper is not None:
            headroom = min(headroom, upper - t_in)
        if lower is not None:
            headroom = min(headroom, t_in - lower)
        return headroom

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------
    def _compute_cost(self, obs) -> float:
        """Mean discomfort across buildings ∈ [0, 1]."""
        self._ensure_base_init()
        buildings = self._get_buildings()
        if not buildings:
            return 0.0
        total = sum(self._building_discomfort(b) for b in buildings)
        return float(total / len(buildings))

    # ------------------------------------------------------------------
    # Label
    # ------------------------------------------------------------------
    def _compute_label(self, obs) -> int:
        self._ensure_base_init()
        buildings = self._get_buildings()
        if not buildings:
            return 0

        # Check for any building already violating with usable storage
        # → cost should own recovery
        for b in buildings:
            headroom = self._building_headroom(b)
            if headroom <= 0:
                # Already violating — cost should drive recovery
                if b.electrical_storage is not None:
                    soc = self._at_timestep(b.electrical_storage.soc)
                    if soc is not None and soc > self._soc_usable_thresh:
                        return 1

        # Frontier: high price + near-boundary + usable storage
        if self._price_threshold is None:
            return 0
        max_price = self._max_current_price(buildings)
        if max_price is None or max_price <= self._price_threshold:
            return 0

        for b in buildings:
            headroom = self._building_headroom(b)
            # Still inside comfort zone but near the boundary
            if headroom <= 0 or headroom >= self._headroom_thresh:
                continue

            # Check electrical storage only (DHW ≠ indoor thermal comfort)
            if b.electrical_storage is not None:
                soc = self._at_timestep(b.electrical_storage.soc)
                if soc is not None and soc > self._soc_usable_thresh:
                    return 1

        return 0


# -----------------------------------------------------------------------
# Factory helpers
# -----------------------------------------------------------------------

def make_cost_vs_comfort_env(
    env_name: str = DEFAULT_SCHEMA,
    episode_time_steps: Optional[int] = None,
    comfort_band: float = DEFAULT_COMFORT_BAND,
    comfort_max_delta: float = DEFAULT_COMFORT_MAX_DELTA,
    price_quantile: float = DEFAULT_PRICE_QUANTILE,
    headroom_thresh: float = DEFAULT_HEADROOM_THRESH,
    soc_usable_thresh: float = DEFAULT_SOC_USABLE_THRESH,
    **kwargs,
):
    """Single raw CityLearn env with Cost-vs-Comfort CMDP wrapper."""
    inner = make_citylearn_inner_env(
        schema=env_name, episode_time_steps=episode_time_steps,
    )
    return CostVsComfortWrapper(
        inner,
        comfort_band=comfort_band,
        comfort_max_delta=comfort_max_delta,
        price_quantile=price_quantile,
        headroom_thresh=headroom_thresh,
        soc_usable_thresh=soc_usable_thresh,
    )


def make_cost_vs_comfort_vec_env(
    env_name: str = DEFAULT_SCHEMA,
    n_envs: int = 1,
    seed: Optional[int] = None,
    episode_time_steps: Optional[int] = None,
    comfort_band: float = DEFAULT_COMFORT_BAND,
    comfort_max_delta: float = DEFAULT_COMFORT_MAX_DELTA,
    price_quantile: float = DEFAULT_PRICE_QUANTILE,
    headroom_thresh: float = DEFAULT_HEADROOM_THRESH,
    soc_usable_thresh: float = DEFAULT_SOC_USABLE_THRESH,
    **kwargs,
):
    """VecEnv of CityLearn envs with Cost-vs-Comfort CMDP wrapper."""
    env_kwargs = {
        "env_name": env_name,
        "episode_time_steps": episode_time_steps,
        "comfort_band": comfort_band,
        "comfort_max_delta": comfort_max_delta,
        "price_quantile": price_quantile,
        "headroom_thresh": headroom_thresh,
        "soc_usable_thresh": soc_usable_thresh,
    }
    return make_cost_vec_env(
        make_cost_vs_comfort_env,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
    )
