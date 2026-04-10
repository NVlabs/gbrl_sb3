##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
"""Scenario C — Electricity Cost vs Peak-Demand Control (CMDP).

reward  =  negative price-weighted net electricity cash-flow  (maximize savings)
cost    =  daily-peak penalty: fires when district-total net consumption
           sets a new rolling daily maximum, penalising demand spikes
           that translate into demand charges.
label   =  1 when  (1) district-total non-shiftable load (NSL) is near or
           above the rolling daily NSL peak,  (2) district-total NSL is
           above the historical ``nsl_quantile`` threshold,  (3) usable
           electrical storage exists (discharge could shave the peak),
           and  (4) price is not extreme (reward does not already
           dominate the decision).

           NEC (net electricity consumption) is action-dependent and
           therefore not valid for pre-step labels; NSL is exogenous and
           fully observable before the agent acts.

All aggregation uses **district total** (sum across buildings), because
demand charges are levied on aggregate grid import, not per-building mean.

Active actions: [dhw_storage, electrical_storage, cooling_device] per
building.  Discharging electrical storage directly reduces net grid import,
shaving the peak; reward prefers *not* discharging when price is low.
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
DEFAULT_PEAK_LOOKBACK = 24          # hours: rolling window for daily peak
DEFAULT_PEAK_PROXIMITY = 0.85       # fraction: nec >= this × daily_peak → near peak
DEFAULT_NSL_QUANTILE = 0.70         # non-shiftable load above this → "high"
DEFAULT_SOC_USABLE_THRESH = 0.15    # SOC above this → storage can discharge
DEFAULT_PRICE_QUANTILE_UPPER = 0.80 # price BELOW this → cost-shaving is relevant
                                    # (at extreme prices, reward already dominates)


# -----------------------------------------------------------------------
# Wrapper
# -----------------------------------------------------------------------

class PeakShavingWrapper(CityLearnBaseWrapper):
    """CMDP: minimise electricity cost subject to peak-demand constraint.

    Cost — daily peak penalty
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    After each step, compute the maximum district-total net electricity
    consumption over the last ``peak_lookback`` hours (rolling daily
    peak).  If the current step's total sets a *new* rolling peak:

        cost  =  clamp( (total_nec - prev_peak) / prev_peak,  0,  1 )

    Otherwise cost is 0.

    Label — peak-shaving decision frontier
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    All four conditions must hold simultaneously:

    1. Current district-total NSL is ≥
       ``peak_proximity × rolling_daily_nsl_peak`` — the exogenous
       demand is near or at the point that could set a new NEC peak.
       (NEC itself is action-dependent and invalid for pre-step labels.)
    2. District-total non-shiftable load is above the
       ``nsl_quantile``-th percentile of historical district-total NSL —
       the spike is exogenous, not caused by the agent's own storage
       charging.
    3. At least one building has usable electrical storage
       (SOC > ``soc_usable_thresh``), giving the agent a discharge lever.
    4. Current max electricity price is **below** the
       ``price_quantile_upper`` threshold — at extreme prices the reward
       signal already encourages reducing consumption, so the label
       should only fire where cost and reward *disagree*.

    No storage → label = 0.
    """

    def __init__(
        self,
        env,
        peak_lookback: int = DEFAULT_PEAK_LOOKBACK,
        peak_proximity: float = DEFAULT_PEAK_PROXIMITY,
        nsl_quantile: float = DEFAULT_NSL_QUANTILE,
        soc_usable_thresh: float = DEFAULT_SOC_USABLE_THRESH,
        price_quantile_upper: float = DEFAULT_PRICE_QUANTILE_UPPER,
    ):
        super().__init__(env)
        self._peak_lookback = peak_lookback
        self._peak_proximity = peak_proximity
        self._nsl_quantile = nsl_quantile
        self._soc_usable_thresh = soc_usable_thresh
        self._price_quantile_upper = price_quantile_upper

        self._nsl_threshold: Optional[float] = None
        self._price_upper_threshold: Optional[float] = None

    # ------------------------------------------------------------------
    # One-time init
    # ------------------------------------------------------------------
    def _scenario_init(self):
        cl_env = self.unwrapped
        if not (hasattr(cl_env, "buildings") and cl_env.buildings):
            return

        # Non-shiftable load threshold from historical district-total NSL.
        # Sum across buildings per hour, then take quantile.
        buildings = cl_env.buildings
        n_hours = None
        for b in buildings:
            nsl = b.non_shiftable_load
            if hasattr(nsl, "__len__") and len(nsl) > 0:
                n_hours = len(nsl) if n_hours is None else min(n_hours, len(nsl))
        if n_hours is not None and n_hours > 0:
            hourly_totals = np.zeros(n_hours)
            for b in buildings:
                nsl = b.non_shiftable_load
                if hasattr(nsl, "__len__") and len(nsl) >= n_hours:
                    hourly_totals += np.array(nsl[:n_hours], dtype=float)
            self._nsl_threshold = float(
                np.quantile(hourly_totals, self._nsl_quantile)
            )

        # Price upper bound
        all_prices = []
        for b in buildings:
            if hasattr(b, "pricing") and b.pricing is not None:
                ep = b.pricing.electricity_pricing
                if hasattr(ep, "__len__") and len(ep) > 0:
                    all_prices.extend(ep)
        if all_prices:
            self._price_upper_threshold = float(
                np.quantile(all_prices, self._price_quantile_upper)
            )

    # ------------------------------------------------------------------
    # Helpers — all use district total (sum across buildings)
    # ------------------------------------------------------------------
    def _total_current_nec(self) -> Optional[float]:
        """District-total net electricity consumption at current timestep."""
        total = 0.0
        n = 0
        for b in self._get_buildings():
            nec = self._at_timestep(b.net_electricity_consumption)
            if nec is not None:
                total += nec
                n += 1
        return float(total) if n > 0 else None

    def _rolling_daily_peak(self) -> Optional[float]:
        """Max district-total NEC over the last ``peak_lookback`` hours
        (excluding the current hour).  Only considers positive totals."""
        ts = self._get_time_step() - 1  # current index
        lookback_start = max(0, ts - self._peak_lookback)
        if lookback_start >= ts:
            return None

        buildings = self._get_buildings()
        if not buildings:
            return None

        peak = None
        for t in range(lookback_start, ts):
            hour_total = 0.0
            n = 0
            for b in buildings:
                nec_arr = b.net_electricity_consumption
                if hasattr(nec_arr, "__len__") and 0 <= t < len(nec_arr):
                    hour_total += float(nec_arr[t])
                    n += 1
            if n > 0 and hour_total > 0:
                if peak is None or hour_total > peak:
                    peak = hour_total
        return peak

    def _total_current_nsl(self) -> Optional[float]:
        """District-total non-shiftable load at current timestep."""
        total = 0.0
        n = 0
        for b in self._get_buildings():
            nsl = b.non_shiftable_load
            ts = self._get_time_step() - 1
            if hasattr(nsl, "__len__") and 0 <= ts < len(nsl):
                total += float(nsl[ts])
                n += 1
        return float(total) if n > 0 else None

    def _total_pre_step_nsl(self) -> Optional[float]:
        """District-total non-shiftable load using pre-step accessor.

        NSL arrays are pre-allocated (full episode from CSV), so
        ``_at_pre_step`` reads ``nsl[ts]`` = the current hour's
        exogenous demand, valid before the agent acts.
        """
        total = 0.0
        n = 0
        for b in self._get_buildings():
            val = self._at_pre_step(b.non_shiftable_load)
            if val is not None:
                total += val
                n += 1
        return float(total) if n > 0 else None

    def _rolling_nsl_peak_pre_step(self) -> Optional[float]:
        """Max district-total NSL over the last ``peak_lookback`` hours
        (excluding current hour), using pre-step timing.

        At pre-step, ``time_step`` has not been incremented, so indices
        ``0 .. time_step-1`` are the committed previous hours.
        """
        ts = self._get_time_step()  # pre-step: not yet incremented
        lookback_start = max(0, ts - self._peak_lookback)
        if lookback_start >= ts:
            return None

        buildings = self._get_buildings()
        if not buildings:
            return None

        peak = None
        for t in range(lookback_start, ts):
            hour_total = 0.0
            n = 0
            for b in buildings:
                nsl = b.non_shiftable_load
                if hasattr(nsl, "__len__") and 0 <= t < len(nsl):
                    hour_total += float(nsl[t])
                    n += 1
            if n > 0 and hour_total > 0:
                if peak is None or hour_total > peak:
                    peak = hour_total
        return peak

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------
    def _compute_cost(self, obs) -> float:
        """Peak-overshoot cost ∈ [0, 1].

        Fires when the current district-total NEC exceeds the rolling
        daily peak.  The excess is normalised by the previous peak.
        """
        self._ensure_base_init()
        current_nec = self._total_current_nec()
        if current_nec is None:
            return 0.0

        prev_peak = self._rolling_daily_peak()
        if prev_peak is None or prev_peak <= 0:
            return 0.0

        if current_nec <= prev_peak:
            return 0.0

        excess = (current_nec - prev_peak) / prev_peak
        return float(min(1.0, excess))

    # ------------------------------------------------------------------
    # Label
    # ------------------------------------------------------------------
    def _compute_label(self, obs) -> int:
        self._ensure_base_init()
        buildings = self._get_buildings()
        if not buildings:
            return 0

        # 1) District-total NSL near rolling NSL peak (pre-step).
        #    NSL is the exogenous demand component, fully observable
        #    before the agent acts.  NEC is action-dependent and invalid.
        current_nsl = self._total_pre_step_nsl()
        if current_nsl is None:
            return 0
        nsl_peak = self._rolling_nsl_peak_pre_step()
        if nsl_peak is None or nsl_peak <= 0:
            return 0
        if current_nsl < self._peak_proximity * nsl_peak:
            return 0

        # 2) District-total non-shiftable load is high (spike is exogenous)
        if self._nsl_threshold is None:
            return 0
        if current_nsl <= self._nsl_threshold:
            return 0

        # 3) Usable storage exists
        has_usable = False
        for b in buildings:
            if b.electrical_storage is not None:
                soc = self._at_timestep(b.electrical_storage.soc)
                if soc is not None and soc > self._soc_usable_thresh:
                    has_usable = True
                    break
        if not has_usable:
            return 0

        # 4) Price is not extreme (reward doesn't already dominate)
        if self._price_upper_threshold is None:
            return 0
        max_price = self._max_pre_step_price(buildings)
        if max_price is not None and max_price >= self._price_upper_threshold:
            return 0

        return 1


# -----------------------------------------------------------------------
# Factory helpers
# -----------------------------------------------------------------------

def make_peak_shaving_env(
    env_name: str = DEFAULT_SCHEMA,
    episode_time_steps: Optional[int] = None,
    peak_lookback: int = DEFAULT_PEAK_LOOKBACK,
    peak_proximity: float = DEFAULT_PEAK_PROXIMITY,
    nsl_quantile: float = DEFAULT_NSL_QUANTILE,
    soc_usable_thresh: float = DEFAULT_SOC_USABLE_THRESH,
    price_quantile_upper: float = DEFAULT_PRICE_QUANTILE_UPPER,
    **kwargs,
):
    """Single raw CityLearn env with Peak-Shaving CMDP wrapper."""
    inner = make_citylearn_inner_env(
        schema=env_name, episode_time_steps=episode_time_steps,
    )
    return PeakShavingWrapper(
        inner,
        peak_lookback=peak_lookback,
        peak_proximity=peak_proximity,
        nsl_quantile=nsl_quantile,
        soc_usable_thresh=soc_usable_thresh,
        price_quantile_upper=price_quantile_upper,
    )


def make_peak_shaving_vec_env(
    env_name: str = DEFAULT_SCHEMA,
    n_envs: int = 1,
    seed: Optional[int] = None,
    episode_time_steps: Optional[int] = None,
    peak_lookback: int = DEFAULT_PEAK_LOOKBACK,
    peak_proximity: float = DEFAULT_PEAK_PROXIMITY,
    nsl_quantile: float = DEFAULT_NSL_QUANTILE,
    soc_usable_thresh: float = DEFAULT_SOC_USABLE_THRESH,
    price_quantile_upper: float = DEFAULT_PRICE_QUANTILE_UPPER,
    **kwargs,
):
    """VecEnv of CityLearn envs with Peak-Shaving CMDP wrapper."""
    env_kwargs = {
        "env_name": env_name,
        "episode_time_steps": episode_time_steps,
        "peak_lookback": peak_lookback,
        "peak_proximity": peak_proximity,
        "nsl_quantile": nsl_quantile,
        "soc_usable_thresh": soc_usable_thresh,
        "price_quantile_upper": price_quantile_upper,
    }
    return make_cost_vec_env(
        make_peak_shaving_env,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
    )
