##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
"""Scenario C — Electricity Cost vs Peak-Demand Control (CMDP).

reward  =  negative price-weighted electricity consumption  (minimize spending)
cost    =  daily-peak penalty: fires when an hourly net consumption sets a
           new daily maximum, penalising the agent for creating demand
           spikes that translate into demand charges.
label   =  1 when  (1) current net consumption is near or above the running
           daily peak,  (2) non-shiftable load is high (the spike is
           exogenous, not caused by storage charging),  (3) usable
           electrical storage exists (discharge could shave the peak),
           and  (4) price is not extreme (reward does not already
           dominate the decision).

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
    After each step, compute the maximum net electricity consumption over
    the last ``peak_lookback`` hours (rolling daily peak).  If the current
    step's consumption sets a *new* rolling peak, cost is:

        cost  =  clamp( (nec - prev_peak) / prev_peak,  0,  1 )

    Otherwise cost is 0.  A fresh rolling peak that exceeds the previous
    one by 100 %+ of the old peak saturates at 1.

    Label — peak-shaving decision frontier
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    All four conditions must hold simultaneously:

    1. Current mean net consumption across buildings is ≥
       ``peak_proximity × rolling_daily_peak`` — the system is near or
       at the point of setting a new peak.
    2. Mean non-shiftable load across buildings is above the
       ``nsl_quantile``-th percentile — the spike is exogenous, not
       caused by the agent's own storage charging.
    3. At least one building has usable electrical storage
       (SOC > ``soc_usable_thresh``), giving the agent a discharge lever.
    4. Current max electricity price is **below** the ``price_quantile_upper``
       threshold — at extreme prices the reward signal already encourages
       reducing consumption, so the label should only fire where cost and
       reward *disagree*.

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

        # Non-shiftable load threshold from historical data
        all_nsl = []
        for b in cl_env.buildings:
            nsl = b.non_shiftable_load
            if hasattr(nsl, "__len__") and len(nsl) > 0:
                all_nsl.extend(nsl)
        if all_nsl:
            self._nsl_threshold = float(
                np.quantile(all_nsl, self._nsl_quantile)
            )

        # Price upper bound
        all_prices = []
        for b in cl_env.buildings:
            if hasattr(b, "pricing") and b.pricing is not None:
                ep = b.pricing.electricity_pricing
                if hasattr(ep, "__len__") and len(ep) > 0:
                    all_prices.extend(ep)
        if all_prices:
            self._price_upper_threshold = float(
                np.quantile(all_prices, self._price_quantile_upper)
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _mean_current_nec(self) -> Optional[float]:
        """Mean net electricity consumption at current timestep."""
        values = []
        for b in self._get_buildings():
            nec = self._at_timestep(b.net_electricity_consumption)
            if nec is not None:
                values.append(nec)
        return float(np.mean(values)) if values else None

    def _rolling_daily_peak(self) -> Optional[float]:
        """Max mean-NEC over the last ``peak_lookback`` hours (excl. current)."""
        ts = self._get_time_step() - 1  # current index (already stepped)
        lookback_start = max(0, ts - self._peak_lookback)
        if lookback_start >= ts:
            return None

        # Compute mean NEC across buildings for each hour in lookback
        buildings = self._get_buildings()
        if not buildings:
            return None

        peak = None
        for t in range(lookback_start, ts):
            hour_sum = 0.0
            n = 0
            for b in buildings:
                nec_arr = b.net_electricity_consumption
                if hasattr(nec_arr, "__len__") and 0 <= t < len(nec_arr):
                    hour_sum += float(nec_arr[t])
                    n += 1
            if n > 0:
                hour_mean = hour_sum / n
                if peak is None or hour_mean > peak:
                    peak = hour_mean
        return peak

    def _mean_current_nsl(self) -> Optional[float]:
        """Mean non-shiftable load at current timestep."""
        values = []
        for b in self._get_buildings():
            nsl = b.non_shiftable_load
            ts = self._get_time_step() - 1
            if hasattr(nsl, "__len__") and 0 <= ts < len(nsl):
                values.append(float(nsl[ts]))
        return float(np.mean(values)) if values else None

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------
    def _compute_cost(self, obs) -> float:
        """Peak-overshoot cost ∈ [0, 1].

        Fires when the current step's mean NEC exceeds the rolling daily
        peak.  The excess is normalised by the previous peak.
        """
        self._ensure_base_init()
        current_nec = self._mean_current_nec()
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

        # 1) Current consumption near daily peak
        current_nec = self._mean_current_nec()
        if current_nec is None:
            return 0
        prev_peak = self._rolling_daily_peak()
        if prev_peak is None or prev_peak <= 0:
            return 0
        if current_nec < self._peak_proximity * prev_peak:
            return 0

        # 2) Non-shiftable load is high (spike is exogenous)
        if self._nsl_threshold is None:
            return 0
        mean_nsl = self._mean_current_nsl()
        if mean_nsl is None or mean_nsl <= self._nsl_threshold:
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
        max_price = self._max_current_price(buildings)
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
