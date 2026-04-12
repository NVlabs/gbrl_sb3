##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
"""CityLearn CMDP scenario wrappers.

All scenarios share the same price-weighted reward from CityLearnBaseWrapper
and differ only in cost function and label logic.

Scenarios
---------
B. **Arbitrage vs Buffer** — electricity arbitrage vs storage resilience
C. **Contract Demand** — electricity cost vs district import cap
D. **Carbon Aware** — electricity cost vs carbon emissions
E. **Peak Ratchet** — electricity cost vs peak import ratchet
F. **Demand Response** — electricity cost vs demand response reserves
G. **Solar Ramp Reserve** — electricity cost vs post-sunset reserve events
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from env.citylearn.base import (
    CityLearnBaseWrapper,
    DEFAULT_SCHEMA,
    make_citylearn_inner_env,
)
from utils.helpers import make_cost_vec_env


# ###########################################################################
# Scenario B — Arbitrage vs Buffer
# ###########################################################################

DEFAULT_AVB_PRICE_QUANTILE = 0.85
DEFAULT_AVB_SOC_USABLE_THRESH = 0.15
DEFAULT_AVB_SOC_SAFETY_LEVEL = 0.30
DEFAULT_AVB_SOC_FRONTIER_BAND = 0.10


class ArbitrageVsBufferWrapper(CityLearnBaseWrapper):
    """CMDP: minimise electricity cost subject to buffer-resilience constraint.

    Cost — critical-battery depletion
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Uses *critical SOC* (min across buildings) instead of mean SOC.
    One depleted battery is exactly where the reserve problem is real;
    mean SOC hides that bottleneck.

        cost  =  clamp((safety − critical_soc) / safety,  0,  1)

    Label — conflict-only frontier (label=1 → cost, label=0 → reward)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Label fires only where reward and cost objectives *disagree*:

        label = 1   iff   critical_soc ≤ safety + band
                          AND price > high_price_threshold
                          AND some battery is dischargeable

    Low-SOC recovery at low price → label=0 (reward and cost both want
    to charge, no conflict).  Only high-price + tight-reserve states
    where reward wants to discharge but cost wants to hold go to label=1.

    No storage → label = 0.
    """

    def __init__(
        self,
        env,
        price_quantile: float = DEFAULT_AVB_PRICE_QUANTILE,
        soc_usable_thresh: float = DEFAULT_AVB_SOC_USABLE_THRESH,
        soc_safety_level: float = DEFAULT_AVB_SOC_SAFETY_LEVEL,
        soc_frontier_band: float = DEFAULT_AVB_SOC_FRONTIER_BAND,
    ):
        super().__init__(env)
        self._price_quantile = price_quantile
        self._soc_usable_thresh = soc_usable_thresh
        self._soc_safety_level = soc_safety_level
        self._soc_frontier_band = soc_frontier_band
        self._price_threshold: Optional[float] = None

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

    # -- Helpers -------------------------------------------------------

    def _critical_electrical_soc(self) -> Optional[float]:
        """Min electrical-storage SOC across buildings (critical battery).

        Uses ``_at_timestep`` (reads ``soc[ts-1]``).  Correct for both
        post-step (cost) and pre-step (label) because SOC arrays are
        written during each step; the last-written value is at ``ts-1``.
        """
        soc_values = []
        for b in self._get_buildings():
            if b.electrical_storage is not None:
                soc = self._at_timestep(b.electrical_storage.soc)
                if soc is not None:
                    soc_values.append(soc)
        return float(np.min(soc_values)) if soc_values else None

    # -- Cost ----------------------------------------------------------

    def _compute_cost(self, obs) -> float:
        """Critical-battery depletion cost ∈ [0, 1]."""
        self._ensure_base_init()
        critical_soc = self._critical_electrical_soc()
        if critical_soc is None:
            return 0.0
        if critical_soc >= self._soc_safety_level:
            return 0.0
        return float(min(
            1.0,
            (self._soc_safety_level - critical_soc) / self._soc_safety_level,
        ))

    # -- Label ---------------------------------------------------------

    def _compute_label(self, obs) -> int:
        self._ensure_base_init()
        buildings = self._get_buildings()
        if not buildings:
            return 0

        critical_soc = self._critical_electrical_soc()
        if critical_soc is None:
            return 0

        # Require dischargeable battery — the conflict is about whether
        # to discharge (reward) or hold (cost).
        has_dischargeable = False
        for b in buildings:
            if b.electrical_storage is not None:
                soc = self._at_timestep(b.electrical_storage.soc)
                if soc is not None and soc > self._soc_usable_thresh:
                    has_dischargeable = True
                    break
        if not has_dischargeable:
            return 0

        # Price gate: conflict only exists when price tempts discharge.
        if self._price_threshold is None:
            return 0
        max_price = self._max_pre_step_price(buildings)
        if max_price is None or max_price <= self._price_threshold:
            return 0

        # Tight-reserve frontier: critical SOC near or below safety.
        frontier_upper = self._soc_safety_level + self._soc_frontier_band
        return 1 if critical_soc <= frontier_upper else 0


# ###########################################################################
# Scenario C — Contract Demand (replaces Peak Shaving)
# ###########################################################################

DEFAULT_CD_CAP_QUANTILE = 0.95
DEFAULT_CD_CAP_MARGIN = 2.5
DEFAULT_CD_FRONTIER_FRAC = 0.90
DEFAULT_CD_PRICE_LOW_QUANTILE = 0.60
DEFAULT_CD_CHARGE_HEADROOM_FRAC = 0.60
DEFAULT_CD_SOC_CHARGE_THRESHOLD = 0.80


class ContractDemandWrapper(CityLearnBaseWrapper):
    """CMDP: minimise electricity cost subject to district import cap.

    A district has a fixed import capacity derived from exogenous load
    statistics.  The reward still wants cheap electricity (arbitrage);
    the cost penalises operating near or above the import cap.

    Cap computation (at init)
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        district_nsl[t] = Σ_i  non_shiftable_load_i[t]
        cap = cap_margin × quantile(district_nsl, cap_quantile)
        frontier = frontier_frac × cap

    The cap is **stationary**: computed once from the exogenous load
    profile, never updated.

    Cost — soft capacity-stress penalty
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        district_import = max(Σ_i NEC_i, 0)
        cost = clamp((district_import − frontier) / (cap − frontier), 0, 1)

    Below the frontier: cost = 0.  At or above the cap: cost ≈ 1.

    Label — cheap charging trap with SOC gate (conflict zone)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        total_charge_power = Σ_i  nominal_power_i
        trap_threshold = frontier − charge_headroom_frac × total_charge_power
        label = 1   iff   price ≤ price_low_threshold
                          AND  district_nsl ≥ trap_threshold
                          AND  mean_soc < soc_charge_threshold

    The trap threshold is derived from the cost boundary (frontier) and
    the actual controllable import (battery charge power).  At the
    threshold, charging ``charge_headroom_frac`` of the fleet would push
    district import to the frontier.  With the default (0.60), labelled
    hours are those where more than 60% of fleet charging threatens the
    cap — the agent cannot freely charge.

    SOC gate: if batteries are nearly full (mean_soc ≥ threshold), the
    agent won't charge much and the conflict disappears → label=0.
    This creates a feedback loop: as the agent learns to pre-charge
    during safe hours, it arrives at trap hours with full batteries,
    label density drops, and the reward policy regains control.
    """

    def __init__(
        self,
        env,
        cap_quantile: float = DEFAULT_CD_CAP_QUANTILE,
        cap_margin: float = DEFAULT_CD_CAP_MARGIN,
        frontier_frac: float = DEFAULT_CD_FRONTIER_FRAC,
        price_low_quantile: float = DEFAULT_CD_PRICE_LOW_QUANTILE,
        charge_headroom_frac: float = DEFAULT_CD_CHARGE_HEADROOM_FRAC,
        soc_charge_threshold: float = DEFAULT_CD_SOC_CHARGE_THRESHOLD,
    ):
        super().__init__(env)
        self._cap_quantile = cap_quantile
        self._cap_margin = cap_margin
        self._frontier_frac = frontier_frac
        self._price_low_quantile = price_low_quantile
        self._charge_headroom_frac = charge_headroom_frac
        self._soc_charge_threshold = soc_charge_threshold

        self._cap: Optional[float] = None
        self._frontier: Optional[float] = None
        self._price_low_threshold: Optional[float] = None
        self._trap_threshold: Optional[float] = None

    def _scenario_init(self):
        cl_env = self.unwrapped
        if not (hasattr(cl_env, "buildings") and cl_env.buildings):
            return

        buildings = cl_env.buildings

        # Compute district NSL from full exogenous schedule and derive cap.
        # Note: b.non_shiftable_load is live-appended (grows each step),
        # but b.energy_simulation._non_shiftable_load holds the full
        # pre-allocated horizon loaded from CSV at init.
        n_hours = None
        n_readable = 0
        for b in buildings:
            es = getattr(b, "energy_simulation", None)
            nsl_full = getattr(es, "_non_shiftable_load", None)
            if nsl_full is not None and hasattr(nsl_full, '__len__') and len(nsl_full) > 0:
                n_hours = len(nsl_full) if n_hours is None else min(n_hours, len(nsl_full))
                n_readable += 1
        if n_hours is None or n_hours == 0:
            raise RuntimeError(
                "ContractDemandWrapper: could not read full-horizon "
                "non_shiftable_load schedule from any building's "
                "energy_simulation. The scenario cannot compute the "
                "import cap and will be non-functional."
            )
        if n_readable != len(buildings):
            raise RuntimeError(
                f"ContractDemandWrapper: only {n_readable}/{len(buildings)} "
                "buildings have a readable non_shiftable_load schedule. "
                "Cap would be computed from a subset — refusing to continue."
            )
        hourly_totals = np.zeros(n_hours)
        for b in buildings:
            es = getattr(b, "energy_simulation", None)
            nsl_full = getattr(es, "_non_shiftable_load", None)
            if nsl_full is not None and hasattr(nsl_full, '__len__') and len(nsl_full) >= n_hours:
                hourly_totals += np.array(nsl_full[:n_hours], dtype=float)
        base = float(np.quantile(hourly_totals, self._cap_quantile))
        self._cap = self._cap_margin * base
        self._frontier = self._frontier_frac * self._cap

        # Trap threshold derived from frontier and actual battery charge power
        total_charge_power = 0.0
        for b in buildings:
            es = b.electrical_storage
            if (es is not None
                    and hasattr(es, 'nominal_power')
                    and es.nominal_power is not None):
                total_charge_power += es.nominal_power
        if total_charge_power > 0:
            self._trap_threshold = max(
                0.0,
                self._frontier - self._charge_headroom_frac * total_charge_power,
            )
        else:
            # Fallback: no battery info, use 50% of frontier
            self._trap_threshold = 0.50 * self._frontier

        # Price threshold for "cheap" gate
        all_prices = []
        for b in buildings:
            if hasattr(b, "pricing") and b.pricing is not None:
                ep = b.pricing.electricity_pricing
                if hasattr(ep, "__len__") and len(ep) > 0:
                    all_prices.extend(ep[:n_hours])
        if all_prices:
            self._price_low_threshold = float(
                np.quantile(all_prices, self._price_low_quantile)
            )

    # -- Helpers -------------------------------------------------------

    def _district_import_post_step(self) -> Optional[float]:
        """District positive import at current timestep (post-step).

        ``max(Σ_i NEC_i, 0)`` — negative (net export) is clipped to 0.
        """
        total = 0.0
        n = 0
        for b in self._get_buildings():
            nec = self._at_timestep(b.net_electricity_consumption)
            if nec is not None:
                total += nec
                n += 1
        if n == 0:
            return None
        return max(total, 0.0)

    def _district_pre_step_nsl(self) -> Optional[float]:
        """District NSL at current timestep (pre-step, from schedule)."""
        total = 0.0
        n = 0
        ts = self._get_time_step()
        for b in self._get_buildings():
            es = getattr(b, "energy_simulation", None)
            sched = getattr(es, "_non_shiftable_load", None)
            if sched is not None and hasattr(sched, "__len__") and ts < len(sched):
                total += float(sched[ts])
                n += 1
        return total if n > 0 else None

    # -- Cost ----------------------------------------------------------

    def _compute_cost(self, obs) -> float:
        """Soft capacity-stress cost ∈ [0, 1]."""
        self._ensure_base_init()
        if self._cap is None or self._frontier is None:
            return 0.0
        if self._cap <= self._frontier:
            return 0.0

        district_import = self._district_import_post_step()
        if district_import is None:
            return 0.0

        if district_import <= self._frontier:
            return 0.0

        excess = (district_import - self._frontier) / (self._cap - self._frontier)
        return float(min(1.0, excess))

    # -- SOC helper ----------------------------------------------------

    def _mean_electrical_soc(self) -> Optional[float]:
        """Mean electrical-storage SOC across buildings."""
        soc_values = []
        for b in self._get_buildings():
            if b.electrical_storage is not None:
                soc = self._at_timestep(b.electrical_storage.soc)
                if soc is not None:
                    soc_values.append(soc)
        return float(np.mean(soc_values)) if soc_values else None

    # -- Label ---------------------------------------------------------

    def _compute_label(self, obs) -> int:
        self._ensure_base_init()
        if self._price_low_threshold is None or self._trap_threshold is None:
            return 0

        # Price gate: only label when electricity is cheap
        buildings = self._get_buildings()
        price = self._max_pre_step_price(buildings)
        if price is None or price > self._price_low_threshold:
            return 0

        # Trap: NSL already high enough that charging pushes toward cap
        nsl = self._district_pre_step_nsl()
        if nsl is None:
            return 0
        if nsl < self._trap_threshold:
            return 0

        # SOC gate: conflict only exists when batteries have room to charge.
        # If batteries are nearly full, agent won't charge much → no
        # additional import → no threat to cap → no conflict.
        mean_soc = self._mean_electrical_soc()
        if mean_soc is None:
            return 0
        return 1 if mean_soc < self._soc_charge_threshold else 0


# ###########################################################################
# Scenario D — Carbon Aware
# ###########################################################################

DEFAULT_CA_CARBON_QUANTILE = 0.70
DEFAULT_CA_PRICE_LOW_QUANTILE = 0.60
DEFAULT_CA_CONFLICT_QUANTILE = 0.80
DEFAULT_CA_SOC_CHARGE_THRESHOLD = 0.80


class CarbonAwareWrapper(CityLearnBaseWrapper):
    """CMDP: minimise electricity cost subject to carbon constraint.

    Cost — dirty-grid import penalty (magnitude-weighted)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        dirty = clamp((carbon − threshold) / (carbon_max − threshold), 0, 1)
        district_import = max(Σ_i NEC_i, 0)
        cost = dirty × clamp(district_import / import_scale, 0, 1)

    where ``import_scale`` is the max of historical district NSL
    (computed at init as a policy-independent proxy).  Using the max
    prevents cost saturation — P95 was too low and clipped most dirty-hour
    imports to 1.0, giving the agent zero gradient.

    This scales cost with actual import magnitude — a building importing
    0.01 kWh contributes far less cost than one importing 10 kWh.

    Label — dirty-load-cheap conflict with SOC gate
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Adds a cheapness gate to the dirty×load proxy::

        price_low = quantile(max_price_series, price_low_quantile)
        cheap = clamp((price_low − price) / (price_low − price_min), 0, 1)
        conflict = dirty × load × cheap
        label = 1   iff   conflict ≥ conflict_threshold
                          AND mean_soc < soc_charge_threshold

    where ``conflict_threshold`` is the ``conflict_quantile``-th percentile
    of the *positive* entries in the full-horizon conflict series.  The
    cheapness factor is 1 at price_min and 0 at or above price_low_threshold
    — a hard gate, not a min-max score over the full range.  This filters
    out hours above the cheap threshold entirely, isolating the genuine
    conflict: cheap electricity tempts import, but the grid is dirty.

    SOC gate: if batteries are nearly full (mean_soc ≥ threshold), the
    agent won't import much additional power (no charging) and the
    conflict disappears → label=0.  This creates a feedback loop: as
    the agent learns to charge during clean hours, it arrives at dirty
    cheap hours with full batteries, label density drops, and the
    reward policy regains control.
    """

    def __init__(
        self,
        env,
        carbon_quantile: float = DEFAULT_CA_CARBON_QUANTILE,
        price_low_quantile: float = DEFAULT_CA_PRICE_LOW_QUANTILE,
        conflict_quantile: float = DEFAULT_CA_CONFLICT_QUANTILE,
        soc_charge_threshold: float = DEFAULT_CA_SOC_CHARGE_THRESHOLD,
    ):
        super().__init__(env)
        self._carbon_quantile = carbon_quantile
        self._price_low_quantile = price_low_quantile
        self._conflict_quantile = conflict_quantile
        self._soc_charge_threshold = soc_charge_threshold

        self._carbon_threshold: Optional[float] = None
        self._carbon_min: Optional[float] = None
        self._carbon_max: Optional[float] = None
        self._import_scale: Optional[float] = None
        self._price_min: Optional[float] = None
        self._price_low_threshold: Optional[float] = None
        self._conflict_threshold: Optional[float] = None

    def _scenario_init(self):
        cl_env = self.unwrapped
        if not (hasattr(cl_env, "buildings") and cl_env.buildings):
            return

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

        # Import scale from full-horizon non-shiftable load schedule
        n_hours = None
        n_readable = 0
        for b in cl_env.buildings:
            es = getattr(b, "energy_simulation", None)
            sched = getattr(es, "_non_shiftable_load", None)
            if sched is not None and hasattr(sched, "__len__") and len(sched) > 0:
                n_hours = len(sched) if n_hours is None else min(n_hours, len(sched))
                n_readable += 1
        if n_hours is None or n_hours == 0:
            raise RuntimeError(
                "CarbonAwareWrapper: could not read full-horizon "
                "non_shiftable_load schedule from any building's "
                "energy_simulation. The scenario cannot compute "
                "import_scale and will be non-functional."
            )
        if n_readable != len(cl_env.buildings):
            raise RuntimeError(
                f"CarbonAwareWrapper: only {n_readable}/{len(cl_env.buildings)} "
                "buildings have a readable non_shiftable_load schedule. "
                "import_scale would be computed from a subset — refusing to continue."
            )
        hourly_nsl = np.zeros(n_hours)
        for b in cl_env.buildings:
            es = getattr(b, "energy_simulation", None)
            sched = getattr(es, "_non_shiftable_load", None)
            if sched is not None and hasattr(sched, "__len__") and len(sched) >= n_hours:
                hourly_nsl += np.asarray(sched[:n_hours], dtype=float)
        self._import_scale = float(np.max(hourly_nsl))

        # Collect price series for cheapness factor
        all_prices = []
        for b in cl_env.buildings:
            if hasattr(b, "pricing") and b.pricing is not None:
                ep = b.pricing.electricity_pricing
                if hasattr(ep, "__len__") and len(ep) >= n_hours:
                    all_prices.append(np.asarray(ep[:n_hours], dtype=float))
        # Use max price across buildings at each timestep (same as label accessor)
        if all_prices:
            max_price_series = np.max(all_prices, axis=0)
            self._price_min = float(np.min(max_price_series))
            self._price_low_threshold = float(
                np.quantile(max_price_series, self._price_low_quantile)
            )

        # Compute conflict threshold from full-horizon exogenous series
        if (self._carbon_threshold is not None
                and self._carbon_max is not None
                and self._carbon_max > self._carbon_threshold
                and self._import_scale > 0
                and self._price_min is not None
                and self._price_low_threshold is not None
                and self._price_low_threshold > self._price_min):
            # Build per-building carbon array aligned with hourly_nsl
            # Use max carbon across buildings at each timestep
            all_ci_arrays = []
            for b in cl_env.buildings:
                if hasattr(b, "carbon_intensity") and b.carbon_intensity is not None:
                    ci = b.carbon_intensity.carbon_intensity
                    if hasattr(ci, "__len__") and len(ci) >= n_hours:
                        all_ci_arrays.append(np.asarray(ci[:n_hours], dtype=float))
            if all_ci_arrays:
                max_ci_series = np.max(all_ci_arrays, axis=0)
                violation_range = self._carbon_max - self._carbon_threshold
                dirty_series = np.clip(
                    (max_ci_series - self._carbon_threshold) / violation_range,
                    0.0, 1.0,
                )
                load_series = np.clip(hourly_nsl / self._import_scale, 0.0, 1.0)
                price_range = self._price_low_threshold - self._price_min
                cheap_series = np.clip(
                    (self._price_low_threshold - max_price_series) / price_range,
                    0.0, 1.0,
                )
                conflict_series = dirty_series * load_series * cheap_series
                # Threshold over positive values only — zero-conflict
                # hours are not informative and dilute the threshold
                positive_conflict = conflict_series[conflict_series > 0]
                if len(positive_conflict) > 0:
                    self._conflict_threshold = float(
                        np.quantile(positive_conflict, self._conflict_quantile)
                    )

    # -- Helpers -------------------------------------------------------

    def _current_carbon_intensity(self, building) -> Optional[float]:
        """Carbon intensity — post-step accessor (for cost)."""
        if not (hasattr(building, "carbon_intensity")
                and building.carbon_intensity is not None):
            return None
        return self._at_timestep(building.carbon_intensity.carbon_intensity)

    def _pre_step_carbon_intensity(self, building) -> Optional[float]:
        """Carbon intensity — pre-step accessor (for label)."""
        if not (hasattr(building, "carbon_intensity")
                and building.carbon_intensity is not None):
            return None
        return self._at_pre_step(building.carbon_intensity.carbon_intensity)

    def _max_current_carbon(self, buildings) -> Optional[float]:
        """Max carbon intensity — post-step (for cost)."""
        max_ci = None
        for b in buildings:
            ci = self._current_carbon_intensity(b)
            if ci is not None and (max_ci is None or ci > max_ci):
                max_ci = ci
        return max_ci

    def _max_pre_step_carbon(self, buildings) -> Optional[float]:
        """Max carbon intensity — pre-step (for label)."""
        max_ci = None
        for b in buildings:
            ci = self._pre_step_carbon_intensity(b)
            if ci is not None and (max_ci is None or ci > max_ci):
                max_ci = ci
        return max_ci

    # -- Cost ----------------------------------------------------------

    def _compute_cost(self, obs) -> float:
        """Carbon cost ∈ [0, 1] — dirty_factor × normalised district import."""
        self._ensure_base_init()
        buildings = self._get_buildings()
        if (not buildings
                or self._carbon_threshold is None
                or self._carbon_max is None
                or self._carbon_max <= self._carbon_threshold
                or self._import_scale is None
                or self._import_scale <= 0):
            return 0.0

        # Dirty factor from max carbon intensity across buildings
        max_ci = self._max_current_carbon(buildings)
        if max_ci is None or max_ci <= self._carbon_threshold:
            return 0.0

        violation_range = self._carbon_max - self._carbon_threshold
        dirty = min(1.0, (max_ci - self._carbon_threshold) / violation_range)

        # District positive import magnitude
        total_nec = 0.0
        n = 0
        for b in buildings:
            nec = self._at_timestep(b.net_electricity_consumption)
            if nec is not None:
                total_nec += nec
                n += 1
        if n == 0:
            return 0.0
        district_import = max(total_nec, 0.0)
        if district_import <= 0:
            return 0.0

        normalised_import = min(1.0, district_import / self._import_scale)
        return float(dirty * normalised_import)

    # -- Label ---------------------------------------------------------

    def _district_pre_step_nsl(self) -> Optional[float]:
        """District NSL at current timestep (pre-step, from schedule)."""
        total = 0.0
        n = 0
        ts = self._get_time_step()
        for b in self._get_buildings():
            es = getattr(b, "energy_simulation", None)
            sched = getattr(es, "_non_shiftable_load", None)
            if sched is not None and hasattr(sched, "__len__") and ts < len(sched):
                total += float(sched[ts])
                n += 1
        return total if n > 0 else None

    def _compute_label(self, obs) -> int:
        self._ensure_base_init()
        buildings = self._get_buildings()
        if (not buildings
                or self._carbon_threshold is None
                or self._carbon_max is None
                or self._carbon_max <= self._carbon_threshold
                or self._import_scale is None
                or self._import_scale <= 0
                or self._price_min is None
                or self._price_low_threshold is None
                or self._price_low_threshold <= self._price_min
                or self._conflict_threshold is None):
            return 0

        max_carbon = self._max_pre_step_carbon(buildings)
        if max_carbon is None:
            return 0

        nsl = self._district_pre_step_nsl()
        if nsl is None:
            return 0

        price = self._max_pre_step_price(buildings)
        if price is None:
            return 0

        # Hard price gate: above price_low_threshold, cheap=0 => conflict=0
        if price >= self._price_low_threshold:
            return 0

        violation_range = self._carbon_max - self._carbon_threshold
        dirty = max(0.0, min(1.0, (max_carbon - self._carbon_threshold) / violation_range))
        load = min(1.0, nsl / self._import_scale)
        price_range = self._price_low_threshold - self._price_min
        cheap = max(0.0, min(1.0, (self._price_low_threshold - price) / price_range))
        conflict = dirty * load * cheap
        if conflict < self._conflict_threshold:
            return 0

        # SOC gate: conflict only exists when batteries have room to charge.
        # If batteries are nearly full, agent won't import extra → no
        # carbon-heavy charging → no conflict.
        mean_soc = self._mean_electrical_soc()
        if mean_soc is None:
            return 0
        return 1 if mean_soc < self._soc_charge_threshold else 0

    # -- SOC helper ----------------------------------------------------

    def _mean_electrical_soc(self) -> Optional[float]:
        """Mean electrical-storage SOC across buildings."""
        soc_values = []
        for b in self._get_buildings():
            if b.electrical_storage is not None:
                soc = self._at_timestep(b.electrical_storage.soc)
                if soc is not None:
                    soc_values.append(soc)
        return float(np.mean(soc_values)) if soc_values else None


# ###########################################################################
# Factory helpers
# ###########################################################################

# -- Arbitrage vs Buffer -----------------------------------------------

def make_arbitrage_vs_buffer_env(
    env_name: str = DEFAULT_SCHEMA,
    episode_time_steps: Optional[int] = None,
    price_quantile: float = DEFAULT_AVB_PRICE_QUANTILE,
    soc_usable_thresh: float = DEFAULT_AVB_SOC_USABLE_THRESH,
    soc_safety_level: float = DEFAULT_AVB_SOC_SAFETY_LEVEL,
    soc_frontier_band: float = DEFAULT_AVB_SOC_FRONTIER_BAND,
    **kwargs,
):
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
    price_quantile: float = DEFAULT_AVB_PRICE_QUANTILE,
    soc_usable_thresh: float = DEFAULT_AVB_SOC_USABLE_THRESH,
    soc_safety_level: float = DEFAULT_AVB_SOC_SAFETY_LEVEL,
    soc_frontier_band: float = DEFAULT_AVB_SOC_FRONTIER_BAND,
    **kwargs,
):
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


# -- Contract Demand ---------------------------------------------------

def make_contract_demand_env(
    env_name: str = DEFAULT_SCHEMA,
    episode_time_steps: Optional[int] = None,
    cap_quantile: float = DEFAULT_CD_CAP_QUANTILE,
    cap_margin: float = DEFAULT_CD_CAP_MARGIN,
    frontier_frac: float = DEFAULT_CD_FRONTIER_FRAC,
    price_low_quantile: float = DEFAULT_CD_PRICE_LOW_QUANTILE,
    charge_headroom_frac: float = DEFAULT_CD_CHARGE_HEADROOM_FRAC,
    **kwargs,
):
    inner = make_citylearn_inner_env(
        schema=env_name, episode_time_steps=episode_time_steps,
    )
    return ContractDemandWrapper(
        inner,
        cap_quantile=cap_quantile,
        cap_margin=cap_margin,
        frontier_frac=frontier_frac,
        price_low_quantile=price_low_quantile,
        charge_headroom_frac=charge_headroom_frac,
    )


def make_contract_demand_vec_env(
    env_name: str = DEFAULT_SCHEMA,
    n_envs: int = 1,
    seed: Optional[int] = None,
    episode_time_steps: Optional[int] = None,
    cap_quantile: float = DEFAULT_CD_CAP_QUANTILE,
    cap_margin: float = DEFAULT_CD_CAP_MARGIN,
    frontier_frac: float = DEFAULT_CD_FRONTIER_FRAC,
    price_low_quantile: float = DEFAULT_CD_PRICE_LOW_QUANTILE,
    charge_headroom_frac: float = DEFAULT_CD_CHARGE_HEADROOM_FRAC,
    **kwargs,
):
    env_kwargs = {
        "env_name": env_name,
        "episode_time_steps": episode_time_steps,
        "cap_quantile": cap_quantile,
        "cap_margin": cap_margin,
        "frontier_frac": frontier_frac,
        "price_low_quantile": price_low_quantile,
        "charge_headroom_frac": charge_headroom_frac,
    }
    return make_cost_vec_env(
        make_contract_demand_env,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
    )


# -- Carbon Aware ------------------------------------------------------

def make_carbon_aware_env(
    env_name: str = DEFAULT_SCHEMA,
    episode_time_steps: Optional[int] = None,
    carbon_quantile: float = DEFAULT_CA_CARBON_QUANTILE,
    price_low_quantile: float = DEFAULT_CA_PRICE_LOW_QUANTILE,
    conflict_quantile: float = DEFAULT_CA_CONFLICT_QUANTILE,
    **kwargs,
):
    inner = make_citylearn_inner_env(
        schema=env_name, episode_time_steps=episode_time_steps,
    )
    return CarbonAwareWrapper(
        inner,
        carbon_quantile=carbon_quantile,
        price_low_quantile=price_low_quantile,
        conflict_quantile=conflict_quantile,
    )


def make_carbon_aware_vec_env(
    env_name: str = DEFAULT_SCHEMA,
    n_envs: int = 1,
    seed: Optional[int] = None,
    episode_time_steps: Optional[int] = None,
    carbon_quantile: float = DEFAULT_CA_CARBON_QUANTILE,
    price_low_quantile: float = DEFAULT_CA_PRICE_LOW_QUANTILE,
    conflict_quantile: float = DEFAULT_CA_CONFLICT_QUANTILE,
    **kwargs,
):
    env_kwargs = {
        "env_name": env_name,
        "episode_time_steps": episode_time_steps,
        "carbon_quantile": carbon_quantile,
        "price_low_quantile": price_low_quantile,
        "conflict_quantile": conflict_quantile,
    }
    return make_cost_vec_env(
        make_carbon_aware_env,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
    )


# ###########################################################################
# Scenario E — Peak Import Ratchet
# ###########################################################################

DEFAULT_PR_PEAK_TARGET_QUANTILE = 0.75
DEFAULT_PR_PEAK_CAP_MARGIN = 1.5
DEFAULT_PR_PRICE_LOW_QUANTILE = 0.60
DEFAULT_PR_CHARGE_HEADROOM_FRAC = 0.50


class PeakRatchetWrapper(CityLearnBaseWrapper):
    """CMDP: minimise electricity cost subject to peak import ratchet.

    The district pays a *peak demand charge* proportional to the highest
    hourly import during the episode.  Once a new peak is set, it never
    decreases within the episode — every subsequent hour below that peak
    is effectively "free" from cost.  This creates a **non-stationary
    decision boundary** that shifts as the episode progresses.

    Observation augmentation
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Appends one feature to the observation: normalised running peak
    (``running_peak / peak_cap``).  The agent must observe the current
    peak to decide whether importing at this step would set a new record.

    Cost — incremental peak penalty
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        district_import = max(Σ_i NEC_i, 0)
        if district_import > running_peak:
            running_peak = district_import
            cost = clamp((district_import − peak_target) / (peak_cap − peak_target), 0, 1)
        else:
            cost = 0

    Cost is only positive when a *new* peak above peak_target is set.
    Running peak resets to 0 on ``env.reset()``.

    Label — ratchet-aware charging trap
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        potential_import = district_nsl + charge_headroom_frac × total_charge_power
        label = 1   iff   price ≤ price_low_threshold
                          AND potential_import > max(running_peak, peak_target)

    Early in the episode when running_peak is low, many cheap + high-load
    hours trigger label=1 (any charging threatens a new peak).  Later,
    after a high-NSL spike has already set the peak, most hours fall below
    it and become label=0 — the reward policy can charge freely.

    **Why Split-RL wins**: CUP uses a fixed λ that cannot adapt to the
    evolving peak.  Its single multiplier over-constrains early (inhibiting
    profitable charging) and under-constrains late (not protecting against
    new peaks near the current record).  Split-RL partitions the state
    space based on {running_peak, nsl, price}, routing only the
    peak-threatening states to the cost policy.
    """

    def __init__(
        self,
        env,
        peak_target_quantile: float = DEFAULT_PR_PEAK_TARGET_QUANTILE,
        peak_cap_margin: float = DEFAULT_PR_PEAK_CAP_MARGIN,
        price_low_quantile: float = DEFAULT_PR_PRICE_LOW_QUANTILE,
        charge_headroom_frac: float = DEFAULT_PR_CHARGE_HEADROOM_FRAC,
    ):
        super().__init__(env)
        self._peak_target_quantile = peak_target_quantile
        self._peak_cap_margin = peak_cap_margin
        self._price_low_quantile = price_low_quantile
        self._charge_headroom_frac = charge_headroom_frac

        self._peak_target: Optional[float] = None
        self._peak_cap: Optional[float] = None
        self._price_low_threshold: Optional[float] = None
        self._total_charge_power: float = 0.0
        self._running_peak: float = 0.0

        # Extend observation space by 1 (normalised running peak)
        low = np.append(self.observation_space.low, np.float32(0.0))
        high = np.append(self.observation_space.high, np.float32(1.0))
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype,
        )

    def _scenario_init(self):
        cl_env = self.unwrapped
        if not (hasattr(cl_env, "buildings") and cl_env.buildings):
            return
        buildings = cl_env.buildings

        # District NSL schedule
        n_hours, hourly_totals = self._read_nsl_schedule(buildings)

        # Peak target and cap
        base = float(np.quantile(hourly_totals, self._peak_target_quantile))
        self._peak_target = base
        self._peak_cap = self._peak_cap_margin * base

        # Total fleet charge power
        for b in buildings:
            es = b.electrical_storage
            if (es is not None
                    and hasattr(es, 'nominal_power')
                    and es.nominal_power is not None):
                self._total_charge_power += es.nominal_power

        # Price threshold
        all_prices = []
        for b in buildings:
            if hasattr(b, "pricing") and b.pricing is not None:
                ep = b.pricing.electricity_pricing
                if hasattr(ep, "__len__") and len(ep) > 0:
                    all_prices.extend(ep[:n_hours])
        if all_prices:
            self._price_low_threshold = float(
                np.quantile(all_prices, self._price_low_quantile)
            )

    @staticmethod
    def _read_nsl_schedule(buildings):
        """Read full-horizon district NSL. Returns (n_hours, hourly_totals)."""
        n_hours = None
        for b in buildings:
            es = getattr(b, "energy_simulation", None)
            sched = getattr(es, "_non_shiftable_load", None)
            if sched is not None and hasattr(sched, "__len__") and len(sched) > 0:
                n_hours = len(sched) if n_hours is None else min(n_hours, len(sched))
        if n_hours is None or n_hours == 0:
            raise RuntimeError("PeakRatchetWrapper: cannot read NSL schedule.")
        hourly_totals = np.zeros(n_hours)
        for b in buildings:
            es = getattr(b, "energy_simulation", None)
            sched = getattr(es, "_non_shiftable_load", None)
            if sched is not None and hasattr(sched, "__len__") and len(sched) >= n_hours:
                hourly_totals += np.asarray(sched[:n_hours], dtype=float)
        return n_hours, hourly_totals

    # -- Helpers -------------------------------------------------------

    def _district_import_post_step(self) -> Optional[float]:
        total = 0.0
        n = 0
        for b in self._get_buildings():
            nec = self._at_timestep(b.net_electricity_consumption)
            if nec is not None:
                total += nec
                n += 1
        return max(total, 0.0) if n > 0 else None

    def _district_pre_step_nsl(self) -> Optional[float]:
        total = 0.0
        n = 0
        ts = self._get_time_step()
        for b in self._get_buildings():
            es = getattr(b, "energy_simulation", None)
            sched = getattr(es, "_non_shiftable_load", None)
            if sched is not None and hasattr(sched, "__len__") and ts < len(sched):
                total += float(sched[ts])
                n += 1
        return total if n > 0 else None

    def _augment_obs(self, obs):
        raw = (self._running_peak / self._peak_cap
               if self._peak_cap and self._peak_cap > 0 else 0.0)
        return np.append(obs, np.float32(min(1.0, raw)))

    # -- Cost ----------------------------------------------------------

    def _compute_cost(self, obs) -> float:
        self._ensure_base_init()
        if self._peak_target is None or self._peak_cap is None:
            return 0.0
        if self._peak_cap <= self._peak_target:
            return 0.0

        district_import = self._district_import_post_step()
        if district_import is None:
            return 0.0

        if district_import <= self._running_peak:
            return 0.0  # below current peak — free

        # New peak being set
        self._running_peak = district_import

        if district_import <= self._peak_target:
            return 0.0

        excess = (district_import - self._peak_target) / (
            self._peak_cap - self._peak_target
        )
        return float(min(1.0, excess))

    # -- Label ---------------------------------------------------------

    def _compute_label(self, obs) -> int:
        self._ensure_base_init()
        if self._price_low_threshold is None or self._peak_target is None:
            return 0

        buildings = self._get_buildings()
        price = self._max_pre_step_price(buildings)
        if price is None or price > self._price_low_threshold:
            return 0

        nsl = self._district_pre_step_nsl()
        if nsl is None:
            return 0

        # Would charging push import above the current ratchet level?
        potential = nsl + self._charge_headroom_frac * self._total_charge_power
        threshold = max(self._running_peak, self._peak_target)
        return 1 if potential > threshold else 0

    # -- Gymnasium API overrides (obs augmentation + ratchet reset) -----

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._ensure_base_init()
        self._running_peak = 0.0
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return self._augment_obs(obs), reward, terminated, truncated, info


# ###########################################################################
# Scenario F — Demand Response Events
# ###########################################################################

DEFAULT_DR_EVENT_STARTS: Tuple[int, ...] = (168, 360, 528)
DEFAULT_DR_EVENT_DURATION = 12
DEFAULT_DR_SOC_TARGET = 0.80
DEFAULT_DR_PREP_WINDOW = 24
DEFAULT_DR_HIGH_PRICE_QUANTILE = 0.60


class DemandResponseWrapper(CityLearnBaseWrapper):
    """CMDP: minimise electricity cost subject to demand response events.

    Pre-announced grid events require battery reserves.  The reward wants
    normal arbitrage; the cost penalises low SOC during events.

    Events are at fixed timesteps within the episode (deterministic,
    known to the agent through the augmented observation).

    Observation augmentation
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Appends one feature: normalised steps until next event
    (``steps_until / episode_length``).  1.0 = far away, 0.0 = imminent
    or currently in event.

    Cost — SOC deficit during events
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        if in_event_window:
            cost = clamp((soc_target − mean_soc) / soc_target, 0, 1)
        else:
            cost = 0

    Label — phase-aware pre-event conflict
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Three phases trigger label=1:

    1. **Prep + low SOC**: reserves below target during prep window.
       Cost branch must charge aggressively (conflicts with reward's
       normal arbitrage timing).
    2. **Prep + high price + adequate SOC**: reserves at target but
       expensive electricity tempts discharge.  Cost branch must hold.
    3. **In-event + low SOC**: event has started with insufficient
       reserves.  Cost branch manages recovery.

    **Why Split-RL wins**: CUP's fixed λ over-conserves far from events
    (suppressing profitable arbitrage) and may under-conserve near events.
    Split-RL routes pre-event high-price states to the cost policy and
    everything else to the reward policy.
    """

    def __init__(
        self,
        env,
        event_starts: Sequence[int] = DEFAULT_DR_EVENT_STARTS,
        event_duration: int = DEFAULT_DR_EVENT_DURATION,
        soc_target: float = DEFAULT_DR_SOC_TARGET,
        prep_window: int = DEFAULT_DR_PREP_WINDOW,
        high_price_quantile: float = DEFAULT_DR_HIGH_PRICE_QUANTILE,
    ):
        super().__init__(env)
        self._event_starts = tuple(sorted(event_starts))
        self._event_duration = event_duration
        self._soc_target = soc_target
        self._prep_window = prep_window
        self._high_price_quantile = high_price_quantile

        self._high_price_threshold: Optional[float] = None
        self._episode_length: int = 720  # updated in _scenario_init

        # Extend observation space (normalised steps-until-event)
        low = np.append(self.observation_space.low, np.float32(0.0))
        high = np.append(self.observation_space.high, np.float32(1.0))
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype,
        )

    def _scenario_init(self):
        cl_env = self.unwrapped
        if not (hasattr(cl_env, "buildings") and cl_env.buildings):
            return
        buildings = cl_env.buildings

        # Episode length from NSL schedule
        for b in buildings:
            es = getattr(b, "energy_simulation", None)
            sched = getattr(es, "_non_shiftable_load", None)
            if sched is not None and hasattr(sched, "__len__") and len(sched) > 0:
                self._episode_length = len(sched)
                break

        # High-price threshold
        all_prices = []
        for b in buildings:
            if hasattr(b, "pricing") and b.pricing is not None:
                ep = b.pricing.electricity_pricing
                if hasattr(ep, "__len__") and len(ep) > 0:
                    all_prices.extend(ep[:self._episode_length])
        if all_prices:
            self._high_price_threshold = float(
                np.quantile(all_prices, self._high_price_quantile)
            )

    # -- Helpers -------------------------------------------------------

    def _in_event_window(self, ts: int) -> bool:
        """True if timestep *ts* is inside any event window."""
        for start in self._event_starts:
            if start <= ts < start + self._event_duration:
                return True
        return False

    def _steps_until_next_event(self, ts: int) -> int:
        """Steps from *ts* to the start of the next event (0 if inside)."""
        for start in self._event_starts:
            if ts < start:
                return start - ts
            if start <= ts < start + self._event_duration:
                return 0  # currently in event
        return self._episode_length  # no more events

    def _mean_electrical_soc(self) -> Optional[float]:
        soc_values = []
        for b in self._get_buildings():
            if b.electrical_storage is not None:
                soc = self._at_timestep(b.electrical_storage.soc)
                if soc is not None:
                    soc_values.append(soc)
        return float(np.mean(soc_values)) if soc_values else None

    def _augment_obs(self, obs):
        ts = self._get_time_step()
        steps = self._steps_until_next_event(ts)
        norm = np.float32(
            min(1.0, steps / self._episode_length)
            if self._episode_length > 0 else 1.0
        )
        return np.append(obs, norm)

    # -- Cost ----------------------------------------------------------

    def _compute_cost(self, obs) -> float:
        self._ensure_base_init()
        ts = self._get_time_step() - 1  # post-step
        if not self._in_event_window(ts):
            return 0.0

        mean_soc = self._mean_electrical_soc()
        if mean_soc is None or mean_soc >= self._soc_target:
            return 0.0

        return float(min(1.0,
            (self._soc_target - mean_soc) / self._soc_target
        ))

    # -- Label ---------------------------------------------------------

    def _compute_label(self, obs) -> int:
        """Phase-aware label for demand response events.

        Three phases trigger label=1:

        1. **Prep + low SOC**: reserves below target during preparation
           window.  Cost branch must charge aggressively (conflicts with
           reward's normal arbitrage timing).
        2. **Prep + high price + adequate SOC**: reserves at target but
           expensive electricity tempts discharge.  Cost branch must hold.
        3. **In-event + low SOC**: post-event-start recovery when reserves
           were insufficient.  Cost branch manages deficit.
        """
        self._ensure_base_init()

        ts = self._get_time_step()  # pre-step
        steps_until = self._steps_until_next_event(ts)

        # Phase 1 & 2: Preparation window — cost branch owns reserves
        if 0 < steps_until <= self._prep_window:
            mean_soc = self._mean_electrical_soc()
            # Low SOC: must charge aggressively regardless of price
            if mean_soc is not None and mean_soc < self._soc_target:
                return 1
            # Adequate SOC + high price: hold reserves, don't discharge
            if self._high_price_threshold is not None:
                buildings = self._get_buildings()
                price = self._max_pre_step_price(buildings)
                if price is not None and price > self._high_price_threshold:
                    return 1
            return 0

        # Phase 3: In-event recovery — cost branch manages SOC deficit
        if steps_until == 0:
            mean_soc = self._mean_electrical_soc()
            if mean_soc is not None and mean_soc < self._soc_target:
                return 1

        return 0

    # -- Gymnasium API overrides (obs augmentation) --------------------

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._ensure_base_init()
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return self._augment_obs(obs), reward, terminated, truncated, info


# ###########################################################################
# Scenario G — Solar Ramp Reserve
# ###########################################################################

DEFAULT_SR_SOC_RESERVE = 0.50
DEFAULT_SR_PREP_HOURS = 4
DEFAULT_SR_BUFFER_HOURS = 4
DEFAULT_SR_HIGH_PRICE_QUANTILE = 0.60


class SolarRampReserveWrapper(CityLearnBaseWrapper):
    """CMDP: minimise electricity cost subject to post-sunset reserve events.

    Each day, solar generation fades to zero at sunset.  The district
    transitions from cheap PV power to expensive grid imports.  Battery
    reserves must bridge this daily "sunset transition" event.

    Unlike a broad-regime approach that penalises all solar hours, this
    scenario defines discrete, predictable daily events from the solar
    schedule:

    - **Pre-sunset prep window**: last ``prep_hours`` of positive solar
      before each daily sunset.  This is where the gradient conflict lives.
    - **Post-sunset buffer window**: first ``buffer_hours`` of zero solar
      after each sunset.  This is where the cost fires.

    Event detection (at init)
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    From the full solar schedule, a sunset transition is any hour *t+1*
    where ``solar[t] > 0`` and ``solar[t+1] == 0``.  Prep and buffer
    windows are anchored to each detected sunset.

    Observation augmentation
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Appends one feature: normalised hours until the next post-sunset
    buffer event (``hours_until / episode_length``).  1.0 = far away,
    0.0 = buffer event imminent or ongoing.

    Cost — reserve deficit during post-sunset buffer
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        if in_buffer_window:
            cost = clamp((soc_reserve − mean_soc) / soc_reserve, 0, 1)
        else:
            cost = 0

    Cost only fires during the specific post-sunset windows, not during
    all solar hours.  This makes the cost sparse and event-driven.

    Label — phase-aware sunset routing
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Three phases trigger label=1:

    1. **Pre-sunset + low SOC**: reserves below target during prep window.
       Cost branch must charge aggressively despite reward preferring
       normal arbitrage timing.
    2. **Pre-sunset + high price + adequate SOC**: reserves at target but
       expensive electricity tempts discharge.  Cost branch must hold.
    3. **In-buffer + low SOC**: post-sunset recovery when reserves were
       insufficient.  Cost branch manages deficit.

    **Why Split-RL wins**: CUP's fixed λ over-conserves during morning
    solar hours (no sunset threat) and under-conserves during late
    afternoon when the agent should be building reserves.  Split-RL
    routes only the sunset-adjacent conflict states to the cost policy,
    leaving the reward policy free to arbitrage during all other hours.
    """

    def __init__(
        self,
        env,
        soc_reserve: float = DEFAULT_SR_SOC_RESERVE,
        prep_hours: int = DEFAULT_SR_PREP_HOURS,
        buffer_hours: int = DEFAULT_SR_BUFFER_HOURS,
        high_price_quantile: float = DEFAULT_SR_HIGH_PRICE_QUANTILE,
    ):
        super().__init__(env)
        self._soc_reserve = soc_reserve
        self._prep_hours = prep_hours
        self._buffer_hours = buffer_hours
        self._high_price_quantile = high_price_quantile

        self._high_price_threshold: Optional[float] = None
        self._episode_length: int = 720
        # Sets of timesteps in each window type (populated in _scenario_init)
        self._prep_timesteps: set = set()
        self._buffer_timesteps: set = set()
        # Sorted list of buffer-start timesteps (for steps_until computation)
        self._buffer_starts: list = []

        # Extend observation space (normalised hours-until-buffer)
        low = np.append(self.observation_space.low, np.float32(0.0))
        high = np.append(self.observation_space.high, np.float32(1.0))
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype,
        )

    def _scenario_init(self):
        cl_env = self.unwrapped
        if not (hasattr(cl_env, "buildings") and cl_env.buildings):
            return
        buildings = cl_env.buildings

        # District solar schedule
        n_hours = None
        for b in buildings:
            es = getattr(b, "energy_simulation", None)
            sg = getattr(es, "_solar_generation", None)
            if sg is not None and hasattr(sg, "__len__") and len(sg) > 0:
                n_hours = len(sg) if n_hours is None else min(n_hours, len(sg))
        if n_hours is None or n_hours == 0:
            return
        self._episode_length = n_hours

        hourly_solar = np.zeros(n_hours)
        for b in buildings:
            es = getattr(b, "energy_simulation", None)
            sg = getattr(es, "_solar_generation", None)
            if sg is not None and hasattr(sg, "__len__") and len(sg) >= n_hours:
                hourly_solar += np.asarray(sg[:n_hours], dtype=float)

        # Detect daily sunset transitions: t+1 is first zero-solar hour
        sunsets = []
        for t in range(n_hours - 1):
            if hourly_solar[t] > 0 and hourly_solar[t + 1] <= 0:
                sunsets.append(t + 1)  # first zero-solar hour

        # Build prep and buffer windows around each sunset
        for sunset_ts in sunsets:
            # Prep: last prep_hours of positive solar before this sunset
            prep_start = max(0, sunset_ts - self._prep_hours)
            for h in range(prep_start, sunset_ts):
                self._prep_timesteps.add(h)
            # Buffer: first buffer_hours of zero solar after sunset
            buffer_end = min(n_hours, sunset_ts + self._buffer_hours)
            for h in range(sunset_ts, buffer_end):
                self._buffer_timesteps.add(h)
            self._buffer_starts.append(sunset_ts)

        self._buffer_starts.sort()

        # High-price threshold
        all_prices = []
        for b in buildings:
            if hasattr(b, "pricing") and b.pricing is not None:
                ep = b.pricing.electricity_pricing
                if hasattr(ep, "__len__") and len(ep) > 0:
                    all_prices.extend(ep[:n_hours])
        if all_prices:
            self._high_price_threshold = float(
                np.quantile(all_prices, self._high_price_quantile)
            )

    # -- Helpers -------------------------------------------------------

    def _steps_until_next_buffer(self, ts: int) -> int:
        """Steps from *ts* to the start of the next buffer window."""
        if ts in self._buffer_timesteps:
            return 0  # currently in buffer
        for bs in self._buffer_starts:
            if ts < bs:
                return bs - ts
        return self._episode_length  # no more buffers

    def _mean_electrical_soc(self) -> Optional[float]:
        soc_values = []
        for b in self._get_buildings():
            if b.electrical_storage is not None:
                soc = self._at_timestep(b.electrical_storage.soc)
                if soc is not None:
                    soc_values.append(soc)
        return float(np.mean(soc_values)) if soc_values else None

    def _augment_obs(self, obs):
        ts = self._get_time_step()
        steps = self._steps_until_next_buffer(ts)
        norm = np.float32(
            min(1.0, steps / self._episode_length)
            if self._episode_length > 0 else 1.0
        )
        return np.append(obs, norm)

    # -- Cost ----------------------------------------------------------

    def _compute_cost(self, obs) -> float:
        self._ensure_base_init()
        ts = self._get_time_step() - 1  # post-step
        if ts not in self._buffer_timesteps:
            return 0.0

        mean_soc = self._mean_electrical_soc()
        if mean_soc is None or mean_soc >= self._soc_reserve:
            return 0.0

        return float(min(1.0,
            (self._soc_reserve - mean_soc) / self._soc_reserve
        ))

    # -- Label ---------------------------------------------------------

    def _compute_label(self, obs) -> int:
        """Phase-aware label for solar ramp reserve events.

        1. Pre-sunset + low SOC → label=1 (charge aggressively).
        2. Pre-sunset + high price + adequate SOC → label=1 (hold).
        3. In-buffer + low SOC → label=1 (recovery).
        """
        self._ensure_base_init()
        ts = self._get_time_step()  # pre-step

        # Phase 1 & 2: Pre-sunset preparation
        if ts in self._prep_timesteps:
            mean_soc = self._mean_electrical_soc()
            # Low SOC: must charge aggressively
            if mean_soc is not None and mean_soc < self._soc_reserve:
                return 1
            # Adequate SOC + high price: hold reserves, don't discharge
            if self._high_price_threshold is not None:
                buildings = self._get_buildings()
                price = self._max_pre_step_price(buildings)
                if price is not None and price > self._high_price_threshold:
                    return 1
            return 0

        # Phase 3: In-buffer recovery
        if ts in self._buffer_timesteps:
            mean_soc = self._mean_electrical_soc()
            if mean_soc is not None and mean_soc < self._soc_reserve:
                return 1

        return 0

    # -- Gymnasium API overrides (obs augmentation) --------------------

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._ensure_base_init()
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return self._augment_obs(obs), reward, terminated, truncated, info


# ###########################################################################
# Factory helpers — new scenarios
# ###########################################################################

# -- Peak Ratchet -------------------------------------------------------

def make_peak_ratchet_env(
    env_name: str = DEFAULT_SCHEMA,
    episode_time_steps: Optional[int] = None,
    peak_target_quantile: float = DEFAULT_PR_PEAK_TARGET_QUANTILE,
    peak_cap_margin: float = DEFAULT_PR_PEAK_CAP_MARGIN,
    price_low_quantile: float = DEFAULT_PR_PRICE_LOW_QUANTILE,
    charge_headroom_frac: float = DEFAULT_PR_CHARGE_HEADROOM_FRAC,
    **kwargs,
):
    inner = make_citylearn_inner_env(
        schema=env_name, episode_time_steps=episode_time_steps,
    )
    return PeakRatchetWrapper(
        inner,
        peak_target_quantile=peak_target_quantile,
        peak_cap_margin=peak_cap_margin,
        price_low_quantile=price_low_quantile,
        charge_headroom_frac=charge_headroom_frac,
    )


def make_peak_ratchet_vec_env(
    env_name: str = DEFAULT_SCHEMA,
    n_envs: int = 1,
    seed: Optional[int] = None,
    episode_time_steps: Optional[int] = None,
    peak_target_quantile: float = DEFAULT_PR_PEAK_TARGET_QUANTILE,
    peak_cap_margin: float = DEFAULT_PR_PEAK_CAP_MARGIN,
    price_low_quantile: float = DEFAULT_PR_PRICE_LOW_QUANTILE,
    charge_headroom_frac: float = DEFAULT_PR_CHARGE_HEADROOM_FRAC,
    **kwargs,
):
    env_kwargs = {
        "env_name": env_name,
        "episode_time_steps": episode_time_steps,
        "peak_target_quantile": peak_target_quantile,
        "peak_cap_margin": peak_cap_margin,
        "price_low_quantile": price_low_quantile,
        "charge_headroom_frac": charge_headroom_frac,
    }
    return make_cost_vec_env(
        make_peak_ratchet_env,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
    )


# -- Demand Response ---------------------------------------------------

def make_demand_response_env(
    env_name: str = DEFAULT_SCHEMA,
    episode_time_steps: Optional[int] = None,
    event_starts: Sequence[int] = DEFAULT_DR_EVENT_STARTS,
    event_duration: int = DEFAULT_DR_EVENT_DURATION,
    soc_target: float = DEFAULT_DR_SOC_TARGET,
    prep_window: int = DEFAULT_DR_PREP_WINDOW,
    high_price_quantile: float = DEFAULT_DR_HIGH_PRICE_QUANTILE,
    **kwargs,
):
    inner = make_citylearn_inner_env(
        schema=env_name, episode_time_steps=episode_time_steps,
    )
    return DemandResponseWrapper(
        inner,
        event_starts=event_starts,
        event_duration=event_duration,
        soc_target=soc_target,
        prep_window=prep_window,
        high_price_quantile=high_price_quantile,
    )


def make_demand_response_vec_env(
    env_name: str = DEFAULT_SCHEMA,
    n_envs: int = 1,
    seed: Optional[int] = None,
    episode_time_steps: Optional[int] = None,
    event_starts: Sequence[int] = DEFAULT_DR_EVENT_STARTS,
    event_duration: int = DEFAULT_DR_EVENT_DURATION,
    soc_target: float = DEFAULT_DR_SOC_TARGET,
    prep_window: int = DEFAULT_DR_PREP_WINDOW,
    high_price_quantile: float = DEFAULT_DR_HIGH_PRICE_QUANTILE,
    **kwargs,
):
    env_kwargs = {
        "env_name": env_name,
        "episode_time_steps": episode_time_steps,
        "event_starts": event_starts,
        "event_duration": event_duration,
        "soc_target": soc_target,
        "prep_window": prep_window,
        "high_price_quantile": high_price_quantile,
    }
    return make_cost_vec_env(
        make_demand_response_env,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
    )


# -- Solar Ramp Reserve ------------------------------------------------

def make_solar_ramp_reserve_env(
    env_name: str = DEFAULT_SCHEMA,
    episode_time_steps: Optional[int] = None,
    soc_reserve: float = DEFAULT_SR_SOC_RESERVE,
    prep_hours: int = DEFAULT_SR_PREP_HOURS,
    buffer_hours: int = DEFAULT_SR_BUFFER_HOURS,
    high_price_quantile: float = DEFAULT_SR_HIGH_PRICE_QUANTILE,
    **kwargs,
):
    inner = make_citylearn_inner_env(
        schema=env_name, episode_time_steps=episode_time_steps,
    )
    return SolarRampReserveWrapper(
        inner,
        soc_reserve=soc_reserve,
        prep_hours=prep_hours,
        buffer_hours=buffer_hours,
        high_price_quantile=high_price_quantile,
    )


def make_solar_ramp_reserve_vec_env(
    env_name: str = DEFAULT_SCHEMA,
    n_envs: int = 1,
    seed: Optional[int] = None,
    episode_time_steps: Optional[int] = None,
    soc_reserve: float = DEFAULT_SR_SOC_RESERVE,
    prep_hours: int = DEFAULT_SR_PREP_HOURS,
    buffer_hours: int = DEFAULT_SR_BUFFER_HOURS,
    high_price_quantile: float = DEFAULT_SR_HIGH_PRICE_QUANTILE,
    **kwargs,
):
    env_kwargs = {
        "env_name": env_name,
        "episode_time_steps": episode_time_steps,
        "soc_reserve": soc_reserve,
        "prep_hours": prep_hours,
        "buffer_hours": buffer_hours,
        "high_price_quantile": high_price_quantile,
    }
    return make_cost_vec_env(
        make_solar_ramp_reserve_env,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
    )


# Backward-compatible aliases (old name → new)
make_solar_contingency_env = make_solar_ramp_reserve_env
make_solar_contingency_vec_env = make_solar_ramp_reserve_vec_env
