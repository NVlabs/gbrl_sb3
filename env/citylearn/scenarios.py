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
A. **Cost vs Comfort** — electricity cost vs thermal comfort
B. **Arbitrage vs Buffer** — electricity arbitrage vs storage resilience
C. **Contract Demand** — electricity cost vs district import cap
D. **Carbon Aware** — electricity cost vs carbon emissions
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


# ###########################################################################
# Scenario A — Cost vs Comfort
# ###########################################################################

# Defaults
DEFAULT_CVC_COMFORT_BAND = 1.0       # °C tolerance around each setpoint
DEFAULT_CVC_COMFORT_MAX_DELTA = 5.0  # °C beyond band that maps to cost = 1.0
DEFAULT_CVC_PRICE_QUANTILE = 0.70    # price above this quantile → "high price"
DEFAULT_CVC_HEADROOM_THRESH = 0.5    # °C: headroom below this → "low thermal slack"


class CostVsComfortWrapper(CityLearnBaseWrapper):
    """CMDP: minimise electricity cost subject to comfort constraint.

    Cost — worst-case thermal discomfort
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For each building the comfort zone is the two-sided band

        [heating_sp − band,   cooling_sp + band]

    where *band* is ``comfort_band`` (default 1 °C).  If only a cooling
    setpoint exists for a given building, only the upper bound is enforced.

    Violation for building *i*:

        v_i  =  max(0, T_in − upper) + max(0, lower − T_in)

    Cost  =  max_i  clamp(v_i / max_delta,  0,  1)

    Using max (not mean) avoids dilution when only one building violates.

    Label — objective routing (label=1 → cost, label=0 → reward)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Two zones route to cost:

    1. **Already violating** — any building has headroom ≤ 0 (outside
       comfort band) → label=1.  Cost should drive recovery.
    2. **Pre-violation frontier** — high price + a building is still
       comfortable but has headroom < ``headroom_thresh`` → label=1.

    No storage gate — the conflict between saving electricity and
    preserving comfort exists regardless of battery SOC because cooling
    control is the primary lever.
    """

    def __init__(
        self,
        env,
        comfort_band: float = DEFAULT_CVC_COMFORT_BAND,
        comfort_max_delta: float = DEFAULT_CVC_COMFORT_MAX_DELTA,
        price_quantile: float = DEFAULT_CVC_PRICE_QUANTILE,
        headroom_thresh: float = DEFAULT_CVC_HEADROOM_THRESH,
    ):
        super().__init__(env)
        self._comfort_band = comfort_band
        self._comfort_max_delta = comfort_max_delta
        self._price_quantile = price_quantile
        self._headroom_thresh = headroom_thresh
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

    # -- Per-building comfort helpers ----------------------------------

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

    # -- Cost ----------------------------------------------------------

    def _compute_cost(self, obs) -> float:
        """Max discomfort across buildings ∈ [0, 1]."""
        self._ensure_base_init()
        buildings = self._get_buildings()
        if not buildings:
            return 0.0
        return float(max(self._building_discomfort(b) for b in buildings))

    # -- Label ---------------------------------------------------------

    def _compute_label(self, obs) -> int:
        self._ensure_base_init()
        buildings = self._get_buildings()
        if not buildings:
            return 0

        # Zone 1: any building already violating comfort band
        for b in buildings:
            if self._building_headroom(b) <= 0:
                return 1

        # Zone 2: pre-violation frontier — high price + low headroom
        if self._price_threshold is None:
            return 0
        max_price = self._max_pre_step_price(buildings)
        if max_price is None or max_price <= self._price_threshold:
            return 0

        for b in buildings:
            headroom = self._building_headroom(b)
            if 0 < headroom < self._headroom_thresh:
                return 1

        return 0


# ###########################################################################
# Scenario B — Arbitrage vs Buffer
# ###########################################################################

DEFAULT_AVB_PRICE_QUANTILE = 0.70
DEFAULT_AVB_SOC_USABLE_THRESH = 0.15
DEFAULT_AVB_SOC_SAFETY_LEVEL = 0.30
DEFAULT_AVB_SOC_FRONTIER_BAND = 0.10


class ArbitrageVsBufferWrapper(CityLearnBaseWrapper):
    """CMDP: minimise electricity cost subject to buffer-resilience constraint.

    Cost — storage buffer depletion
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    When the mean electrical-storage SOC across buildings drops below
    ``soc_safety_level``, cost rises linearly to 1.0 at SOC = 0:

        cost  =  clamp((safety − mean_soc) / safety,  0,  1)

    Label — objective routing (label=1 → cost, label=0 → reward)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1. SOC > safety + band → label=0 (reward owns).
    2. safety ≤ SOC ≤ safety + band AND high price + dischargeable storage
       → label=1 (frontier).
    3. SOC < safety with storage → label=1 (recovery; lever is *charge*).

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

    def _mean_electrical_soc(self) -> Optional[float]:
        """Mean electrical-storage SOC across buildings that have batteries.

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
        return float(np.mean(soc_values)) if soc_values else None

    # -- Cost ----------------------------------------------------------

    def _compute_cost(self, obs) -> float:
        """Buffer depletion cost ∈ [0, 1]."""
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

    # -- Label ---------------------------------------------------------

    def _compute_label(self, obs) -> int:
        self._ensure_base_init()
        buildings = self._get_buildings()
        if not buildings:
            return 0

        mean_soc = self._mean_electrical_soc()
        if mean_soc is None:
            return 0

        has_storage = False
        has_dischargeable = False
        for b in buildings:
            if b.electrical_storage is not None:
                has_storage = True
                soc = self._at_timestep(b.electrical_storage.soc)
                if soc is not None and soc > self._soc_usable_thresh:
                    has_dischargeable = True

        # Recovery regime: SOC below safety.  Lever is *charge*, so
        # only require that storage exists (not that it is dischargeable).
        if mean_soc < self._soc_safety_level:
            return 1 if has_storage else 0

        # Frontier zone: near safety + high price.  Lever is *discharge*,
        # so require dischargeable energy.
        if not has_dischargeable:
            return 0
        if self._price_threshold is None:
            return 0
        max_price = self._max_pre_step_price(buildings)
        if max_price is None or max_price <= self._price_threshold:
            return 0

        frontier_upper = self._soc_safety_level + self._soc_frontier_band
        if mean_soc <= frontier_upper:
            return 1

        return 0


# ###########################################################################
# Scenario C — Contract Demand (replaces Peak Shaving)
# ###########################################################################

DEFAULT_CD_CAP_QUANTILE = 0.95
DEFAULT_CD_CAP_MARGIN = 2.5
DEFAULT_CD_FRONTIER_FRAC = 0.90
DEFAULT_CD_NSL_WARNING_QUANTILE = 0.90


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

    Label — action-contingent peak-shave frontier
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Two-zone danger label gated by battery usability:

    * **Warning zone**: current district NSL ≥ ``nsl_warning_quantile``-th
      percentile of the full-horizon NSL schedule.
    * **Recovery zone**: previous-step district import ≥ frontier.

    Either zone triggers label = 1 **only if** at least one battery has
    SOC above a minimum discharge threshold (5%).  Empty batteries mean
    the agent has no lever — label = 0 avoids misleading the policy.
    """

    def __init__(
        self,
        env,
        cap_quantile: float = DEFAULT_CD_CAP_QUANTILE,
        cap_margin: float = DEFAULT_CD_CAP_MARGIN,
        frontier_frac: float = DEFAULT_CD_FRONTIER_FRAC,
        nsl_warning_quantile: float = DEFAULT_CD_NSL_WARNING_QUANTILE,
    ):
        super().__init__(env)
        self._cap_quantile = cap_quantile
        self._cap_margin = cap_margin
        self._frontier_frac = frontier_frac
        self._nsl_warning_quantile = nsl_warning_quantile

        self._cap: Optional[float] = None
        self._frontier: Optional[float] = None
        self._nsl_warning_threshold: Optional[float] = None

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

        # NSL warning threshold from the same district total schedule
        self._nsl_warning_threshold = float(
            np.quantile(hourly_totals, self._nsl_warning_quantile)
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

    def _district_import_prev_step(self) -> Optional[float]:
        """District positive import from the previous committed step.

        At pre-step time, ``time_step`` has NOT been incremented, so
        ``_at_timestep`` reads ``series[ts-1]`` = the last committed
        transition.  This is the most recent *observable* district
        import — action-dependent but already committed.
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

    def _district_current_nsl(self) -> Optional[float]:
        """District non-shiftable load at the current pre-step timestep.

        Uses ``_latest()`` on the live-appended NSL series so it reads
        the most recent exogenous value *before* the action is applied.
        """
        total = 0.0
        n = 0
        for b in self._get_buildings():
            nsl = getattr(b, "non_shiftable_load", None)
            if nsl is not None and hasattr(nsl, "__len__") and len(nsl) > 0:
                val = self._latest(nsl)
                if val is not None:
                    total += val
                    n += 1
        if n == 0:
            return None
        return total

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

    # -- Label ---------------------------------------------------------

    def _compute_label(self, obs) -> int:
        self._ensure_base_init()
        if self._frontier is None:
            return 0

        buildings = self._get_buildings()
        if not buildings:
            return 0

        # Zone 1 — warning: exogenous NSL is high (early predictor)
        in_warning = False
        if self._nsl_warning_threshold is not None:
            current_nsl = self._district_current_nsl()
            if current_nsl is not None and current_nsl >= self._nsl_warning_threshold:
                in_warning = True

        # Zone 2 — recovery: already near/above capacity frontier
        in_recovery = False
        prev_import = self._district_import_prev_step()
        if prev_import is not None and prev_import >= self._frontier:
            in_recovery = True

        if not (in_warning or in_recovery):
            return 0

        # Battery must have usable charge (lever: discharge to reduce import)
        min_soc_frac = 0.05
        has_usable_battery = False
        for b in buildings:
            es = b.electrical_storage
            if es is None or es.capacity <= 0:
                continue
            ts = self._get_time_step()
            if ts > 0 and hasattr(es.soc, '__len__') and ts - 1 < len(es.soc):
                soc_frac = es.soc[ts - 1] / es.capacity
                if soc_frac >= min_soc_frac:
                    has_usable_battery = True
                    break
        return 1 if has_usable_battery else 0


# ###########################################################################
# Scenario D — Carbon Aware
# ###########################################################################

DEFAULT_CA_CARBON_QUANTILE = 0.70


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

    Label — action-contingent dirty-grid frontier
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    All three conditions must hold:

    1. Carbon intensity above ``carbon_quantile``-th percentile (dirty grid).
    2. District NSL above median — high import means the agent's battery
       action materially affects cost.  Low-NSL dirty hours have near-zero
       cost regardless of action (label=0 avoids misleading the policy).
    3. At least one battery has SOC ≥ 5% (the actual discharge lever).

    Low NSL or empty battery → label = 0.
    """

    def __init__(
        self,
        env,
        carbon_quantile: float = DEFAULT_CA_CARBON_QUANTILE,
    ):
        super().__init__(env)
        self._carbon_quantile = carbon_quantile

        self._carbon_threshold: Optional[float] = None
        self._carbon_min: Optional[float] = None
        self._carbon_max: Optional[float] = None
        self._import_scale: Optional[float] = None
        self._nsl_median: Optional[float] = None

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
        self._nsl_median = float(np.median(hourly_nsl))

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
        if not buildings:
            return 0

        # Condition 1: high carbon (danger region)
        if self._carbon_threshold is None:
            return 0
        max_carbon = self._max_pre_step_carbon(buildings)
        if max_carbon is None or max_carbon <= self._carbon_threshold:
            return 0

        # Condition 2: district NSL above median (import is material)
        if self._nsl_median is not None:
            nsl = self._district_pre_step_nsl()
            if nsl is not None and nsl < self._nsl_median:
                return 0

        # Condition 3: battery has usable charge (discharge lever)
        min_soc_frac = 0.05
        has_usable_battery = False
        ts = self._get_time_step()
        for b in buildings:
            es = b.electrical_storage
            if es is None or es.capacity <= 0:
                continue
            if ts > 0 and hasattr(es.soc, '__len__') and ts - 1 < len(es.soc):
                soc_frac = es.soc[ts - 1] / es.capacity
                if soc_frac >= min_soc_frac:
                    has_usable_battery = True
                    break
        return 1 if has_usable_battery else 0


# ###########################################################################
# Factory helpers
# ###########################################################################

# -- Cost vs Comfort ---------------------------------------------------

def make_cost_vs_comfort_env(
    env_name: str = DEFAULT_SCHEMA,
    episode_time_steps: Optional[int] = None,
    comfort_band: float = DEFAULT_CVC_COMFORT_BAND,
    comfort_max_delta: float = DEFAULT_CVC_COMFORT_MAX_DELTA,
    price_quantile: float = DEFAULT_CVC_PRICE_QUANTILE,
    headroom_thresh: float = DEFAULT_CVC_HEADROOM_THRESH,
    **kwargs,
):
    inner = make_citylearn_inner_env(
        schema=env_name, episode_time_steps=episode_time_steps,
    )
    return CostVsComfortWrapper(
        inner,
        comfort_band=comfort_band,
        comfort_max_delta=comfort_max_delta,
        price_quantile=price_quantile,
        headroom_thresh=headroom_thresh,
    )


def make_cost_vs_comfort_vec_env(
    env_name: str = DEFAULT_SCHEMA,
    n_envs: int = 1,
    seed: Optional[int] = None,
    episode_time_steps: Optional[int] = None,
    comfort_band: float = DEFAULT_CVC_COMFORT_BAND,
    comfort_max_delta: float = DEFAULT_CVC_COMFORT_MAX_DELTA,
    price_quantile: float = DEFAULT_CVC_PRICE_QUANTILE,
    headroom_thresh: float = DEFAULT_CVC_HEADROOM_THRESH,
    **kwargs,
):
    env_kwargs = {
        "env_name": env_name,
        "episode_time_steps": episode_time_steps,
        "comfort_band": comfort_band,
        "comfort_max_delta": comfort_max_delta,
        "price_quantile": price_quantile,
        "headroom_thresh": headroom_thresh,
    }
    return make_cost_vec_env(
        make_cost_vs_comfort_env,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
    )


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
    nsl_warning_quantile: float = DEFAULT_CD_NSL_WARNING_QUANTILE,
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
        nsl_warning_quantile=nsl_warning_quantile,
    )


def make_contract_demand_vec_env(
    env_name: str = DEFAULT_SCHEMA,
    n_envs: int = 1,
    seed: Optional[int] = None,
    episode_time_steps: Optional[int] = None,
    cap_quantile: float = DEFAULT_CD_CAP_QUANTILE,
    cap_margin: float = DEFAULT_CD_CAP_MARGIN,
    frontier_frac: float = DEFAULT_CD_FRONTIER_FRAC,
    nsl_warning_quantile: float = DEFAULT_CD_NSL_WARNING_QUANTILE,
    **kwargs,
):
    env_kwargs = {
        "env_name": env_name,
        "episode_time_steps": episode_time_steps,
        "cap_quantile": cap_quantile,
        "cap_margin": cap_margin,
        "frontier_frac": frontier_frac,
        "nsl_warning_quantile": nsl_warning_quantile,
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
    **kwargs,
):
    inner = make_citylearn_inner_env(
        schema=env_name, episode_time_steps=episode_time_steps,
    )
    return CarbonAwareWrapper(
        inner,
        carbon_quantile=carbon_quantile,
    )


def make_carbon_aware_vec_env(
    env_name: str = DEFAULT_SCHEMA,
    n_envs: int = 1,
    seed: Optional[int] = None,
    episode_time_steps: Optional[int] = None,
    carbon_quantile: float = DEFAULT_CA_CARBON_QUANTILE,
    **kwargs,
):
    env_kwargs = {
        "env_name": env_name,
        "episode_time_steps": episode_time_steps,
        "carbon_quantile": carbon_quantile,
    }
    return make_cost_vec_env(
        make_carbon_aware_env,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
    )
