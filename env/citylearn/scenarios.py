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

DEFAULT_AVB_SOC_SAFETY_LEVEL = 0.30
DEFAULT_AVB_PEAK_PRICE_QUANTILE = 0.92
DEFAULT_AVB_PREP_HOURS = 6
DEFAULT_AVB_HIGH_PRICE_QUANTILE = 0.75
DEFAULT_AVB_EMERGENCY_SOC = 0.20


class ArbitrageVsBufferWrapper(CityLearnBaseWrapper):
    """CMDP: minimise electricity cost subject to buffer-resilience constraint.

    Event-driven design around **peak-price periods** detected from the
    electricity pricing schedule.  During on-peak hours the reward wants
    to discharge (sell expensive) while the cost wants to hold reserves.

    Event detection (at init)
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    Peak-price periods are contiguous runs of hours where price exceeds
    the ``peak_price_quantile``-th percentile.  A prep window of
    ``prep_hours`` precedes each peak period.

    Observation augmentation
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Appends one feature: normalised steps until the next peak event
    (``steps_until / episode_length``).  1.0 = far away, 0.0 = imminent
    or currently in peak.

    Cost — battery depletion
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        cost = clamp((safety − mean_soc) / safety, 0, 1)

    Cost fires at every timestep.  All algorithms (CUP, CPO, etc.) see
    the same always-on constraint signal.

    Label — hard-routing state partition
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    label=1 routes only in-peak states to the cost branch:

    1. **In-peak + high price** (frontier): sell temptation during
       peak — cost branch holds reserves.  The high-price threshold
       is Q(0.75) of *peak-hour* prices (not global), so it is
       discriminative within peak periods.
    2. **In-peak + emergency low SOC** (backstop): reserves critically
       depleted during peak — cost branch owns corrective control.

    Prep-window labels and mild-SOC recovery are removed to keep
    label density low and stable (~3-4%).  Off-peak states are
    always label=0: reward branch owns freely.
    """

    def __init__(
        self,
        env,
        soc_safety_level: float = DEFAULT_AVB_SOC_SAFETY_LEVEL,
        peak_price_quantile: float = DEFAULT_AVB_PEAK_PRICE_QUANTILE,
        prep_hours: int = DEFAULT_AVB_PREP_HOURS,
        high_price_quantile: float = DEFAULT_AVB_HIGH_PRICE_QUANTILE,
        emergency_soc: float = DEFAULT_AVB_EMERGENCY_SOC,
    ):
        super().__init__(env)
        self._soc_safety_level = soc_safety_level
        self._peak_price_quantile = peak_price_quantile
        self._emergency_soc = emergency_soc
        self._prep_hours = prep_hours
        self._high_price_quantile = high_price_quantile

        self._high_price_threshold: Optional[float] = None
        self._episode_length: int = 720
        self._peak_timesteps: set = set()
        self._prep_timesteps: set = set()
        self._peak_starts: list = []

        # Extend observation space by 1 (normalised steps-until-peak)
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

        # Build max-price series across buildings (two-pass for safety)
        raw_arrays = []
        for b in buildings:
            if hasattr(b, "pricing") and b.pricing is not None:
                ep = b.pricing.electricity_pricing
                if hasattr(ep, "__len__") and len(ep) > 0:
                    raw_arrays.append(np.asarray(ep, dtype=float))
        if not raw_arrays:
            return
        n_hours = min(len(a) for a in raw_arrays)
        if n_hours == 0:
            return
        price_arrays = [a[:n_hours] for a in raw_arrays]
        self._episode_length = n_hours
        max_price_series = np.max(price_arrays, axis=0)

        # Peak-price threshold
        peak_threshold = float(np.quantile(max_price_series, self._peak_price_quantile))

        # Detect peak-price periods as contiguous runs at or above threshold.
        # Using >= is robust to discrete pricing where the quantile can land
        # exactly on the top tier (> would yield zero peaks in that case).
        is_peak = max_price_series >= peak_threshold
        in_run = False
        run_start = 0
        peak_runs = []  # list of (start, end) tuples
        for t in range(n_hours):
            if is_peak[t] and not in_run:
                run_start = t
                in_run = True
            elif not is_peak[t] and in_run:
                peak_runs.append((run_start, t))
                in_run = False
        if in_run:
            peak_runs.append((run_start, n_hours))

        # Build peak and prep timestep sets
        for start, end in peak_runs:
            for h in range(start, end):
                self._peak_timesteps.add(h)
            self._peak_starts.append(start)
            prep_start = max(0, start - self._prep_hours)
            for h in range(prep_start, start):
                # Don't mark prep hours that overlap with a previous peak
                if h not in self._peak_timesteps:
                    self._prep_timesteps.add(h)
        self._peak_starts.sort()

        # High-price threshold from *peak-hour* prices only.
        # Global Q(0.75) is trivially exceeded by peak hours (which are
        # Q(0.92)+), collapsing "in_peak + high_price" to just "in_peak".
        # Computing from the peak-hour sub-distribution makes the frontier
        # discriminative within peak periods.
        peak_indices = sorted(self._peak_timesteps)
        if peak_indices:
            peak_prices = max_price_series[peak_indices]
            self._high_price_threshold = float(
                np.quantile(peak_prices, self._high_price_quantile)
            )

    # -- Helpers -------------------------------------------------------

    def _mean_electrical_soc(self) -> Optional[float]:
        """Mean electrical-storage SOC across buildings (post-step)."""
        soc_values = []
        for b in self._get_buildings():
            if b.electrical_storage is not None:
                soc = self._at_timestep(b.electrical_storage.soc)
                if soc is not None:
                    soc_values.append(soc)
        return float(np.mean(soc_values)) if soc_values else None

    def _mean_electrical_soc_pre_step(self) -> Optional[float]:
        """Mean electrical-storage SOC aligned with current observation (pre-step)."""
        soc_values = []
        for b in self._get_buildings():
            if b.electrical_storage is not None:
                soc = self._at_pre_step(b.electrical_storage.soc)
                if soc is not None:
                    soc_values.append(soc)
        return float(np.mean(soc_values)) if soc_values else None

    def _steps_until_next_peak(self, ts: int) -> int:
        """Steps from *ts* to the start of the next peak period."""
        if ts in self._peak_timesteps:
            return 0
        for ps in self._peak_starts:
            if ts < ps:
                return ps - ts
        return self._episode_length

    def _augment_obs(self, obs):
        ts = self._get_time_step()
        steps = self._steps_until_next_peak(ts)
        norm = np.float32(
            min(1.0, steps / self._episode_length)
            if self._episode_length > 0 else 1.0
        )
        return np.append(obs, norm)

    # -- Cost ----------------------------------------------------------

    def _compute_cost(self, obs) -> float:
        """Battery depletion cost ∈ [0, 1]."""
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
        """Hard-routing label: cost branch owns in-peak states only.

        Frontier: in-peak + high price → cost branch holds.
        Emergency: in-peak + critically low SOC → cost branch recovers.
        """
        self._ensure_base_init()
        ts = self._get_time_step()  # pre-step

        if ts not in self._peak_timesteps:
            return 0

        # Frontier: high price sell temptation during peak
        if self._high_price_threshold is not None:
            buildings = self._get_buildings()
            price = self._max_pre_step_price(buildings)
            if price is not None and price > self._high_price_threshold:
                return 1

        # Emergency backstop: critically low SOC during peak
        mean_soc = self._mean_electrical_soc_pre_step()
        if mean_soc is not None and mean_soc < self._emergency_soc:
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
# Scenario C — Contract Demand (replaces Peak Shaving)
# ###########################################################################

DEFAULT_CD_CAP_QUANTILE = 0.95
DEFAULT_CD_CAP_MARGIN = 2.5
DEFAULT_CD_FRONTIER_FRAC = 0.90
DEFAULT_CD_NSL_EVENT_QUANTILE = 0.80
DEFAULT_CD_SOC_TARGET = 0.50
DEFAULT_CD_PREP_HOURS = 6
DEFAULT_CD_HIGH_PRICE_QUANTILE = 0.60
DEFAULT_CD_LOW_PRICE_QUANTILE = 0.25


class ContractDemandWrapper(CityLearnBaseWrapper):
    """CMDP: minimise electricity cost subject to district import cap.

    Event-driven design around **congestion periods** detected from the
    non-shiftable-load (NSL) schedule.  During high-demand hours the
    district grid is stressed; battery charging adds to import and
    threatens the contracted capacity limit.

    Event detection (at init)
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    Congestion events are contiguous runs of hours where district NSL
    exceeds the ``nsl_event_quantile``-th percentile.  A prep window of
    ``prep_hours`` precedes each congestion event.

    Observation augmentation
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Appends one feature: normalised steps until the next congestion event
    (``steps_until / episode_length``).  1.0 = far away, 0.0 = imminent
    or currently in congestion.

    Cap / frontier (at init, for cost)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        district_nsl[t] = Σ_i  non_shiftable_load_i[t]
        cap = cap_margin × quantile(district_nsl, cap_quantile)
        frontier = frontier_frac × cap

    Cost — soft capacity-stress penalty (unchanged)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        district_import = max(Σ_i NEC_i, 0)
        cost = clamp((district_import − frontier) / (cap − frontier), 0, 1)

    Label — congestion + short prep indicator
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        label = 1   iff   timestep ∈ congestion_window
                          OR within 3 hours before congestion start

    During high-demand congestion hours and the 3-hour preparation
    window before each event, battery charging increases district
    import and risks exceeding the contracted capacity.  The safety
    policy owns these timesteps; the reward policy owns all other
    hours.  Both windows are pre-computed from the known NSL
    schedule (observable state) and are trajectory-independent.
    """

    def __init__(
        self,
        env,
        cap_quantile: float = DEFAULT_CD_CAP_QUANTILE,
        cap_margin: float = DEFAULT_CD_CAP_MARGIN,
        frontier_frac: float = DEFAULT_CD_FRONTIER_FRAC,
        nsl_event_quantile: float = DEFAULT_CD_NSL_EVENT_QUANTILE,
        soc_target: float = DEFAULT_CD_SOC_TARGET,
        prep_hours: int = DEFAULT_CD_PREP_HOURS,
        high_price_quantile: float = DEFAULT_CD_HIGH_PRICE_QUANTILE,
        low_price_quantile: float = DEFAULT_CD_LOW_PRICE_QUANTILE,
    ):
        super().__init__(env)
        self._cap_quantile = cap_quantile
        self._cap_margin = cap_margin
        self._frontier_frac = frontier_frac
        self._nsl_event_quantile = nsl_event_quantile
        self._soc_target = soc_target
        self._prep_hours = prep_hours
        self._high_price_quantile = high_price_quantile
        self._low_price_quantile = low_price_quantile

        self._cap: Optional[float] = None
        self._frontier: Optional[float] = None
        self._high_price_threshold: Optional[float] = None
        self._low_price_threshold: Optional[float] = None
        self._episode_length: int = 720
        self._congestion_timesteps: set = set()
        self._prep_timesteps: set = set()
        self._congestion_starts: list = []

        # Extend observation space by 1 (normalised steps-until-congestion)
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

        # ---- Read full-horizon NSL schedule ----
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

        self._episode_length = n_hours

        # ---- Cap and frontier for cost ----
        base = float(np.quantile(hourly_totals, self._cap_quantile))
        self._cap = self._cap_margin * base
        self._frontier = self._frontier_frac * self._cap

        # ---- Detect congestion events from NSL schedule ----
        nsl_threshold = float(np.quantile(hourly_totals, self._nsl_event_quantile))
        is_congested = hourly_totals >= nsl_threshold
        in_run = False
        run_start = 0
        congestion_runs = []
        for t in range(n_hours):
            if is_congested[t] and not in_run:
                run_start = t
                in_run = True
            elif not is_congested[t] and in_run:
                congestion_runs.append((run_start, t))
                in_run = False
        if in_run:
            congestion_runs.append((run_start, n_hours))

        for start, end in congestion_runs:
            for h in range(start, end):
                self._congestion_timesteps.add(h)
            self._congestion_starts.append(start)
            prep_start = max(0, start - self._prep_hours)
            for h in range(prep_start, start):
                if h not in self._congestion_timesteps:
                    self._prep_timesteps.add(h)
        self._congestion_starts.sort()

        # ---- Price thresholds from per-timestep max price series ----
        raw_price_arrays = []
        for b in buildings:
            if hasattr(b, "pricing") and b.pricing is not None:
                ep = b.pricing.electricity_pricing
                if hasattr(ep, "__len__") and len(ep) > 0:
                    raw_price_arrays.append(np.asarray(ep[:n_hours], dtype=float))
        if raw_price_arrays:
            max_price_series = np.max(raw_price_arrays, axis=0)
            self._high_price_threshold = float(
                np.quantile(max_price_series, self._high_price_quantile)
            )
            self._low_price_threshold = float(
                np.quantile(max_price_series, self._low_price_quantile)
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

    def _mean_electrical_soc(self) -> Optional[float]:
        """Mean electrical-storage SOC across buildings (post-step)."""
        soc_values = []
        for b in self._get_buildings():
            if b.electrical_storage is not None:
                soc = self._at_timestep(b.electrical_storage.soc)
                if soc is not None:
                    soc_values.append(soc)
        return float(np.mean(soc_values)) if soc_values else None

    def _mean_electrical_soc_pre_step(self) -> Optional[float]:
        """Mean electrical-storage SOC aligned with current observation (pre-step)."""
        soc_values = []
        for b in self._get_buildings():
            if b.electrical_storage is not None:
                soc = self._at_pre_step(b.electrical_storage.soc)
                if soc is not None:
                    soc_values.append(soc)
        return float(np.mean(soc_values)) if soc_values else None

    def _steps_until_next_congestion(self, ts: int) -> int:
        """Steps from *ts* to the start of the next congestion period."""
        if ts in self._congestion_timesteps:
            return 0
        for cs in self._congestion_starts:
            if ts < cs:
                return cs - ts
        return self._episode_length

    def _augment_obs(self, obs):
        ts = self._get_time_step()
        steps = self._steps_until_next_congestion(ts)
        norm = np.float32(
            min(1.0, steps / self._episode_length)
            if self._episode_length > 0 else 1.0
        )
        return np.append(obs, norm)

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
        """label=1 during congestion hours and 3h pre-congestion prep."""
        self._ensure_base_init()
        ts = self._get_time_step()  # pre-step
        if ts in self._congestion_timesteps:
            return 1
        # 3-hour prep window before each congestion event
        for cs in self._congestion_starts:
            if cs > ts + 3:
                break  # sorted — no further starts within 3h
            if ts < cs:  # within 3h before this congestion start
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
# Scenario D — Carbon Aware
# ###########################################################################

DEFAULT_CA_CARBON_QUANTILE = 0.70
DEFAULT_CA_CARBON_EVENT_QUANTILE = 0.80
DEFAULT_CA_SOC_TARGET = 0.50
DEFAULT_CA_PREP_HOURS = 6
DEFAULT_CA_HIGH_PRICE_QUANTILE = 0.60
DEFAULT_CA_LOW_PRICE_QUANTILE = 0.25


class CarbonAwareWrapper(CityLearnBaseWrapper):
    """CMDP: minimise electricity cost subject to carbon constraint.

    Event-driven design around **dirty-grid periods** detected from the
    carbon intensity schedule.  During high-carbon hours the reward still
    wants cheap electricity (which may be dirty grid power); the cost
    penalises importing when the grid is dirty.

    Event detection (at init)
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    Dirty-grid events are contiguous runs of hours where (max) carbon
    intensity exceeds the ``carbon_event_quantile``-th percentile.
    A prep window of ``prep_hours`` precedes each dirty event.

    Observation augmentation
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Appends one feature: normalised steps until the next dirty-grid event
    (``steps_until / episode_length``).  1.0 = far away, 0.0 = imminent
    or currently in a dirty event.

    Cost — dirty-grid import penalty (unchanged)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        dirty = clamp((carbon − threshold) / (carbon_max − threshold), 0, 1)
        district_import = max(Σ_i NEC_i, 0)
        cost = dirty × clamp(district_import / import_scale, 0, 1)

    Label — dirty-grid + cheap-price indicator
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        label = 1   iff   timestep ∈ (dirty_window ∪ prep_window)
                          AND price ≤ low_price_threshold

    During dirty-grid periods (and their preparation windows), cheap
    electricity prices tempt the reward policy to charge — but importing
    during dirty hours incurs carbon cost.  The safety policy owns
    these conflict states; the reward policy owns all other hours.
    Both the dirty/prep windows and the price threshold are pre-computed
    from known schedules (observable state, trajectory-independent).
    """

    def __init__(
        self,
        env,
        carbon_quantile: float = DEFAULT_CA_CARBON_QUANTILE,
        carbon_event_quantile: float = DEFAULT_CA_CARBON_EVENT_QUANTILE,
        soc_target: float = DEFAULT_CA_SOC_TARGET,
        prep_hours: int = DEFAULT_CA_PREP_HOURS,
        high_price_quantile: float = DEFAULT_CA_HIGH_PRICE_QUANTILE,
        low_price_quantile: float = DEFAULT_CA_LOW_PRICE_QUANTILE,
    ):
        super().__init__(env)
        self._carbon_quantile = carbon_quantile
        self._carbon_event_quantile = carbon_event_quantile
        self._soc_target = soc_target
        self._prep_hours = prep_hours
        self._high_price_quantile = high_price_quantile
        self._low_price_quantile = low_price_quantile

        self._carbon_threshold: Optional[float] = None
        self._carbon_max: Optional[float] = None
        self._import_scale: Optional[float] = None
        self._high_price_threshold: Optional[float] = None
        self._low_price_threshold: Optional[float] = None
        self._episode_length: int = 720
        self._dirty_timesteps: set = set()
        self._prep_timesteps: set = set()
        self._dirty_starts: list = []

        # Extend observation space by 1 (normalised steps-until-dirty)
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

        # ---- Carbon intensity schedules (two-pass for safety) ----
        raw_ci = []
        for b in buildings:
            if hasattr(b, "carbon_intensity") and b.carbon_intensity is not None:
                ci = b.carbon_intensity.carbon_intensity
                if hasattr(ci, "__len__") and len(ci) > 0:
                    raw_ci.append(np.asarray(ci, dtype=float))
        if not raw_ci:
            return
        n_hours = min(len(a) for a in raw_ci)
        if n_hours == 0:
            return
        ci_arrays = [a[:n_hours] for a in raw_ci]

        max_ci_series = np.max(ci_arrays, axis=0)

        # Cost thresholds
        self._carbon_threshold = float(np.quantile(max_ci_series, self._carbon_quantile))
        self._carbon_max = float(np.max(max_ci_series))

        # ---- Import scale from NSL schedule ----
        n_readable = 0
        nsl_n_hours = None
        for b in buildings:
            es = getattr(b, "energy_simulation", None)
            sched = getattr(es, "_non_shiftable_load", None)
            if sched is not None and hasattr(sched, "__len__") and len(sched) > 0:
                nsl_n_hours = len(sched) if nsl_n_hours is None else min(nsl_n_hours, len(sched))
                n_readable += 1
        if nsl_n_hours is None or nsl_n_hours == 0:
            raise RuntimeError(
                "CarbonAwareWrapper: could not read full-horizon "
                "non_shiftable_load schedule from any building's "
                "energy_simulation. The scenario cannot compute "
                "import_scale and will be non-functional."
            )
        if n_readable != len(buildings):
            raise RuntimeError(
                f"CarbonAwareWrapper: only {n_readable}/{len(buildings)} "
                "buildings have a readable non_shiftable_load schedule. "
                "import_scale would be computed from a subset — refusing to continue."
            )
        hourly_nsl = np.zeros(nsl_n_hours)
        for b in buildings:
            es = getattr(b, "energy_simulation", None)
            sched = getattr(es, "_non_shiftable_load", None)
            if sched is not None and hasattr(sched, "__len__") and len(sched) >= nsl_n_hours:
                hourly_nsl += np.asarray(sched[:nsl_n_hours], dtype=float)
        self._import_scale = float(np.max(hourly_nsl))
        self._episode_length = n_hours

        # ---- Detect dirty-grid events from carbon schedule ----
        event_threshold = float(np.quantile(max_ci_series, self._carbon_event_quantile))
        is_dirty = max_ci_series >= event_threshold
        in_run = False
        run_start = 0
        dirty_runs = []
        for t in range(n_hours):
            if is_dirty[t] and not in_run:
                run_start = t
                in_run = True
            elif not is_dirty[t] and in_run:
                dirty_runs.append((run_start, t))
                in_run = False
        if in_run:
            dirty_runs.append((run_start, n_hours))

        for start, end in dirty_runs:
            for h in range(start, end):
                self._dirty_timesteps.add(h)
            self._dirty_starts.append(start)
            prep_start = max(0, start - self._prep_hours)
            for h in range(prep_start, start):
                if h not in self._dirty_timesteps:
                    self._prep_timesteps.add(h)
        self._dirty_starts.sort()

        # ---- Price thresholds from per-timestep max price series ----
        raw_price_arrays = []
        for b in buildings:
            if hasattr(b, "pricing") and b.pricing is not None:
                ep = b.pricing.electricity_pricing
                if hasattr(ep, "__len__") and len(ep) > 0:
                    raw_price_arrays.append(np.asarray(ep[:n_hours], dtype=float))
        if raw_price_arrays:
            max_price_series = np.max(raw_price_arrays, axis=0)
            self._high_price_threshold = float(
                np.quantile(max_price_series, self._high_price_quantile)
            )
            self._low_price_threshold = float(
                np.quantile(max_price_series, self._low_price_quantile)
            )

    # -- Helpers -------------------------------------------------------

    def _current_carbon_intensity(self, building) -> Optional[float]:
        """Carbon intensity — post-step accessor (for cost)."""
        if not (hasattr(building, "carbon_intensity")
                and building.carbon_intensity is not None):
            return None
        return self._at_timestep(building.carbon_intensity.carbon_intensity)

    def _max_current_carbon(self, buildings) -> Optional[float]:
        """Max carbon intensity — post-step (for cost)."""
        max_ci = None
        for b in buildings:
            ci = self._current_carbon_intensity(b)
            if ci is not None and (max_ci is None or ci > max_ci):
                max_ci = ci
        return max_ci

    def _mean_electrical_soc(self) -> Optional[float]:
        """Mean electrical-storage SOC across buildings (post-step)."""
        soc_values = []
        for b in self._get_buildings():
            if b.electrical_storage is not None:
                soc = self._at_timestep(b.electrical_storage.soc)
                if soc is not None:
                    soc_values.append(soc)
        return float(np.mean(soc_values)) if soc_values else None

    def _mean_electrical_soc_pre_step(self) -> Optional[float]:
        """Mean electrical-storage SOC aligned with current observation (pre-step)."""
        soc_values = []
        for b in self._get_buildings():
            if b.electrical_storage is not None:
                soc = self._at_pre_step(b.electrical_storage.soc)
                if soc is not None:
                    soc_values.append(soc)
        return float(np.mean(soc_values)) if soc_values else None

    def _steps_until_next_dirty(self, ts: int) -> int:
        """Steps from *ts* to the start of the next dirty-grid period."""
        if ts in self._dirty_timesteps:
            return 0
        for ds in self._dirty_starts:
            if ts < ds:
                return ds - ts
        return self._episode_length

    def _augment_obs(self, obs):
        ts = self._get_time_step()
        steps = self._steps_until_next_dirty(ts)
        norm = np.float32(
            min(1.0, steps / self._episode_length)
            if self._episode_length > 0 else 1.0
        )
        return np.append(obs, norm)

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

        max_ci = self._max_current_carbon(buildings)
        if max_ci is None or max_ci <= self._carbon_threshold:
            return 0.0

        violation_range = self._carbon_max - self._carbon_threshold
        dirty = min(1.0, (max_ci - self._carbon_threshold) / violation_range)

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

    def _compute_label(self, obs) -> int:
        """label=1 during dirty/prep windows when price is cheap."""
        self._ensure_base_init()
        ts = self._get_time_step()  # pre-step

        if ts not in self._dirty_timesteps and ts not in self._prep_timesteps:
            return 0

        if self._low_price_threshold is not None:
            buildings = self._get_buildings()
            price = self._max_pre_step_price(buildings)
            if price is not None and price <= self._low_price_threshold:
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
# Factory helpers
# ###########################################################################

# -- Arbitrage vs Buffer -----------------------------------------------

def make_arbitrage_vs_buffer_env(
    env_name: str = DEFAULT_SCHEMA,
    episode_time_steps: Optional[int] = None,
    soc_safety_level: float = DEFAULT_AVB_SOC_SAFETY_LEVEL,
    peak_price_quantile: float = DEFAULT_AVB_PEAK_PRICE_QUANTILE,
    prep_hours: int = DEFAULT_AVB_PREP_HOURS,
    high_price_quantile: float = DEFAULT_AVB_HIGH_PRICE_QUANTILE,
    emergency_soc: float = DEFAULT_AVB_EMERGENCY_SOC,
    **kwargs,
):
    inner = make_citylearn_inner_env(
        schema=env_name, episode_time_steps=episode_time_steps,
    )
    return ArbitrageVsBufferWrapper(
        inner,
        soc_safety_level=soc_safety_level,
        peak_price_quantile=peak_price_quantile,
        prep_hours=prep_hours,
        high_price_quantile=high_price_quantile,
        emergency_soc=emergency_soc,
    )


def make_arbitrage_vs_buffer_vec_env(
    env_name: str = DEFAULT_SCHEMA,
    n_envs: int = 1,
    seed: Optional[int] = None,
    episode_time_steps: Optional[int] = None,
    soc_safety_level: float = DEFAULT_AVB_SOC_SAFETY_LEVEL,
    peak_price_quantile: float = DEFAULT_AVB_PEAK_PRICE_QUANTILE,
    prep_hours: int = DEFAULT_AVB_PREP_HOURS,
    high_price_quantile: float = DEFAULT_AVB_HIGH_PRICE_QUANTILE,
    emergency_soc: float = DEFAULT_AVB_EMERGENCY_SOC,
    **kwargs,
):
    env_kwargs = {
        "env_name": env_name,
        "episode_time_steps": episode_time_steps,
        "soc_safety_level": soc_safety_level,
        "peak_price_quantile": peak_price_quantile,
        "prep_hours": prep_hours,
        "high_price_quantile": high_price_quantile,
        "emergency_soc": emergency_soc,
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
    nsl_event_quantile: float = DEFAULT_CD_NSL_EVENT_QUANTILE,
    soc_target: float = DEFAULT_CD_SOC_TARGET,
    prep_hours: int = DEFAULT_CD_PREP_HOURS,
    high_price_quantile: float = DEFAULT_CD_HIGH_PRICE_QUANTILE,
    low_price_quantile: float = DEFAULT_CD_LOW_PRICE_QUANTILE,
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
        nsl_event_quantile=nsl_event_quantile,
        soc_target=soc_target,
        prep_hours=prep_hours,
        high_price_quantile=high_price_quantile,
        low_price_quantile=low_price_quantile,
    )


def make_contract_demand_vec_env(
    env_name: str = DEFAULT_SCHEMA,
    n_envs: int = 1,
    seed: Optional[int] = None,
    episode_time_steps: Optional[int] = None,
    cap_quantile: float = DEFAULT_CD_CAP_QUANTILE,
    cap_margin: float = DEFAULT_CD_CAP_MARGIN,
    frontier_frac: float = DEFAULT_CD_FRONTIER_FRAC,
    nsl_event_quantile: float = DEFAULT_CD_NSL_EVENT_QUANTILE,
    soc_target: float = DEFAULT_CD_SOC_TARGET,
    prep_hours: int = DEFAULT_CD_PREP_HOURS,
    high_price_quantile: float = DEFAULT_CD_HIGH_PRICE_QUANTILE,
    low_price_quantile: float = DEFAULT_CD_LOW_PRICE_QUANTILE,
    **kwargs,
):
    env_kwargs = {
        "env_name": env_name,
        "episode_time_steps": episode_time_steps,
        "cap_quantile": cap_quantile,
        "cap_margin": cap_margin,
        "frontier_frac": frontier_frac,
        "nsl_event_quantile": nsl_event_quantile,
        "soc_target": soc_target,
        "prep_hours": prep_hours,
        "high_price_quantile": high_price_quantile,
        "low_price_quantile": low_price_quantile,
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
    carbon_event_quantile: float = DEFAULT_CA_CARBON_EVENT_QUANTILE,
    soc_target: float = DEFAULT_CA_SOC_TARGET,
    prep_hours: int = DEFAULT_CA_PREP_HOURS,
    high_price_quantile: float = DEFAULT_CA_HIGH_PRICE_QUANTILE,
    low_price_quantile: float = DEFAULT_CA_LOW_PRICE_QUANTILE,
    **kwargs,
):
    inner = make_citylearn_inner_env(
        schema=env_name, episode_time_steps=episode_time_steps,
    )
    return CarbonAwareWrapper(
        inner,
        carbon_quantile=carbon_quantile,
        carbon_event_quantile=carbon_event_quantile,
        soc_target=soc_target,
        prep_hours=prep_hours,
        high_price_quantile=high_price_quantile,
        low_price_quantile=low_price_quantile,
    )


def make_carbon_aware_vec_env(
    env_name: str = DEFAULT_SCHEMA,
    n_envs: int = 1,
    seed: Optional[int] = None,
    episode_time_steps: Optional[int] = None,
    carbon_quantile: float = DEFAULT_CA_CARBON_QUANTILE,
    carbon_event_quantile: float = DEFAULT_CA_CARBON_EVENT_QUANTILE,
    soc_target: float = DEFAULT_CA_SOC_TARGET,
    prep_hours: int = DEFAULT_CA_PREP_HOURS,
    high_price_quantile: float = DEFAULT_CA_HIGH_PRICE_QUANTILE,
    low_price_quantile: float = DEFAULT_CA_LOW_PRICE_QUANTILE,
    **kwargs,
):
    env_kwargs = {
        "env_name": env_name,
        "episode_time_steps": episode_time_steps,
        "carbon_quantile": carbon_quantile,
        "carbon_event_quantile": carbon_event_quantile,
        "soc_target": soc_target,
        "prep_hours": prep_hours,
        "high_price_quantile": high_price_quantile,
        "low_price_quantile": low_price_quantile,
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
DEFAULT_PR_PRICE_LOW_QUANTILE = 0.35
DEFAULT_PR_CHARGE_HEADROOM_FRAC = 0.50
DEFAULT_PR_RATCHET_MARGIN = 1.15


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

    Label — high-demand indicator
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        label = 1   iff   district_nsl > peak_target

    The label is **trajectory-independent**: it depends only on the
    pre-computed NSL schedule, not on the running peak (which evolves
    with the agent's actions).  During high-demand hours (base load
    above the 75th-percentile), any battery charging adds to district
    import and risks setting a new peak.  The safety policy owns
    these timesteps to limit charging; the reward policy owns all
    lower-demand hours.
    """

    def __init__(
        self,
        env,
        peak_target_quantile: float = DEFAULT_PR_PEAK_TARGET_QUANTILE,
        peak_cap_margin: float = DEFAULT_PR_PEAK_CAP_MARGIN,
        price_low_quantile: float = DEFAULT_PR_PRICE_LOW_QUANTILE,
        charge_headroom_frac: float = DEFAULT_PR_CHARGE_HEADROOM_FRAC,
        ratchet_margin: float = DEFAULT_PR_RATCHET_MARGIN,
    ):
        super().__init__(env)
        self._peak_target_quantile = peak_target_quantile
        self._peak_cap_margin = peak_cap_margin
        self._price_low_quantile = price_low_quantile
        self._charge_headroom_frac = charge_headroom_frac
        self._ratchet_margin = ratchet_margin

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

        # Price threshold — per-timestep max across buildings
        # (matches _max_pre_step_price used at decision time)
        raw_price_arrays = []
        for b in buildings:
            if hasattr(b, "pricing") and b.pricing is not None:
                ep = b.pricing.electricity_pricing
                if hasattr(ep, "__len__") and len(ep) > 0:
                    raw_price_arrays.append(np.asarray(ep[:n_hours], dtype=float))
        if raw_price_arrays:
            max_price_series = np.max(raw_price_arrays, axis=0)
            self._price_low_threshold = float(
                np.quantile(max_price_series, self._price_low_quantile)
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
        """label=1 when district base demand exceeds peak target."""
        self._ensure_base_init()
        if self._peak_target is None:
            return 0

        nsl = self._district_pre_step_nsl()
        if nsl is None:
            return 0

        return 1 if nsl > self._peak_target else 0

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
DEFAULT_DR_PREP_WINDOW = 3
DEFAULT_DR_HIGH_PRICE_QUANTILE = 0.75
DEFAULT_DR_EMERGENCY_SOC = 0.40


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

    Label — event-window indicator
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        label = 1   iff   timestep ∈ event_window

    During demand response events, the safety policy controls
    battery actions to maintain reserves.  The reward policy
    owns all non-event hours and is free to charge cheaply
    (building SOC reserves that benefit the subsequent event).
    Event windows are fixed and known at episode start.
    """

    def __init__(
        self,
        env,
        event_starts: Sequence[int] = DEFAULT_DR_EVENT_STARTS,
        event_duration: int = DEFAULT_DR_EVENT_DURATION,
        soc_target: float = DEFAULT_DR_SOC_TARGET,
        prep_window: int = DEFAULT_DR_PREP_WINDOW,
        high_price_quantile: float = DEFAULT_DR_HIGH_PRICE_QUANTILE,
        emergency_soc: float = DEFAULT_DR_EMERGENCY_SOC,
    ):
        super().__init__(env)
        self._event_starts = tuple(sorted(event_starts))
        self._event_duration = event_duration
        self._soc_target = soc_target
        self._prep_window = prep_window
        self._high_price_quantile = high_price_quantile
        self._emergency_soc = emergency_soc

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

        # High-price threshold — per-timestep max across buildings
        # (matches _max_pre_step_price used at decision time)
        raw_price_arrays = []
        for b in buildings:
            if hasattr(b, "pricing") and b.pricing is not None:
                ep = b.pricing.electricity_pricing
                if hasattr(ep, "__len__") and len(ep) > 0:
                    raw_price_arrays.append(np.asarray(ep[:self._episode_length], dtype=float))
        if raw_price_arrays:
            max_price_series = np.max(raw_price_arrays, axis=0)
            self._high_price_threshold = float(
                np.quantile(max_price_series, self._high_price_quantile)
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
        """Mean electrical-storage SOC (post-step)."""
        soc_values = []
        for b in self._get_buildings():
            if b.electrical_storage is not None:
                soc = self._at_timestep(b.electrical_storage.soc)
                if soc is not None:
                    soc_values.append(soc)
        return float(np.mean(soc_values)) if soc_values else None

    def _mean_electrical_soc_pre_step(self) -> Optional[float]:
        """Mean electrical-storage SOC aligned with current observation (pre-step)."""
        soc_values = []
        for b in self._get_buildings():
            if b.electrical_storage is not None:
                soc = self._at_pre_step(b.electrical_storage.soc)
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
        """label=1 during demand response events."""
        self._ensure_base_init()
        ts = self._get_time_step()  # pre-step
        return 1 if self._in_event_window(ts) else 0

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
DEFAULT_SR_PREP_HOURS = 2
DEFAULT_SR_BUFFER_HOURS = 4
DEFAULT_SR_HIGH_PRICE_QUANTILE = 0.75
DEFAULT_SR_EMERGENCY_SOC = 0.25


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

    Label — buffer-window indicator
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        label = 1   iff   timestep ∈ buffer_window ∪ prep_window

    During post-sunset buffer windows and the preparation hours
    preceding them, any battery action affects whether reserves
    are available when solar generation drops.  The safety policy
    owns these timesteps; the reward policy owns all other hours.
    Both windows are pre-computed from the known solar schedule
    (observable state) and are trajectory-independent.
    """

    def __init__(
        self,
        env,
        soc_reserve: float = DEFAULT_SR_SOC_RESERVE,
        prep_hours: int = DEFAULT_SR_PREP_HOURS,
        buffer_hours: int = DEFAULT_SR_BUFFER_HOURS,
        high_price_quantile: float = DEFAULT_SR_HIGH_PRICE_QUANTILE,
        emergency_soc: float = DEFAULT_SR_EMERGENCY_SOC,
    ):
        super().__init__(env)
        self._soc_reserve = soc_reserve
        self._prep_hours = prep_hours
        self._buffer_hours = buffer_hours
        self._high_price_quantile = high_price_quantile
        self._emergency_soc = emergency_soc

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

        # High-price threshold — per-timestep max across buildings
        # (matches _max_pre_step_price used at decision time)
        raw_price_arrays = []
        for b in buildings:
            if hasattr(b, "pricing") and b.pricing is not None:
                ep = b.pricing.electricity_pricing
                if hasattr(ep, "__len__") and len(ep) > 0:
                    raw_price_arrays.append(np.asarray(ep[:n_hours], dtype=float))
        if raw_price_arrays:
            max_price_series = np.max(raw_price_arrays, axis=0)
            self._high_price_threshold = float(
                np.quantile(max_price_series, self._high_price_quantile)
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
        """Mean electrical-storage SOC (post-step)."""
        soc_values = []
        for b in self._get_buildings():
            if b.electrical_storage is not None:
                soc = self._at_timestep(b.electrical_storage.soc)
                if soc is not None:
                    soc_values.append(soc)
        return float(np.mean(soc_values)) if soc_values else None

    def _mean_electrical_soc_pre_step(self) -> Optional[float]:
        """Mean electrical-storage SOC aligned with current observation (pre-step)."""
        soc_values = []
        for b in self._get_buildings():
            if b.electrical_storage is not None:
                soc = self._at_pre_step(b.electrical_storage.soc)
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
        """label=1 during buffer and prep windows."""
        self._ensure_base_init()
        ts = self._get_time_step()  # pre-step
        if ts in self._buffer_timesteps or ts in self._prep_timesteps:
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
    ratchet_margin: float = DEFAULT_PR_RATCHET_MARGIN,
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
        ratchet_margin=ratchet_margin,
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
    ratchet_margin: float = DEFAULT_PR_RATCHET_MARGIN,
    **kwargs,
):
    env_kwargs = {
        "env_name": env_name,
        "episode_time_steps": episode_time_steps,
        "peak_target_quantile": peak_target_quantile,
        "peak_cap_margin": peak_cap_margin,
        "price_low_quantile": price_low_quantile,
        "charge_headroom_frac": charge_headroom_frac,
        "ratchet_margin": ratchet_margin,
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
    emergency_soc: float = DEFAULT_DR_EMERGENCY_SOC,
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
        emergency_soc=emergency_soc,
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
    emergency_soc: float = DEFAULT_DR_EMERGENCY_SOC,
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
        "emergency_soc": emergency_soc,
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
    emergency_soc: float = DEFAULT_SR_EMERGENCY_SOC,
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
        emergency_soc=emergency_soc,
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
    emergency_soc: float = DEFAULT_SR_EMERGENCY_SOC,
    **kwargs,
):
    env_kwargs = {
        "env_name": env_name,
        "episode_time_steps": episode_time_steps,
        "soc_reserve": soc_reserve,
        "prep_hours": prep_hours,
        "buffer_hours": buffer_hours,
        "high_price_quantile": high_price_quantile,
        "emergency_soc": emergency_soc,
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
