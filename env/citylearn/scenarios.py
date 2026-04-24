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
Labels follow a strict pre-violation decision frontier principle:
**label=1 ONLY at state-observable, action-contingent, pre-violation
decision points where the next action changes the reward-vs-cost
tradeoff.**  The agent's state must be close enough to the cost
boundary that the next action determines whether cost fires.

PeakGuard — **Arbitrage vs Buffer**
    Frontier: peak price + SOC in [safety, safety+margin].
    Reward wants to discharge (sell expensive), but one more discharge
    would push SOC below the safety level.  Label=1 at this narrow
    pre-violation band only.

CapFrontier — **Contract Demand**
    Frontier: congestion + cheap price + import headroom tight.
    Reward wants to charge (cheap), but charging would push district
    import past the contracted cap.  Label=1 at this triple condition.

DirtyReserve — **Dirty Window Reserve**
    Frontier: dirty hours + SOC in [target, target+margin] + expensive
    price.  Reward wants to discharge (sell high), but discharge would
    deplete reserves needed during dirty grid hours.

PeakRatchet — **Peak Ratchet**
    Frontier: baseline import near target + cheap price.  Reward wants
    to charge (cheap), but charging would push import above the peak
    target.  Import must be BELOW target (pre-violation).

EventReserve — **Demand Response**
    Frontier: event window + SOC in [target, target+margin].  During
    DR events, discharge would cross the reserve boundary.

SunsetBridge — **Solar Ramp Reserve**
    Frontier: post-sunset buffer + SOC in [reserve, reserve+margin].
    During buffer hours, discharge would cross the reserve boundary.
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
# Scenario PeakGuard — Arbitrage vs Buffer
# ###########################################################################

DEFAULT_AVB_SOC_SAFETY_LEVEL = 0.50
DEFAULT_AVB_PEAK_PRICE_QUANTILE = 0.92
DEFAULT_AVB_HIGH_PRICE_QUANTILE = 0.75
DEFAULT_AVB_EMERGENCY_SOC = 0.35
DEFAULT_AVB_PREP_HOURS = 6


class ArbitrageVsBufferWrapper(CityLearnBaseWrapper):
    """CMDP: minimise electricity cost subject to buffer-resilience constraint.

    Event-driven design around **peak-price periods** detected from the
    electricity pricing schedule.  During on-peak hours the reward wants
    to discharge (sell expensive) while the cost wants to hold reserves.

    Event detection (at init)
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    Peak-price periods are contiguous runs of hours where price exceeds
    the ``peak_price_quantile``-th percentile.

    Observation augmentation
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Appends one feature: normalised steps until the next peak event
    (``steps_until / episode_length``).  1.0 = far away, 0.0 = imminent
    or currently in peak.

    Cost — battery depletion (always-on)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        cost = clamp((safety − mean_soc) / safety, 0, 1)

    Cost fires at every timestep.  All algorithms see the same
    always-on constraint signal.

    Label — high-precision binary (no multi-label)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        SOC < safety                              →  label = 1
        SOC ∈ [safety, safety+m] + peak + high_price → label = 1
        SOC > safety+m                            →  label = 0
        Otherwise                                 →  label = 0

    Binary labels make precision critical: label=1 starves reward.
    Only label=1 at active violations or narrow conflict frontiers.
    """

    def __init__(
        self,
        env,
        soc_safety_level: float = DEFAULT_AVB_SOC_SAFETY_LEVEL,
        peak_price_quantile: float = DEFAULT_AVB_PEAK_PRICE_QUANTILE,
        high_price_quantile: float = DEFAULT_AVB_HIGH_PRICE_QUANTILE,
        emergency_soc: float = DEFAULT_AVB_EMERGENCY_SOC,
        prep_hours: int = DEFAULT_AVB_PREP_HOURS,
    ):
        super().__init__(env)
        self._soc_safety_level = soc_safety_level
        self._peak_price_quantile = peak_price_quantile
        self._emergency_soc = emergency_soc
        self._high_price_quantile = high_price_quantile
        self._prep_hours = prep_hours

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

    def _compute_label(self, obs):
        """High-precision binary label for ArbitrageVsBuffer.

        label=1 only for:
          - active SOC safety violation (anywhere in episode)
          - peak-price discharge temptation when SOC is near safety

        Everything else is reward-owned (label=0).
        """
        self._ensure_base_init()
        ts = self._get_time_step()  # pre-step

        mean_soc = self._mean_electrical_soc_pre_step()
        if mean_soc is None:
            return 0

        soc_margin = 0.07

        # Active always-on cost violation: cost branch owns recovery.
        if mean_soc < self._soc_safety_level:
            return 1

        # Safely above frontier: reward branch owns.
        if mean_soc > self._soc_safety_level + soc_margin:
            return 0

        # Near safety during peak + high price: reward wants to discharge,
        # cost wants to preserve buffer.
        price = self._max_pre_step_price(self._get_buildings())
        high_price = (
            price is not None
            and self._high_price_threshold is not None
            and price >= self._high_price_threshold
        )

        if ts in self._peak_timesteps and high_price:
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
# Scenario CapFrontier — Contract Demand (replaces Peak Shaving)
# ###########################################################################

DEFAULT_CD_CAP_QUANTILE = 0.95
DEFAULT_CD_CAP_MARGIN = 1.10
DEFAULT_CD_FRONTIER_FRAC = 0.90
DEFAULT_CD_NSL_EVENT_QUANTILE = 0.80
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
    exceeds the ``nsl_event_quantile``-th percentile.

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

    Label — two-tier: cost-relevant vs safe
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        Congestion + headroom ≤ charge_push + low_price →  1
        Otherwise                                        →  0
        Otherwise                            →  0

    Never label=1.  ``cf_coef`` controls the balance.
    """

    def __init__(
        self,
        env,
        cap_quantile: float = DEFAULT_CD_CAP_QUANTILE,
        cap_margin: float = DEFAULT_CD_CAP_MARGIN,
        frontier_frac: float = DEFAULT_CD_FRONTIER_FRAC,
        nsl_event_quantile: float = DEFAULT_CD_NSL_EVENT_QUANTILE,
        low_price_quantile: float = DEFAULT_CD_LOW_PRICE_QUANTILE,
    ):
        super().__init__(env)
        self._cap_quantile = cap_quantile
        self._cap_margin = cap_margin
        self._frontier_frac = frontier_frac
        self._nsl_event_quantile = nsl_event_quantile
        self._low_price_quantile = low_price_quantile

        self._cap: Optional[float] = None
        self._frontier: Optional[float] = None
        self._low_price_threshold: Optional[float] = None
        self._total_charge_power: float = 0.0
        self._episode_length: int = 720
        self._congestion_timesteps: set = set()
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
        self._congestion_starts.sort()

        # ---- Total fleet charge power (for headroom estimation) ----
        for b in buildings:
            es = b.electrical_storage
            if (es is not None
                    and hasattr(es, 'nominal_power')
                    and es.nominal_power is not None):
                self._total_charge_power += es.nominal_power

        # ---- Low-price threshold from per-timestep max price series ----
        raw_price_arrays = []
        for b in buildings:
            if hasattr(b, "pricing") and b.pricing is not None:
                ep = b.pricing.electricity_pricing
                if hasattr(ep, "__len__") and len(ep) > 0:
                    raw_price_arrays.append(np.asarray(ep[:n_hours], dtype=float))
        if raw_price_arrays:
            max_price_series = np.max(raw_price_arrays, axis=0)
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

    def _district_pre_step_nsl(self) -> Optional[float]:
        """District NSL at pre-step timestep (from schedule)."""
        ts = self._get_time_step()
        total = 0.0
        n = 0
        for b in self._get_buildings():
            es = getattr(b, "energy_simulation", None)
            sched = getattr(es, "_non_shiftable_load", None)
            if sched is not None and hasattr(sched, "__len__") and ts < len(sched):
                total += float(sched[ts])
                n += 1
        return total if n > 0 else None

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

    def _compute_label(self, obs):
        """Always label=0: battery cannot meaningfully control import cost.

        District import is dominated by non-shiftable load; battery
        charge/discharge is a tiny fraction.  Labels only interfere
        with reward learning without reducing cost.
        """
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
# Scenario DirtyReserve — Dirty Window Reserve
# ###########################################################################

DEFAULT_CA_CARBON_EVENT_QUANTILE = 0.65
DEFAULT_CA_SOC_TARGET = 0.50
DEFAULT_CA_PREP_HOURS = 6
DEFAULT_CA_HIGH_PRICE_QUANTILE = 0.60
DEFAULT_CA_LOW_PRICE_QUANTILE = 0.40


class DirtyWindowReserveWrapper(CityLearnBaseWrapper):
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

    Cost — SOC deficit during dirty-grid periods
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        if timestep ∈ dirty_window:
            cost = clamp((soc_target − mean_soc) / soc_target, 0, 1)
        else:
            cost = 0

    Cost fires ONLY during dirty-grid windows.  Inside those windows,
    low battery reserves are penalized.  The agent should have charged
    during preceding clean-grid hours; discharging during dirty windows
    depletes reserves and forces reliance on dirty grid import.
    The gradient cancellation: reward says discharge (sell at high
    price during dirty periods), cost says hold SOC.

    Label — frontier + exclusive violation control
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        Dirty + SOC < target                 →  label = 1
        Dirty + SOC ∈ [target, target+m] + expensive  →  label = 1
        Dirty + SOC < target + violation               →  label = 1
        Dirty + SOC ∈ [target, target+m] + high_price   →  label = 1
        Imminent (≤2 steps) + SOC < target + high_price →  label = 1
        Otherwise                                       →  label = 0
        Otherwise                            →  label = 0
    """

    def __init__(
        self,
        env,
        carbon_event_quantile: float = DEFAULT_CA_CARBON_EVENT_QUANTILE,
        soc_target: float = DEFAULT_CA_SOC_TARGET,
        prep_hours: int = DEFAULT_CA_PREP_HOURS,
        high_price_quantile: float = DEFAULT_CA_HIGH_PRICE_QUANTILE,
        low_price_quantile: float = DEFAULT_CA_LOW_PRICE_QUANTILE,
    ):
        super().__init__(env)
        self._carbon_event_quantile = carbon_event_quantile
        self._soc_target = soc_target
        self._prep_hours = prep_hours
        self._high_price_quantile = high_price_quantile
        self._low_price_quantile = low_price_quantile

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
        """SOC deficit cost ∈ [0, 1] — fires only during dirty windows.

        Zero outside dirty-grid periods.  Inside dirty windows, low
        battery reserves are penalized.  Discharging for profit during
        dirty hours depletes SOC and forces future dirty grid import.
        """
        self._ensure_base_init()
        ts = self._get_time_step() - 1  # post-step
        if ts not in self._dirty_timesteps:
            return 0.0

        mean_soc = self._mean_electrical_soc()
        if mean_soc is None or mean_soc >= self._soc_target:
            return 0.0

        return float(min(1.0,
            (self._soc_target - mean_soc) / self._soc_target
        ))

    # -- Label ---------------------------------------------------------

    def _compute_label(self, obs):
        """High-precision binary label for DirtyWindowReserve.

        label=1 only for:
          - active dirty-window reserve violation
          - dirty-window high-price discharge temptation near target
          - imminent dirty event while below target and price is high

        Everything else is reward-owned (label=0).
        """
        self._ensure_base_init()
        ts = self._get_time_step()  # pre-step

        mean_soc = self._mean_electrical_soc_pre_step()
        if mean_soc is None:
            return 0

        soc_margin = 0.07

        price = self._max_pre_step_price(self._get_buildings())
        high_price = (
            price is not None
            and self._high_price_threshold is not None
            and price >= self._high_price_threshold
        )

        # --- Dirty event active ---
        if ts in self._dirty_timesteps:
            # Active cost violation.
            if mean_soc < self._soc_target:
                return 1
            # Near target only matters if reward is tempted to discharge.
            if mean_soc <= self._soc_target + soc_margin and high_price:
                return 1
            return 0

        # --- Imminent dirty event: only if reserve deficient + price conflict ---
        steps = self._steps_until_next_dirty(ts)
        if 0 < steps <= 2 and mean_soc < self._soc_target and high_price:
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
    high_price_quantile: float = DEFAULT_AVB_HIGH_PRICE_QUANTILE,
    emergency_soc: float = DEFAULT_AVB_EMERGENCY_SOC,
    **kwargs,
):
    env_kwargs = {
        "env_name": env_name,
        "episode_time_steps": episode_time_steps,
        "soc_safety_level": soc_safety_level,
        "peak_price_quantile": peak_price_quantile,
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
        "low_price_quantile": low_price_quantile,
    }
    return make_cost_vec_env(
        make_contract_demand_env,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
    )


# -- Dirty Window Reserve ----------------------------------------------

def make_dirty_window_reserve_env(
    env_name: str = DEFAULT_SCHEMA,
    episode_time_steps: Optional[int] = None,
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
    return DirtyWindowReserveWrapper(
        inner,
        carbon_event_quantile=carbon_event_quantile,
        soc_target=soc_target,
        prep_hours=prep_hours,
        high_price_quantile=high_price_quantile,
        low_price_quantile=low_price_quantile,
    )


def make_dirty_window_reserve_vec_env(
    env_name: str = DEFAULT_SCHEMA,
    n_envs: int = 1,
    seed: Optional[int] = None,
    episode_time_steps: Optional[int] = None,
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
        "carbon_event_quantile": carbon_event_quantile,
        "soc_target": soc_target,
        "prep_hours": prep_hours,
        "high_price_quantile": high_price_quantile,
        "low_price_quantile": low_price_quantile,
    }
    return make_cost_vec_env(
        make_dirty_window_reserve_env,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
    )


# ###########################################################################
# Scenario PeakRatchet — Peak Import Ratchet
# ###########################################################################

DEFAULT_PR_PEAK_TARGET_QUANTILE = 0.90  # quantile of net baseline import
DEFAULT_PR_PEAK_CAP_MARGIN = 1.0       # factor on charge_power added to target
DEFAULT_PR_PRICE_LOW_QUANTILE = 0.35
DEFAULT_PR_CHARGE_HEADROOM_FRAC = 0.10
DEFAULT_PR_RATCHET_MARGIN = 1.05


class PeakRatchetWrapper(CityLearnBaseWrapper):
    """CMDP: minimise electricity cost subject to peak import exceedance.

    Each hour, the district pays a *demand charge* proportional to how
    much its net import exceeds a capacity target.  The running peak
    (highest import so far) is tracked in the observation so the agent
    can learn the irreversible consequences of its charging decisions.

    Observation augmentation
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Appends one feature to the observation: normalised running peak
    (``running_peak / peak_cap``).  The running peak never decreases
    within an episode and resets to 0 on ``env.reset()``.

    Cost — per-step peak exceedance
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        district_import = max(Σ_i NEC_i, 0)
        running_peak = max(running_peak, district_import)   # observation only

        if district_import > peak_target:
            cost = clamp((district_import − peak_target) / (peak_cap − peak_target), 0, 1)
        else:
            cost = 0

    Cost fires each step where the *current* import exceeds the target.
    A zero-action policy sees cost only during natural NSL spikes
    (~10 % at Q90).  A greedy charge policy pushes many more steps
    over the target, producing dramatically higher cumulative cost.

    Label — frontier + exclusive violation control
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        import >= peak_target                            →  label = 1
        0 < headroom <= charge_push * margin + cheap     →  label = 1
        low_price + headroom ≤ charge_push * margin →  label = 1
        Otherwise                                    →  label = 0
        Otherwise                                        →  label = 0
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
        self._baseline_import: Optional[np.ndarray] = None

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

        # District solar schedule
        hourly_solar = np.zeros(n_hours)
        for b in buildings:
            es = getattr(b, "energy_simulation", None)
            sg = getattr(es, "_solar_generation", None)
            if sg is not None and hasattr(sg, "__len__") and len(sg) >= n_hours:
                hourly_solar += np.asarray(sg[:n_hours], dtype=float)

        # Net baseline import: what NEC would be with zero battery action
        self._baseline_import = np.maximum(hourly_totals - hourly_solar, 0.0)

        # Peak target from baseline import (solar-adjusted, not raw NSL)
        self._peak_target = float(
            np.quantile(self._baseline_import, self._peak_target_quantile)
        )

        # Total fleet charge power
        for b in buildings:
            es = b.electrical_storage
            if (es is not None
                    and hasattr(es, 'nominal_power')
                    and es.nominal_power is not None):
                self._total_charge_power += es.nominal_power

        # Cap: target + battery's max import contribution
        self._peak_cap = (
            self._peak_target
            + self._peak_cap_margin * self._total_charge_power
        )

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

        # Update running peak (ratchet — never decreases, observation only)
        if district_import > self._running_peak:
            self._running_peak = district_import

        # Per-step cost: fires when current import exceeds target
        if district_import <= self._peak_target:
            return 0.0

        excess = (district_import - self._peak_target) / (
            self._peak_cap - self._peak_target
        )
        return float(min(1.0, excess))

    # -- Label ---------------------------------------------------------

    def _compute_label(self, obs):
        """High-precision binary label for PeakRatchet.

        label=1 only when reward is likely to charge because price is
        cheap, and charging could push district import through the peak
        target.  High-price near-cap states stay label=0 because reward
        and cost are already aligned (prefer discharge / avoid import).
        """
        self._ensure_base_init()
        if self._peak_target is None:
            return 0

        ts = self._get_time_step()

        # Price check: only label=1 when price is cheap (reward wants to charge).
        price = self._max_pre_step_price(self._get_buildings())
        low_price = (
            price is not None
            and self._price_low_threshold is not None
            and price <= self._price_low_threshold
        )
        if not low_price:
            return 0

        if (self._baseline_import is not None
                and 0 <= ts < len(self._baseline_import)):
            current_import = float(self._baseline_import[ts])
        else:
            current_import = self._district_pre_step_nsl() or 0.0

        charge_push = self._charge_headroom_frac * self._total_charge_power
        headroom = self._peak_target - current_import

        # Cheap price + charging could exceed target.
        if headroom <= charge_push * self._ratchet_margin:
            return 1

        return 0

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
# Scenario EventReserve — Demand Response Events
# ###########################################################################

DEFAULT_DR_EVENT_STARTS: Optional[Tuple[int, ...]] = None  # auto-detect at peak prices
DEFAULT_DR_N_EVENTS = 3
DEFAULT_DR_EVENT_DURATION = 8
DEFAULT_DR_SOC_TARGET = 0.90
DEFAULT_DR_PREP_WINDOW = 4
DEFAULT_DR_HIGH_PRICE_QUANTILE = 0.75


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

    Label — frontier + exclusive violation control
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        In event + SOC <= target+m           →  label = 1
        1-2 steps before event + SOC <= target+m + expensive → label = 1
        In event + SOC < target             →  label = 1
        In event + SOC ≤ target+m            →  label = 1
        1-2 steps before + SOC < target      →  label = 1
        Otherwise                            →  label = 0
        Otherwise                            →  label = 0
    """

    def __init__(
        self,
        env,
        event_starts: Optional[Sequence[int]] = DEFAULT_DR_EVENT_STARTS,
        event_duration: int = DEFAULT_DR_EVENT_DURATION,
        soc_target: float = DEFAULT_DR_SOC_TARGET,
        prep_window: int = DEFAULT_DR_PREP_WINDOW,
        high_price_quantile: float = DEFAULT_DR_HIGH_PRICE_QUANTILE,
        n_events: int = DEFAULT_DR_N_EVENTS,
    ):
        super().__init__(env)
        self._event_starts = tuple(sorted(event_starts)) if event_starts else ()
        self._n_events = n_events
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

            # Auto-detect: place events at peak-price windows
            if not self._event_starts:
                n_valid = len(max_price_series) - self._event_duration + 1
                if n_valid > 0:
                    window_sums = np.convolve(
                        max_price_series,
                        np.ones(self._event_duration),
                        mode='valid',
                    )
                    # One event per episode segment (temporal spread)
                    detected: list = []
                    seg_len = max(1, n_valid // self._n_events)
                    for seg_i in range(self._n_events):
                        s = seg_i * seg_len
                        e = min((seg_i + 1) * seg_len, n_valid)
                        if s < e:
                            detected.append(s + int(np.argmax(window_sums[s:e])))
                    self._event_starts = tuple(sorted(detected))

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

    def _compute_label(self, obs):
        """High-precision binary label for DemandResponse.

        label=1 only for:
          - active event reserve violation
          - event-window near-violation (narrow margin)
          - very imminent event while below target

        No broad prep-window labels.  target=0.90 already makes this
        scenario highly cost-conservative.
        """
        self._ensure_base_init()
        ts = self._get_time_step()  # pre-step

        mean_soc = self._mean_electrical_soc_pre_step()
        if mean_soc is None:
            return 0

        # 0.15 is too large for target=0.90; use narrow margin.
        soc_margin = 0.03
        steps = self._steps_until_next_event(ts)

        # --- Inside DR event ---
        if steps == 0:
            # Active cost violation.
            if mean_soc < self._soc_target:
                return 1
            # Very narrow near-boundary protection.
            if mean_soc <= self._soc_target + soc_margin:
                return 1
            return 0

        # --- Very imminent event: hard cost ownership ---
        if 0 < steps <= 2 and mean_soc < self._soc_target:
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
# Scenario SunsetBridge — Solar Ramp Reserve
# ###########################################################################

DEFAULT_SR_SOC_RESERVE = 0.70
DEFAULT_SR_PREP_HOURS = 3
DEFAULT_SR_BUFFER_HOURS = 6


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

    Label — frontier + exclusive violation control
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        In buffer + SOC <= reserve+m  →  label = 1
        1 step before buffer + SOC <= reserve+m  →  label = 1
        In buffer + SOC < reserve            →  label = 1
        In buffer + SOC ≤ reserve+m + high_price → label = 1
        1 step before + SOC < reserve        →  label = 1
        Prep + SOC < reserve + high_price    →  label = 1
        Otherwise                            →  label = 0
        Otherwise                      →  label = 0
    """

    def __init__(
        self,
        env,
        soc_reserve: float = DEFAULT_SR_SOC_RESERVE,
        prep_hours: int = DEFAULT_SR_PREP_HOURS,
        buffer_hours: int = DEFAULT_SR_BUFFER_HOURS,
    ):
        super().__init__(env)
        self._soc_reserve = soc_reserve
        self._prep_hours = prep_hours
        self._buffer_hours = buffer_hours

        self._episode_length: int = 720
        # Sets of timesteps in each window type (populated in _scenario_init)
        self._prep_timesteps: set = set()
        self._buffer_timesteps: set = set()
        # Sorted list of buffer-start timesteps (for steps_until computation)
        self._buffer_starts: list = []
        self._high_price_threshold: Optional[float] = None

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

        # ---- Price threshold for discharge-temptation detection ----
        raw_price_arrays = []
        for b in buildings:
            if hasattr(b, "pricing") and b.pricing is not None:
                ep = b.pricing.electricity_pricing
                if hasattr(ep, "__len__") and len(ep) > 0:
                    raw_price_arrays.append(np.asarray(ep[:n_hours], dtype=float))
        if raw_price_arrays:
            max_price_series = np.max(raw_price_arrays, axis=0)
            self._high_price_threshold = float(
                np.quantile(max_price_series, 0.65)
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

    def _compute_label(self, obs):
        """High-precision binary label for SolarRampReserve.

        label=1 only for:
          - active buffer reserve violation
          - buffer near-violation under high-price discharge temptation
          - one step before buffer while below reserve
          - prep window only if below reserve AND high price conflict

        Everything else is reward-owned (label=0).
        """
        self._ensure_base_init()
        ts = self._get_time_step()  # pre-step

        mean_soc = self._mean_electrical_soc_pre_step()
        if mean_soc is None:
            return 0

        soc_margin = 0.05
        steps = self._steps_until_next_buffer(ts)

        price = self._max_pre_step_price(self._get_buildings())
        high_price = (
            price is not None
            and self._high_price_threshold is not None
            and price >= self._high_price_threshold
        )

        # --- In post-sunset buffer ---
        if steps == 0:
            # Active cost violation.
            if mean_soc < self._soc_reserve:
                return 1
            # Near reserve only matters when reward is tempted to discharge.
            if mean_soc <= self._soc_reserve + soc_margin and high_price:
                return 1
            return 0

        # --- One step before buffer: urgent if already below reserve ---
        if steps == 1 and mean_soc < self._soc_reserve:
            return 1

        # --- Prep window: only cost-own if reward is actively conflicting ---
        if 1 < steps <= self._prep_hours and mean_soc < self._soc_reserve and high_price:
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
    event_starts: Optional[Sequence[int]] = DEFAULT_DR_EVENT_STARTS,
    event_duration: int = DEFAULT_DR_EVENT_DURATION,
    soc_target: float = DEFAULT_DR_SOC_TARGET,
    prep_window: int = DEFAULT_DR_PREP_WINDOW,
    high_price_quantile: float = DEFAULT_DR_HIGH_PRICE_QUANTILE,
    n_events: int = DEFAULT_DR_N_EVENTS,
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
        n_events=n_events,
    )


def make_demand_response_vec_env(
    env_name: str = DEFAULT_SCHEMA,
    n_envs: int = 1,
    seed: Optional[int] = None,
    episode_time_steps: Optional[int] = None,
    event_starts: Optional[Sequence[int]] = DEFAULT_DR_EVENT_STARTS,
    event_duration: int = DEFAULT_DR_EVENT_DURATION,
    soc_target: float = DEFAULT_DR_SOC_TARGET,
    prep_window: int = DEFAULT_DR_PREP_WINDOW,
    high_price_quantile: float = DEFAULT_DR_HIGH_PRICE_QUANTILE,
    n_events: int = DEFAULT_DR_N_EVENTS,
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
        "n_events": n_events,
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
    )


def make_solar_ramp_reserve_vec_env(
    env_name: str = DEFAULT_SCHEMA,
    n_envs: int = 1,
    seed: Optional[int] = None,
    episode_time_steps: Optional[int] = None,
    soc_reserve: float = DEFAULT_SR_SOC_RESERVE,
    prep_hours: int = DEFAULT_SR_PREP_HOURS,
    buffer_hours: int = DEFAULT_SR_BUFFER_HOURS,
    **kwargs,
):
    env_kwargs = {
        "env_name": env_name,
        "episode_time_steps": episode_time_steps,
        "soc_reserve": soc_reserve,
        "prep_hours": prep_hours,
        "buffer_hours": buffer_hours,
    }
    return make_cost_vec_env(
        make_solar_ramp_reserve_env,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
    )


# Backward-compatible aliases (old name → new)
CarbonAwareWrapper = DirtyWindowReserveWrapper
make_carbon_aware_env = make_dirty_window_reserve_env
make_carbon_aware_vec_env = make_dirty_window_reserve_vec_env
make_solar_contingency_env = make_solar_ramp_reserve_env
make_solar_contingency_vec_env = make_solar_ramp_reserve_vec_env
