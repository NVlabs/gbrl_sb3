##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
"""CityLearn environment wrappers for constrained RL experiments.

Six CMDP scenarios are provided (all share the same price-weighted reward):

PeakGuard. **Arbitrage vs Buffer** (``arbitrage_vs_buffer``):
   cost   = critical-battery depletion (min SOC below safety threshold),
   label  = high price + tight reserve frontier + dischargeable storage.

CapFrontier. **Contract Demand** (``contract_demand``):
   cost   = soft penalty for district import near/above fixed cap,
   label  = near-cap import + low price + usable storage.

DirtyReserve. **Dirty Window Reserve** (``dirty_window_reserve`` / ``carbon_aware``):
   cost   = SOC deficit during dirty-grid windows,
   label  = dirty + expensive price (discharge temptation vs hold SOC).

PeakRatchet. **Peak Ratchet** (``peak_ratchet``):
   cost   = incremental peak penalty on new district import records,
   label  = cheap price + potential import above evolving ratchet.

EventReserve. **Demand Response** (``demand_response``):
   cost   = SOC deficit during pre-announced grid events,
   label  = phase-aware pre-event conflict (prep/hold/recovery).

SunsetBridge. **Solar Ramp Reserve** (``solar_ramp_reserve``):
   cost   = reserve deficit during post-sunset buffer windows,
   label  = phase-aware sunset routing (prep/hold/recovery).

All share ``CityLearnBaseWrapper`` (base.py) which provides price-weighted
electricity-cost reward and raw building-state helpers.  All scenario
wrappers live in scenarios.py.

Usage from train.py
-------------------
Pass ``--env_type citylearn`` and select the scenario via ``env_kwargs``:

    --env_kwargs '{"scenario": "arbitrage_vs_buffer"}'   # default
    --env_kwargs '{"scenario": "contract_demand"}'
    --env_kwargs '{"scenario": "dirty_window_reserve"}'  # or "carbon_aware"
    --env_kwargs '{"scenario": "peak_ratchet"}'
    --env_kwargs '{"scenario": "demand_response"}'
    --env_kwargs '{"scenario": "solar_ramp_reserve"}'
"""

from env.citylearn.base import (
    CityLearnBaseWrapper,
    DEFAULT_SCHEMA,
    make_citylearn_inner_env,
)
from env.citylearn.scenarios import (
    ArbitrageVsBufferWrapper,
    ContractDemandWrapper,
    DirtyWindowReserveWrapper,
    CarbonAwareWrapper,  # backward-compatible alias
    PeakRatchetWrapper,
    DemandResponseWrapper,
    SolarRampReserveWrapper,
    make_arbitrage_vs_buffer_env,
    make_arbitrage_vs_buffer_vec_env,
    make_contract_demand_env,
    make_contract_demand_vec_env,
    make_dirty_window_reserve_env,
    make_dirty_window_reserve_vec_env,
    make_carbon_aware_env,  # backward-compatible alias
    make_carbon_aware_vec_env,  # backward-compatible alias
    make_peak_ratchet_env,
    make_peak_ratchet_vec_env,
    make_demand_response_env,
    make_demand_response_vec_env,
    make_solar_ramp_reserve_env,
    make_solar_ramp_reserve_vec_env,
)

SCENARIO_VEC_FACTORIES = {
    "arbitrage_vs_buffer": make_arbitrage_vs_buffer_vec_env,
    "contract_demand": make_contract_demand_vec_env,
    "carbon_aware": make_dirty_window_reserve_vec_env,
    "dirty_window_reserve": make_dirty_window_reserve_vec_env,
    "peak_ratchet": make_peak_ratchet_vec_env,
    "demand_response": make_demand_response_vec_env,
    "solar_ramp_reserve": make_solar_ramp_reserve_vec_env,
}

__all__ = [
    "CityLearnBaseWrapper",
    "ArbitrageVsBufferWrapper",
    "ContractDemandWrapper",
    "DirtyWindowReserveWrapper",
    "CarbonAwareWrapper",  # backward-compatible alias
    "PeakRatchetWrapper",
    "DemandResponseWrapper",
    "SolarRampReserveWrapper",
    "DEFAULT_SCHEMA",
    "make_citylearn_inner_env",
    "make_arbitrage_vs_buffer_env",
    "make_arbitrage_vs_buffer_vec_env",
    "make_contract_demand_env",
    "make_contract_demand_vec_env",
    "make_dirty_window_reserve_env",
    "make_dirty_window_reserve_vec_env",
    "make_carbon_aware_env",  # backward-compatible alias
    "make_carbon_aware_vec_env",  # backward-compatible alias
    "make_peak_ratchet_env",
    "make_peak_ratchet_vec_env",
    "make_demand_response_env",
    "make_demand_response_vec_env",
    "make_solar_ramp_reserve_env",
    "make_solar_ramp_reserve_vec_env",
    "SCENARIO_VEC_FACTORIES",
]
