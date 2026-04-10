##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
"""CityLearn environment wrappers for constrained RL experiments.

Four CMDP scenarios are provided (all share the same price-weighted reward):

A. **Cost vs Comfort** (``cost_vs_comfort``):
   cost   = max thermal discomfort across buildings,
   label  = violation or high-price + low-headroom frontier.

B. **Arbitrage vs Buffer** (``arbitrage_vs_buffer``):
   cost   = storage buffer depletion (SOC below safety threshold),
   label  = high price + SOC near safety frontier + usable storage.

C. **Contract Demand** (``contract_demand``):
   cost   = soft penalty for district import near/above fixed cap,
   label  = near-cap import + low price + usable storage.

D. **Carbon Aware** (``carbon_aware``):
   cost   = dirty-factor × normalised district import magnitude,
   label  = high carbon + low price + usable storage.

All share ``CityLearnBaseWrapper`` (base.py) which provides price-weighted
electricity-cost reward and raw building-state helpers.  All scenario
wrappers live in scenarios.py.

Usage from train.py
-------------------
Pass ``--env_type citylearn`` and select the scenario via ``env_kwargs``:

    --env_kwargs '{"scenario": "arbitrage_vs_buffer"}'   # default
    --env_kwargs '{"scenario": "contract_demand"}'
    --env_kwargs '{"scenario": "carbon_aware"}'
    --env_kwargs '{"scenario": "cost_vs_comfort"}'
"""

from env.citylearn.base import (
    CityLearnBaseWrapper,
    DEFAULT_SCHEMA,
    make_citylearn_inner_env,
)
from env.citylearn.scenarios import (
    CostVsComfortWrapper,
    ArbitrageVsBufferWrapper,
    ContractDemandWrapper,
    CarbonAwareWrapper,
    make_cost_vs_comfort_env,
    make_cost_vs_comfort_vec_env,
    make_arbitrage_vs_buffer_env,
    make_arbitrage_vs_buffer_vec_env,
    make_contract_demand_env,
    make_contract_demand_vec_env,
    make_carbon_aware_env,
    make_carbon_aware_vec_env,
)

SCENARIO_VEC_FACTORIES = {
    "cost_vs_comfort": make_cost_vs_comfort_vec_env,
    "arbitrage_vs_buffer": make_arbitrage_vs_buffer_vec_env,
    "contract_demand": make_contract_demand_vec_env,
    "carbon_aware": make_carbon_aware_vec_env,
}

__all__ = [
    "CityLearnBaseWrapper",
    "CostVsComfortWrapper",
    "ArbitrageVsBufferWrapper",
    "ContractDemandWrapper",
    "CarbonAwareWrapper",
    "DEFAULT_SCHEMA",
    "make_citylearn_inner_env",
    "make_cost_vs_comfort_env",
    "make_cost_vs_comfort_vec_env",
    "make_arbitrage_vs_buffer_env",
    "make_arbitrage_vs_buffer_vec_env",
    "make_contract_demand_env",
    "make_contract_demand_vec_env",
    "make_carbon_aware_env",
    "make_carbon_aware_vec_env",
    "SCENARIO_VEC_FACTORIES",
]
