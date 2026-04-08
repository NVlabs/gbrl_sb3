##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
"""CityLearn environment wrappers for constrained RL experiments.

Three CMDP scenarios are provided (all share the same price-weighted reward):

1. **Arbitrage vs Buffer** (``arbitrage_vs_buffer``):
   cost   = storage buffer depletion (SOC below safety threshold),
   label  = high price + SOC near safety frontier + usable storage.

2. **Peak Shaving** (``peak_shaving``):
   cost   = daily peak demand overshoot,
   label  = near daily peak + high non-shiftable load + storage available
            + price not extreme.

3. **Carbon Aware** (``carbon_aware``):
   cost   = carbon-weighted grid import,
   label  = high carbon + low price + usable storage (the disagreement).

All share the same base wrapper (``CityLearnBaseWrapper``) which provides
explicit price-weighted electricity-cost reward and raw building-state
helpers.

Usage from train.py
-------------------
Pass ``--env_type citylearn`` and select the scenario via ``env_kwargs``:

    --env_kwargs '{"scenario": "arbitrage_vs_buffer"}'   # default
    --env_kwargs '{"scenario": "peak_shaving"}'
    --env_kwargs '{"scenario": "carbon_aware"}'
"""

from env.citylearn.base import (
    CityLearnBaseWrapper,
    DEFAULT_SCHEMA,
    make_citylearn_inner_env,
)
from env.citylearn.arbitrage_vs_buffer import (
    ArbitrageVsBufferWrapper,
    make_arbitrage_vs_buffer_env,
    make_arbitrage_vs_buffer_vec_env,
)
from env.citylearn.peak_shaving import (
    PeakShavingWrapper,
    make_peak_shaving_env,
    make_peak_shaving_vec_env,
)
from env.citylearn.carbon_aware import (
    CarbonAwareWrapper,
    make_carbon_aware_env,
    make_carbon_aware_vec_env,
)

SCENARIO_VEC_FACTORIES = {
    "arbitrage_vs_buffer": make_arbitrage_vs_buffer_vec_env,
    "peak_shaving": make_peak_shaving_vec_env,
    "carbon_aware": make_carbon_aware_vec_env,
}

__all__ = [
    "CityLearnBaseWrapper",
    "ArbitrageVsBufferWrapper",
    "PeakShavingWrapper",
    "CarbonAwareWrapper",
    "DEFAULT_SCHEMA",
    "make_citylearn_inner_env",
    "make_arbitrage_vs_buffer_env",
    "make_arbitrage_vs_buffer_vec_env",
    "make_peak_shaving_env",
    "make_peak_shaving_vec_env",
    "make_carbon_aware_env",
    "make_carbon_aware_vec_env",
    "SCENARIO_VEC_FACTORIES",
]
