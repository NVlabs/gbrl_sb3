##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
"""CityLearn environment wrappers for constrained RL experiments.

Two CMDP scenarios are provided:

1. **Cost vs Comfort** (``cost_vs_comfort``):
   reward = negative electricity cost,
   cost   = thermal discomfort,
   label  = high price + low thermal headroom + usable storage.

2. **Arbitrage vs Buffer** (``arbitrage_vs_buffer``):
   reward = negative electricity cost,
   cost   = storage buffer depletion,
   label  = high price + usable storage.

Both share the same base wrapper (``CityLearnBaseWrapper``) which provides
explicit price-weighted electricity-cost reward and raw building-state
helpers.

Usage from train.py
-------------------
Pass ``--env_type citylearn`` and select the scenario via ``env_kwargs``:

    --env_kwargs '{"scenario": "cost_vs_comfort"}'      # default
    --env_kwargs '{"scenario": "arbitrage_vs_buffer"}'
"""

from env.citylearn.base import (
    CityLearnBaseWrapper,
    DEFAULT_SCHEMA,
    make_citylearn_inner_env,
)
from env.citylearn.cost_vs_comfort import (
    CostVsComfortWrapper,
    make_cost_vs_comfort_env,
    make_cost_vs_comfort_vec_env,
)
from env.citylearn.arbitrage_vs_buffer import (
    ArbitrageVsBufferWrapper,
    make_arbitrage_vs_buffer_env,
    make_arbitrage_vs_buffer_vec_env,
)

SCENARIO_VEC_FACTORIES = {
    "cost_vs_comfort": make_cost_vs_comfort_vec_env,
    "arbitrage_vs_buffer": make_arbitrage_vs_buffer_vec_env,
}

__all__ = [
    "CityLearnBaseWrapper",
    "CostVsComfortWrapper",
    "ArbitrageVsBufferWrapper",
    "DEFAULT_SCHEMA",
    "make_citylearn_inner_env",
    "make_cost_vs_comfort_env",
    "make_cost_vs_comfort_vec_env",
    "make_arbitrage_vs_buffer_env",
    "make_arbitrage_vs_buffer_vec_env",
    "SCENARIO_VEC_FACTORIES",
]
