##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################

from gymnasium import spaces
from typing import Tuple, Dict, Any
import numpy as np

from bsuite.utils import gym_wrapper
import dm_env
from dm_env import specs

_GymTimestep = Tuple[np.ndarray, float, bool, Dict[str, Any]]

class GymnasiumFromDMEnv(gym_wrapper.GymFromDMEnv):
    """A wrapper that converts a dm_env.Environment to an OpenAI gym.Env."""

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, env: dm_env.Environment):
        super().__init__(env)

    @property
    def action_space(self) -> spaces.Discrete:
        action_spec = self._env.action_spec()  # type: specs.DiscreteArray
        return spaces.Discrete(action_spec.num_values)

    def reset(self, seed: int = None) -> np.ndarray:
        obs =  super().reset()
        return obs, {}

    @property
    def observation_space(self) -> spaces.Box:
        obs_spec = self._env.observation_spec()  # type: specs.Array
        if isinstance(obs_spec, specs.BoundedArray):
            return spaces.Box(
                low=float(obs_spec.minimum),
                high=float(obs_spec.maximum),
                shape=tuple(dim for dim in obs_spec.shape if dim != 1),
                dtype=obs_spec.dtype)
        return spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=tuple(dim for dim in obs_spec.shape if dim != 1),
            dtype=obs_spec.dtype)

    def step(self, action: int) -> _GymTimestep:
        timestep = self._env.step(action)
        self._last_observation = timestep.observation
        reward = timestep.reward or 0.
        if timestep.last():
            self.game_over = True
        return timestep.observation, reward, timestep.last(), False, {}

    @property
    def reward_range(self) -> Tuple[float, float]:
        reward_spec = self._env.reward_spec()
        if isinstance(reward_spec, specs.BoundedArray):
            return reward_spec.minimum, reward_spec.maximum
        return -float('inf'), float('inf')

    def __getattr__(self, attr):
        """Delegate attribute access to underlying environment."""
        return getattr(self._env, attr)