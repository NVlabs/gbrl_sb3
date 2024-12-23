##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################

import gymnasium as gym 
from typing import Any
import numpy as np 
from gymnasium.utils import seeding
from gymnasium.envs.registration import register
from gymnasium.core import ObsType, ActType


class Extrapolation(gym.Env):
    def __init__(self, discrete_actions: bool = False, dS: float = 0.1, goal: float = 0.0, max_steps: int = 100, obs_range: tuple = (-1, 1)):
        self.observation_space = gym.spaces.Box(low=-np.inf,
            high=np.inf,
            shape=(1, ),
            dtype=float,)
         
        if discrete_actions:
            self.action_space =  gym.spaces.Discrete(2)
            self.goal_tolerance = dS
        else:
            self.action_space = gym.spaces.Box(low=-1,
                high=1,
                shape=(1, ),
                dtype=float,)
            self.goal_tolerance = 0.05
        self.discrete_actions = discrete_actions
        self.dS = np.array([dS])
        self.goal = goal
        self.obs = None
        self.step_count = 0
        self.max_steps = max_steps
        self.obs_range = obs_range
            
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]: 
        
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        obs = np.array([self._np_random.uniform(*self.obs_range)])
        self.obs = obs.copy()
        self.step_count = 0
        return obs, {}

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        if not isinstance(action, np.ndarray) and not isinstance(self.action_space, gym.spaces.Discrete):
            # action = np.clip(action, -1, 1)
            action = -1 if action == 0 else 1
        # if isinstance(action, np.ndarray) or not isinstance(self.action_space, gym.spaces.Discrete):
        #     # action = np.clip(action, -1, 1)
        # else:
            

        observation = self.obs + action * self.dS if self.discrete_actions else self.obs + action
        self.step_count += 1
        self.obs  = observation.copy()
        terminated = abs(self.obs - self.goal) <= self.goal_tolerance
        reward = -(self.obs - self.goal)**2
        # reward = 1 if terminated else 0
        truncated =  self.step_count >= self.max_steps
        return observation, reward, bool(terminated), bool(truncated), {}
    

def register_extrapolation_tests():
    # PutNear
    # ----------------------------------------

    register(
        id="Extrapolation-Discrete-train",
        entry_point="env.extrapolation:Extrapolation",
        kwargs={"discrete_actions": True},
    )
    register(
        id="Extrapolation-Discrete-test",
        entry_point="env.extrapolation:Extrapolation",
        kwargs={"discrete_actions": True, "obs_range": (-10, 10)},
    )
    register(
        id="Extrapolation-Continuous-train",
        entry_point="env.extrapolation:Extrapolation",
        kwargs={"discrete_actions": False},
    )
    register(
        id="Extrapolation-Continuous-test",
        entry_point="env.extrapolation:Extrapolation",
        kwargs={"discrete_actions": False, "obs_range": (-10, 10)},
    )
