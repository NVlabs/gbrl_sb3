##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
from gymnasium.envs.registration import register


class SymbolicSwappingEnv(gym.Env):
    def __init__(self, is_mixed: bool = False):
        super(SymbolicSwappingEnv, self).__init__()

        self.L_max = 11
        self.L_min = 2

        # Action space: (pick block index, place on target index)
        self.action_space = spaces.Discrete(self.L_max)

        # Observation space: color, size, placed_order for each block
        obs_shape = self.L_max if is_mixed else self.L_max*self.L_max

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.single)

        self.is_mixed = is_mixed
        self.max_steps = 50

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        # Initialize block states: [color, size, placed_order (0 = unplaced)]
        self.L = np.random.randint(self.L_min, high=self.L_max + 1)
        self.state = list(range(1, self.L+1))
        np.random.shuffle(self.state)
        self.correct_state = np.sort(self.state)
        return self._get_obs(), {}
    
    def _get_obs(self):
        # Encode state and add goal/constraint
        if self.is_mixed:
            obs = []
            for l in range(self.L_max):
                if l < self.L:
                    element = '='*self.state[l]
                    obs.append(element.encode('utf-8'))
                else:
                    obs.append('Null'.encode('utf-8'))
            obs = np.array(obs, dtype=object)
        else:
            obs = []
            for l in range(self.L_max):
                one_hot = [0] * self.L_max
                if l < self.L:
                    one_hot[self.state[l] - 1] = 1
                obs.extend(one_hot)
            obs = np.array(obs, dtype=np.single)
        return obs

    def step(self, action):
        self.step_count += 1
        reward = 0  
        terminated = False 
        truncated = False 

        if action >= self.L - 1:
            if action == self.L_max - 1: # reverse list
                self.state.reverse()
            else: # invalid
                terminated = True
        else: # swap elements
            tmp = self.state[action]
            self.state[action] = self.state[action + 1]
            self.state[action + 1] = tmp


        if np.array_equal(self.correct_state, self.state):
            terminated = True 
            reward = 1.0 - 0.9 * (self.step_count / self.max_steps)

        if self.step_count >= self.max_steps:
            truncated = True 
        return self._get_obs(), reward, terminated, truncated, {}


def register_symswap_tests():
    # PutNear
    # ----------------------------------------

    register(
        id="SymSwap-v0",
        entry_point="env.symbolic_swaping:SymbolicSwappingEnv",
        kwargs={},
    )