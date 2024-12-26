##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register

class EquationEnv(gym.Env):
    """
    Custom Gym Environment for Chain Construction Task.
    The agent must construct a valid sequence of actions (e.g., A -> B -> C)
    based on logical dependencies.
    """
    
    def __init__(self):
        super(EquationEnv, self).__init__()

        # actions are add/ subtract/ divide/ multiply and multiply by -1
        
        self.action_space = spaces.Discrete(4*9 + 1) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3, ), dtype=float)
        self.step_count = 0
        self.max_steps = 50

    
    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        nums = [] 
        for _ in range(3):
            digit = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
            sign = np.random.choice([-1, 1])
            nums.append(digit*sign)
        self.step_count = 0
        self.state = np.array(nums, dtype=np.single)
        return self.state, {}
    
    def _gen_state(self, action):
        state = self.state
        action_type = action // 9
        action_number = action % 9
        if action_type == 5:
            state = -state 
        elif action_type == 0:
            state[1] += action_number + 1
            state[2] += action_number + 1
        elif action_type == 1:
            state[1] -= action_number+ 1
            state[2] -= action_number + 1
        elif action_type == 2:
            state = state / (action_number + 1)
        else:
            state = state * (action_number + 1)
        self.state = state
        return state
    
    def step(self, action):
        """Take an action and return the next state, reward, and done flag."""
        reward = 0
        terminated = False 
        truncated = False
        info = {}
        prev_state = self.state.copy()
        state = self._gen_state(action)
        if state[0] == 1 and state[1] == 0:  # Isolating x condition
            reward = 1.0
            terminated = True
        # elif state[1] == 0 and prev_state[1] != 0:  # First time isolating the constant
        #     reward = 0.5
        # elif prev_state[1] == 0 and state[1] == 0:  # Repeating an invalid step
        #     terminated = True

        if np.isnan(state).any() or np.isinf(state).any():
            reward = -1
            terminated = True
            
        if state[0] == 0:
            terminated = True

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True
        return state, reward, terminated, truncated, info
    

def register_equation_tests():
    # PutNear
    # ----------------------------------------

    register(
        id="Equation-v0",
        entry_point="env.equation:EquationEnv",
        kwargs={},
    )