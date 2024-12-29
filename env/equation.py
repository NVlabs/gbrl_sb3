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
    
    def __init__(self, with_history: bool = False, is_mixed: bool = False):
        super(EquationEnv, self).__init__()

        # actions are add/ subtract/ divide/ multiply and multiply by -1
        
        # self.action_space = spaces.Discrete(4*9 + 1) 
        # self.action_space = spaces.Discrete(4*9 + 1) 
        self.n_action_types = 4
        self.digits = 9
        self.inverse = 2
        self.with_history = with_history

        self.coef = 3
        shape = self.coef
        if self.with_history:
            if is_mixed:
                shape = self.coef + 3
            else:
                shape = self.coef + self.n_action_types + self.digits + self.inverse
        self.action_space = spaces.MultiDiscrete([self.n_action_types, self.digits, self.inverse])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(shape, ), dtype=float)
        self.step_count = 0
        self.max_steps = 50
        self.is_mixed = is_mixed

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        nums = [] 
        for _ in range(3):
            digit = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
            sign = np.random.choice([-1, 1])
            nums.append(digit*sign)
        if self.with_history:
            if self.is_mixed:
                nums.append('None'.encode('utf-8'))
                nums.append(0)
                nums.append('None'.encode('utf-8'))
            else:
                one_hot = [0] * self.n_action_types
                nums.extend(one_hot)
                one_hot = [0] * self.digits
                nums.extend(one_hot)
                one_hot = [0] * self.inverse
                nums.extend(one_hot)
        self.step_count = 0
        self.state = np.array(nums, dtype=object if self.is_mixed else np.single)
        return self.state, {}
    
    def _gen_state(self, action):
        state = self.state
        action_type, action_number, minus_1 = action
        # action_type = action // 9
        # action_number = action % 9
        if minus_1:
            state[:self.coef] = -state[:self.coef] 
        elif action_type == 0:
            state[1] += action_number + 1
            state[2] += action_number + 1
        elif action_type == 1:
            state[1] -= action_number + 1
            state[2] -= action_number + 1
        elif action_type == 2:
            state[:self.coef] = state[:self.coef] / (action_number + 1)
        else:
            state[:self.coef] = state[:self.coef] * (action_number + 1)
        
        if self.with_history:
            if self.is_mixed: 
                state[self.coef] = str(action_type).encode('utf-8')
                state[self.coef + 1] = action_number + 1
                state[self.coef + 2] = str(bool(minus_1)).encode('utf-8')
            else:
                state[self.coef:] = 0
                state[self.coef + action_type] = 1 
                state[self.coef + self.n_action_types + action_number] = 1 
                state[self.coef + self.n_action_types + self.digits + minus_1] = 1 

        self.state = state
        return state
    
    def step(self, action):
        """Take an action and return the next state, reward, and done flag."""

        reward = 0
        terminated = False 
        truncated = False
        info = {}
        state = self._gen_state(action)
        if state[0] == 1 and state[1] == 0:  # Isolating x condition
            reward = 1.0
            terminated = True
        # elif state[1] == 0 and prev_state[1] != 0:  # First time isolating the constant
        #     reward = 0.5
        # elif prev_state[1] == 0 and state[1] == 0:  # Repeating an invalid step
        #     terminated = True

        if not self.is_mixed and (np.isnan(state).any() or np.isinf(state).any()):
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
    register(
        id="Equation-v1",
        entry_point="env.equation:EquationEnv",
        kwargs={'with_history': True},
    )