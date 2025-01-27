##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register


class LinearEquationEnv(gym.Env):
    """
    Custom Gym Environment for isolating a linear equation: start with: ax + b = c
    Goal: Isolate x
    """
    
    def __init__(self, with_history: bool = False, is_mixed: bool = False):
        super(LinearEquationEnv, self).__init__()

        # actions are add/ subtract/ divide/ multiply and multiply by -1
        
        self.n_action_types = 3
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
        action_type, action_number, sign_idx = action
        # action_type = action // 9
        # action_number = action % 9
        sign = 2*sign_idx - 1
        # if minus_1:
        #     state[:self.coef] = -state[:self.coef] 
        if action_type == 0:
            state[1] += sign*(action_number + 1)
            state[2] += sign*(action_number + 1)
        elif action_type == 1:
            state[:self.coef] = sign*state[:self.coef] / (action_number + 1)
        else:
            state[:self.coef] = sign*state[:self.coef] * (action_number + 1)
        
        if self.with_history:
            if self.is_mixed: 
                state[self.coef] = str(action_type).encode('utf-8')
                state[self.coef + 1] = action_number + 1
                state[self.coef + 2] = str(bool(sign_idx)).encode('utf-8')
            else:
                state[self.coef:] = 0
                state[self.coef + action_type] = 1 
                state[self.coef + self.n_action_types + action_number] = 1 
                state[self.coef + self.n_action_types + self.digits + sign_idx] = 1 

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
            reward = 1.0 - 0.9 * (self.step_count / self.max_steps)
            terminated = True

        if not self.is_mixed and (np.isnan(state).any() or np.isinf(state).any()):
            terminated = True
            
        if state[0] == 0:
            terminated = True

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True
        return state, reward, terminated, truncated, info

class BalancedTwoVariableLinearEquationEnv(gym.Env):
    """
    Custom Gym Environment for isolating a linear equation: start with: ax + b = cy + d
    Goal: isolate y to the form = y = (d - b - ax) / c
    """
    
    def __init__(self):
        super(BalancedTwoVariableLinearEquationEnv, self).__init__()

        # actions are add/ subtract/ divide/ multiply /add x / subtract x /add y /subtract y and multiply by -1
        self.indices = {'a': 0, 'b': 1, 'y': 2, 'c': 3, 'd': 4, 'x': 5}
        self.n_action_types = 5
        self.digits = 9
        self.sign_type = 2
        self.coef = 6
        shape = self.coef
        
        self.action_space = spaces.MultiDiscrete([self.n_action_types, self.digits, self.sign_type])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(shape, ), dtype=float)
        self.step_count = 0
        self.max_steps = 50

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        nums = [] 
        for i in range(6):
            digit = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
            sign = np.random.choice([-1, 1])
            if i == 2 or i == 5:
                nums.append(0)
            else:
                nums.append(digit*sign)
        

        self.step_count = 0
        self.state = np.array(nums, dtype=np.single)
        self.x_bonus_given = False
        self.y_bonus_given = False
        self.moved_b = False
        return self.state, {}

    def _get_bool(self, value):
        if isinstance(value, str):
            return True if value == 'True' else False 
        return bool(value)
    
    def _gen_state(self, action):
        # ax + b + y_place_holder = cy + d + x_place_holder
        indices = self.indices
        state = self.state
        self.indices = {'a': 0, 'b': 1, 'y': 2, 'c': 3, 'd': 4, 'x': 5, 'x_pos': 6, 'y_pos': 7}

        action_type, action_number, sign_type = action

        sign = 2*sign_type - 1

        if action_type == 0:
            state[indices['b']] += sign*(action_number + 1)
            state[indices['d']] += sign*(action_number + 1)
        elif action_type == 1:
            state[:self.coef] = sign*state[:self.coef] / (action_number + 1)
        elif action_type == 2:
            state[:self.coef] = sign*state[:self.coef] * (action_number + 1)
        elif action_type == 3:
            state[indices['a']] += sign*(action_number + 1)
            state[indices['x']] += sign*(action_number + 1)
        elif action_type == 4:
            state[indices['y']] += sign*(action_number + 1)
            state[indices['c']] += sign*(action_number + 1)
        

        self.state = state
        return state
    
    def step(self, action):
        """Take an action and return the next state, reward, and done flag."""
        indices = self.indices
        reward = 0
        terminated = False 
        truncated = False
        info = {}
        state = self._gen_state(action)
        # ax + b + y_place_holder = cy + d + x_place_holder
        if state[indices['a']] == 0 and state[indices['b']] == 0 and state[indices['y']] == 1 and state[indices['c']] == 0:  # Isolating x condition
            reward = 1.0 - 0.9 * (self.step_count / self.max_steps) - 0.3  # remove bonuses
            terminated = True

        if state[indices['a']] == 0 and not self.x_bonus_given:
            reward += 0.1
            self.x_bonus_given = True

        if state[indices['c']] == 0 and not self.y_bonus_given:
            reward += 0.1
            self.y_bonus_given = True

        if state[indices['b']] == 0 and not self.moved_b:
            reward += 0.1
            self.moved_b = True


        if (np.isnan(state).any() or np.isinf(state).any()):
            # reward = -1
            terminated = True

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True
        return state, reward, terminated, truncated, info

class TwoVariableLinearEquationEnv(gym.Env):
    """
    Custom Gym Environment for isolating a linear equation: start with: ax + by + c = d
    Goal: Isolate x
    """
    
    def __init__(self):
        super(TwoVariableLinearEquationEnv, self).__init__()

        # actions are add/ subtract/ divide/ multiply /add x / subtract x and multiply by -1
        
        self.n_action_types = 4
        self.digits = 9
        self.sign_type = 2
        self.coef = 5 
        shape = self.coef
        self.indices = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'x': 4}
        
        self.action_space = spaces.MultiDiscrete([self.n_action_types, self.digits, self.sign_type])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(shape, ), dtype=float)
        self.step_count = 0
        self.max_steps = 50

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        nums = [] 
        for _ in range(4):
            digit = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
            sign = np.random.choice([-1, 1])
            nums.append(digit*sign)
        
        nums.append(0)

        self.step_count = 0
        self.state = np.array(nums, dtype=np.single)
        self.moved_x = False
        self.moved_c = False
        return self.state, {}
    
    def _gen_state(self, action):
        # ax + by + c = d + place_holder_x
        state = self.state.copy()
        action_type, action_number, sign_type = action
        sign = 2*sign_type - 1
        if action_type == 0:
            state[self.indices['c']] += sign*(action_number + 1)
            state[self.indices['d']] += sign*(action_number + 1)
        elif action_type == 1:
            state = sign*state / (action_number + 1)
        elif action_type == 2:
            state = sign*state * (action_number + 1)
        elif action_type == 3:
            state[self.indices['a']] += sign*(action_number + 1)
            state[self.indices['x']] += sign*(action_number + 1)
        self.state = state.copy()
        return state
    
    def step(self, action):
        """Take an action and return the next state, reward, and done flag."""

        reward = 0
        terminated = False 
        truncated = False
        info = {}
        state = self._gen_state(action)

        if state[self.indices['a']] == 0 and not self.moved_x:
            reward += 0.1
            self.moved_x = True
        if state[self.indices['c']] == 0 and not self.moved_c:
            reward += 0.1
            self.moved_c = True

        if state[self.indices['a']] == 0 and state[self.indices['b']] == 1 and state[self.indices['c']] == 0:  # Isolating x condition
            reward = 1.0 - 0.9 * (self.step_count / self.max_steps) - 0.2 # remove bonus
            terminated = True

        if (np.isnan(state).any() or np.isinf(state).any()):
            terminated = True
            
        if state[self.indices['b']] == 0:
            terminated = True

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True
        return state, reward, terminated, truncated, info
    

def register_equation_tests():
    # PutNear
    # ----------------------------------------

    register(
        id="LinearEquation-v0",
        entry_point="env.equation:LinearEquationEnv",
        kwargs={},
    )
    register(
        id="StrLinearEquation-v0",
        entry_point="env.equation:StrLinearEquationEnv",
        kwargs={},
    )
    register(
        id="TwoVariableLinearEquation-v0",
        entry_point="env.equation:TwoVariableLinearEquationEnv",
        kwargs={},
    )
    register(
        id="BalancedTwoVariableLinearEquation-v0",
        entry_point="env.equation:BalancedTwoVariableLinearEquationEnv",
        kwargs={},
    )