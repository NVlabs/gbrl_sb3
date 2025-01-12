##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register




class LinearEquationEnv(gym.Env):
    """
    Custom Gym Environment for isolating a linear equation: start with: ax + b = c
    Goal: Isolate x
    """
    
    def __init__(self, with_history: bool = False, is_mixed: bool = False):
        super(LinearEquationEnv, self).__init__()

        # actions are add/ subtract/ divide/ multiply and multiply by -1
        
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
            reward = 1.0 - 0.9 * (self.step_count / self.max_steps)
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


class StrLinearEquationEnv(gym.Env):
    """
    Custom Gym Environment for isolating a linear equation: start with: ax + b = c
    Goal: Isolate x
    """
    
    def __init__(self, is_mixed: bool = False):
        super(StrLinearEquationEnv, self).__init__()

        # actions are add/ subtract/ divide/ multiply
        
        self.n_action_types = 3
        self.digits = 9
        self.inverse = 2
        self.max_length = 14
        self.vocabulary = list('0123456789+-=/x ')  # Added space for padding
        self.char_to_idx = {char: i for i, char in enumerate(self.vocabulary)}
        self.idx_to_char = {i: char for char, i in self.char_to_idx.items()}

        shape = self.max_length*len(self.vocabulary) if not is_mixed else self.max_length
        self.action_space = spaces.MultiDiscrete([self.n_action_types, self.digits, self.inverse])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(shape, ), dtype=float)
        self.step_count = 0
        self.max_steps = 50
        self.max_num = 100
        self.is_mixed = is_mixed

    def _get_observation(self):
        if self.is_mixed:
           return np.array([s.encode('utf-8') for s in self.state], dtype=object)
        state = []
        for i in range(self.max_length):
            one_hot = [0] * len(self.vocabulary) 
            one_hot[self.char_to_idx[self.state[i]]] = 1
            state.extend(one_hot)
        return np.array(state, dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        state = [] 
        # Format: sign digit/digit x sign digit/digit = sign digit/digit
        # sign 0 digit 1 / 2 digit 3 x 4       
        # sign 5 digit 6 / 7 digit 8 = 9
        # sign 10 digit 11 / 12 digit 13        
    
        for i in range(3):
            digit = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
            sign = np.random.choice([-1, 1])
            sign_ = '-' if sign == -1 else ' '
            if sign == 1 and i == 1:
                sign_ = '+'
            
            dig = str(digit) 
            
            state.extend([
                sign_,
                dig,
                ' ',
                ' '
            ])
            if i == 0:
                state.append('x')
            elif i == 1:
                state.append('=') 

        self.step_count = 0
        self.state = state
        self.constant_on_left = True
        self.x_was_valid = state[1] == '1' and state[0] == ' '
        self.progress_reward = 0.0
        self.past_b = abs(int(state[6]))
        return self._get_observation(), {}
    
    def _add_digit(self, sign_char, numerator, denominator, digit):
        sign = -1 if sign_char == '-' else 1
        numer = int(numerator)
        res = sign*numer + digit if denominator == ' ' else sign*numer + digit*int(denominator)
        res_sign = '-' if res < 0 else ' '
        res = abs(res)
        res = min(res, self.max_num)
        return str(res), res_sign
    
    def _subtract_digit(self, sign_char, numerator, denominator, digit):
        sign = -1 if sign_char == '-' else 1
        numer = int(numerator)
        res = sign*numer - digit if denominator == ' ' else sign*numer - digit*int(denominator)
        res_sign = '-' if res < 0 else ' '
        res = abs(res)
        res = min(res, self.max_num)
        return str(res), res_sign

    def _divide_digit(self, numerator, denominator, digit):
        if numerator == '0':
            return '0', ' ', ' '
        if denominator == ' ':
            return numerator, '/', str(digit)
        res = int(denominator)*digit
        res = min(res, self.max_num)
        if res < 0:
            print(f"res: {res} of {denominator}*{digit} < 0")
        return numerator, '/', str(res)

    def _multiply_digit(self, numerator, digit):
        if numerator == '0':
            return '0'

        numer = int(numerator)
        res = numer * digit
        res = min(res, self.max_num)
        return str(res)
        
    def _set_gcd(self, numerator, divide_char, denominator):
        numer = abs(int(numerator))
        if divide_char == ' ':
            return numerator, ' ', ' ', numer > self.max_num
        elif denominator == '1' or denominator == ' ':
            return numerator, ' ', ' ', numer > self.max_num
        denom = abs(int(denominator))
        gcd = math.gcd(numer, denom)
        if gcd == 0:
            print(f"GCD failure: numerator={numer}, denominator={denom}")
        numer = numer // gcd
        denom = denom // gcd 
        large_res = numer >= self.max_num or denom >= self.max_num
        if denom == 1:
            return str(numer), ' ', ' ', large_res
        return str(numer), '/', str(denom), large_res

    def _gen_state(self, action):
        state = self.state
        action_type, action_number, action_sign = action
        # Format: sign digit/digit x sign digit/digit = sign digit/digit
        # sign 0 digit 1 / 2 digit 3 x 4       
        # sign 5 digit 6 / 7 digit 8 = 9
        action_sign = bool(action_sign)
        # sign 10 digit 11 / 12 digit 13 
        large_res = False

        if action_type == 0:
            if action_sign:
                state[6], state[5] = self._add_digit(state[5], state[6], state[8], action_number + 1)
                state[11], state[10] = self._add_digit(state[10], state[11], state[13], action_number + 1)
            else:
                state[6], state[5] = self._subtract_digit(state[5], state[6], state[8], action_number + 1)
                state[11], state[10] = self._subtract_digit(state[10], state[11], state[13], action_number + 1)
        elif action_type == 1:
            state[1], state[2], state[3] = self._divide_digit(state[1], state[3], action_number + 1)
            state[6], state[7], state[8] = self._divide_digit(state[6], state[8], action_number + 1)
            state[11], state[12], state[13] = self._divide_digit(state[11], state[13], action_number + 1)
        else:
            state[1] = self._multiply_digit(state[1], action_number + 1)
            state[6] = self._multiply_digit(state[6], action_number + 1)
            state[11] = self._multiply_digit(state[11], action_number + 1)
        if action_type in [1, 2] and not action_sign:
            state[0] = '-' if state[0] == ' ' else ' '
            state[5] = '-' if state[5] == '+' or state[5] == ' '  else '+'
            state[10] = '-' if state[10] == ' ' else ' '

        if state[6] == '0':
            state[5] = ' '
            state[7] = ' '
            state[8] = ' '

        if state[5] == ' ':
            state[5] = '+'
        if state[11] == '0':
            state[10] = ' '
            state[12] = ' '
            state[13] = ' '
        
        state[1], state[2], state[3], large_res = self._set_gcd(state[1], state[2], state[3])
        state[6], state[7], state[8], large_res = self._set_gcd(state[6], state[7], state[8])
        state[11], state[12], state[13], large_res = self._set_gcd(state[11], state[12], state[13])
        
        self.state = state
        return self._get_observation(), large_res
    
    def step(self, action):
        """Take an action and return the next state, reward, and done flag."""

        reward = 0
        terminated = False 
        truncated = False
        info = {}
        prev_state = self.state.copy()
        obs, large_res = self._gen_state(action)
        state = self.state
        
        x_positive = state[0] == ' '
        x_valid = x_positive and state[1] == '1' and state[2] == ' ' and state[3] == ' '
        constant_valid = state[6] == '0'

        # if constant_valid and self.constant_on_left:
        #     self.constant_on_left = False 
        #     reward += 0.1
        #     self.progress_reward += 0.1
        # if x_valid and not self.x_was_valid:
        #     reward += 0.1
        #     self.progress_reward += 0.1
        #     self.x_was_valid = True

        # if state[7] == ' ' and abs(int(state[6])) < self.past_b:
        #     reward = 0.05 / self.max_steps
        #     self.progress_reward += reward
        #     self.past_b = abs(int(state[6]))
        
        action_type, action_number, action_sign = action
            # self.x_was_valid = True
        if x_valid and constant_valid:  # Isolating x condition
            reward = 1.0 - 0.9 * (self.step_count / self.max_steps)
            # reward = 1.0 - 0.9 * (self.step_count / self.max_steps) 
            terminated = True
        # else:
        #     if constant_valid and action_type in [0]:
        #         reward -= 0.05
        #     if not constant_valid and state[7] == ' ' and action_type in [1, 2]:
        #         reward -= 0.05
                # terminated = True
            # elif x_positive and minus_1:
            #     reward -= 0.1


        
            # if self.fraction_bonus_given:
            #     reward -= 0.1       
        if large_res:
            terminated = True

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True
        # if terminated or truncated:
        #     reward -= self.progress_reward
        # if terminated or truncated:
            # action_type, action_number, action_sign = action
            # cats = {0: 'add', 1: 'subtract', 2: 'divide', 3: 'mult'}
            # print(f' reward: {reward}, prev_state: {prev_state}, action_type: {cats[action_type]} action_number: {action_number + 1}, minus-1: {bool(minus_1)}, state: {state}, terminated:{terminated}, truncated: {truncated}')
        return obs, reward, terminated, truncated, info

class BalancedTwoVariableLinearEquationEnv(gym.Env):
    """
    Custom Gym Environment for isolating a linear equation: start with: ax + b = cy + d
    Goal: isolate y to the form = y = (d - b - ax) / c
    """
    
    def __init__(self, is_mixed: bool = False):
        super(BalancedTwoVariableLinearEquationEnv, self).__init__()

        # actions are add/ subtract/ divide/ multiply /add x / subtract x /add y /subtract y and multiply by -1
        self.indices = {'a': 0, 'b': 1, 'y': 2, 'c': 3, 'd': 4, 'x': 5, 'x_pos': 6, 'y_pos': 7}
        self.n_action_types = 4
        self.digits = 9
        self.additional = 6
        self.coef = 6
        shape = self.coef + 2
        
        self.action_space = spaces.MultiDiscrete([self.n_action_types, self.digits, self.additional])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(shape, ), dtype=float)
        self.step_count = 0
        self.max_steps = 50
        self.is_mixed = is_mixed

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
        
        nums.extend([
                     'True'.encode('utf-8') if self.is_mixed else 1,
                     'False'.encode('utf-8') if self.is_mixed else 1,
                     ])

        self.step_count = 0
        self.state = np.array(nums, dtype=object if self.is_mixed else np.single)
        self.prev_x_pos = True
        self.prev_y_pos = True
        self.x_bonus_given = False
        self.y_bonus_given = False
        return self.state, {}
    
    def _gen_state(self, action):
        # ax + b + y_place_holder = cy + d + x_place_holder
        indices = self.indices
        state = self.state
        x_on_left = state[indices['x_pos']]
        if self.is_mixed:
            x_on_left = True if x_on_left == 'True' else False 
        else:
            x_on_left = bool(x_on_left)
        y_on_right = state[indices['y_pos']]
        if self.is_mixed:
            y_on_right = True if y_on_right == 'True' else False 
        else:
            y_on_right = bool(y_on_right)
        action_type, action_number, additional = action
        if additional == 0:
            if action_type == 0:
                state[indices['b']] += action_number + 1
                state[indices['d']] += action_number + 1
            elif action_type == 1:
                state[indices['b']] -= action_number + 1
                state[indices['d']] -= action_number + 1
            elif action_type == 2:
                state[:self.coef] = state[:self.coef] / (action_number + 1)
            else:
                state[:self.coef] = state[:self.coef] * (action_number + 1)
        elif additional == 1:
            state[:self.coef] = -state[:self.coef] 
        elif additional == 2: # subtract ax 
            if x_on_left:
                state[indices['x']] -= state[indices['a']]
                state[indices['a']] = 0
                if self.is_mixed:
                    state[indices['x_pos']] = 'False'
                else:
                    state[indices['x_pos']] = 0
            else:
                state[indices['a']] -= state[indices['x']]
                state[indices['x']] = 0
                if self.is_mixed:
                    state[indices['x_pos']] = 'True'
                else:
                    state[indices['x_pos']] = 1
        elif additional == 3: # add ax 
            if x_on_left:
                state[indices['x']] += state[indices['a']]
                state[indices['a']] = 0
                if self.is_mixed:
                    state[indices['x_pos']] = 'False'
                else:
                    state[indices['x_pos']] = 0
            else:
                state[indices['a']] += state[indices['x']]
                state[indices['x']] = 0
                if self.is_mixed:
                    state[indices['x_pos']] = 'True'
                else:
                    state[indices['x_pos']] = 1
        elif additional == 4: # subtract cy
            # ax + b + y_place_holder = cy + d + x_place_holder
            if y_on_right:
                state[indices['y']] -= state[indices['c']]
                state[indices['c']] = 0
                if self.is_mixed:
                    state[indices['y_pos']] = 'False'
                else:
                    state[indices['y_pos']] = 0
            else:
                state[indices['c']] -= state[indices['y']]
                state[indices['y']] = 0
                if self.is_mixed:
                    state[indices['y_pos']] = 'True'
                else:
                    state[indices['y_pos']] = 1
        else: # add cy
            if y_on_right:
                state[indices['y']] += state[indices['c']]
                state[indices['c']] = 0
                if self.is_mixed:
                    state[indices['y_pos']] = 'False'
                else:
                    state[indices['y_pos']] = 0
            else:
                state[indices['c']] += state[indices['y']]
                state[indices['y']] = 0
                if self.is_mixed:
                    state[indices['y_pos']] = 'True'
                else:
                    state[indices['y_pos']] = 1
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
        x_on_left = state[indices['x_pos']]
        if self.is_mixed:
            x_on_left = True if x_on_left == 'True' else False 
        else:
            x_on_left = bool(x_on_left)
        y_on_right = state[indices['y_pos']]
        if self.is_mixed:
            y_on_right = True if y_on_right == 'True' else False 
        else:
            y_on_right = bool(y_on_right)
        # ax + b + y_place_holder = cy + d + x_place_holder
        if state[indices['a']] == 0 and state[indices['b']] == 0 and state[indices['y']] == 1 and not x_on_left:  # Isolating x condition
            reward = 1.0 - 0.9 * (self.step_count / self.max_steps) - 0.1*2 # remove bonuses
            terminated = True

        if self.prev_x_pos and not x_on_left and not self.x_bonus_given:
            reward += 0.1
            self.x_bonus_given = True
        if not self.prev_x_pos and x_on_left:
            reward += -0.5
        if self.prev_y_pos and not y_on_right and not self.y_bonus_given:
            reward += 0.1
            self.y_bonus_given = True
        if not self.prev_y_pos and y_on_right:
            reward += -0.5

        if not self.is_mixed and (np.isnan(state).any() or np.isinf(state).any()):
            reward = -1
            terminated = True
            

        self.prev_x_pos = x_on_left
        self.prev_y_pos = y_on_right

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True
        return state, reward, terminated, truncated, info

class TwoVariableLinearEquationEnv(gym.Env):
    """
    Custom Gym Environment for isolating a linear equation: start with: ax + by + c = d
    Goal: Isolate x
    """
    
    def __init__(self, is_mixed: bool = False):
        super(TwoVariableLinearEquationEnv, self).__init__()

        # actions are add/ subtract/ divide/ multiply /add x / subtract x and multiply by -1
        
        self.n_action_types = 4
        self.digits = 9
        self.additional = 4
        self.coef = 5 
        shape = self.coef + 1
        
        self.action_space = spaces.MultiDiscrete([self.n_action_types, self.digits, self.additional])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(shape, ), dtype=float)
        self.step_count = 0
        self.max_steps = 50
        self.is_mixed = is_mixed

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        nums = [] 
        for _ in range(4):
            digit = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
            sign = np.random.choice([-1, 1])
            nums.append(digit*sign)
        
        nums.extend([0,
                     'True'.encode('utf-8') if self.is_mixed else 1])

        self.step_count = 0
        self.state = np.array(nums, dtype=object if self.is_mixed else np.single)
        self.prev_x_pos = True
        self.bonus_given = False
        return self.state, {}
    
    def _gen_state(self, action):
        state = self.state
        x_on_left = state[5]
        if self.is_mixed:
            x_on_left = True if x_on_left == 'True' else False 
        else:
            x_on_left = bool(x_on_left)
        action_type, action_number, additional = action
        if additional == 0:
            if action_type == 0:
                state[2] += action_number + 1
                state[3] += action_number + 1
            elif action_type == 1:
                state[2] -= action_number + 1
                state[3] -= action_number + 1
            elif action_type == 2:
                state[:self.coef] = state[:self.coef] / (action_number + 1)
            else:
                state[:self.coef] = state[:self.coef] * (action_number + 1)
        elif additional == 1:
            state[:self.coef] = -state[:self.coef] 
        elif additional == 2:
            if x_on_left:
                state[4] -= state[0]
                state[0] = 0
                if self.is_mixed:
                    state[5] = 'False'
                else:
                    state[5] = 0
            else:
                state[0] -= state[4]
                state[4] = 0
                if self.is_mixed:
                    state[5] = 'True'
                else:
                    state[5] = 1
        else:
            if x_on_left:
                state[4] += state[0]
                state[0] = 0
                if self.is_mixed:
                    state[5] = 'False'
                else:
                    state[5] = 0
            else:
                state[0] += state[4]
                state[4] = 0
                if self.is_mixed:
                    state[5] = 'True'
                else:
                    state[5] = 1
        self.state = state
        return state
    
    def step(self, action):
        """Take an action and return the next state, reward, and done flag."""

        reward = 0
        terminated = False 
        truncated = False
        info = {}
        state = self._gen_state(action)
        x_on_left = state[5]
        if self.is_mixed:
            x_on_left = True if x_on_left == 'True' else False 
        else:
            x_on_left = bool(x_on_left)

        if state[0] == 0 and state[1] == 1 and state[2] == 0 and not x_on_left:  # Isolating x condition
            reward = 1.0 - 0.9 * (self.step_count / self.max_steps) - 0.1 # remove bonus
            terminated = True

        if self.prev_x_pos and not x_on_left and not self.bonus_given:
            reward += 0.1
            self.bonus_given = True
        if not self.prev_x_pos and x_on_left:
            reward += -0.5

        if not self.is_mixed and (np.isnan(state).any() or np.isinf(state).any()):
            reward = -1
            terminated = True
            
        if state[1] == 0:
            terminated = True

        self.prev_x_pos = x_on_left

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