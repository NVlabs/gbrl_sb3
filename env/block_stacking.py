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

class BlockStackingEnv(gym.Env):
    def __init__(self, is_mixed: bool = False):
        super(BlockStackingEnv, self).__init__()
        self.color_constraints = ["red first", "blue first", "green first", "red last", "green last", "blue last", "red middle", "blue middle", "green middle", "no constraint"]
        self.size_constraints = ["small first", "medium first", "large first", "small last", "large last", "medium last", "small middle", "medium middle", "large middle", 'no constraint']
        self.blocks = ['red small', 
                       'red medium',
                       'red large',
                       'blue small',
                       'blue medium',
                       'blue large',
                       'green small',
                       'green medium',
                       'green large',
                       'no block'
        ]
        self.blocks_dict = {k: i for i, k in enumerate(self.blocks)}
        # Environment parameters
          # Number of blocks
        self.colors = ["red", "blue", "green"]
        self.sizes = ["small", "medium", "large"]
        self.num_blocks = len(self.blocks) - 1
        self.blocks_to_place = 3
        self.goals = ["one of each color", "all same color", 'all red', 'all green', 'all blue', "one of each size", "all small", 'all medium', 'all large', 'all same size']
        self.goal_constraints = {"one of each color": 'color',
                                 "all same color": 'size',
                                 "all red": 'size',
                                 "all blue": 'size',
                                 "all green": 'size',
                                 "one of each size": 'color',
                                 "all small": 'color',
                                 "all medium": 'color',
                                 "all large": 'color',
                                 "all same size": 'color',
        }
        self.color_const_idx = {v: k for k, v in enumerate(self.color_constraints)}
        self.size_const_idx = {v: k for k, v in enumerate(self.size_constraints)}
        self.goal_idx = {v: k for k, v in enumerate(self.goals)}

        # Action space: (pick block index, place on target index)
        self.action_space = spaces.MultiDiscrete([len(self.blocks), 3])

        # Observation space: color, size, placed_order for each block
        obs_shape = 3 if is_mixed else 3 * len(self.blocks)
        constraint_shape = 3 if is_mixed else len(self.goals) + len(self.color_constraints) + len(self.size_constraints)
        # Plus one-hot encoded goal and constraint

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape + constraint_shape,), dtype=np.single)

        self.state = None
        self.goal = None
        self.color_constraint = None
        self.size_constraint = None
        self.is_mixed = is_mixed
        self.max_steps = 50

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        # Initialize block states: [color, size, placed_order (0 = unplaced)]
        self.state = ['no block'] * 3
        # Randomly select a goal and a constraint
        self.goal = np.random.choice(self.goals)
        constraint_type = self.goal_constraints[self.goal]
        if constraint_type == 'color':
            self.size_constraint = 'no constraint'
            self.color_constraint = np.random.choice(self.color_constraints)
        else:
            self.size_constraint = np.random.choice(self.size_constraints)
            self.color_constraint = 'no constraint'
        return self._get_obs(), {}
    
    def _get_obs(self):
        # Encode state and add goal/constraint
        obs = None 
        if self.is_mixed:
            state = self.state.copy() 
            state.extend([self.goal, self.color_constraint, self.size_constraint])
            obs = np.array([s.encode('utf-8') for s in state], dtype=object)
        else:
            state = []
            for s in self.state:
                one_hot = [0] * len(self.blocks)
                one_hot[self.blocks_dict[s]] = 1
                state.extend(one_hot)
            one_hot = [0]* len(self.goals)
            one_hot[self.goal_idx[self.goal]] = 1
            state.extend(one_hot)
            one_hot = [0]* len(self.color_constraints)
            one_hot[self.color_const_idx[self.color_constraint]] = 1
            state.extend(one_hot)
            one_hot = [0]* len(self.size_constraints)
            one_hot[self.size_const_idx[self.size_constraint]] = 1
            state.extend(one_hot)
            obs = np.array(state, dtype=np.single)
        return obs

    def step(self, action):
        pick_block, place_target = action
        self.step_count += 1
        reward = 0  
        terminated = False 
        truncated = False 

        if self.state[place_target] == 'no block':
            self.state[place_target] = self.blocks[pick_block]
        else:
            if self.blocks[pick_block] == 'no block':
                self.state[place_target] = 'no block'
            else:
                reward -= 0.1

        if not self._check_constraint():
            terminated = True 
            reward -= 0.1
        

        if self.step_count >= self.max_steps:
            truncated = True 
        # Check if the goal is satisfied
        if not truncated and not terminated and self._check_goal():
            reward = 1.0 - 0.9 * (self.step_count / self.max_steps)
            terminated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _check_constraint(self):
        passed = True
        
        if self.color_constraint != 'no constraint':
            constraint = self.color_constraint
            if constraint == 'red first':
                relevant_state = self.state[0]
                if relevant_state != 'no block' and 'red' not in relevant_state:
                    passed = False 
            if constraint == 'green first':
                relevant_state = self.state[0]
                if relevant_state != 'no block' and 'green' not in relevant_state:
                    passed = False 
            if constraint == 'blue first':
                relevant_state = self.state[0]
                if relevant_state != 'no block' and 'blue' not in relevant_state:
                    passed = False 
            if constraint == 'red middle':
                relevant_state = self.state[1]
                if relevant_state != 'no block' and 'red' not in relevant_state:
                    passed = False 
            if constraint == 'green middle':
                relevant_state = self.state[1]
                if relevant_state != 'no block' and 'green' not in relevant_state:
                    passed = False 
            if constraint == 'blue middle':
                relevant_state = self.state[1]
                if relevant_state != 'no block' and 'blue' not in relevant_state:
                    passed = False 
            if constraint == 'red last':
                relevant_state = self.state[2]
                if relevant_state != 'no block' and 'red' not in relevant_state:
                    passed = False 
            if constraint == 'green last':
                relevant_state = self.state[2]
                if relevant_state != 'no block' and 'green' not in relevant_state:
                    passed = False 
            if constraint == 'blue last':
                relevant_state = self.state[2]
                if relevant_state != 'no block' and 'blue' not in relevant_state:
                    passed = False 
        elif self.size_constraint != 'no constraint':
            constraint = self.size_constraint
            if constraint == 'small first':
                relevant_state = self.state[0]
                if relevant_state != 'no block' and 'small' not in relevant_state:
                    passed = False 
            if constraint == 'medium first':
                relevant_state = self.state[0]
                if relevant_state != 'no block' and 'medium' not in relevant_state:
                    passed = False 
            if constraint == 'large first':
                relevant_state = self.state[0]
                if relevant_state != 'no block' and 'large' not in relevant_state:
                    passed = False 
            if constraint == 'small middle':
                relevant_state = self.state[1]
                if relevant_state != 'no block' and 'small' not in relevant_state:
                    passed = False 
            if constraint == 'medium middle':
                relevant_state = self.state[1]
                if relevant_state != 'no block' and 'medium' not in relevant_state:
                    passed = False 
            if constraint == 'large middle':
                relevant_state = self.state[1]
                if relevant_state != 'no block' and 'large' not in relevant_state:
                    passed = False 
            if constraint == 'small last':
                relevant_state = self.state[2]
                if relevant_state != 'no block' and 'small' not in relevant_state:
                    passed = False 
            if constraint == 'medium last':
                relevant_state = self.state[2]
                if relevant_state != 'no block' and 'medium' not in relevant_state:
                    passed = False 
            if constraint == 'large last':
                relevant_state = self.state[2]
                if relevant_state != 'no block' and 'large' not in relevant_state:
                    passed = False 
        return passed

    def _check_goal(self):
        goal = self.goal
        condition = True 
        if goal == "one of each color":
            for color in ['red', 'blue', 'green']:
                for s in self.state:
                    if color not in s:
                        condition = False 
                        break 
                if not condition:
                    break
        elif goal == "all same color":
            color = self.state[0]
            if color == 'no block':
                return False 
            color = color.split()[0]
            for s in self.state:
                if color not in s:
                    condition = False 
                    break 
        elif goal == "all red":
              for s in self.state:
                if 'red' not in s:
                    condition = False 
                    break 
        elif goal == "all blue":
              for s in self.state:
                if 'blue' not in s:
                    condition = False 
                    break 
        elif goal == "all green":
              for s in self.state:
                if 'green' not in s:
                    condition = False 
                    break 
        elif goal == "one of each size":
            for box_size in ['small', 'medium', 'large']:
                for s in self.state:
                    if box_size not in s:
                        condition = False 
                        break 
                if not condition:
                    break
        elif goal == "all same size":
            box_size = self.state[0]
            if box_size == 'no block':
                return False 
            box_size = box_size.split()[1]
            for s in self.state:
                if box_size not in s:
                    condition = False 
                    break 
        elif goal == "all small":
              for s in self.state:
                if 'small' not in s:
                    condition = False 
                    break 
        elif goal == "all large":
              for s in self.state:
                if 'large' not in s:
                    condition = False 
                    break 
        elif goal == "all medium":
              for s in self.state:
                if 'medium' not in s:
                    condition = False 
                    break 
        return condition

def register_blocks_tests():
    # PutNear
    # ----------------------------------------

    register(
        id="Blocks-v0",
        entry_point="env.block_stacking:BlockStackingEnv",
        kwargs={},
    )