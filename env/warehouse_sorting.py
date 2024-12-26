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

class WarehouseSortingEnv(gym.Env):
    """
    Custom Gym Environment for Chain Construction Task.
    The agent must construct a valid sequence of actions (e.g., A -> B -> C)
    based on logical dependencies.
    """
    features = ['deadline', 'weight', 'is_fragile', 'priority']
    n_features = len(features)
    def __init__(self, n_items: bool):
        super(WarehouseSortingEnv, self).__init__()

        # actions are add/ subtract/ divide/ multiply and multiply by -1
        
        self.action_space = spaces.Discrete(n_items) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=((self.n_features + 1) * n_items, ), dtype=float)
        self.item_idx = 0
        self.n_items = n_items

    
    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        self.item_idx = 0
        items = []
        for i in range(self.n_items):
                deadline = np.random.choice([1, 2, 3])
                weight = np.random.normal(loc=5, scale=3)
                weight = max(weight, 1)
                is_fragile = np.random.choice([0, 1])  # Fixed typo in np.random.choice
                priority = 0
                items.append((i, deadline, weight, is_fragile, priority))
            
        # Sort items by deadline, fragility (descending), and weight (descending)
        sorted_items = sorted(
            items, 
            key=lambda x: (x[1], -x[3], -x[2])
        )
        
        # Get the sorted indices (1-based)
        self.sorted_indices = [item[0] + 1 for item in sorted_items]
        # Extract attributes without the original index for state representation
        item_attributes = [list(item[1:]) for item in items]
        item_attributes = np.array(item_attributes).flatten()
        active_index = np.zeros(self.n_items)
        active_index[self.item_idx] = 1
        self.state = np.concatenate([item_attributes, active_index], axis=0)
        return self.state, {}
    
    def _gen_state(self, action):
        state = self.state.copy()
        priority_idx = self.item_idx * self.n_features + self.features.index('priority')
        state[priority_idx] = action + 1
        # deactivate current active item
        state[self.n_items*self.n_features + self.item_idx] = 0 
        self.item_idx += 1
        return state

    def step(self, action):
        """Take an action and return the next state, reward, and done flag."""
        reward = 0
        terminated = False 
        truncated = False
        info = {}

        state = self._gen_state(action)

        if self.item_idx >= self.n_items:
            terminated = True
            for i in range(self.n_items):
                priority_idx = i * self.n_features + self.features.index('priority')
                priority = state[priority_idx]
                if priority == self.sorted_indices[i]:
                    reward += 1.0 / self.n_items
        else: 
            active_idx = self.n_items*self.n_features + self.item_idx
            state[active_idx] = 1
        self.state = state
        return state, reward, terminated, truncated, info
    

def register_warehouse_sorting_tests():
    # PutNear
    # ----------------------------------------

    register(
        id="warehousesorting-3-v0",
        entry_point="env.warehouse_sorting:WarehouseSortingEnv",
        kwargs={'n_items': 3},
    )
    register(
        id="warehousesorting-4-v0",
        entry_point="env.warehouse_sorting:WarehouseSortingEnv",
        kwargs={'n_items': 4},
    )
    register(
        id="warehousesorting-5-v0",
        entry_point="env.warehouse_sorting:WarehouseSortingEnv",
        kwargs={'n_items': 5},
    )
    register(
        id="warehousesorting-6-v0",
        entry_point="env.warehouse_sorting:WarehouseSortingEnv",
        kwargs={'n_items': 6},
    )
    register(
        id="warehousesorting-7-v0",
        entry_point="env.warehouse_sorting:WarehouseSortingEnv",
        kwargs={'n_items': 7},
    )