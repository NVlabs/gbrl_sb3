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
        
        self.action_space = spaces.MultiDiscrete([n_items, n_items])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=((self.n_features) * n_items, ), dtype=float)
        self.n_items = n_items
        self.max_steps = 50

    
    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        self.step_count = 0
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
        self.state = np.array(item_attributes).flatten()

        self.wasted_actions = 0
        
        # Track assigned items and priorities
        self.assigned_actions = set()
        return self.state, {}
    
    def _gen_state(self, priority, item_idx):
        state = self.state.copy()
        priority_idx = item_idx * self.n_features + self.features.index('priority')
        state[priority_idx] = priority + 1
        return state
    


    def _calculate_reward(self):
        """Evaluate correctness and penalize inefficiency."""
        correct_count = 0
        for i in range(self.n_items):
            priority_idx = i * self.n_features + self.features.index('priority')
            assigned_priority = self.state[priority_idx]
            optimal_priority = self.sorted_indices[i]
            if assigned_priority == optimal_priority:
                correct_count += 1
        
        # Reward formula
        # reward = np.exp(-np.linalg.norm(correct_count - self.n_items) - 0.9 * self.step_count / self.max_steps - 0.5 * self.wasted_actions)
        reward = correct_count/self.n_items - 0.9 * self.step_count / self.max_steps - 0.5 * self.wasted_actions
        terminated = correct_count == self.n_items
        return reward, terminated

    def step(self, action):
        """Take an action and return the next state, reward, and done flag."""
        priority, item_idx = action

        action_hash = (item_idx, priority)

        if action_hash in self.assigned_actions:
            self.wasted_actions += 1 

        self.state = self._gen_state(priority, item_idx)
        reward, terminated = self._calculate_reward()

        self.step_count += 1

        self.assigned_actions.add(action_hash)

        truncated = self.step_count >= self.max_steps
        
        return self.state, reward, terminated, truncated, {}
    

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