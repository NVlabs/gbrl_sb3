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
    Enhanced Warehouse Sorting Task Environment.
    Logical rules create irregular and non-linear decision boundaries,
    favoring tree-based models.
    """
    features = ['deadline', 'weight', 'is_fragile', 'priority', 'urgency', 'handling_time']
    n_features = len(features)
    
    def __init__(self, n_items: int):
        super(WarehouseSortingEnv, self).__init__()
        self.n_items = n_items
        self.max_steps = 50
        
        # Actions: (priority, item_idx)
        self.action_space = spaces.MultiDiscrete([n_items, n_items])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=((self.n_features) * n_items,), dtype=float
        )
        
    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.correct_priorities = set()
        self.assigned_actions = set()
        
        items = []
        for i in range(self.n_items):
            deadline = np.random.choice([1, 2, 3])
            weight = np.random.normal(loc=5, scale=3)
            weight = max(weight, 1)
            is_fragile = np.random.choice([0, 1])
            urgency = np.random.choice([0, 1])
            handling_time = (weight * 0.5) + (is_fragile * 2)
            priority = 0
            
            items.append((i, deadline, weight, is_fragile, urgency, handling_time, priority))
        
        # Clear and Logical Sorting Rules
        sorted_items = sorted(
            items,
            key=lambda x: (
                -x[4],                    # Urgency first (highest priority)
                x[1] + x[5],              # Deadline adjusted by handling time
                -x[2] if x[3] == 1 else x[2]  # Weight prioritized differently for fragile items
            )
        )
        
        self.sorted_indices = [item[0] + 1 for item in sorted_items]
        item_attributes = [list(item[1:]) for item in items]
        self.state = np.array(item_attributes).flatten()
        
        return self.state, {}
    
    def _gen_state(self, priority, item_idx):
        state = self.state.copy()
        priority_idx = item_idx * self.n_features + self.features.index('priority')
        state[priority_idx] = priority + 1
        return state
    
    def _calculate_reward(self):
        correct_count = 0
        for i in range(self.n_items):
            priority_idx = i * self.n_features + self.features.index('priority')
            assigned_priority = self.state[priority_idx]
            optimal_priority = self.sorted_indices[i]
            
            if assigned_priority == optimal_priority and assigned_priority not in self.correct_priorities:
                correct_count += 1
                self.correct_priorities.add(assigned_priority)

        reward = (
            correct_count / self.n_items
            - 0.9 * self.step_count / self.max_steps
        )
        return reward
    
    def step(self, action):
        priority, item_idx = action
        action_hash = (item_idx, priority)
        reward = 0
        truncated = False
        
        if action_hash in self.assigned_actions:
            truncated = True
        else:
            self.state = self._gen_state(priority, item_idx)
            
        
        self.step_count += 1
        self.assigned_actions.add(action_hash)
        terminated = self.step_count >= self.max_steps
        if terminated and not truncated:
            reward, terminated = self._calculate_reward()
        
        return self.state, reward, terminated, truncated, {}

    def render(self, mode='human'):
        print(f"Step: {self.step_count}, State: {self.state}")

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