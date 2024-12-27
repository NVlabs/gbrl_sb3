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

    
    def step(self, action):
        """
        - Each item can give reward only once when it first gets the correct priority.
        - If the action (item_idx, priority) is repeated, truncate episode immediately.
        """
        priority, item_idx = action
        action_hash = (item_idx, priority)
        reward = 0.0
        truncated = False

        # 1. Check if repeated action => truncate
        if action_hash in self.assigned_actions:
            truncated = True
        else:
            # 2. Otherwise, apply the priority assignment
            self.state = self._gen_state(priority, item_idx)
            
            # 3. Check if this assignment is correct for the *first time*
            #    'optimal_priority' is how we decided this item should be ranked.
            #    Typically, "self.sorted_indices[item_idx]" or a similar mapping.
            assigned_priority = priority + 1   # If you store priority as (priority+1) in state
            optimal_priority = self.sorted_indices[item_idx]
            
            # If the assignment matches the optimal AND we haven't rewarded this item yet
            if assigned_priority == optimal_priority and item_idx not in self.correct_priorities:
                # Mark item as "correctly prioritized"
                self.correct_priorities.add(item_idx)
                # Give incremental reward = fraction of total
                # so eventually you can sum to 1.0 if all items are correct
                reward = 1.0 / self.n_items

        # 4. Update bookkeeping
        self.step_count += 1
        self.assigned_actions.add(action_hash)

        # 5. Check if we've run out of steps or if all items are correct
        #    You might want to terminate if all items are correct:
        all_correct = (len(self.correct_priorities) == self.n_items)
        terminated = (self.step_count >= self.max_steps) or all_correct

        # 6. Optionally apply final penalty only at termination (and not truncated)
        if terminated and not truncated:
            # For example, subtract a fraction of the steps used
            # This encourages finishing in fewer steps
            time_penalty = 0.9 * (self.step_count / self.max_steps)
            reward -= time_penalty

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