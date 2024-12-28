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
    Enhanced Warehouse Sorting Environment.
    - Three customer types with distinct sorting preferences.
    - Explicit sorting with item positioning.
    - Sparse reward: Given only at the end of the episode.
    - Clear rules: Easy for humans and GBTs, difficult for NNs.
    """
    features = ['distance', 'weight', 'is_fragile', 'position', 'is_sorted']
    n_features = len(features)
    
    def __init__(self, n_items: int = 5, max_steps: int = 10, is_mixed: bool = False):
        super(WarehouseSortingEnv, self).__init__()
        self.n_items = n_items
        self.max_steps = max_steps
        self.is_mixed = is_mixed
        
        # Action: Select an item and target position
        self.action_space = spaces.MultiDiscrete([self.n_items, self.n_items])
        
        # Observation: Item attributes + Position + Customer Type
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.n_items * (self.n_features + 1), ),  # Each item + customer type
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment state."""
        super().reset(seed=seed)
        self.step_count = 0
        self.items = []
        
        for i in range(self.n_items):
            distance = np.random.uniform(1, 100)  # Distance in arbitrary units
            weight = np.random.uniform(1, 10)  # Weight in arbitrary units
            is_fragile = np.random.choice([0, 1])  # Fragile (0 or 1)
            
            self.items.append({
                'distance': distance,
                'weight': weight,
                'is_fragile': is_fragile,
                'position': i,  # Initial order is unsorted
                'is_sorted': 0
            })
        
        # Randomly assign a customer type (1, 2, or 3)
        self.customer_type = np.random.choice([1, 2, 3])
        
        # Optimal sorting order based on customer preferences
        if self.customer_type == 1:
            self.optimal_order = sorted(
                range(self.n_items),
                key=lambda i: (self.items[i]['distance'], self.items[i]['weight'], -self.items[i]['is_fragile'])
            )
        elif self.customer_type == 2:
            self.optimal_order = sorted(
                range(self.n_items),
                key=lambda i: (self.items[i]['distance'], -self.items[i]['is_fragile'], self.items[i]['weight'])
            )
        else:  # Customer 3
            self.optimal_order = sorted(
                range(self.n_items),
                key=lambda i: (-self.items[i]['is_fragile'], self.items[i]['weight'])
            )
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Construct the observation space."""
        obs = []
        for i, item in enumerate(self.items):
            if self.is_mixed:
                obs.extend([
                    item['distance'] / 100,  # Normalize
                    item['weight'] / 10,     # Normalize
                    str(bool(item['is_fragile'])).encode('utf-8'),
                    item['position'],
                    str(bool(item['is_sorted'])).encode('utf-8'),
                    str(self.customer_type).encode('utf-8')   # Encode customer type
                ])
            else:
                obs.extend([
                    item['distance'] / 100,  # Normalize
                    item['weight'] / 10,     # Normalize
                    item['is_fragile'],
                    item['position'] / self.n_items,  # Normalize position
                    item['is_sorted'],
                    self.customer_type / 3   # Encode customer type
                ])
        return np.array(obs, dtype=object if self.is_mixed else np.float32)
    
    def step(self, action):
        """
        Perform an action: Select an item and place it in a position.
        """
        item_idx, target_position = action
        reward = 0
        terminated = False
        info = {}
        
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        
        # Check if the move is valid
        if self.items[item_idx]['is_sorted']:
            reward -= 0.1  # Penalty for trying to sort an already sorted item
        else:
            # Update item position and mark as sorted
            self.items[item_idx]['position'] = target_position
            self.items[item_idx]['is_sorted'] = 1
        
        # Check if all items are sorted
        all_sorted = all(item['is_sorted'] for item in self.items)
        if all_sorted and not truncated:
            terminated = True
            # Evaluate final reward based on correct order
            final_order = [item['position'] for item in self.items]
            correct = sum(1 for i, idx in enumerate(final_order) if idx == self.optimal_order[i])
            reward = correct / self.n_items  # Fraction of correctly prioritized items
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def render(self, mode='human'):
        print(f"Customer Type: {self.customer_type}")
        print(f"Step: {self.step_count}")
        for i, item in enumerate(self.items):
            print(f"Item {i}: {item}")

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