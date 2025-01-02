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


class CompareWithEnv(gym.Env):
    """
    A toy RL environment focused on sequential and dynamic logical inference.
    The agent compares cells based on logical rules and makes sequential decisions.
    """
    def __init__(self, n_cells: int = 5, is_mixed: bool = False):
        super(CompareWithEnv, self).__init__()
        
        self.n_cells = n_cells
        self.current_step = 0
        self.is_mixed = is_mixed
        self.max_steps = n_cells * 10
        
        # Observation space: Single observation for the current state
        self.observation_space = spaces.Box(low=-1, high=10, shape=(4 if self.is_mixed else 1 + 3 + n_cells + 1,), dtype=np.float32)
        
        # Action space: n_cells + 2 (target cell or binary value to set)
        self.action_space = spaces.Discrete(n_cells + 2)
        
        self.done = False
        self.total_reward = 0
        
    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        self.current_step = 0
        self.step_count = 0

        # Reinitialize cells
        self.state = [
            {
                'state_value': np.random.uniform(0, 10),
                'logical_category': np.random.choice(['IS_GREATER', 'IS_EQUAL', 'IS_LESS']),
                'target_cell': np.random.randint(0, self.n_cells),
                'visible': False
            }
            for _ in range(self.n_cells)
        ]
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Return the current observation."""
        current_cell = self.state[self.current_step]
        target_cell = self.state[current_cell['target_cell']]  # Hidden by default
        
        if self.is_mixed:
            logical_category = current_cell['logical_category'].encode('utf-8')
            return np.array([
                current_cell['state_value'],
                logical_category,
                f"cell_{current_cell['target_cell']}".encode('utf-8'),
                target_cell['state_value' ] if current_cell['visible'] else -1 ,
            ], dtype=object)
        else:
            logical_category_map = {
                'IS_GREATER': 0,
                'IS_EQUAL': 1,
                'IS_LESS': 2
            }
            logical_category = logical_category_map[current_cell['logical_category']]
            obs = [current_cell['state_value']]
            logical_cat = [3] * 3
            logical_cat[logical_category] = 1
            target_cell_encoding = [0] * self.n_cells
            target_cell_encoding[current_cell['target_cell']] = 1
            obs.extend(logical_cat)
            obs.extend(target_cell_encoding)
            obs.append(target_cell['state_value'] if current_cell['visible'] else -1)
            return np.array(obs, dtype=np.single)

    def step(self, action):
        terminated = False
        truncated = False
        info = {}
        reward = 0
        self.step_count += 1
        """Take one step in the environment based on the agent's action."""
        if action < self.n_cells:
            # Target Cell Selection
            selected_target = action
            current_cell = self.state[self.current_step]
            if self.state[self.current_step]['visible']:
                reward = -1.0 / self.n_cells
            elif selected_target == current_cell['target_cell']:
                self.state[self.current_step]['visible'] = True
            else:
                reward = -1.0 / self.n_cells
        else:
            # Value Setting Action (0 or 1)
            if self.state[self.current_step]['visible']:
                value_to_set = action - self.n_cells
                current_cell = self.state[self.current_step]
                target_cell = self.state[current_cell['target_cell']] 
                
                correct = False
                if current_cell['logical_category'] == 'IS_GREATER':
                    correct = (current_cell['state_value'] > target_cell['state_value']) == bool(value_to_set)
                elif current_cell['logical_category'] == 'IS_EQUAL':
                    correct = (current_cell['state_value'] == target_cell['state_value'] ) == bool(value_to_set)
                elif current_cell['logical_category'] == 'IS_LESS':
                    correct = (current_cell['state_value'] < target_cell['state_value']) == bool(value_to_set)
                
                if correct:
                    reward = 1 / self.n_cells
                    # print('got correct prediction')
                    self.current_step += 1
                else:
                    # print(f"incorrect prediciton predicted")
                    # reward = -1.0 / self.n_cells
                    self.current_step += 2
            # else:
            #      reward = -1.0 / self.n_cells
        
        if self.current_step >= self.n_cells:
            terminated = True
            self.current_step = self.n_cells - 1
        if self.step_count >= self.max_steps: 
            truncated = True
        obs = self._get_observation()
        return obs, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """Render the current state of the environment."""
        print(f"Step: {self.current_step}, Total Reward: {self.reward}")
        print(f"Current Cell: {self.state[self.current_step]}")
    
    def close(self):
        """Close the environment."""
        pass

def register_logical_tests():
    # PutNear
    # ----------------------------------------

    register(
        id="Compare-v0",
        entry_point="env.logical_chain:CompareWithEnv",
        kwargs={'n_cells': 5},
    )