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


class MatrixInversionEnv(gym.Env):
    """
    Custom Gym Environment for Chain Construction Task.
    The agent must construct a valid sequence of actions (e.g., A -> B -> C)
    based on logical dependencies.
    """

    def __init__(self, N: int):
        super(MatrixInversionEnv, self).__init__()

        # Observation Space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2*N*N, ), dtype=np.float32)
        self.N = N
        # Action Space: Operation, Row1, Row2, Scalar
        self.action_space = spaces.MultiDiscrete([3, N, N, 10])
        self.scalar_values = [-5, -2, -1, 1, 2, 5, -0.2, -0.5, 0.5, 0.2]
        self.max_steps = 50

    def generate_valid_matrix(self):
        matrix = np.eye(self.N)
        for _ in range(5):  # Random row operations
            operation = np.random.choice(['swap', 'scale', 'add'])
            row1, row2 = np.random.choice(self.N, size=2, replace=False)
            scalar = np.random.choice(self.scalar_values)
            
            if operation == 'swap':
                matrix[[row1, row2]] = matrix[[row2, row1]]
            elif operation == 'scale' and scalar != 0:
                matrix[row1] *= scalar
            elif operation == 'add' and row1 != row2:
                matrix[row1] += matrix[row2]
        
        if np.linalg.det(matrix) == 0:
            return self.generate_valid_matrix()
        return matrix
    
    def _get_state(self):
        return self.augmented_matrix.flatten()
    
    def reset(self, seed=None, options=None):
        self.base_matrix = self.generate_valid_matrix()
        self.augmented_matrix = np.hstack((self.base_matrix, np.eye(self.N)))
        self.step_count = 0
        return self._get_state(), {}
    
    def step(self, action):
        operation, row1, row2, scalar_idx = action
        scalar = self.scalar_values[scalar_idx]
        self.step_count += 1
        
        # Apply row operations on the augmented matrix
        if operation == 0:  # Swap Rows
            if row1 != row2:
                self.augmented_matrix[[row1, row2]] = self.augmented_matrix[[row2, row1]]
        elif operation == 1:  # Scale Row
            if scalar != 0:
                self.augmented_matrix[row1] *= scalar
        elif operation == 2:  # Add Row
            if row1 != row2:
                self.augmented_matrix[row1] += scalar * self.augmented_matrix[row2]

        
        # Calculate Reward
        left_matrix = self.augmented_matrix[:, :self.N]
        progress_error = np.linalg.norm(left_matrix.flatten() - np.eye(self.N).flatten())
        reward = np.exp(-progress_error)  # Closer to identity → higher reward - 0.9 * (self.step_count / self.max_steps)
        
        # Check Termination and Truncation
        terminated = np.allclose(left_matrix, np.eye(self.N), atol=1e-2)
        truncated = self.step_count >= self.max_steps
        
        if terminated:
            reward += 1  - 0.9 * (self.step_count  / self.max_steps)  # Success bonus
        
        return self._get_state(), reward, terminated, truncated, {}


def register_mat_inv_tests():
    # PutNear
    # ----------------------------------------

    register(
        id="mat_inv-2-v0",
        entry_point="env.matrix_inversion:MatrixInversionEnv",
        kwargs={'N': 2},
    )
    register(
        id="mat_inv-3-v0",
        entry_point="env.matrix_inversion:MatrixInversionEnv",
        kwargs={'N': 3},
    )
    