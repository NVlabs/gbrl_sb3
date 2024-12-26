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
        """
        Generate a valid invertible matrix by applying meaningful, non-redundant Gaussian elimination operations.
        """
        matrix = np.eye(self.N)  # Start with an identity matrix
        self.creation_steps = []  # Log steps for optimal inversion
        
        used_scalars = set()
        used_swaps = set()
        
        # Step 1: Apply N scaling operations
        for i in range(self.N):
            scalar = np.random.choice(self.scalar_values)
            if scalar != 1 and scalar not in used_scalars:
                matrix[i] *= scalar
                self.creation_steps.append(('scale', i, None, scalar))
                used_scalars.add(scalar)
        
            # Step 2: Apply N-1 swap operations
            for i in range(self.N - 1):
                row1, row2 = i, i + 1
                if (row1, row2) not in used_swaps:
                    matrix[[row1, row2]] = matrix[[row2, row1]]
                    self.creation_steps.append(('swap', row1, row2, None))
                    used_swaps.add((row1, row2))
            
            # Step 3: Apply N(N-1)/2 meaningful row addition operations
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    scalar = np.random.choice(self.scalar_values)
                    if scalar != 0:
                        matrix[j] += scalar * matrix[i]
                        self.creation_steps.append(('add', j, i, scalar))
            
            # Step 4: Validate Invertibility
            if np.linalg.det(matrix) == 0:
                return self.generate_valid_matrix()
            
            self.optimal_steps = len(self.creation_steps)
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
        # Check Termination and Truncation
        terminated = np.allclose(left_matrix, np.eye(self.N), atol=1e-2)
        truncated = self.step_count >= self.max_steps
        reward = np.exp(-np.linalg.norm(left_matrix.flatten() - np.eye(self.N).flatten()) -0.9 * (self.step_count  / self.optimal_steps))
        # print(reward)
        
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
    
    register(
        id="mat_inv-4-v0",
        entry_point="env.matrix_inversion:MatrixInversionEnv",
        kwargs={'N': 4},
    )
    
    register(
        id="mat_inv-5-v0",
        entry_point="env.matrix_inversion:MatrixInversionEnv",
        kwargs={'N': 5},
    )
    
    register(
        id="mat_inv-6-v0",
        entry_point="env.matrix_inversion:MatrixInversionEnv",
        kwargs={'N': 6},
    )
    
    register(
        id="mat_inv-7-v0",
        entry_point="env.matrix_inversion:MatrixInversionEnv",
        kwargs={'N': 7},
    )
    