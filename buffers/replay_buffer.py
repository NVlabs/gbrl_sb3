##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
from typing import (Any, Callable, Dict, Generator, List, NamedTuple, Optional,
                    OrderedDict, Type, Union)

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer, ReplayBuffer
from stable_baselines3.common.type_aliases import (GymEnv, MaybeCallback,
                                                   Schedule)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.util import obs_space_info

from utils.wrappers import categorical_dtype


class AWRReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    next_observations: th.Tensor
    actions: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor

class CategoricalReplayBufferSamples(NamedTuple):
    observations: np.ndarray
    next_observations: np.ndarray
    actions: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor

RETURN_TYPES = ['monte-carlo', 'gae']
class AWRReplayBuffer(ReplayBuffer):
    """
    Replay buffer used in off-policy algorithms like AWR.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        gamma: float, 
        gae_lambda: float,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        return_type: str = 'monte-carlo',
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        env: Optional[VecNormalize] = None,
    ):
        assert return_type in RETURN_TYPES, f"return_type must be in {RETURN_TYPES}"
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=False, handle_timeout_termination=True)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.gamma = gamma 
        self.gae_lambda = gae_lambda
        self.last_start_pos = np.zeros(self.n_envs, dtype=np.int32)
        self.return_type = return_type
        self.env = env

        self.valid_pos = 0

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        if self.return_type == 'monte-carlo':
            # monte-carlo returns
            norm_rewards = self._normalize_reward(reward.reshape(-1, self.n_envs), self.env if isinstance(self.env, VecNormalize) else None)
            self.returns[self.pos] = norm_rewards
            for env_id in range(self.n_envs):
                # Calculate discounted rewards from the last start position to the current position
                # Handle the buffer wrap-around case
                start_pos = self.last_start_pos[env_id]
                if self.pos > start_pos:
                    # Standard case: no wrap-around
                    # Calculate discounted rewards from last_start_pos to current pos
                    length = self.pos - start_pos
                    disc_rew = (self.gamma ** np.arange(1, length + 1))[::-1] * norm_rewards[:, env_id]
                    self.returns[start_pos:self.pos, env_id] += disc_rew
                elif self.pos < start_pos:
                    # Handle wrap-around
                    print("wrap-around")
                    remaining_length = self.buffer_size - start_pos
                    wraparound_length = self.pos
                    total_length = wraparound_length + remaining_length
                    
                    disc_rew = (self.gamma ** np.arange(1, total_length + 1))[::-1] * norm_rewards[:, env_id]
                    self.returns[start_pos:, env_id] += disc_rew[:remaining_length]
                    self.returns[:self.pos, env_id] += disc_rew[remaining_length:total_length]
                # Update latest start position
                if self.dones[self.pos, env_id]:
                    self.last_start_pos[env_id] = (self.pos + 1) % self.buffer_size
        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0


        self.valid_pos = self.buffer_size if self.full else self.pos

    def add_advantages_returns(self, values: np.array, next_values: np.array, env: Optional[VecNormalize] = None) -> None:
        """Compute Lambda return"""
        assert len(values) == self.valid_pos

        self.advantages = np.zeros((self.valid_pos, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.valid_pos, self.n_envs), dtype=np.float32)
        last_gae_lam = np.zeros(self.n_envs, dtype=np.float32)

        rewards = self._normalize_reward(self.rewards[:self.valid_pos, :].reshape(-1, self.n_envs), env)
        non_terminal = 1.0 - self.dones[:self.valid_pos, :] * (1.0 - self.timeouts[:self.valid_pos, :])

        for step in reversed(range(self.valid_pos)):
            delta = rewards[step] + self.gamma * next_values[step] * non_terminal[step] -values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * non_terminal[step] * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + values


    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> AWRReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> AWRReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            next_obs,
            self.actions[batch_inds, env_indices, :],
            self.advantages[batch_inds, env_indices],
            self.returns[batch_inds, env_indices],
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return AWRReplayBufferSamples(*tuple(map(self.to_torch, data)))



class CategoricalAWRReplayBuffer(AWRReplayBuffer):
    """
    Replay buffer used in off-policy algorithms like AWR.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        gamma: float, 
        gae_lambda: float,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        return_type: str = 'monte-carlo',
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, gamma, gae_lambda, device, n_envs=n_envs, return_type= return_type, optimize_memory_usage=optimize_memory_usage, handle_timeout_termination=handle_timeout_termination)
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=categorical_dtype)
        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=categorical_dtype)
    
    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            if array.dtype == categorical_dtype:
                return np.copy(array)
            return th.tensor(array, device=self.device)
        if array.dtype == categorical_dtype:
            return array
        return th.as_tensor(array, device=self.device)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> AWRReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :]
        else:
            next_obs = self.next_observations[batch_inds, env_indices, :]

        data = (
            self.observations[batch_inds, env_indices, :],
            next_obs,
            self.actions[batch_inds, env_indices, :],
            self.advantages[batch_inds, env_indices],
            self.returns[batch_inds, env_indices],
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return AWRReplayBufferSamples(*tuple(map(self.to_torch, data)))


class CategoricalReplayBuffer(ReplayBuffer):
    """
    Replay buffer used in off-policy algorithms like DQN.
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, handle_timeout_termination=handle_timeout_termination)
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=categorical_dtype)
        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=categorical_dtype)

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            if array.dtype == categorical_dtype:
                return np.copy(array)
            return th.tensor(array, device=self.device)
        if array.dtype == categorical_dtype:
            return array
        return th.as_tensor(array, device=self.device)


    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> CategoricalReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> CategoricalReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            next_obs,
            self.actions[batch_inds, env_indices, :],
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return CategoricalReplayBufferSamples(*tuple(map(self.to_torch, data)))