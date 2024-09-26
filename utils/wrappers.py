##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
from typing import Any, Callable, Dict, List, OrderedDict

import gymnasium as gym
import numpy as np
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, EpisodicLifeEnv, FireResetEnv, MaxAndSkipEnv, NoopResetEnv, StickyActionEnv
import torch as th
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT, STATE_TO_IDX
from minigrid.wrappers import ObservationWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.util import obs_space_info
from stable_baselines3.common.type_aliases import AtariResetReturn, AtariStepReturn

from gym.core import ObsType

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}
MAX_TEXT_LENGTH = 128 - 1

numerical_dtype = np.dtype('float32')
categorical_dtype = np.dtype('S128')  

MIXED_ENVS = ['Gopher']

class CategoricalObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.image_shape = self.observation_space['image'].shape
        self.flattened_shape = self.image_shape[0]*self.image_shape[1] + 2
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.flattened_shape, ), dtype=np.float32)
        env.is_categorical = True
         
    def observation(self, observation):
        # Transform the observation in some way
        categorical_array = np.empty(self.flattened_shape, dtype=categorical_dtype)
        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                category = f"{str(IDX_TO_OBJECT[observation['image'][i, j, 0]])},{str(IDX_TO_COLOR[observation['image'][i, j, 1]])},{str(IDX_TO_STATE[observation['image'][i, j, 2]])}"
                categorical_array[i*self.image_shape[1] + j] = category.encode('utf-8')
        categorical_array[self.image_shape[0]*self.image_shape[1]] = str(observation['direction']).encode('utf-8')
        categorical_array[self.image_shape[0]*self.image_shape[1] + 1] = observation['mission'].encode('utf-8')

        return np.ascontiguousarray(categorical_array)

    def reset(self, seed: int = None):
        observation, info = self.env.reset(seed=seed)
        return self.observation(observation), info

class CategoricalDummyVecEnv(DummyVecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``Cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: a list of functions
        that return environments to vectorize
    :raises ValueError: If the same environment instance is passed as the output of two or more different env_fn.
    """

    actions: np.ndarray

    def __init__(self, env_fns: List[Callable[[], gym.Env]], is_mixed: bool = False):
        self.envs = [_patch_env(fn()) for fn in env_fns]
        if len(set([id(env.unwrapped) for env in self.envs])) != len(self.envs):
            raise ValueError(
                "You tried to create multiple environments, but the function to create them returned the same instance "
                "instead of creating different objects. "
                "You are probably using `make_vec_env(lambda: env)` or `DummyVecEnv([lambda: env] * n_envs)`. "
                "You should replace `lambda: env` by a `make_env` function that "
                "creates a new instance of the environment at every call "
                "(using `gym.make()` for instance). You can take a look at the documentation for an example. "
                "Please read https://github.com/DLR-RM/stable-baselines3/issues/1151 for more information."
            )
        env = self.envs[0]
        super().__init__(env_fns)
        obs_space = env.observation_space
        self.keys, shapes, _ = obs_space_info(obs_space)
        self.is_mixed = is_mixed
        self.is_categorical = True

        self.buf_obs = OrderedDict([(k, np.zeros((self.num_envs, *tuple(shapes[k])), dtype=object if is_mixed else categorical_dtype)) for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]

class CategoricalMaxAndSkipEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Return only every ``skip``-th frame (frameskipping)
    and return the max between the two last frames.

    :param env: Environment to wrap
    :param skip: Number of ``skip``-th frame
        The same action will be taken ``skip`` times.
    :param env: Environment dtype
    """

    def __init__(self, env: gym.Env, skip: int = 4, wrapper_dtype: np.dtype = np.float32) -> None:
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        assert env.observation_space.dtype is not None, "No dtype specified for the observation space"
        assert env.observation_space.shape is not None, "No shape defined for the observation space"
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=wrapper_dtype)
        self._skip = skip

    def step(self, action: int) -> AtariStepReturn:
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info
class AtariRamWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Atari 2600 preprocessings

    Specifically:

    * Noop reset: obtain initial state by taking random number of no-ops on reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost.
    * Resize to a square image: 84x84 by default
    * Grayscale observation
    * Clip reward to {-1, 0, 1}
    * Sticky actions: disabled by default

    See https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
    for a visual explanation.

    .. warning::
        Use this wrapper only with Atari v4 without frame skip: ``env_id = "*NoFrameskip-v4"``.

    :param env: Environment to wrap
    :param noop_max: Max number of no-ops
    :param frame_skip: Frequency at which the agent experiences the game.
        This correspond to repeating the action ``frame_skip`` times.
    :param screen_size: Resize Atari frame
    :param terminal_on_life_loss: If True, then step() returns done=True whenever a life is lost.
    :param clip_reward: If True (default), the reward is clip to {-1, 0, 1} depending on its sign.
    :param action_repeat_probability: Probability of repeating the last action
    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        terminal_on_life_loss: bool = True,
        clip_reward: bool = True,
        action_repeat_probability: float = 0.0,
    ) -> None:
        if action_repeat_probability > 0.0:
            env = StickyActionEnv(env, action_repeat_probability)
        if noop_max > 0:
            env = NoopResetEnv(env, noop_max=noop_max)
        # frame_skip=1 is the same as no frame-skip (action repeat)
        if frame_skip > 1:
            env = CategoricalMaxAndSkipEnv(env, skip=frame_skip, wrapper_dtype=object if hasattr(env, 'is_mixed') and env.is_mixed else env.observation_space.dtype)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():  # type: ignore[attr-defined]
            env = FireResetEnv(env)
        if clip_reward:
            env = ClipRewardEnv(env)

        super().__init__(env)


class NeuroSymbolicAtariWrapper(ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        flattened_shape = env.observation_space.shape[0]
        if len(env.observation_space.shape) > 1:
            flattened_shape = env.observation_space.shape[1]*env.observation_space.shape[2] + env.observation_space.shape[1] - 1
        if self.env.game_name == 'Gopher':
            flattened_shape = 12
            env.is_mixed = True
        env.observation_space = gym.spaces.Box(low=0, high=255, shape=(flattened_shape, ), dtype=np.float32 )
        
    def observation(self, observation: np.ndarray):
        frame_t = observation[-1][:, :2]
        frame_prev_t = observation[-2][:, :2]
        
        
        if self.env.game_name == 'Gopher':
            return gopher_extraction(frame_t)
            # prev_positions = gopher_extraction(frame_prev_t)
            # delta_frame = positions - prev_positions
            # other_positions = positions[1:, :2]
            # distances = np.linalg.norm(other_positions - player_position, axis=1)
            # pos_and_speed = np.concatenate([positions, delta_frame], axis=-1).flatten()
            # return np.concatenate([pos_and_speed, distances])
        else:
            player_position = frame_t[0, :2]
            delta_frame = frame_t - frame_prev_t
            other_positions = frame_t[1:, :2]
            distances = np.linalg.norm(other_positions - player_position, axis=1)
            pos_and_speed = np.concatenate([frame_t, delta_frame], axis=-1).flatten()
            return np.concatenate([pos_and_speed, distances])
        
    def reset(self,  *, seed: int = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed)
        return self.observation(observation), info

def gopher_extraction(positions: np.ndarray) -> np.ndarray:
    def get_orientation(player_x, object_x):
        if player_x > object_x:
            return 'right'
        elif player_x < object_x:
            return 'left'
        else:
            return 'below'
    
    player_position = positions[0]
    empty_block_first_idx = 5
    empty_blocks = positions[empty_block_first_idx:]
    unique_x, unique_idx = np.unique(empty_blocks[:, 0], return_index=True)
    counts = np.sum(empty_blocks[:, 0][:, None] == empty_blocks[unique_idx, 0], axis=0)
    counts[empty_blocks[unique_idx][:, 0] == 0] = 0
    aligned_blocks_exist = (counts == 3).any()
    almost_aligned_block_exist = (counts == 2).any()

    nearest_aligned_pos = np.array([0, 0])
    aligned_orientation = 'None'
    aligned_dist = 0
    if aligned_blocks_exist:
        aligned_x = unique_x[counts == 3]
        aligned_pos = empty_blocks[np.isin(empty_blocks[:, 0], aligned_x)]
        aligned_dist = np.linalg.norm(player_position - aligned_pos, axis=1)
        nearest_aligned_pos = aligned_pos[np.argmin(aligned_dist)]
        aligned_dist = np.min(aligned_dist)
        aligned_orientation = get_orientation(player_position[0], aligned_x[0])

    nearest_almost_aligned_pos = np.array([0, 0])
    almost_aligned_orientation = 'None'
    almost_aligned_dist = 0
    if almost_aligned_block_exist:
        almost_aligned_x = unique_x[counts == 2]
        almost_aligned_pos = empty_blocks[np.isin(empty_blocks[:, 0], almost_aligned_x)]
        almost_aligned_dist = np.linalg.norm(player_position - almost_aligned_pos, axis=1)
        nearest_almost_aligned_pos = almost_aligned_pos[np.argmin(almost_aligned_dist)]
        almost_aligned_dist = np.min(almost_aligned_dist)
        almost_aligned_orientation = get_orientation(player_position[0], almost_aligned_x[0])
    
    return np.array([player_position[0], player_position[1], str(aligned_blocks_exist), nearest_aligned_pos[0], nearest_aligned_pos[1],
                     aligned_dist, aligned_orientation, str(almost_aligned_block_exist), nearest_almost_aligned_pos[0], 
                     nearest_almost_aligned_pos[1], almost_aligned_dist, almost_aligned_orientation], dtype=object)
    # fixed_positions = np.zeros((len(positions), 3), dtype=numerical_dtype)
    # x_positions = positions[:, 0]
    # _, unique_idx = np.unique(x_positions, return_index=True)
    # unique_idx = sorted(unique_idx)
    # unique_y = positions[unique_idx, 1]
    # counts = np.sum(positions[:, 0][:, None] == x_positions[unique_idx], axis=0)
    # counts[positions[unique_idx][:, 0] == 0] = 0
    
    # final_positions = np.column_stack((x_positions[unique_idx], unique_y, counts))
    # fixed_positions[unique_idx, :2] = positions[unique_idx, :2]
    # fixed_positions[unique_idx, 2] = counts
    # return fixed_positions