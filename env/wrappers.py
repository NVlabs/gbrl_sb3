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
from gym.core import ObsType
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT, STATE_TO_IDX
from minigrid.wrappers import ObservationWrapper
from env.ocatari import (gopher_extraction,
                         asterix_extraction,
                         general_extraction,
                         breakout_extraction,
                         bowling_extraction,
                         alien_extraction,
                         tennis_extraction,
                         kangaroo_extraction,
                         space_invaders_extraction,
                         pong_extraction,
                         ATARI_GENERAL_EXTRACTION
                                       )
from stable_baselines3.common.atari_wrappers import (ClipRewardEnv,
                                                     EpisodicLifeEnv,
                                                     FireResetEnv,
                                                     NoopResetEnv,
                                                     StickyActionEnv)
from stable_baselines3.common.type_aliases import (
                                                   AtariStepReturn)
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.util import obs_space_info
import matplotlib.pyplot as plt

def save_rendered_frame(env, frame_number=None):
    # Render the environment's current state
    frame = env.render()  # 'rgb_array' gives a numpy array of the image
    plt.imshow(frame)
    plt.axis('off')  # Turn off the axis
    # Save image as 'frame_number.png'
    name = 'frame'
    if frame_number:
        name += f'_{frame_number}'
    name += '.png'
    plt.savefig(name, bbox_inches='tight', pad_inches=0)
    plt.close()


IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}
MAX_TEXT_LENGTH = 128 - 1

numerical_dtype = np.dtype('float32')
categorical_dtype = np.dtype('S128')  


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
    def __init__(self, env: gym.Env, is_mixed: bool = True):
        super().__init__(env)
        flattened_shape = env.observation_space.shape[0]
        if len(env.observation_space.shape) > 1:
            flattened_shape = env.observation_space.shape[1]*env.observation_space.shape[2] + env.observation_space.shape[1] - 1
        if self.env.game_name == 'Gopher':
            flattened_shape = 25 if is_mixed else 115
        elif self.env.game_name == 'Breakout':
            flattened_shape = 163 if is_mixed else 201
        elif self.env.game_name == 'Alien':
            flattened_shape = 34 if is_mixed else 288
        elif self.env.game_name == 'Kangaroo':
            flattened_shape = 112 if is_mixed else 1034
        elif self.env.game_name == 'SpaceInvaders':
            flattened_shape = 142 if is_mixed else 1304
        elif self.env.game_name == 'Pong':
            flattened_shape = 121 if is_mixed else 189
        elif self.env.game_name == 'Assault':
            flattened_shape = 44 if is_mixed else 420
        elif self.env.game_name == 'Asterix':
            flattened_shape = 80 if is_mixed else 777
        elif self.env.game_name == 'Bowling':
            flattened_shape = 41 if is_mixed else 391
        elif self.env.game_name == 'Freeway':
            flattened_shape = 38 if is_mixed else 362
        elif self.env.game_name == 'Tennis':
            flattened_shape = 21 if is_mixed else 193
        elif self.env.game_name == 'Boxing':
            flattened_shape = 8 if is_mixed else 72
        else:
            flattened_shape = len(env.max_objects)*2
        env.is_mixed = is_mixed
        env.observation_space = gym.spaces.Box(low=0, high=255, shape=(flattened_shape, ), dtype=np.float32)
        
    def observation(self, observation: np.ndarray):
        if self.env.game_name == 'Gopher':
            return gopher_extraction(observation, self.env.is_mixed)
        elif self.env.game_name == 'Breakout':
            return breakout_extraction(observation, self.env.is_mixed)
        elif self.env.game_name == 'Pong':
            return pong_extraction(observation, self.env.is_mixed)
        elif self.env.game_name == 'Alien':
            return alien_extraction(observation, self.env.is_mixed)
        elif self.env.game_name == 'Kangaroo':
            return kangaroo_extraction(observation, self.env.is_mixed)
        elif self.env.game_name == 'SpaceInvaders':
            return space_invaders_extraction(observation, self.env.is_mixed)
        elif self.env.game_name == 'Tennis':
            return tennis_extraction(observation, self.env.is_mixed)
        elif self.env.game_name == 'Bowling':
            return bowling_extraction(observation, self.env.is_mixed)
        elif self.env.game_name == 'Asterix':
            return asterix_extraction(observation, self.env.is_mixed)
        elif self.env.game_name in ATARI_GENERAL_EXTRACTION:
            return general_extraction(observation, self.env.is_mixed)
        else:
            frame_t = observation[-1][:, :2]
            return frame_t.flatten()
        
    def reset(self,  *, seed: int = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed)
        return self.observation(observation), info
