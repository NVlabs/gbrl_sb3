##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
import time
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple, SupportsFloat

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gbrl.common.utils import categorical_dtype
from gymnasium.core import ActType, ObsType
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT, STATE_TO_IDX
from minigrid.wrappers import FullyObsWrapper, ObservationWrapper
from stable_baselines3.common.atari_wrappers import (ClipRewardEnv,
                                                     EpisodicLifeEnv,
                                                     FireResetEnv,
                                                     NoopResetEnv,
                                                     StickyActionEnv)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import AtariStepReturn
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.util import obs_space_info


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


class MiniGridCategoricalObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.image_shape = self.observation_space['image'].shape
        self.flattened_shape = self.image_shape[0]*self.image_shape[1] + 1
        if not isinstance(env, FullyObsWrapper):
            self.flattened_shape += 1
        self.is_mixed = False
        env.is_categorical = True
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.flattened_shape, ), dtype=np.float32)

    def observation(self, observation):
        # Transform the observation in some way
        categorical_array = np.empty(self.flattened_shape, dtype=categorical_dtype if not self.is_mixed else object)
        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                if isinstance(self.env, FullyObsWrapper) and \
                   str(IDX_TO_OBJECT[observation['image'][i, j, 0]]) == 'agent':
                    category = f"{str(IDX_TO_OBJECT[observation['image'][i, j, 0]])}," + \
                               f"{str(IDX_TO_COLOR[observation['image'][i, j, 1]])}," + \
                               f"{str(observation['image'][i, j, 2])}"
                else:
                    category = f"{str(IDX_TO_OBJECT[observation['image'][i, j, 0]])}," \
                               f"{str(IDX_TO_COLOR[observation['image'][i, j, 1]])}," \
                               f"{str(IDX_TO_STATE[observation['image'][i, j, 2]])}"
                categorical_array[i*self.image_shape[1] + j] = category.encode('utf-8')
        categorical_array[self.image_shape[0]*self.image_shape[1]] = str(observation['direction']).encode('utf-8')
        categorical_array[self.image_shape[0]*self.image_shape[1] + 1] = observation['mission'].encode('utf-8')
        # if self.env.env.env.env.carrying is not None:
        #     print()
        return np.ascontiguousarray(categorical_array)

    def reset(self, seed: int = None):
        observation, info = self.env.reset(seed=seed)
        return self.observation(observation), info


class CostMonitor(Monitor):
    def __init__(        self,
        env: gym.Env,
        filename: Optional[str] = None,
        allow_early_resets: bool = True,
        reset_keywords: Tuple[str, ...] = (),
        info_keywords: Tuple[str, ...] = (),
        override_existing: bool = True,):

        super().__init__(env, filename, allow_early_resets, reset_keywords, info_keywords, override_existing)
        self.episode_costs: List[float] = []
        self.costs: List[float] = []
        self.episode_scalarization: List[float] = []
        self.scalarizations = []
        self.cost_limit = 0.1

    def reset(self, **kwargs) -> Tuple[ObsType, Dict[str, Any]]:
        self.costs = []
        self.scalarizations = []
        return super().reset(**kwargs)

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, terminated, truncated, info = self.env.step(action)
        cost = info.get('cost', 0.0)
        
        def get_scalarization(reward: float, cost: float, cost_limit: float) -> float:
            if cost <= cost_limit:
                return reward
            else:
                return -cost - 1.5
            
        self.rewards.append(float(reward))
        self.costs.append(float(cost))
        self.scalarizations.append(get_scalarization(float(reward), float(cost), self.cost_limit))
        if terminated or truncated:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_cost = sum(self.costs)
            ep_len = len(self.rewards)
            ep_scalarization = sum(self.scalarizations)
            ep_info = {"r": round(ep_rew, 6), "c": round(ep_cost, 6), "s": round(ep_scalarization, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_costs.append(ep_cost)
            self.episode_lengths.append(ep_len)
            self.episode_scalarization.append(ep_scalarization)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, terminated, truncated, info

    def get_episode_costs(self) -> List[float]:
        """
        Returns the rewards of all the episodes

        :return:
        """
        return self.episode_costs



class MiniGridIndexCategoricalObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.image_shape = self.observation_space['image'].shape
        self.flattened_shape = self.image_shape[0]*self.image_shape[1]*3 + 2
        env.is_categorical = True
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.flattened_shape, ), dtype=np.float32)

    def observation(self, observation):
        # Transform the observation in some way
        n_categories = 3
        categorical_array = np.empty(self.flattened_shape, dtype=categorical_dtype)
        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                categorical_array[(i*self.image_shape[1] + j)*n_categories] = str(
                    IDX_TO_OBJECT[observation['image'][i, j, 0]]).encode('utf-8')
                categorical_array[(i*self.image_shape[1] + j)*n_categories + 1] = str(
                    IDX_TO_COLOR[observation['image'][i, j, 1]]).encode('utf-8')
                categorical_array[(i*self.image_shape[1] + j)*n_categories + 2] = str(
                    IDX_TO_STATE[observation['image'][i, j, 2]]).encode('utf-8')

        categorical_array[-2] = str(observation['direction']).encode('utf-8')
        categorical_array[-1] = observation['mission'].encode('utf-8')
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

    def __init__(self, env_fns: List[Callable[[], gym.Env]], is_mixed: bool = False, **kwargs):
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
        self.is_mixed = is_mixed or (hasattr(env, 'is_mixed') and env.is_mixed)
        self.is_categorical = True

        self.buf_obs = OrderedDict([(k, np.zeros((self.num_envs, *tuple(shapes[k])),
                                                 dtype=object if self.is_mixed else categorical_dtype)) for
                                    k in self.keys])
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

    See
    https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
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
            env = CategoricalMaxAndSkipEnv(env, skip=frame_skip, wrapper_dtype=object if hasattr(env, 'is_mixed') and
                                           env.is_mixed else env.observation_space.dtype)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():  # type: ignore[attr-defined]
            env = FireResetEnv(env)
        if clip_reward:
            env = ClipRewardEnv(env)

        super().__init__(env)
