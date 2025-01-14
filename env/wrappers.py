##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
import operator
import time
from copy import deepcopy
from functools import reduce
from typing import (Any, Callable, Dict, List, Optional, OrderedDict,
                    SupportsFloat, Tuple, Type)

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gym.core import ObsType
from gymnasium import spaces
from gymnasium.core import ActType, WrapperObsType
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT, STATE_TO_IDX
from minigrid.wrappers import ObservationWrapper
from stable_baselines3.common.atari_wrappers import (ClipRewardEnv,
                                                     EpisodicLifeEnv,
                                                     FireResetEnv,
                                                     NoopResetEnv,
                                                     StickyActionEnv)
from stable_baselines3.common.monitor import Monitor, ResultsWriter
from stable_baselines3.common.type_aliases import AtariStepReturn
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import (VecEnv,
                                                           VecEnvIndices,
                                                           VecEnvObs,
                                                           VecEnvStepReturn)
from minigrid.wrappers import FullyObsWrapper
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.util import (copy_obs_dict, dict_to_obs,
                                                   obs_space_info)

from env.ocatari import (ATARI_GENERAL_EXTRACTION, alien_extraction,
                         asterix_extraction, bowling_extraction,
                         breakout_extraction, general_extraction,
                         gopher_extraction, kangaroo_extraction,
                         pong_extraction, space_invaders_extraction,
                         tennis_extraction)

COOPERATIVE_GAMES = 'hanabi'

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


class MiniGridFlatObsWrapper(ObservationWrapper):
    """ Identical to FlatObsWrapper from MiniGrid but also allows distances to be incorporated in the observation space
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array.

    This wrapper is not applicable to BabyAI environments, given that these have their own language component.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import FlatObsWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> env_obs = FlatObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> obs.shape
        (2835,)
    """

    def __init__(self, env, maxStrLen=96):
        super().__init__(env)

        self.maxStrLen = maxStrLen
        self.numCharCodes = 28

        imgSpace = env.observation_space.spaces["image"]
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        distanceSize = 0
        if 'distances' in env.observation_space.spaces.keys():
            distanceSpace = env.observation_space.spaces['distances']
            distanceSize = reduce(operator.mul, distanceSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize + self.numCharCodes * self.maxStrLen + distanceSize,),
            dtype="uint8",
        )

        self.cachedStr: str = None

    def observation(self, obs):
        image = obs["image"]
        mission = obs["mission"]

        # Cache the last-encoded mission string
        if mission != self.cachedStr:
            assert (
                len(mission) <= self.maxStrLen
            ), f"mission string too long ({len(mission)} chars)"
            mission = mission.lower()

            strArray = np.zeros(
                shape=(self.maxStrLen, self.numCharCodes), dtype="float32"
            )

            for idx, ch in enumerate(mission):
                if ch >= "a" and ch <= "z":
                    chNo = ord(ch) - ord("a")
                elif ch == " ":
                    chNo = ord("z") - ord("a") + 1
                elif ch == ",":
                    chNo = ord("z") - ord("a") + 2
                else:
                    raise ValueError(
                        f"Character {ch} is not available in mission string."
                    )
                assert chNo < self.numCharCodes, "%s : %d" % (ch, chNo)
                strArray[idx, chNo] = 1

            self.cachedStr = mission
            self.cachedArray = strArray
        if 'distances' in obs:
            obs = np.concatenate((image.flatten(), self.cachedArray.flatten(), obs['distances']))
        else:
            obs = np.concatenate((image.flatten(), self.cachedArray.flatten()))

        return obs
    


class FlatObsWrapperWithDirection(ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array.

    This wrapper is not applicable to BabyAI environments, given that these have their own language component.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import FlatObsWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> env_obs = FlatObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> obs.shape
        (2835,)
    """

    def __init__(self, env, maxStrLen=96):
        super().__init__(env)

        self.maxStrLen = maxStrLen
        self.numCharCodes = 28

        imgSpace = env.observation_space.spaces["image"]
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize + self.numCharCodes * self.maxStrLen + 1,),
            dtype="uint8",
        )

        self.cachedStr: str = None

    def observation(self, obs):
        image = obs["image"]
        mission = obs["mission"]
        direction = obs["direction"]

        # Cache the last-encoded mission string
        if mission != self.cachedStr:
            assert (
                len(mission) <= self.maxStrLen
            ), f"mission string too long ({len(mission)} chars)"
            mission = mission.lower()

            strArray = np.zeros(
                shape=(self.maxStrLen, self.numCharCodes), dtype="float32"
            )

            for idx, ch in enumerate(mission):
                if ch >= "a" and ch <= "z":
                    chNo = ord(ch) - ord("a")
                elif ch == " ":
                    chNo = ord("z") - ord("a") + 1
                elif ch == ",":
                    chNo = ord("z") - ord("a") + 2
                else:
                    raise ValueError(
                        f"Character {ch} is not available in mission string."
                    )
                assert chNo < self.numCharCodes, "%s : %d" % (ch, chNo)
                strArray[idx, chNo] = 1

            self.cachedStr = mission
            self.cachedArray = strArray

        obs = np.concatenate((image.flatten(), self.cachedArray.flatten(), np.array([direction])))

        return obs
    
class FlatObsWrapperWithDirectionCategoricalInfo(ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array.

    This wrapper is not applicable to BabyAI environments, given that these have their own language component.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import FlatObsWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> env_obs = FlatObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> obs.shape
        (2835,)
    """

    def __init__(self, env, maxStrLen=96):
        super().__init__(env)

        self.maxStrLen = maxStrLen
        self.numCharCodes = 28

        imgSpace = env.observation_space.spaces["image"]
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize + self.numCharCodes * self.maxStrLen + 1,),
            dtype="uint8",
        )

        self.cachedStr: str = None

        self.image_shape = imgSpace.shape
        self.flattened_shape = self.image_shape[0]*self.image_shape[1] + 1

        if not isinstance(env, FullyObsWrapper):
            self.flattened_shape += 1

    def observation(self, obs):
        image = obs["image"]
        mission = obs["mission"]
        direction = obs["direction"]

        # Cache the last-encoded mission string
        if mission != self.cachedStr:
            assert (
                len(mission) <= self.maxStrLen
            ), f"mission string too long ({len(mission)} chars)"
            mission = mission.lower()

            strArray = np.zeros(
                shape=(self.maxStrLen, self.numCharCodes), dtype="float32"
            )

            for idx, ch in enumerate(mission):
                if ch >= "a" and ch <= "z":
                    chNo = ord(ch) - ord("a")
                elif ch == " ":
                    chNo = ord("z") - ord("a") + 1
                elif ch == ",":
                    chNo = ord("z") - ord("a") + 2
                else:
                    raise ValueError(
                        f"Character {ch} is not available in mission string."
                    )
                assert chNo < self.numCharCodes, "%s : %d" % (ch, chNo)
                strArray[idx, chNo] = 1

            self.cachedStr = mission
            self.cachedArray = strArray

        obs = np.concatenate((image.flatten(), self.cachedArray.flatten(), np.array([direction])))
        return obs

    def step(
        self, action: ActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`step` using :meth:`self.observation` on the returned observations."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        info['cat_obs'] = self._cat_obs(observation)
        return self.observation(observation), reward, terminated, truncated, info
    
    def reset(self, seed: int = None):
        observation, info = self.env.reset(seed=seed)
        info['cat_obs'] = self._cat_obs(observation)
        return self.observation(observation), info
    
    def _cat_obs(self, observation):
                # Transform the observation in some way
        categorical_array = np.empty(self.flattened_shape, dtype= object)
        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                if isinstance(self.env, FullyObsWrapper) and str(IDX_TO_OBJECT[observation['image'][i, j, 0]]) == 'agent':
                   category = f"{str(IDX_TO_OBJECT[observation['image'][i, j, 0]])},{str(IDX_TO_COLOR[observation['image'][i, j, 1]])},{str(observation['image'][i, j, 2])}"   
                else:
                    category = f"{str(IDX_TO_OBJECT[observation['image'][i, j, 0]])},{str(IDX_TO_COLOR[observation['image'][i, j, 1]])},{str(IDX_TO_STATE[observation['image'][i, j, 2]])}"
                categorical_array[i*self.image_shape[1] + j] = category.encode('utf-8')
        if isinstance(self.env, FullyObsWrapper):
            categorical_array[self.image_shape[0]*self.image_shape[1]] = observation['mission'].encode('utf-8')
        else:
            categorical_array[self.image_shape[0]*self.image_shape[1]] = str(observation['direction']).encode('utf-8')
            categorical_array[self.image_shape[0]*self.image_shape[1] + 1] = observation['mission'].encode('utf-8')
        return np.ascontiguousarray(categorical_array)
    
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
                if isinstance(self.env, FullyObsWrapper) and str(IDX_TO_OBJECT[observation['image'][i, j, 0]]) == 'agent':
                   category = f"{str(IDX_TO_OBJECT[observation['image'][i, j, 0]])},{str(IDX_TO_COLOR[observation['image'][i, j, 1]])},{str(observation['image'][i, j, 2])}"   
                else:
                    category = f"{str(IDX_TO_OBJECT[observation['image'][i, j, 0]])},{str(IDX_TO_COLOR[observation['image'][i, j, 1]])},{str(IDX_TO_STATE[observation['image'][i, j, 2]])}"
                categorical_array[i*self.image_shape[1] + j] = category.encode('utf-8')
        else:
            categorical_array[self.image_shape[0]*self.image_shape[1]] = str(observation['direction']).encode('utf-8')
            categorical_array[self.image_shape[0]*self.image_shape[1] + 1] = observation['mission'].encode('utf-8')
        return np.ascontiguousarray(categorical_array)

    def reset(self, seed: int = None):
        observation, info = self.env.reset(seed=seed)
        return self.observation(observation), info
    
class MiniGridOneHotObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_colors = len(IDX_TO_COLOR)
        self.n_objects = len(IDX_TO_OBJECT)
        self.n_states = len(IDX_TO_STATE)

        self.image_shape = self.observation_space['image'].shape
        self.flattened_shape = self.image_shape[0]*self.image_shape[1]*self.n_colors*self.n_objects*self.n_states
        self.flattened_shape += 4 + 3 # assuming 4 directions and 3 missions
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.flattened_shape, ), dtype=np.float32)
          
    def observation(self, observation):
        # Transform the observation in some way
        categorical_array = np.zeros((self.image_shape[0], self.image_shape[1], self.n_objects, self.n_colors, self.n_states), dtype=np.single)
        extra_array = np.zeros(4 + 3, dtype=np.single)
        missions = {'get a red ball': 0, 'get a green ball': 1, 'get a blue ball': 2}
        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                color_ = observation['image'][i, j, 1]
                object_ = observation['image'][i, j, 0]
                state_ = observation['image'][i, j, 2]
                categorical_array[i, j, object_, color_, state_] = 1
        extra_array[-3 + missions[observation['mission']]] = 1
        extra_array[observation['direction']] = 1
        # categorical_array[-1 - 3 - directions[observation['direction']]] = 1
            # categorical_array[self.image_shape[0]*self.image_shape[1]] = str(observation['direction']).encode('utf-8')
            # categorical_array[self.image_shape[0]*self.image_shape[1] + 1] = observation['mission'].encode('utf-8')
        return np.ascontiguousarray(np.concatenate([categorical_array.flatten(), extra_array]))

    def reset(self, seed: int = None):
        observation, info = self.env.reset(seed=seed)
        return self.observation(observation), info
    
    
class PointMazeObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.flattened_shape = 6
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.flattened_shape, ), dtype=np.float32)
         
    def observation(self, obs): 
        full_obs = np.concatenate([obs['observation'], obs['desired_goal'] - obs['achieved_goal']], axis=0)
        return np.ascontiguousarray(full_obs)

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
        self.is_mixed = is_mixed or(hasattr(env, 'is_mixed') and env.is_mixed) 
        self.is_categorical = True

        self.buf_obs = OrderedDict([(k, np.zeros((self.num_envs, *tuple(shapes[k])), dtype=object if self.is_mixed else categorical_dtype)) for k in self.keys])
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

class HighWayWrapper(ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        flattened_shape = env.observation_space.shape[0] * env.observation_space.shape[1]
        env.observation_space = gym.spaces.Box(low=0, high=255, shape=(flattened_shape, ), dtype=np.float32)

    def observation(self, observation: np.ndarray):
        return observation.flatten()

    def reset(self,  *, seed: int = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed)
        return self.observation(observation), info

class NeuroSymbolicAtariWrapper(ObservationWrapper):
    def __init__(self, env: gym.Env, is_mixed: bool = False, **kwargs):
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
            # flattened_shape = 21 if is_mixed else 191
            flattened_shape = 22 if is_mixed else 192
        elif self.env.game_name == 'Assault':
            flattened_shape = 44 if is_mixed else 420
        elif self.env.game_name == 'Asterix':
            flattened_shape = 80 if is_mixed else 777
        elif self.env.game_name == 'Bowling':
            flattened_shape = 41 if is_mixed else 391
        elif self.env.game_name == 'Freeway':
            flattened_shape = 38 if is_mixed else 362
        elif self.env.game_name == 'Tennis':
            flattened_shape = 25 if is_mixed else 197
        elif self.env.game_name == 'Boxing':
            flattened_shape = 8 if is_mixed else 72
        else:
            flattened_shape = len(env.max_objects)*2
        env.is_mixed = is_mixed
        self.min_value = kwargs.get('min_value', 0)
        self.max_value = kwargs.get('max_value', 255)
        self.useless_value = float(np.random.randint(self.min_value, self.max_value))
        env.observation_space = gym.spaces.Box(low=0, high=255, shape=(flattened_shape, ), dtype=np.float32)
        
    def observation(self, observation: np.ndarray):
        if self.env.game_name == 'Gopher':
            return gopher_extraction(observation, self.env.is_mixed)
        elif self.env.game_name == 'Breakout':
            return breakout_extraction(observation, self.env.is_mixed)
        elif self.env.game_name == 'Pong':
            return pong_extraction(observation, self.env.is_mixed, self.useless_value)
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

    def step(
        self, action: ActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`step` using :meth:`self.observation` on the returned observations."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self.useless_value = float(np.random.randint(self.min_value, self.max_value))
        return self.observation(observation), reward, terminated, truncated, info
        
    def reset(self,  *, seed: int = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed)
        return self.observation(observation), info



class MultiPlayerMonitor(Monitor):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: The environment
    :param filename: the location to save a log file, can be None for no log
    :param allow_early_resets: allows the reset of the environment before it is done
    :param reset_keywords: extra keywords for the reset call,
        if extra parameters are needed at reset
    :param info_keywords: extra information to log, from the information return of env.step()
    :param override_existing: appends to file if ``filename`` exists, otherwise
        override existing files (default)
    """

    EXT = "monitor.csv"

    def __init__(
        self,
        env: gym.Env,
        filename: Optional[str] = None,
        allow_early_resets: bool = True,
        reset_keywords: Tuple[str, ...] = (),
        info_keywords: Tuple[str, ...] = (),
        override_existing: bool = True,
    ):
        super().__init__(env=env, filename=filename, allow_early_resets=allow_early_resets,
                         reset_keywords=reset_keywords, info_keywords=info_keywords, 
                         override_existing=override_existing
                         )
        self.active_player = 'player_0'

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, terminated, truncated, info = self.env.step(action)
        if info['player'] == self.active_player or info['game_name'] in COOPERATIVE_GAMES:
            self.rewards.append(float(reward))
        if terminated or truncated:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        """
        Closes the environment
        """
        super().close()
        if self.results_writer is not None:
            self.results_writer.close()

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps

        :return:
        """
        return self.total_steps

    def get_episode_rewards(self) -> List[float]:
        """
        Returns the rewards of all the episodes

        :return:
        """
        return self.episode_returns

    def get_episode_lengths(self) -> List[int]:
        """
        Returns the number of timesteps of all the episodes

        :return:
        """
        return self.episode_lengths

    def get_episode_times(self) -> List[float]:
        """
        Returns the runtime in seconds of all the episodes

        :return:
        """
        return self.episode_times
    

    

class CARLDummyVecEnv(DummyVecEnv):
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

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
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
        self.num_envs = len(env_fns)
        self.observation_space = env.observation_space['obs']
        self.action_space = env.action_space
        # store info returned by the reset method
        self.reset_infos: List[Dict[str, Any]] = [{} for _ in range(len(env_fns))]
        # seeds to be used in the next call to env.reset()
        self._seeds: List[Optional[int]] = [None for _ in range(len(env_fns))]

        try:
            render_modes = self.get_attr("render_mode")
        except AttributeError:
            import warnings
            warnings.warn("The `render_mode` attribute is not defined in your environment. It will be set to None.")
            render_modes = [None for _ in range(self.num_envs)]

        assert all(
            render_mode == render_modes[0] for render_mode in render_modes
        ), "render_mode mode should be the same for all environments"
        self.render_mode = render_modes[0]

        render_modes = []
        if self.render_mode is not None:
            if self.render_mode == "rgb_array":
                # SB3 uses OpenCV for the "human" mode
                render_modes = ["human", "rgb_array"]
            else:
                render_modes = [self.render_mode]

        self.metadata = {"render_modes": render_modes}

        obs_space = env.observation_space['obs']
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = OrderedDict([(k, np.zeros((self.num_envs, *tuple(shapes[k])), dtype=dtypes[k])) for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self.metadata = env.metadata


    def step_wait(self) -> VecEnvStepReturn:
        # Avoid circular imports
        for env_idx in range(self.num_envs):
            obs_and_context, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            self.buf_infos[env_idx]['context'] = obs_and_context['context']
            obs = obs_and_context['obs']
            # convert to SB3 VecEnv api
            self.buf_dones[env_idx] = terminated or truncated
            # See https://github.com/openai/gym/issues/3102
            # Gym 0.26 introduces a breaking change
            self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs_and_context, self.reset_infos[env_idx] = self.envs[env_idx].reset()
                obs = obs_and_context['obs']
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

    def reset(self) -> VecEnvObs:
        for env_idx in range(self.num_envs):
            obs_and_context, self.reset_infos[env_idx] = self.envs[env_idx].reset(seed=self._seeds[env_idx])
            self.reset_infos[env_idx]['context'] = obs_and_context['context']
            obs = obs_and_context['obs']
            self._save_obs(env_idx, obs)
        # Seeds are only used once
        self._reset_seeds()
        return self._obs_from_buf()

class CARLDummyVecEnvWithContext(DummyVecEnv):
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

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
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
        self.num_envs = len(env_fns)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (env.observation_space['obs'].shape[0] + 1, ), float)
        self.action_space = env.action_space
        # store info returned by the reset method
        self.reset_infos: List[Dict[str, Any]] = [{} for _ in range(len(env_fns))]
        # seeds to be used in the next call to env.reset()
        self._seeds: List[Optional[int]] = [None for _ in range(len(env_fns))]

        try:
            render_modes = self.get_attr("render_mode")
        except AttributeError:
            import warnings
            warnings.warn("The `render_mode` attribute is not defined in your environment. It will be set to None.")
            render_modes = [None for _ in range(self.num_envs)]

        assert all(
            render_mode == render_modes[0] for render_mode in render_modes
        ), "render_mode mode should be the same for all environments"
        self.render_mode = render_modes[0]

        render_modes = []
        if self.render_mode is not None:
            if self.render_mode == "rgb_array":
                # SB3 uses OpenCV for the "human" mode
                render_modes = ["human", "rgb_array"]
            else:
                render_modes = [self.render_mode]

        self.metadata = {"render_modes": render_modes}

        obs_space = self.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = OrderedDict([(k, np.zeros((self.num_envs, *tuple(shapes[k])), dtype=dtypes[k])) for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self.metadata = env.metadata


    def step_wait(self) -> VecEnvStepReturn:
        # Avoid circular imports
        for env_idx in range(self.num_envs):
            obs_and_context, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            self.buf_infos[env_idx]['context'] = obs_and_context['context']
            obs = obs_and_context['obs']
            obs = np.concatenate([obs, np.array([self.buf_infos[env_idx]['context_id']])], axis=0)
            
            # convert to SB3 VecEnv api
            self.buf_dones[env_idx] = terminated or truncated
            # See https://github.com/openai/gym/issues/3102
            # Gym 0.26 introduces a breaking change
            self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs_and_context, self.reset_infos[env_idx] = self.envs[env_idx].reset()
                obs = obs_and_context['obs']
                obs = np.concatenate([obs, np.array([self.buf_infos[env_idx]['context_id']])], axis=0)
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

    def reset(self) -> VecEnvObs:
        for env_idx in range(self.num_envs):
            obs_and_context, self.reset_infos[env_idx] = self.envs[env_idx].reset(seed=self._seeds[env_idx])
            self.reset_infos[env_idx]['context'] = obs_and_context['context']
            obs = obs_and_context['obs']
            obs = np.concatenate([obs, np.array([self.reset_infos[env_idx]['context_id']])], axis=0)
            self._save_obs(env_idx, obs)
        # Seeds are only used once
        self._reset_seeds()
        return self._obs_from_buf()


class SepsisObservationWrapper(ObservationWrapper):
    def __init__(self, env, one_hot: bool = False):
        super().__init__(env)

        self.flattened_shape = 716 if one_hot else 1
        self.info_shape = 48 
        self.one_hot = one_hot
        self.is_mixed = not one_hot
        self.n_actions = env.action_space.n
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.flattened_shape + self.info_shape, ), dtype=np.single)
        self.action_mask = None
        
         
    def observation(self, observation, state_vector, sofa_score):
        # Transform the observation in some way

        if self.one_hot:
            obs = np.zeros(self.flattened_shape, dtype=np.single)
            obs[observation] = 1
        else:
            obs = np.array([str(observation).encode('utf-8')], dtype=object)
        obs = np.concatenate([obs, state_vector, np.array([sofa_score])], axis=0, dtype=np.single if self.one_hot else object)
        return np.ascontiguousarray(obs)
    
    def step(
        self, action: ActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`step` using :meth:`self.observation` on the returned observations."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.action_mask = info['admissible_actions']
        return self.observation(observation, info['state_vector'], info['sofa_score'] ), reward, terminated, truncated, info

    def reset(self, seed: int = None):
        observation, info = self.env.reset(seed=seed)
        self.action_mask = info['admissible_actions']
        return self.observation(observation, info['state_vector'], info['sofa_score']), info

    def action_masks(self):
        """Separate function used in order to access the action mask."""
        mask = np.zeros(self.n_actions, dtype=bool)
        # Step 2: Set valid indices to True
        mask[self.action_mask] = True
        return mask