from typing import Any, Callable, Dict, List, OrderedDict

import gymnasium as gym
import numpy as np
import torch as th
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT, STATE_TO_IDX
from minigrid.wrappers import ObservationWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.util import obs_space_info

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
        super().__init__(env_fns)
        obs_space = env.observation_space
        self.keys, shapes, _ = obs_space_info(obs_space)

        self.buf_obs = OrderedDict([(k, np.zeros((self.num_envs, *tuple(shapes[k])), dtype=categorical_dtype)) for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
