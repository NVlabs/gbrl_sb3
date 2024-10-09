##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
import os
import random
from typing import Any, Callable, Dict, List, Optional, Type, Union

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env

from utils.wrappers import AtariRamWrapper, CategoricalDummyVecEnv, NeuroSymbolicAtariWrapper

from ocatari import OCAtari

MINIGRID_VALID_ACTIONS = { 
    'MiniGrid-Empty-Random-5x5-v0': [0, 1 ,2],
    'MiniGrid-Empty-5x5-v0': [0, 1 ,2],
    'MiniGrid-Empty-6x6-v0': [0, 1 ,2],
    'MiniGrid-Empty-Random-6x6-v0': [0, 1 ,2],
    'MiniGrid-Empty-Random-8x8-v0': [0, 1 ,2],
    'MiniGrid-Empty-Random-16x16-v0': [0, 1 ,2],
    'MiniGrid-Fetch-5x5-N2-v0': [0, 1 ,2, 3],
    'MiniGrid-Fetch-6x6-N2-v0': [0, 1 ,2, 3],
    'MiniGrid-Fetch-8x8-N2-v0': [0, 1 ,2, 3],
    'MiniGrid-FourRooms-v0': [0, 1 ,2],
    'MiniGrid-GoToDoor-5x5-v0': [0, 1 ,2, 6],
    'MiniGrid-GoToDoor-6x6-v0': [0, 1 ,2, 6],
    'MiniGrid-GoToDoor-8x8-v0': [0, 1 ,2, 6],
    'MiniGrid-GoToObject-6x6-N2-v0': [0, 1 ,2, 6],
    'MiniGrid-GoToObject-8x8-N2-v0': [0, 1 ,2, 6],
    'MiniGrid-KeyCorridorS3R1-v0': [0, 1 ,2, 3],
    'MiniGrid-KeyCorridorS3R2-v0': [0, 1 ,2, 3],
    'MiniGrid-KeyCorridorS3R3-v0': [0, 1 ,2, 3],
    'MiniGrid-KeyCorridorS4R3-v0': [0, 1 ,2, 3],
    'MiniGrid-KeyCorridorS5R3-v0': [0, 1 ,2, 3],
    'MiniGrid-KeyCorridorS6R3-v0': [0, 1 ,2, 3],
    'MiniGrid-LavaGapS5-v0': [0, 1 ,2],
    'MiniGrid-LavaGapS6-v0': [0, 1 ,2],
    'MiniGrid-LavaGapS7-v0': [0, 1 ,2],
    'MiniGrid-LockedRoom-v0': [0, 1 ,2, 3, 5],
    'MiniGrid-MemoryS17Random-v0': [0, 1 ,2, 3, 5],
    'MiniGrid-MemoryS13Random-v0': [0, 1 ,2, 3, 5],
    'MiniGrid-MemoryS13-v0': [0, 1 ,2, 3, 5],
    'MiniGrid-MemoryS11-v0': [0, 1 ,2, 3, 5],
    'MiniGrid-MultiRoom-N2-S4-v0': [0, 1 ,2, 5],
    'MiniGrid-MultiRoom-N4-S5-v0': [0, 1 ,2, 5],
    'MiniGrid-MultiRoom-N6-v0': [0, 1 ,2, 5],
    'MiniGrid-ObstructedMaze-1Dlhb-v0': [0, 1 ,2, 3, 4, 5, 6],
    'MiniGrid-ObstructedMaze-2Dlh-v0': [0, 1, 2, 3, 4, 5, 6],
    'MiniGrid-ObstructedMaze-Full-v0': [0, 1 ,2, 3, 4, 5, 6],
    'MiniGrid-PutNear-6x6-N2-v0': [0, 1 ,2, 3, 4],
    'MiniGrid-PutNear-8x8-N3-v0': [0, 1 ,2, 3, 4],
    'MiniGrid-RedBlueDoors-6x6-v0': [0, 1 ,2, 5],
    'MiniGrid-RedBlueDoors-8x8-v0': [0, 1 ,2, 5],
    'MiniGrid-Unlock-v0': [0, 1 ,2, 5],
    'MiniGrid-UnlockPickup-v0': [0, 1 ,2, 3, 5],
    'MiniGrid-Dynamic-Obstacles-5x5-v0': [0, 1 ,2],
    'MiniGrid-Dynamic-Obstacles-6x6-v0': [0, 1 ,2],
    'MiniGrid-Dynamic-Obstacles-8x8-v0': [0, 1 ,2],
    'MiniGrid-Dynamic-Obstacles-16x16-v0': [0, 1 ,2],
    'MiniGrid-Dynamic-Obstacles-Random-5x5-v0': [0, 1 ,2],
    'MiniGrid-Dynamic-Obstacles-Random-6x6-v0': [0, 1 ,2],
    'MiniGrid-DoorKey-5x5-v0': [0, 1 ,2, 3, 5],
    'MiniGrid-DoorKey-6x6-v0': [0, 1 ,2, 3, 5],
    'MiniGrid-DoorKey-8x8-v0': [0, 1 ,2, 3, 5],
    'MiniGrid-DoorKey-16x16-v0': [0, 1 ,2, 3, 5],
    'MiniGrid-DistShift1-v0': [0, 1 ,2, 3],
    'MiniGrid-DistShift2-v0': [0, 1 ,2, 3],
    'MiniGrid-LavaCrossingS9N1-v0': [0, 1 ,2],
    'MiniGrid-LavaCrossingS9N2-v0': [0, 1 ,2],
    'MiniGrid-LavaCrossingS9N3-v0': [0, 1 ,2],
    'MiniGrid-LavaCrossingS11N5-v0': [0, 1 ,2],
    'MiniGrid-SimpleCrossingS9N1-v0': [0, 1 ,2],
    'MiniGrid-SimpleCrossingS9N2-v0': [0, 1 ,2],
    'MiniGrid-SimpleCrossingS9N3-v0': [0, 1 ,2],
    'MiniGrid-SimpleCrossingS11N5-v0': [0, 1 ,2],
    'MiniGrid-BlockedUnlockPickup-v0': [0, 1 ,2, 3],
}

def get_minigrid_valid_actions(env_names):
    if isinstance(env_names, str):
        return MINIGRID_VALID_ACTIONS[env_names]
    valid_actions = []
    for env_name in env_names:
        valid_actions.extend(MINIGRID_VALID_ACTIONS[env_name])
    return sorted(list(set(valid_actions)))
    

def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    # Force conversion to float
    initial_value_ = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value_

    return func

def convert_clip_range(clip_range: Union[str, int]):
    if clip_range is None:
        return None
    if isinstance(clip_range, str) and 'lin_' in clip_range:
        return linear_schedule(clip_range.replace('lin_', ''))
    return float(clip_range)

def print_stats(values):
    return f'{np.mean(values):.3f}'+'\u00B1'+f'{np.std(values):.3f}'

def get_tree_theta_and_value(tree, state, output_dim):
    #print('printing tree:', tree)
    theta = tree.predict_theta(state, len(state), output_dim)
    value = tree.predict_value(state, len(state))
    return theta, value

def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)

    th.backends.cudnn.enabled = True
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True


def make_ocvec_env(
    env_id: Union[str, Callable[..., gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored ``VecEnv``.
    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    :param env_id: either the env ID, the env class or a callable returning an env
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
        Note: the wrapper specified by this parameter will be applied after the ``Monitor`` wrapper.
        if some cases (e.g. with TimeLimit wrapper) this can lead to undesired behavior.
        See here for more details: https://github.com/DLR-RM/stable-baselines3/issues/894
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :param wrapper_kwargs: Keyword arguments to pass to the ``Wrapper`` class constructor.
    :return: The wrapped environment
    """
    env_kwargs = env_kwargs or {}
    vec_env_kwargs = vec_env_kwargs or {}
    monitor_kwargs = monitor_kwargs or {}
    wrapper_kwargs = wrapper_kwargs or {}
    assert vec_env_kwargs is not None  # for mypy

    def make_env(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            # For type checker:
            assert monitor_kwargs is not None
            assert wrapper_kwargs is not None
            assert env_kwargs is not None

            if isinstance(env_id, str):
                # if the render mode was not specified, we set it to `rgb_array` as default.
                
                kwargs = {"render_mode": "rgb_array", "obs_mode": "obj"}
                kwargs.update(env_kwargs)
                env = OCAtari(env_id, **kwargs)
            else:
                env = env_id(**env_kwargs)
                # Patch to support gym 0.21/0.26 and gymnasium
                env = _patch_env(env)

            if seed is not None:
                # Note: here we only seed the action space
                # We will seed the env at the next reset
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None and monitor_dir is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            env = NeuroSymbolicAtariWrapper(env, vec_env_kwargs.get('is_mixed', False))
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    vec_env = vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)
    # Prepare the seeds for the first reset
    vec_env.seed(seed)
    return vec_env

def make_ram_ocatari_env(
    env_id: Union[str, Callable[..., gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Union[Type[DummyVecEnv], Type[SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:


    return make_ocvec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        start_index=start_index,
        monitor_dir=monitor_dir,
        wrapper_class=AtariRamWrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
        monitor_kwargs=monitor_kwargs,
        wrapper_kwargs=wrapper_kwargs,
    )


def make_ram_atari_env(
    env_id: Union[str, Callable[..., gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Union[Type[DummyVecEnv], Type[SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored VecEnv for Atari.
    It is a wrapper around ``make_vec_env`` that includes common preprocessing for Atari games.

    :param env_id: either the env ID, the env class or a callable returning an env
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_kwargs: Optional keyword argument to pass to the ``AtariRamWrapper``
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :return: The wrapped environment
    """
    return make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        start_index=start_index,
        monitor_dir=monitor_dir,
        wrapper_class=AtariRamWrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
        monitor_kwargs=monitor_kwargs,
        wrapper_kwargs=wrapper_kwargs,
    )

def make_vec_from_different_envs(
    env_ids: List[Union[str, Callable[..., gym.Env]]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Union[CategoricalDummyVecEnv, DummyVecEnv]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored ``VecEnv``.
    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    :param env_id: either the env ID, the env class or a callable returning an env
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
        Note: the wrapper specified by this parameter will be applied after the ``Monitor`` wrapper.
        if some cases (e.g. with TimeLimit wrapper) this can lead to undesired behavior.
        See here for more details: https://github.com/DLR-RM/stable-baselines3/issues/894
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :param wrapper_kwargs: Keyword arguments to pass to the ``Wrapper`` class constructor.
    :return: The wrapped environment
    """
    env_kwargs = env_kwargs or {}
    vec_env_kwargs = vec_env_kwargs or {}
    monitor_kwargs = monitor_kwargs or {}
    wrapper_kwargs = wrapper_kwargs or {}
    assert vec_env_kwargs is not None  # for mypy

    def make_env(rank: int, env_id: Union[str, Callable[..., gym.Env]]):
        def _init() -> gym.Env:
            # For type checker:
            assert monitor_kwargs is not None
            assert wrapper_kwargs is not None
            assert env_kwargs is not None

            if isinstance(env_id, str):
                # if the render mode was not specified, we set it to `rgb_array` as default.
                kwargs = {"render_mode": "rgb_array"}
                kwargs.update(env_kwargs)
                try:
                    env = gym.make(env_id, **kwargs)  # type: ignore[arg-type]
                except TypeError:
                    env = gym.make(env_id, **env_kwargs)
            else:
                env = env_id(**env_kwargs)
                # Patch to support gym 0.21/0.26 and gymnasium
                env = _patch_env(env)

            if seed is not None:
                # Note: here we only seed the action space
                # We will seed the env at the next reset
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None and monitor_dir is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = CategoricalDummyVecEnv

    envs = []
    env_idx = 0
    for env_id in env_ids:
        for _ in range(n_envs):
            envs.append(make_env(env_idx + start_index, env_id))
            env_idx += 1
    vec_env = vec_env_cls(envs, **vec_env_kwargs)
    # Prepare the seeds for the first reset
    vec_env.seed(seed)
    return vec_env




