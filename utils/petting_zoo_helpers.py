##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
from pettingzoo.utils import BaseWrapper
import os 
import gymnasium as gym
from typing import Union, Callable, Optional, Dict, Any, Type
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np
from sb3_contrib.common.wrappers import ActionMasker

from pettingzoo.classic import chess_v6, tictactoe_v3

PETTINGZOO_ENVS = {'chess': chess_v6, 'tictactoe_v3': tictactoe_v3} 







class PettingzooGymEnv(gym.Env):
    def __init__(self, env_id: str, **env_kwargs):
        self.env = PETTINGZOO_ENVS[env_id].env(**env_kwargs)  # type: ignore[arg-type]
        # self.env = OpenSpielCompatibilityV0(game_name=env_id, **env_kwargs)  # type: ignore[arg-type]
        self.observation_space = self.env.observation_space(self.env.possible_agents[0])["observation"]
        # self.observation_space = self.env.observation_space(self.env.possible_agents[0])
        self.action_space = self.env.action_space(self.env.possible_agents[0])
        self.num_envs = 1
        self.env_id = env_id
    
    def reset(self, seed=None, options=None):
        self.env.reset(seed, options)
        return self.observe(self.env.agent_selection), {}
    
    def seed(self, seed):
        pass
    
    def observe(self, agent):
        """Return only raw observation, removing action mask."""
        return self.env.observe(agent)["observation"]

    def render(self):
        return self.env.render()
    
    def step(self, action):
        """Gymnasium-like step function, returning observation, reward, termination, truncation, info."""
        self.env.step(action)
        return self.env.last()
    
    def action_mask(self):
        """Separate function used in order to access the action mask."""
        return self.env.observe(self.env.agent_selection)["action_mask"]
    
    def close(self):
        return self.env.close()





def make_petting_zoo_env(
    env_id: str,
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
            # if the render mode was not specified, we set it to `rgb_array` as default.
            kwargs = {"render_mode": "rgb_array"}
            # kwargs = {"render_mode": "ansi"}
            kwargs.update(env_kwargs)
            try:
                env = PettingzooGymEnv(env_id, **kwargs)  # type: ignore[arg-type]
            except TypeError:
                env = PettingzooGymEnv(env_id, **env_kwargs)
            env = ActionMasker(env, mask_fn)

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
        vec_env_cls = DummyVecEnv

    vec_env = vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)
    # Prepare the seeds for the first reset
    vec_env.seed(seed)
    return vec_env

