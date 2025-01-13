##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
import argparse
import glob
import os
import sys
from pathlib import Path
import torch as th

import numpy as np

ROOT_PATH = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_PATH))

import warnings

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.save_util import load_from_pkl
from stable_baselines3.common.vec_env import (DummyVecEnv, VecFrameStack,
                                              VecNormalize)

from env.ocatari import MIXED_ATARI_ENVS
from env.wrappers import CategoricalDummyVecEnv, CategoricalObservationWrapper, FlatObsWrapperWithDirection
from utils.helpers import make_ram_atari_env, make_ram_ocatari_env, make_carl_env
from utils.shap_visualization import MiniGridShapVisualizationWrapper, PolicyDeepExplainer, ShapVecVideoRecorder
from env.minigrid import register_minigrid_tests
warnings.filterwarnings("ignore")

from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.ppo.ppo import PPO

from algos.a2c import A2C_GBRL
from algos.awr import AWR_GBRL
from algos.awr_nn import AWR
from algos.dqn import DQN_GBRL
from algos.ppo import PPO_GBRL
from algos.sac import SAC_GBRL
from config.args import json_string_to_dict
from typing import Union, Dict, Callable, Optional, Any, Tuple, List
import gymnasium as gym 
from stable_baselines3.common.vec_env import VecEnv, VecMonitor, is_vecenv_wrapped

NAME_TO_ALGO = {'ppo_gbrl': PPO_GBRL, 'a2c_gbrl': A2C_GBRL, 'sac_gbrl': SAC_GBRL, 'awr_gbrl': AWR_GBRL,'ppo_nn': PPO, 'a2c_nn': A2C, 'dqn_gbrl': DQN_GBRL, 'awr_nn': AWR, 'dqn_nn': DQN}
CATEGORICAL_ALGOS = [algo for algo in NAME_TO_ALGO if 'gbrl' in algo]
ON_POLICY_ALGOS = ['ppo_gbrl', 'a2c_gbrl']
OFF_POLICY_ALGOS = ['sac_gbrl', 'dqn_gbrl', 'awr_gbrl']

MINIGRID_ACTIONS = {0: 'go left', 
                    1: 'go right',
                    2: 'go forward',
                    3: 'pickup object',
                    4: 'drop object',
                    5: 'toggle/activate object',
                    6: 'done'}


def shap_evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    algo_type: str,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    background_obs = None
    if 'nn' in algo_type:
        background_env = env.env.env
        observations = background_env.reset()
        background_obs = observations.copy()
        episode_starts = np.ones((env.num_envs,), dtype=bool)
        states = None
        while len(background_obs) < 200:
            actions, states = model.predict(
                            observations,  # type: ignore[arg-type]
                            state=states,
                            episode_start=episode_starts,
                            deterministic=deterministic,
                            )
            new_observations, rewards, dones, infos = background_env.step(actions)
            background_obs = np.concatenate([background_obs, new_observations], axis=0)
        background_obs = th.tensor(background_obs).to(model.device)

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    if isinstance(observations, tuple):
        observations = observations[0]
    states = None
    n_episode = 0
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        shap_values = None
        if 'gbrl' in algo_type.lower():
            shap_values = -model.policy.model.shap(observations)[:, :, actions]
        else:
            shap_observations = th.tensor(observations).to(model.device)
            e = PolicyDeepExplainer(model.policy, background_obs)
            shap_values = e.shap_values(shap_observations)[:, :, actions]

        env.set_shap_values(shap_values, MINIGRID_ACTIONS[actions[0]])
        
        if render:
            env.render()
        new_observations, rewards, dones, infos = env.step(actions)
        
        current_rewards += rewards
        current_lengths += 1
        if dones.any():
            print(f"Episode: {n_episode + 1}/{n_eval_episodes} ")
            n_episode += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations
        if not env.recording:
            print(f'Finished recording after {len(episode_rewards)} episodes')
            break


    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_type', type=str, choices=['atari', 'ocatari', 'minigrid', 'gym', 'mujoco', 'football', 'carl']) 
    parser.add_argument('--algo_type', type=str, choices=['ppo_nn', 'ppo_gbrl', 'a2c_gbrl', 'sac_gbrl', 'awr_gbrl', 'dqn_gbrl', 'a2c_nn', 'awr_nn', 'dqn_nn']) 
    parser.add_argument('--env_name', type=str)  
    parser.add_argument('--name_str', type=str, default='eval')  
    parser.add_argument('--folder_path', type=str, default=str(ROOT_PATH / 'saved_models'))
    # env args
    # parser.add_argument('--total_n_steps', type=int)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--n_eval_episodes', type=int, default=1000)
    # parser.add_argument('--n_eval_episodes', type=int, default=10000)
    # parser.add_argument('--video_length', type=int, default=250)
    parser.add_argument('--video_length', type=int, default=100)
    parser.add_argument('--atari_wrapper_kwargs', type=json_string_to_dict)
    parser.add_argument('--env_kwargs', type=json_string_to_dict)
    parser.add_argument('--last_checkpoint', action="store_true")
    parser.add_argument('--record', action="store_true")
    parser.add_argument('--render', action="store_true")
    parser.add_argument('--use_box', action="store_true")
    parser.add_argument('--eval_env', type=str)
    parser.add_argument('--plot_path', type=str)
    parser.add_argument('--deterministic', action="store_true")
    args = parser.parse_args()

    if args.eval_env is None:
        args.eval_env = args.env_name

    model_fullname = args.model_name
    save_path = os.path.join(args.folder_path, args.env_type, args.env_name,args.algo_type)
    if args.checkpoint:
        model_fullname = args.model_name + "_" + args.checkpoint + "_steps"
        model_path = os.path.join(save_path, model_fullname)
        
    if args.last_checkpoint:
        checkpoints = glob.glob(os.path.join(save_path, f"{args.model_name}_*_steps.zip"))
        if len(checkpoints) == 0:
            raise ValueError(f"No checkpoint found for {args.algo_type} on {args.env_type}, path: {save_path}")

        def step_count(checkpoint_path: str) -> int:
            # path follow the pattern "*_steps.zip", we count from the back to ignore any other _ in the path
            return int(checkpoint_path.split("_")[-2])
        model_fullname = args.model_name + "_final_checkpoint" 
        checkpoints = sorted(checkpoints, key=step_count)
        model_path = checkpoints[-1]
    model_underscores = model_path.split("_")
    vecnormalize_path = "_".join(model_underscores[:-2] + ['vecnormalize'] + model_underscores[-2:])
    vecnormalize_path = vecnormalize_path.replace(".zip", ".pkl")

    eval_env = None


    register_minigrid_tests()
    wrapper_class = CategoricalObservationWrapper if args.algo_type in CATEGORICAL_ALGOS else FlatObsWrapperWithDirection
    vec_env_cls= CategoricalDummyVecEnv if args.algo_type in CATEGORICAL_ALGOS else DummyVecEnv
    eval_kwargs = {} if args.env_kwargs is None else args.env_kwargs.copy()
    eval_env = make_vec_env(args.eval_env, n_envs=1, env_kwargs=eval_kwargs, wrapper_class=wrapper_class, vec_env_cls=vec_env_cls)

    eval_env = MiniGridShapVisualizationWrapper(eval_env, plot_path=args.plot_path, algo_type=args.algo_type)
    if os.path.exists(vecnormalize_path):
        eval_env = VecNormalize.load(vecnormalize_path, eval_env)
        eval_env.training = False
    
    if args.record:
        video_path = ROOT_PATH  / f'videos/{args.env_type}/{args.env_name}/{args.algo_type}'
        if not os.path.exists(video_path):
            os.makedirs(video_path, exist_ok=True)
        name_prefix = f'{args.name_str}_{model_fullname}'
        eval_env = ShapVecVideoRecorder(eval_env, video_folder=video_path, record_video_trigger=lambda x: x == 0, name_prefix=name_prefix, video_length=args.video_length)

    # set_seed(args.seed)
    
    algo = NAME_TO_ALGO[args.algo_type].load(
        path=model_path,
        env=eval_env,
        device=args.device,
        force_reset= True)

    episode_rewards, episode_lengths = shap_evaluate_policy(
        algo,
        eval_env,
        args.algo_type,
        n_eval_episodes=args.n_eval_episodes,
        render=args.render,
        deterministic=args.deterministic,
        return_episode_rewards=True,
    )
    mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
    mean_length, std_length = np.mean(episode_lengths), np.std(episode_lengths)
    print(f'Evaluation results over {args.n_eval_episodes} - EP reward: {mean_reward:.2f} ' + u"\u00B1" + f' {std_reward:.2f} EP length: {mean_length:.2f} ' + u"\u00B1" + f' {std_length:.2f}')
