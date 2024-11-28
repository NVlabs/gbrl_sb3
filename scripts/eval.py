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

import numpy as np

ROOT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_PATH))

import warnings

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.save_util import load_from_pkl
from stable_baselines3.common.vec_env import (DummyVecEnv, VecFrameStack,
                                              VecNormalize, VecVideoRecorder)

from env.ocatari import MIXED_ATARI_ENVS
from env.wrappers import CategoricalDummyVecEnv, CategoricalObservationWrapper
from utils.helpers import make_ram_atari_env, make_ram_ocatari_env, make_carl_env
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

NAME_TO_ALGO = {'ppo_gbrl': PPO_GBRL, 'a2c_gbrl': A2C_GBRL, 'sac_gbrl': SAC_GBRL, 'awr_gbrl': AWR_GBRL,'ppo_nn': PPO, 'a2c_nn': A2C, 'dqn_gbrl': DQN_GBRL, 'awr_nn': AWR, 'dqn_nn': DQN}
CATEGORICAL_ALGOS = [algo for algo in NAME_TO_ALGO if 'gbrl' in algo]
ON_POLICY_ALGOS = ['ppo_gbrl', 'a2c_gbrl']
OFF_POLICY_ALGOS = ['sac_gbrl', 'dqn_gbrl', 'awr_gbrl']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_type', type=str, choices=['atari', 'ocatari', 'minigrid', 'gym', 'mujoco', 'football', 'carl']) 
    parser.add_argument('--algo_type', type=str, choices=['ppo_nn', 'ppo_gbrl', 'a2c_gbrl', 'sac_gbrl', 'awr_gbrl', 'dqn_gbrl', 'a2c_nn', 'awr_nn', 'dqn_nn']) 
    parser.add_argument('--env_name', type=str)  
    parser.add_argument('--folder_path', type=str, default=str(ROOT_PATH / 'saved_models'))
    # env args
    # parser.add_argument('--total_n_steps', type=int)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--checkpoint', type=str)
    # parser.add_argument('--n_eval_episodes', type=int, default=10)
    parser.add_argument('--n_eval_episodes', type=int, default=1000)
    parser.add_argument('--video_length', type=int, default=2000)
    parser.add_argument('--atari_wrapper_kwargs', type=json_string_to_dict)
    parser.add_argument('--env_kwargs', type=json_string_to_dict)
    parser.add_argument('--last_checkpoint', action="store_true")
    parser.add_argument('--record', action="store_true")
    parser.add_argument('--render', action="store_true")
    parser.add_argument('--deterministic', action="store_true")
    args = parser.parse_args()

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
    if 'atari' in args.env_type:
        env_kwargs = {'full_action_space': False}
        vec_env_cls = None
        vec_env_kwargs = None
        if args.env_type == "ocatari":
            make_ram_atari_env = make_ram_ocatari_env
            print("Using Ocatari environment")
            vec_env_cls  = CategoricalDummyVecEnv if args.env_name.split('-')[0] in MIXED_ATARI_ENVS and args.algo_type in CATEGORICAL_ALGOS else vec_env_cls
            vec_env_kwargs = {'is_mixed': True} if args.env_name.split('-')[0] in MIXED_ATARI_ENVS and args.algo_type in CATEGORICAL_ALGOS else vec_env_kwargs
        eval_env = make_ram_atari_env(args.env_name, n_envs=1, wrapper_kwargs=args.atari_wrapper_kwargs, env_kwargs=env_kwargs, vec_env_cls=vec_env_cls, vec_env_kwargs=vec_env_kwargs) 

        if args.atari_wrapper_kwargs and 'frame_stack' in args.atari_wrapper_kwargs:
            eval_env = VecFrameStack(eval_env, n_stack=args.atari_wrapper_kwargs['frame_stack'])
    elif args.env_type == 'minigrid':
        from minigrid.wrappers import FlatObsWrapper
        register_minigrid_tests()
        wrapper_class = CategoricalObservationWrapper if args.algo_type in CATEGORICAL_ALGOS else FlatObsWrapper
        vec_env_cls= CategoricalDummyVecEnv if args.algo_type in CATEGORICAL_ALGOS else DummyVecEnv
        eval_env = make_vec_env(args.env_name, n_envs=1, env_kwargs=args.env_kwargs, wrapper_class=wrapper_class, vec_env_cls=vec_env_cls)
    elif args.env_type == 'football':
        try:
            from env.football import FootballGymSB3
        except ModuleNotFoundError:
            print("Could not find gfootball! please run pip install gfootball") 
        if args.env_kwargs is None:
            args.env_kwargs = {}
        args.env_kwargs['env_name'] = args.env_name
        eval_env = make_vec_env(FootballGymSB3, n_envs=1, env_kwargs=args.env_kwargs)
    elif args.env_type == 'mujoco' or args.env_type == 'gym':
        eval_env = make_vec_env(args.env_name, n_envs=1, env_kwargs=args.env_kwargs)
    elif args.env_type == 'carl':
        eval_env = make_carl_env(args.env_name, n_envs=1, env_kwargs=args.env_kwargs)
    else:
        print("Invalid env_type!")

    if os.path.exists(vecnormalize_path):
        eval_env = VecNormalize.load(vecnormalize_path, eval_env)
        eval_env.training = False


    if args.record:
        video_path = ROOT_PATH  / f'videos/{args.env_type}/{args.env_name}/{args.algo_type}'
        if not os.path.exists(video_path):
            os.makedirs(video_path, exist_ok=True)
        eval_env = VecVideoRecorder(eval_env, video_folder=video_path, record_video_trigger=lambda x: x == 0, name_prefix=f'eval_{model_fullname}', video_length=args.video_length)

    # set_seed(args.seed)
    
    algo = NAME_TO_ALGO[args.algo_type].load(
        path=model_path,
        env=eval_env,
        device=args.device,
        force_reset= True)

    episode_rewards, episode_lengths = evaluate_policy(
        algo,
        eval_env,
        n_eval_episodes=args.n_eval_episodes,
        render=args.render,
        deterministic=args.deterministic,
        return_episode_rewards=True,
    )
    mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
    mean_length, std_length = np.mean(episode_lengths), np.std(episode_lengths)
    print(f'Evaluation results over {args.n_eval_episodes} - EP reward: {mean_reward:.2f} ' + u"\u00B1" + f' {std_reward:.2f} EP length: {mean_length:.2f} ' + u"\u00B1" + f' {std_length:.2f}')
