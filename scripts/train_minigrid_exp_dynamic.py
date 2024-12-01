##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
import os
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_PATH))

import sys
import warnings

from stable_baselines3.common.callbacks import (
    CallbackList, CheckpointCallback)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (DummyVecEnv,
                                              VecNormalize)

from callback.callbacks import (ActorCriticCompressionCallback,
                                OffPolicyDistillationCallback,
                                OnPolicyDistillationCallback,
                                StopTrainingOnNoImprovementInTraining,
                                MultiEvalCallback,
                                ChangeEnvCallback)
from utils.helpers import set_seed
from env.wrappers import (CategoricalDummyVecEnv,
                          CategoricalObservationWrapper,
                          )
from env.ocatari import MIXED_ATARI_ENVS
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
from algos.ppo_selfplay import PPO_GBRL_SelfPlay, PPO_SelfPlay
from algos.sac import SAC_GBRL
from config.args import parse_args, process_logging, process_policy_kwargs

NAME_TO_ALGO = {'ppo_gbrl': PPO_GBRL, 'a2c_gbrl': A2C_GBRL, 'sac_gbrl': SAC_GBRL, 'awr_gbrl': AWR_GBRL,'ppo_nn': PPO, 'a2c_nn': A2C, 'dqn_gbrl': DQN_GBRL, 'awr_nn': AWR, 'dqn_nn': DQN}
CATEGORICAL_ALGOS = [algo for algo in NAME_TO_ALGO if 'gbrl' in algo]
ON_POLICY_ALGOS = ['ppo_gbrl', 'a2c_gbrl']
OFF_POLICY_ALGOS = ['sac_gbrl', 'dqn_gbrl', 'awr_gbrl']

def change_undersampling_rate(env, init_rate = 1, new_rate = 10):
    if env.env.env.env.env.undersampling_rate is None or env.env.env.env.env.undersampling_rate  == 1:
        env.env.env.env.env.undersampling_rate = new_rate 
        print(f"Changed undersampling_rate from {init_rate} to {new_rate}")
    else: 
        env.env.env.env.env.undersampling_rate = init_rate
        print(f"Changed undersampling_rate from {new_rate} to {init_rate}")

if __name__ == '__main__':
    args = parse_args()
    callback_list = []
    if args.distil and args.distil_kwargs:
        if args.algo_type in ON_POLICY_ALGOS:
            callback_list.append(OnPolicyDistillationCallback(args.distil_kwargs, args.distil_kwargs.get('distil_verbose', 0)))
        elif args.algo_type in OFF_POLICY_ALGOS:
            callback_list.append(OffPolicyDistillationCallback(args.distil_kwargs, args.distil_kwargs.get('distil_verbose', 0)))
    if args.compress and args.compress_kwargs:
        args.compress_kwargs['capacity'] = int(args.compress_kwargs['capacity'] / args.num_envs)
        callback_list.append(ActorCriticCompressionCallback(args.compress_kwargs, args.compress_kwargs.get('compress_verbose', 0)))
        
    tensorboard_log = process_logging(args, callback_list)
    env, eval_env = None, None
    if args.env_kwargs is None:
        args.env_kwargs = {}
    learn_kwargs = {}

    from minigrid.wrappers import FlatObsWrapper
    register_minigrid_tests()
    wrapper_class = CategoricalObservationWrapper if args.algo_type in CATEGORICAL_ALGOS else FlatObsWrapper
    vec_env_cls= CategoricalDummyVecEnv if args.algo_type in CATEGORICAL_ALGOS else DummyVecEnv
    env = make_vec_env(args.env_name, n_envs=args.num_envs, seed=args.seed, env_kwargs=args.env_kwargs, wrapper_class=wrapper_class, vec_env_cls=vec_env_cls)

    callback_list.append(ChangeEnvCallback(int(2500000 / args.num_envs), change_undersampling_rate))
    if args.wrapper == 'normalize':
        args.wrapper_kwargs['gamma'] = args.gamma
        env = VecNormalize(env, **args.wrapper_kwargs)
        eval_wrapper_kwargs = args.wrapper_kwargs.copy()
        eval_wrapper_kwargs['training'] = False 
        eval_wrapper_kwargs['norm_reward'] = False 
    for i, ball_color in enumerate(['red', 'green', 'blue']):
        eval_env_kwargs = args.env_kwargs.copy()
        eval_env_kwargs['test_box_idx'] = i
        eval_env = make_vec_env(args.env_name, n_envs=1, env_kwargs=eval_env_kwargs, wrapper_class=wrapper_class, vec_env_cls=vec_env_cls)
        if args.wrapper == 'normalize':
            eval_env = VecNormalize(eval_env, **eval_wrapper_kwargs)
        callback_list.append(MultiEvalCallback(
                            ball_color,
                            eval_env,
                            callback_on_new_best=None,
                            callback_after_eval=None,
                            best_model_save_path=None,
                            log_path=None,
                            eval_freq=int(args.eval_kwargs.get('eval_freq', 10000) / args.num_envs),
                            n_eval_episodes=args.eval_kwargs.get('n_eval_episodes', 50),
                            verbose=args.eval_kwargs.get('verbose', 1), 
                            ))


    undersampling_rate = args.env_kwargs.get('undersampling_rate', 1)
    if args.save_every and args.save_every > 0 and args.specific_seed == args.seed:
        # callback_list.append(CheckpointCallback(save_freq=int(args.save_every / args.num_envs), save_path=os.path.join(args.save_path, f'{args.env_type}/{args.env_name}/{args.algo_type}'), name_prefix=f'{args.save_name}_{int(args.range[0])}_{int(args.range[1])}_seed_{args.seed}', verbose=1, save_vecnormalize=True if args.env_type != 'football' else False))
        callback_list.append(CheckpointCallback(save_freq=int(args.save_every / args.num_envs), save_path=os.path.join(args.save_path, f'{args.env_type}/{args.env_name}/{args.algo_type}'), name_prefix=f'{args.save_name}_ur_{undersampling_rate}_seed_{args.seed}', verbose=1, save_vecnormalize=True if args.env_type != 'football' else False))
    if args.no_improvement_kwargs:
        callback_list.append(StopTrainingOnNoImprovementInTraining(**args.no_improvement_kwargs, verbose=args.verbose))

    callback = None
    if callback_list:
        callback = CallbackList(callback_list)
    set_seed(args.seed)
    
    algo_kwargs = process_policy_kwargs(args)
    print(f"Training with algo_kwargs: {algo_kwargs}")
    
    algo = NAME_TO_ALGO[args.algo_type](env=env, tensorboard_log=tensorboard_log, _init_setup_model=True, **algo_kwargs)

    algo.learn(total_timesteps=args.total_n_steps, callback=callback, log_interval=args.log_interval, progress_bar=False, **learn_kwargs)
