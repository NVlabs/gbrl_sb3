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
    CallbackList, CheckpointCallback, EvalCallback,
    StopTrainingOnNoModelImprovement)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (DummyVecEnv, VecFrameStack,
                                              VecNormalize, VecVideoRecorder)

from callback.callbacks import (ActorCriticCompressionCallback,
                                OffPolicyDistillationCallback,
                                OnPolicyDistillationCallback,
                                StopTrainingOnNoImprovementInTraining)
from utils.helpers import (make_ram_atari_env, 
                           make_ram_ocatari_env, 
                           set_seed,
                           make_openspiel_env, 
                           make_bsuite_env,
                           make_carl_env,
                           make_sepsis_env,
                           make_highway_env)
from env.wrappers import (CategoricalDummyVecEnv,
                          CategoricalObservationWrapper,
                          PointMazeObservationWrapper,
                          FlatObsWrapperWithDirection,
                          SepsisObservationWrapper,
                          HighWayWrapper)
from env.ocatari import MIXED_ATARI_ENVS
from env.minigrid import register_minigrid_tests
from env.equation import register_equation_tests
from env.warehouse_sorting import register_warehouse_sorting_tests
from env.matrix_inversion import register_mat_inv_tests
from env.pipeline_opt import register_pipeline_opt_tests
from env.bank_validation import register_bank_validation_tests

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
    if 'atari' in args.env_type:
        env_kwargs = {'full_action_space': False}
        vec_env_cls = None
        vec_env_kwargs = None
        if args.env_type == "ocatari":
            make_ram_atari_env = make_ram_ocatari_env
            print("Using Ocatari environment")
            vec_env_cls  = CategoricalDummyVecEnv if args.env_name.split('-')[0] in MIXED_ATARI_ENVS and args.algo_type in CATEGORICAL_ALGOS else vec_env_cls
            vec_env_kwargs = {'is_mixed': True} if args.env_name.split('-')[0] in MIXED_ATARI_ENVS and args.algo_type in CATEGORICAL_ALGOS else {}
        env = make_ram_atari_env(args.env_name, n_envs=args.num_envs, seed=args.seed, wrapper_kwargs=args.atari_wrapper_kwargs, env_kwargs=env_kwargs, vec_env_cls=vec_env_cls, vec_env_kwargs=vec_env_kwargs, neurosymbolic_kwargs=args.neurosymbolic_kwargs) 
        if args.evaluate:
            eval_env = make_ram_atari_env(args.env_name, n_envs=1, wrapper_kwargs=args.atari_wrapper_kwargs, env_kwargs=env_kwargs, vec_env_cls=vec_env_cls, vec_env_kwargs=vec_env_kwargs, neurosymbolic_kwargs=args.neurosymbolic_kwargs) 
        if args.atari_wrapper_kwargs and 'frame_stack' in args.atari_wrapper_kwargs:
            env = VecFrameStack(env, n_stack=args.atari_wrapper_kwargs['frame_stack'])
            if eval_env:
                eval_env = VecFrameStack(eval_env, n_stack=args.atari_wrapper_kwargs['frame_stack'])
    elif args.env_type == 'minigrid':
        register_minigrid_tests()
        # wrapper_class = CategoricalObservationWrapper if args.algo_type in CATEGORICAL_ALGOS else MiniGridFlatObsWrapper
        wrapper_class = CategoricalObservationWrapper if args.algo_type in CATEGORICAL_ALGOS else FlatObsWrapperWithDirection
        vec_env_cls= CategoricalDummyVecEnv if args.algo_type in CATEGORICAL_ALGOS else DummyVecEnv
        # vec_env_kwargs = {'is_mixed': True}
        env = make_vec_env(args.env_name, n_envs=args.num_envs, seed=args.seed, env_kwargs=args.env_kwargs, wrapper_class=wrapper_class, vec_env_cls=vec_env_cls)
        if args.evaluate:
            eval_kwargs = args.env_kwargs.copy()
            if 'SpuriousFetch' in args.env_name:
                eval_kwargs['train'] = False
            eval_env = make_vec_env(args.env_name, n_envs=1, env_kwargs=eval_kwargs, wrapper_class=wrapper_class, vec_env_cls=vec_env_cls)
    elif args.env_type == 'highway':
        env = make_highway_env(args.env_name, n_envs=args.num_envs, seed=args.seed, env_kwargs=args.env_kwargs, wrapper_class=HighWayWrapper)
        if args.evaluate:
            eval_env = make_highway_env(args.env_name, n_envs=1, env_kwargs=args.env_kwargs, wrapper_class=HighWayWrapper)
    elif args.env_type == 'football':
        try:
            from env.football import FootballGymSB3
        except ModuleNotFoundError:
            print("Could not find gfootball! please run pip install gfootball") 
        args.env_kwargs['env_name'] = args.env_name
        env = make_vec_env(FootballGymSB3, n_envs=args.num_envs, seed=args.seed, env_kwargs=args.env_kwargs)
        if args.evaluate:
            eval_env = make_vec_env(FootballGymSB3, n_envs=1, env_kwargs=args.env_kwargs)
    elif args.env_type == 'mujoco' or args.env_type == 'gym':
        env = make_vec_env(args.env_name, n_envs=args.num_envs, seed=args.seed, env_kwargs=args.env_kwargs)
        if args.evaluate:
            eval_env = make_vec_env(args.env_name, n_envs=1, env_kwargs=args.env_kwargs)
    elif args.env_type == 'equation':
        register_equation_tests()
        args.env_kwargs['is_mixed'] = True if args.algo_type in CATEGORICAL_ALGOS and 'v1' in args.env_name else False
        vec_env_cls= CategoricalDummyVecEnv if args.algo_type in CATEGORICAL_ALGOS and 'v1' in args.env_name else DummyVecEnv
        env = make_vec_env(args.env_name, n_envs=args.num_envs, seed=args.seed, env_kwargs=args.env_kwargs, vec_env_cls=vec_env_cls)
        if args.evaluate:
            eval_env = make_vec_env(args.env_name, n_envs=1, env_kwargs=args.env_kwargs, vec_env_cls=vec_env_cls)
    elif args.env_type == 'bank':
        register_bank_validation_tests()
        args.env_kwargs['is_mixed'] = True if args.algo_type in CATEGORICAL_ALGOS and 'v1' in args.env_name else False
        vec_env_cls= CategoricalDummyVecEnv if args.algo_type in CATEGORICAL_ALGOS and 'v1' in args.env_name else DummyVecEnv
        env = make_vec_env(args.env_name, n_envs=args.num_envs, seed=args.seed, env_kwargs=args.env_kwargs, vec_env_cls=vec_env_cls)
        if args.evaluate:
            eval_env = make_vec_env(args.env_name, n_envs=1, env_kwargs=args.env_kwargs, vec_env_cls=vec_env_cls)
    elif args.env_type == 'pipeline_opt':
        register_pipeline_opt_tests()
        args.env_kwargs['one_hot_task_types'] = False if args.algo_type in CATEGORICAL_ALGOS else True
        vec_env_cls= CategoricalDummyVecEnv if args.algo_type in CATEGORICAL_ALGOS else DummyVecEnv
        env = make_vec_env(args.env_name, n_envs=args.num_envs, seed=args.seed, env_kwargs=args.env_kwargs, vec_env_cls=vec_env_cls)
        if args.evaluate:
            eval_env = make_vec_env(args.env_name, n_envs=1, env_kwargs=args.env_kwargs, vec_env_cls=vec_env_cls)
    elif args.env_type == 'warehouse_sorting':
        register_warehouse_sorting_tests()
        args.env_kwargs['is_mixed'] = True if args.algo_type in CATEGORICAL_ALGOS else False
        vec_env_cls= CategoricalDummyVecEnv if args.algo_type in CATEGORICAL_ALGOS else DummyVecEnv
        env = make_vec_env(args.env_name, n_envs=args.num_envs, seed=args.seed, env_kwargs=args.env_kwargs, vec_env_cls=vec_env_cls)
        if args.evaluate:
            eval_env = make_vec_env(args.env_name, n_envs=1, env_kwargs=args.env_kwargs, vec_env_cls=vec_env_cls)
    elif args.env_type == 'matrix_inv':
        register_mat_inv_tests()
        env = make_vec_env(args.env_name, n_envs=args.num_envs, seed=args.seed, env_kwargs=args.env_kwargs)
        if args.evaluate:
            eval_env = make_vec_env(args.env_name, n_envs=1, env_kwargs=args.env_kwargs)
    elif args.env_type == 'bsuite':
        env = make_bsuite_env(args.env_name, n_envs=args.num_envs, seed=args.seed, env_kwargs=args.env_kwargs)
        if args.evaluate:
            eval_env = make_bsuite_env(args.env_name, n_envs=1, env_kwargs=args.env_kwargs)
    elif args.env_type == 'carl':
        env = make_carl_env(args.env_name, n_envs=args.num_envs, seed=args.seed, env_kwargs=args.env_kwargs)
        if args.evaluate:
            eval_env = make_carl_env(args.env_name, n_envs=1, env_kwargs=args.env_kwargs)
    elif args.env_type == 'robotics':
        import gymnasium as gym
        import gymnasium_robotics
        gym.register_envs(gymnasium_robotics)
        wrapper_class = PointMazeObservationWrapper
        env = make_vec_env(args.env_name, n_envs=args.num_envs, seed=args.seed, env_kwargs=args.env_kwargs, wrapper_class=wrapper_class)
        if args.evaluate:
            eval_env = make_vec_env(args.env_name, n_envs=1, env_kwargs=args.env_kwargs, wrapper_class=wrapper_class)
    elif args.env_type == 'sepsis':
        wrapper_class = SepsisObservationWrapper
        vec_env_cls = CategoricalDummyVecEnv if args.algo_type in CATEGORICAL_ALGOS else DummyVecEnv
        wrapper_kwargs = {'one_hot': args.algo_type not in CATEGORICAL_ALGOS}
        env = make_sepsis_env(args.env_name, n_envs=args.num_envs, seed=args.seed, vec_env_cls=vec_env_cls, env_kwargs=args.env_kwargs, wrapper_class=wrapper_class, wrapper_kwargs=wrapper_kwargs)
        if args.evaluate:
            eval_env = make_sepsis_env(args.env_name, n_envs=1, env_kwargs=args.env_kwargs, vec_env_cls=vec_env_cls, wrapper_kwargs=wrapper_kwargs, wrapper_class=wrapper_class)
    elif args.env_type == 'openspiel':
        learn_kwargs['use_masking'] = True
        args.env_kwargs['is_mixed'] = True if args.algo_type in CATEGORICAL_ALGOS else False
        args.env_kwargs['config'] = args.openspiel_config if args.openspiel_config else {} 
        args.env_kwargs['config']['seed'] = args.seed
        vec_env_cls  = CategoricalDummyVecEnv if args.algo_type in CATEGORICAL_ALGOS else None
        vec_env_kwargs = {'is_mixed': True} if args.algo_type in CATEGORICAL_ALGOS else None
        env = make_openspiel_env(args.env_name, n_envs=args.num_envs, env_kwargs=args.env_kwargs, vec_env_cls=vec_env_cls, vec_env_kwargs=vec_env_kwargs)
        if args.evaluate:
            eval_env = make_openspiel_env(args.env_name, n_envs=1, env_kwargs=args.env_kwargs, vec_env_cls=vec_env_cls, vec_env_kwargs=vec_env_kwargs)
        NAME_TO_ALGO['ppo_nn'] = PPO_SelfPlay
        NAME_TO_ALGO['ppo_gbrl'] = PPO_GBRL_SelfPlay
    else:
        print("Invalid env_type!")

    if args.wrapper == 'normalize':
        args.wrapper_kwargs['gamma'] = args.gamma
        env = VecNormalize(env, **args.wrapper_kwargs)
        if eval_env is not None:
            args.wrapper_kwargs['training'] = False 
            args.wrapper_kwargs['norm_reward'] = False 
            eval_env = VecNormalize(eval_env, **args.wrapper_kwargs)
    if args.save_every and args.save_every > 0 and args.specific_seed == args.seed:
        callback_list.append(CheckpointCallback(save_freq=int(args.save_every / args.num_envs), save_path=os.path.join(args.save_path, f'{args.env_type}/{args.env_name}/{args.algo_type}'), name_prefix=f'{args.save_name}_seed_{args.seed}', verbose=1, save_vecnormalize=True if args.env_type != 'football' else False))
    if args.no_improvement_kwargs:
        callback_list.append(StopTrainingOnNoImprovementInTraining(**args.no_improvement_kwargs, verbose=args.verbose))
    if eval_env is not None:
        stop_train_callback = None
        if args.eval_kwargs and args.eval_kwargs.get('stop_train', False):
            stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=args.eval_kwargs.get('max_no_improvement_evals', 3),
                                                                    min_evals=args.eval_kwargs.get('min_evals', 5), verbose=1)
        if args.eval_kwargs.get('record', False):
            video_path = ROOT_PATH  / f'videos/{args.env_type}/{args.env_name}/{args.algo_type}'
            if not os.path.exists(video_path):
                os.makedirs(video_path, exist_ok=True)
            
            eval_env = VecVideoRecorder(eval_env, video_folder=video_path,  record_video_trigger=lambda x: x == args.eval_kwargs['eval_freq'], name_prefix=f'{args.save_name}_seed_{args.seed}_eval', video_length=args.eval_kwargs.get('video_length', 2000))
        # save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=os.path.join(args.save_path, f'{args.env_type}/{args.env_name}/{args.algo_type}'))
        callback_list.append(EvalCallback(
                                    eval_env,
                                    callback_on_new_best=None,
                                    callback_after_eval=stop_train_callback,
                                    # best_model_save_path=os.path.join(args.save_path, f'{args.env_type}/{args.env_name}/{args.algo_type}'),
                                    best_model_save_path=None,
                                    log_path=None,
                                    eval_freq=int(args.eval_kwargs.get('eval_freq', 10000) / args.num_envs),
                                    n_eval_episodes=args.eval_kwargs.get('n_eval_episodes', 5),
                                    verbose=args.eval_kwargs.get('verbose', 1),
                                    ))   
    callback = None
    if callback_list:
        callback = CallbackList(callback_list)
    set_seed(args.seed)
    
    algo_kwargs = process_policy_kwargs(args)
    print(f"Training with algo_kwargs: {algo_kwargs}")
    
    algo = NAME_TO_ALGO[args.algo_type](env=env, tensorboard_log=tensorboard_log, _init_setup_model=True, **algo_kwargs)

    algo.learn(total_timesteps=args.total_n_steps, callback=callback, log_interval=args.log_interval, progress_bar=False, **learn_kwargs)

    if args.save_every > 0:
        print("End of training save")
        name_prefix = f'{args.env_type}/{args.env_name}/{args.algo_type}'
        model_path = os.path.join(args.save_path, f"{name_prefix}/{args.save_name}_seed_{args.seed}_{args.total_n_steps}_steps.zip")
        algo.save(model_path)
        if args.verbose >= 2:
            print(f"Saving model checkpoint to {model_path}")
        save_vec_normalize = True if args.env_type != 'football' else False
        if save_vec_normalize and algo.get_vec_normalize_env() is not None:
                # Save the VecNormalize statistics
            vec_normalize_path = os.path.join(args.save_path, f"{name_prefix}_vecnormalize_{args.total_n_steps}_steps.pkl")
            algo.get_vec_normalize_env().save(vec_normalize_path)
            if args.verbose >= 2:
                print(f"Saving model VecNormalize to {vec_normalize_path}")
