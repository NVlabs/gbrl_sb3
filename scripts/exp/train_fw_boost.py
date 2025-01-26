import os
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_PATH))

import argparse
import sys
import warnings

# from rl_zoo3.callbacks import SaveVecNormalizeCallback
from stable_baselines3.common.callbacks import (
    CallbackList, CheckpointCallback, EvalCallback,
    StopTrainingOnNoModelImprovement)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import (DummyVecEnv, VecFrameStack,
                                              VecNormalize, VecVideoRecorder)

from callback.callbacks import (OffPolicyDistillationCallback,
                                OnPolicyDistillationCallback,
                                StopTrainingOnNoImprovementInTraining)
from env.football import FootballGymSB3
# from env.football import FootballGymSB3
from utils.helpers import make_ram_atari_env, set_seed


warnings.filterwarnings("ignore")

import stable_baselines3

print(f"SB3 Version: {stable_baselines3.__version__}")

ENV_NAME = 'Pendulum-v1'
# ENV_NAME = 'academy_empty_goal'

from algos.ppo import PPO_GBRL
from config.args import process_logging, str2bool
from policies.actor_critic_policy import ActorCriticPolicy
from policies.actor_critic_policy_fw_boost import ActorCriticPolicyBOOST

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--env_name', type=str, default=ENV_NAME)  
    parser.add_argument('--env_type', type=str, default='gym')  
    parser.add_argument('--fw_type', type=str, default='gbrl', choices=['gbrl', 'cb', 'xgb'])  
    parser.add_argument('--algo_type', type=str)  
    # env args
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--max_depth', type=int, default=4)
    parser.add_argument('--its_per_grad', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.033)
    parser.add_argument('--total_n_steps', type=int, default=1000000)

    parser.add_argument('--wandb', type=str2bool, default=False) 

    parser.add_argument('--run_name', type=str, help='name of run', default='test')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--group_name', type=str, help='wandb group name')
    parser.add_argument('--project', type=str)
    parser.add_argument('--entity', type=str)
    args = parser.parse_args()
    callback_list = []

    device = args.device if args.fw_type != 'cb' else 'cpu'
    # device = 'cpu'

    env_kwargs = {'Pendulum-v1': {'n_epochs': 20,
                                  'max_depth': 4,
                                  'n_steps': 256,
                                  'num_envs': 16,
                                  'policy_lr': 0.031246793805561623,
                                  'value_lr': 0.01315225738736567,
                                  'max_policy_grad_norm': 100,
                                  'max_value_grad_norm': 10,
                                  'gamma': 0.9147245889494424,
                                  'gae_lambda': 0.9375823088645378 ,
                                  'log_std_init': -2,
                                  'clip_range': 0.2,
                                  'ent_coef': 0,
                                  'batch_size': 512,
                                  'wrapper': 'normalize'}}
    
    args.algo_type = args.fw_type
    tensorboard_log = process_logging(args, callback_list)
    env = make_vec_env(args.env_name, n_envs=env_kwargs[args.env_name]['num_envs'], seed=args.seed, env_kwargs=None)
    if env_kwargs[args.env_name]['wrapper'] == 'normalize':
        wrapper_kwargs = {'gamma': env_kwargs[args.env_name]['gamma'],
                          'training': True, 'norm_obs': False, 'norm_reward': True}
        env = VecNormalize(env, **wrapper_kwargs)
    callback = None
    if callback_list:
        callback = CallbackList(callback_list)
    set_seed(args.seed)

    
    
    algo_kwargs = { 
            "clip_range": env_kwargs[args.env_name]['clip_range'],
            "normalize_advantage": True,
            "n_epochs": env_kwargs[args.env_name]['n_epochs'],
            # "is_categorical": False,
            "ent_coef": env_kwargs[args.env_name]['ent_coef'],
            "n_steps": env_kwargs[args.env_name]['n_steps'],
            "batch_size": env_kwargs[args.env_name]['batch_size'],
            "gae_lambda": env_kwargs[args.env_name]['gae_lambda'],
            "gamma": env_kwargs[args.env_name]['gamma'],
            "total_n_steps": args.total_n_steps,
            "policy_kwargs": {
                "log_std_init": env_kwargs[args.env_name]['log_std_init'],
                # "shared_tree_struct": False,
                "shared_tree_struct": True,
                "tree_struct": {
                    "max_depth": env_kwargs[args.env_name]['max_depth'],
                    "n_bins": 256,
                    "min_data_in_leaf": 0,
                    "par_th": 2,
                    "grow_policy": "oblivious",
                }, 
                "tree_optimizer": {
                    "gbrl_params": {
                        "split_score_func": "cosine",
                        'control_variates': False,
                        "generator_type": "Quantile",
                    }, 
                    "policy_optimizer": {
                        "policy_algo": "SGD",
                        "policy_lr": env_kwargs[args.env_name]['policy_lr'],
                    }, 
                    "value_optimizer": {
                        "value_algo": "SGD",
                        "value_lr": env_kwargs[args.env_name]['value_lr']
                    },
                },
            },
            "device": device,
            "seed": args.seed,
        }
    
    if args.fw_type != 'gbrl':
        algo_kwargs['policy_kwargs']['fw_type'] = args.fw_type
        algo_kwargs['policy_kwargs']['its_per_grad'] = args.its_per_grad
    algo = PPO_GBRL(policy=ActorCriticPolicy if args.fw_type == 'gbrl' else ActorCriticPolicyBOOST, env=env, _init_setup_model=True, tensorboard_log=tensorboard_log, **algo_kwargs)

    algo.learn(total_timesteps=args.total_n_steps, callback=callback, log_interval=args.log_interval, progress_bar=False)
