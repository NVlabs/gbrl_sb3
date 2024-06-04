##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
import argparse
import json
from pathlib import Path
from typing import Callable, Union

import torch as th
import yaml
from torch.nn import ReLU, Tanh
from torch.optim import SGD, Adam

str2Opt = {'adam': Adam, 'sgd': SGD}
str2Activation = {'relu': ReLU, 'tanh': Tanh}
import wandb
from utils.helpers import convert_clip_range

ROOT_PATH = Path(__file__).parent

def get_value(x):
    if isinstance(x, dict):
        for key, value in x.items():
            x[key] = get_value(value)
    elif isinstance(x, str):
        if x.lower() in str2Activation:
            return str2Activation[x.lower()]
        if x.lower() in str2Opt:
            return str2Opt[x.lower()]
    return x


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule Taken from RL_Zoo3.

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

def preprocess_lr(lr: Union[str, float]):
    if isinstance(lr, str):
        if ("_") in lr:
            _, initial_value = lr.split("_")
            initial_value = float(initial_value)
            return linear_schedule(initial_value)
        return float(lr)
    return lr

def json_string_to_dict(json_string):
    """Convert a JSON string to a dictionary."""
    if json_string is None or json_string.lower() in ('null','none'):
        return None
    try:
        loaded_dict = json.loads(json_string)
        for key in loaded_dict: 
            loaded_dict[key] = get_value(loaded_dict[key])
        return loaded_dict
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError('Invalid JSON format for dictionary.')

from wandb.integration.sb3 import WandbCallback


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('null','none'):
        return None
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        # raise argparse.ArgumentTypeError('Boolean value expected.')
        return v

def load_yaml_defaults(yaml_file: str = None):
    if yaml_file is None:
        yaml_file = ROOT_PATH / 'defaults.yaml'
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)
    
def parse_args():
    parser = argparse.ArgumentParser()
    # mandatory arguments
    parser.add_argument('--env_type', type=str, choices=['atari', 'minigrid', 'gym', 'mujoco', 'football']) 
    parser.add_argument('--algo_type', type=str, choices=['ppo_nn', 'ppo_gbrl', 'a2c_gbrl', 'sac_gbrl', 'awr_gbrl', 'dqn_gbrl', 'a2c_nn', 'awr_nn', 'dqn_nn']) 
    parser.add_argument('--env_name', type=str)  
    # env args
    parser.add_argument('--seed', type=int)
    parser.add_argument('--verbose', type=int)
    parser.add_argument('--total_n_steps', type=int)
    parser.add_argument('--num_envs', type=int)
    # env parameters
    parser.add_argument('--log_interval', type=int)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'])
    parser.add_argument('--evaluate', type=str2bool)
    parser.add_argument('--env_kwargs', type=json_string_to_dict)
    parser.add_argument('--eval_kwargs', type=json_string_to_dict)
    parser.add_argument('--no_improvement_kwargs', type=json_string_to_dict) # training no improvement callback
    parser.add_argument('--wrapper', type=str)
    parser.add_argument('--wrapper_kwargs', type=json_string_to_dict)
    parser.add_argument('--atari_wrapper_kwargs', type=json_string_to_dict)
    
    # logging args   
    parser.add_argument('--wandb', type=str2bool) 
    parser.add_argument('--run_name', type=str, help='name of run')
    parser.add_argument('--group_name', type=str, help='wandb group name')
    parser.add_argument('--project', type=str)
    parser.add_argument('--entity', type=str)
    # algo parameters
    parser.add_argument('--normalize_advantage', type=str2bool)
    parser.add_argument('--policy_kwargs', type=json_string_to_dict)
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--n_steps', type=int)
    parser.add_argument('--ent_coef', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--gae_lambda', type=float)
    parser.add_argument('--target_kl', type=float)
    parser.add_argument('--log_std_init', type=float)
    parser.add_argument('--squash', type=float) # squash continuous actions using tanh
    parser.add_argument('--normalize_policy_grads', type=str2bool)
    # NN parameters
    parser.add_argument('--learning_rate', type=str)
    parser.add_argument('--use_sde', type=str2bool)
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--sde_sample_freq', type=int)
    parser.add_argument('--stats_window_size', type=int)
    ## A2C NN
    parser.add_argument('--use_rms_prop', type=str2bool)
    parser.add_argument('--rms_prop_eps', type=float)
    # PPO_GBRL params
    parser.add_argument('--policy_algo', type=str)
    parser.add_argument('--policy_lr', type=str)
    parser.add_argument('--stop_policy_lr', type=float)
    parser.add_argument('--policy_beta_1', type=float)
    parser.add_argument('--policy_beta_2', type=float)
    parser.add_argument('--policy_eps', type=float)
    parser.add_argument('--policy_shrinkage', type=float)

    parser.add_argument('--value_algo', type=str)
    parser.add_argument('--value_lr', type=str)
    parser.add_argument('--stop_value_lr', type=float)
    parser.add_argument('--value_beta_1', type=float)
    parser.add_argument('--value_beta_2', type=float)
    parser.add_argument('--value_eps', type=float)
    parser.add_argument('--value_shrinkage', type=float)
   
    parser.add_argument('--log_std_lr', type=str)
    parser.add_argument('--min_log_std_lr', type=float)
    
    parser.add_argument('--max_value_grad_norm', type=float)
    parser.add_argument('--max_policy_grad_norm', type=float)
    parser.add_argument('--vf_coef', type=float)
    parser.add_argument('--clip_range')
    parser.add_argument('--clip_range_vf')
    parser.add_argument('--log_std_grad_clip', type=float)
    parser.add_argument('--fixed_std', type=str2bool)
    parser.add_argument('--policy_bound_loss_weight', type=float)
    #SAC GBRL Params
    parser.add_argument('--mu_algo', type=str)
    parser.add_argument('--std_algo', type=str)
    parser.add_argument('--weights_algo', type=str)
    parser.add_argument('--bias_algo', type=str)
    parser.add_argument('--critic_algo', type=str)

    parser.add_argument('--critic_lr', type=float)
    parser.add_argument('--mu_lr', type=float)
    parser.add_argument('--std_lr', type=float)
    parser.add_argument('--weights_lr', type=float)
    parser.add_argument('--bias_lr', type=float)
    parser.add_argument('--ent_lr', type=float)

    parser.add_argument('--mu_beta_1', type=float)
    parser.add_argument('--mu_beta_2', type=float)
    parser.add_argument('--mu_eps', type=float)
    parser.add_argument('--mu_shrinkage', type=float)
    parser.add_argument('--std_beta_1', type=float)
    parser.add_argument('--std_beta_2', type=float)
    parser.add_argument('--std_eps', type=float)
    parser.add_argument('--std_shrinkage', type=float)
    parser.add_argument('--weights_beta_1', type=float)
    parser.add_argument('--weights_beta_2', type=float)
    parser.add_argument('--weights_eps', type=float)
    parser.add_argument('--weights_shrinkage', type=float)
    parser.add_argument('--bias_beta_1', type=float)
    parser.add_argument('--bias_beta_2', type=float)
    parser.add_argument('--bias_eps', type=float)
    parser.add_argument('--bias_shrinkage', type=float)
    parser.add_argument('--critic_beta_1', type=float)
    parser.add_argument('--critic_beta_2', type=float)
    parser.add_argument('--critic_eps', type=float)
    parser.add_argument('--critic_shrinkage', type=float)

    parser.add_argument('--tau', type=float)
    parser.add_argument('--learning_starts', type=int)
    parser.add_argument('--train_freq', type=int)
    parser.add_argument('--gradient_steps', type=int)
    parser.add_argument('--n_critics', type=int)
    parser.add_argument('--target_entropy')
    parser.add_argument('--target_update_interval', type=int)
    parser.add_argument('--buffer_size', type=int)
    parser.add_argument('--max_q_grad_norm', type=float)
    parser.add_argument('--q_func_type', type=str, choices=['linear', 'tanh', 'quadratic'])
    # AWR Params
    parser.add_argument('--beta', type=float)
    parser.add_argument('--policy_gradient_steps', type=int)
    parser.add_argument('--value_gradient_steps', type=int)
    parser.add_argument('--value_batch_size', type=int)
    parser.add_argument('--weights_max', type=float)
    parser.add_argument('--reward_mode', type=str, choices=['gae', 'monte-carlo'])
    # AWR NN
    parser.add_argument('--actor_learning_rate', type=str)
    parser.add_argument('--critic_learning_rate', type=str)
    # DQN Params
    parser.add_argument('--exploration_fraction', type=float)
    parser.add_argument('--exploration_initial_eps', type=float)
    parser.add_argument('--exploration_final_eps', type=float)
    parser.add_argument('--normalize_q_grads', type=str2bool)
    # GBT params
    parser.add_argument('--min_data_in_leaf', type=int)
    parser.add_argument('--max_depth', type=int)
    parser.add_argument('--n_bins', type=int)
    parser.add_argument('--par_th', type=int)
    parser.add_argument('--control_variates', type=str2bool)
    parser.add_argument('--shared_tree_struct', type=str2bool)
    parser.add_argument('--split_score_func', type=str, choices=['cosine', 'L2', 'l2', 'COSINE', 'Cosine'])
    parser.add_argument('--generator_type', type=str, choices=['Quantile', 'quantile', 'l2', 'Uniform', 'uniform'])
    parser.add_argument('--grow_policy', type=str, choices=['Oblivious', 'Greedy', 'greedy', 'oblivious'])
    # Saving params
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--save_every', type=int)
    # Distillation Params
    parser.add_argument('--distil', type=str2bool)
    parser.add_argument('--distil_kwargs', type=json_string_to_dict)
    
    args = parser.parse_args()

    defaults = load_yaml_defaults()

    return get_defaults(args, defaults)

def get_defaults(args, defaults):
    # Set hardcoded defaults
    args.env_type = args.env_type if args.env_type else 'gym'
    args.algo_type = args.algo_type if args.algo_type else 'ppo_gbrl'
    args.env_name = args.env_name if args.env_name else 'CartPole-v1'
    # Set defaults from YAML
    args.seed = args.seed if args.seed is not None else defaults['env']['seed']
    args.verbose = args.verbose if args.verbose is not None else defaults['env']['verbose']
    args.total_n_steps = args.total_n_steps if args.total_n_steps is not None else defaults['env']['total_n_steps']
    args.num_envs = args.num_envs if args.num_envs is not None else defaults['env']['num_envs']
    args.device = args.device if args.device is not None else defaults['env']['device']
    args.log_interval = args.log_interval if args.log_interval is not None else defaults['env']['log_interval']
    args.evaluate = args.evaluate if args.evaluate is not None else defaults['env']['evaluate']
    # Wrapper and env_kwargs
    args.wrapper = args.wrapper if args.wrapper is not None else defaults['env']['wrapper']
    args.env_kwargs = args.env_kwargs if args.env_kwargs is not None else defaults['env']['env_kwargs']
    args.eval_kwargs = args.eval_kwargs if args.eval_kwargs else defaults['env']['eval_kwargs']
    args.no_improvement_kwargs = args.no_improvement_kwargs if args.no_improvement_kwargs else defaults['env'].get('no_improvement_kwargs', None)
    # Wrapper kwargs
    if args.wrapper == 'normalize' and args.wrapper_kwargs is None:
        args.wrapper_kwargs = defaults[args.algo_type]['wrapper_kwargs']
    args.atari_wrapper_kwargs = args.atari_wrapper_kwargs if args.atari_wrapper_kwargs else defaults[args.algo_type]['atari_wrapper_kwargs']
    # Logging defaults
    args.wandb = args.wandb if args.wandb is not None else False
    args.run_name = args.run_name if args.run_name is not None else defaults['logging']['run_name']
    args.group_name = args.group_name if args.group_name is not None else defaults['logging']['group_name']
    args.project = args.project if args.project is not None else defaults['logging']['project']
    args.entity = args.entity if args.entity is not None else defaults['logging']['entity']
    # Algorithm specific defaults
    algo_defaults = defaults.get(args.algo_type, {})
    args.normalize_advantage = args.normalize_advantage if args.normalize_advantage is not None else algo_defaults.get('normalize_advantage', False)
    args.n_epochs = args.n_epochs if args.n_epochs is not None else algo_defaults.get('n_epochs', 10)
    args.n_steps = args.n_steps if args.n_steps is not None else algo_defaults.get('n_steps', 512)
    args.ent_coef = args.ent_coef if args.ent_coef is not None else algo_defaults.get('ent_coef', 0.01)
    args.batch_size = args.batch_size if args.batch_size is not None else algo_defaults.get('batch_size', 64)
    args.gamma = args.gamma if args.gamma is not None else algo_defaults.get('gamma', 0.99)
    args.gae_lambda = args.gae_lambda if args.gae_lambda is not None else algo_defaults.get('gae_lambda', 0.95)
    args.target_kl = args.target_kl if args.target_kl is not None else algo_defaults.get('target_kl', None)
    args.log_std_init = args.log_std_init if args.log_std_init is not None else algo_defaults.get('log_std_init', -2)
    args.squash = args.squash if args.squash is not None else algo_defaults.get('squash', False)
    args.normalize_policy_grads = args.normalize_policy_grads if args.normalize_policy_grads is not None else algo_defaults.get('normalize_policy_grads', False)
    args.vf_coef = args.vf_coef if args.vf_coef is not None else algo_defaults.get('vf_coef', 0.5)
    args.clip_range = convert_clip_range(args.clip_range) if args.clip_range is not None else algo_defaults.get('clip_range', 0.2)
    args.clip_range_vf = convert_clip_range(args.clip_range_vf) if args.clip_range_vf is not None else algo_defaults.get('clip_range_vf', 0.2)
    args.max_policy_grad_norm = args.max_policy_grad_norm if args.max_policy_grad_norm is not None else algo_defaults.get('max_policy_grad_norm', 150.0)
    args.max_value_grad_norm = args.max_value_grad_norm if args.max_value_grad_norm is not None else algo_defaults.get('max_value_grad_norm', 100.0)
    args.fixed_std = args.fixed_std if args.fixed_std is not None else algo_defaults.get('fixed_std', False)
    args.policy_bound_loss_weight = args.policy_bound_loss_weight if args.policy_bound_loss_weight is not None else algo_defaults.get('policy_bound_loss_weight', None)
    args.min_log_std_lr = args.min_log_std_lr if args.min_log_std_lr is not None else algo_defaults.get('min_log_std_lr', 1.e-6)
    args.log_std_lr = args.log_std_lr if args.log_std_lr is not None else algo_defaults.get('log_std_lr', 1.e-2)
    args.buffer_size = args.buffer_size if args.buffer_size is not None else algo_defaults.get('buffer_size', 1000000)
    args.train_freq = args.train_freq if args.train_freq is not None else algo_defaults.get('train_freq', 1)
    args.learning_starts = args.learning_starts if args.learning_starts is not None else algo_defaults.get('learning_starts', 10000)
    args.gradient_steps = args.gradient_steps if args.gradient_steps is not None else algo_defaults.get('gradient_steps', 50)
    args.beta = args.beta if args.beta is not None else algo_defaults.get('beta', 1.0)
    args.policy_gradient_steps = args.policy_gradient_steps if args.policy_gradient_steps is not None else algo_defaults.get('policy_gradient_steps', 1000)
    args.value_gradient_steps = args.value_gradient_steps if args.value_gradient_steps is not None else algo_defaults.get('value_gradient_steps', 200)
    args.weights_max = args.weights_max if args.weights_max is not None else algo_defaults.get('weights_max', 20)
    args.value_batch_size = args.value_batch_size if args.value_batch_size is not None else algo_defaults.get('value_batch_size', 8192)
    args.reward_mode = args.reward_mode if args.reward_mode is not None else algo_defaults.get('reward_mode', 'gae')
    args.tau = args.tau if args.tau is not None else algo_defaults.get('tau', 0.005)
    # NN Params
    args.learning_rate = preprocess_lr(args.learning_rate if args.learning_rate is not None else algo_defaults.get('learning_rate', 1.e-2))
    args.actor_learning_rate = preprocess_lr(args.actor_learning_rate if args.actor_learning_rate is not None else algo_defaults.get('actor_learning_rate', 1.e-2))
    args.critic_learning_rate = preprocess_lr(args.critic_learning_rate if args.critic_learning_rate is not None else algo_defaults.get('critic_learning_rate', 1.e-2))
    args.use_sde = args.use_sde if args.use_sde is not None else algo_defaults.get('use_sde', False)
    args.max_grad_norm = args.max_grad_norm if args.max_grad_norm is not None else algo_defaults.get('max_grad_norm', 0.5)
    args.sde_sample_freq = args.sde_sample_freq if args.sde_sample_freq is not None else algo_defaults.get('sde_sample_freq', -1)
    args.stats_window_size = args.stats_window_size if args.stats_window_size is not None else algo_defaults.get('stats_window_size', 100)
    args.use_rms_prop = args.use_rms_prop if args.use_rms_prop is not None else algo_defaults.get('use_rms_prop', False)
    args.rms_prop_eps = args.rms_prop_eps if args.rms_prop_eps is not None else algo_defaults.get('rms_prop_eps', 1.0e-5)
    # Optimizer Params
    tree_optimizer_defaults = defaults['tree_optimizer']
    args.policy_algo = args.policy_algo if args.policy_algo is not None else tree_optimizer_defaults['policy_optimizer']['algo']
    args.policy_lr = args.policy_lr if args.policy_lr is not None else tree_optimizer_defaults['policy_optimizer']['lr']
    args.policy_beta_1 = args.policy_beta_1 if args.policy_beta_1 is not None else tree_optimizer_defaults['policy_optimizer'].get('beta_1', 0.9)
    args.policy_beta_2 = args.policy_beta_2 if args.policy_beta_2 is not None else tree_optimizer_defaults['policy_optimizer'].get('beta_2', 0.999)
    args.policy_eps = args.policy_eps if args.policy_eps is not None else tree_optimizer_defaults['policy_optimizer'].get('eps', 1.0e-5)
    args.value_algo = args.value_algo if args.value_algo is not None else tree_optimizer_defaults['value_optimizer']['algo']
    args.value_lr = args.value_lr if args.value_lr is not None else tree_optimizer_defaults['value_optimizer']['lr']
    args.value_beta_1 = args.value_beta_1 if args.value_beta_1 is not None else tree_optimizer_defaults['value_optimizer'].get('beta_1', 0.9)
    args.value_beta_2 = args.value_beta_2 if args.value_beta_2 is not None else tree_optimizer_defaults['value_optimizer'].get('beta_2', 0.999)
    args.value_eps = args.value_eps if args.value_eps is not None else tree_optimizer_defaults['value_optimizer'].get('eps', 1.0e-5)
    args.mu_algo = args.mu_algo if args.mu_algo is not None else tree_optimizer_defaults['mu_optimizer']['algo']
    args.mu_lr = args.mu_lr if args.mu_lr is not None else tree_optimizer_defaults['mu_optimizer']['lr']
    args.mu_beta_1 = args.mu_beta_1 if args.mu_beta_1 is not None else tree_optimizer_defaults['mu_optimizer'].get('beta_1', 0.9)
    args.mu_beta_2 = args.mu_beta_2 if args.mu_beta_2 is not None else tree_optimizer_defaults['mu_optimizer'].get('beta_2', 0.999)
    args.mu_eps = args.mu_eps if args.mu_eps is not None else tree_optimizer_defaults['mu_optimizer'].get('eps', 1.0e-5)
    args.std_algo = args.std_algo if args.std_algo is not None else tree_optimizer_defaults['std_optimizer']['algo']
    args.std_lr = args.std_lr if args.std_lr is not None else tree_optimizer_defaults['std_optimizer']['lr']
    args.std_beta_1 = args.std_beta_1 if args.std_beta_1 is not None else tree_optimizer_defaults['std_optimizer'].get('beta_1', 0.9)
    args.std_beta_2 = args.std_beta_2 if args.std_beta_2 is not None else tree_optimizer_defaults['std_optimizer'].get('beta_2', 0.999)
    args.std_eps = args.std_eps if args.std_eps is not None else tree_optimizer_defaults['std_optimizer'].get('eps', 1.0e-5)
    args.weights_algo = args.weights_algo if args.weights_algo is not None else tree_optimizer_defaults['weights_optimizer']['algo']
    args.weights_lr = args.weights_lr if args.weights_lr is not None else tree_optimizer_defaults['weights_optimizer']['lr']
    args.weights_beta_1 = args.weights_beta_1 if args.weights_beta_1 is not None else tree_optimizer_defaults['weights_optimizer'].get('beta_1', 0.9)
    args.weights_beta_2 = args.weights_beta_2 if args.weights_beta_2 is not None else tree_optimizer_defaults['weights_optimizer'].get('beta_2', 0.999)
    args.weights_eps = args.weights_eps if args.weights_eps is not None else tree_optimizer_defaults['weights_optimizer'].get('eps', 1.0e-5)
    args.bias_algo = args.bias_algo if args.bias_algo is not None else tree_optimizer_defaults['bias_optimizer']['algo']
    args.bias_lr = args.bias_lr if args.bias_lr is not None else tree_optimizer_defaults['bias_optimizer']['lr']
    args.bias_beta_1 = args.bias_beta_1 if args.bias_beta_1 is not None else tree_optimizer_defaults['bias_optimizer'].get('beta_1', 0.9)
    args.bias_beta_2 = args.bias_beta_2 if args.bias_beta_2 is not None else tree_optimizer_defaults['bias_optimizer'].get('beta_2', 0.999)
    args.bias_eps = args.bias_eps if args.bias_eps is not None else tree_optimizer_defaults['bias_optimizer'].get('eps', 1.0e-5)
    args.critic_algo = args.critic_algo if args.critic_algo is not None else tree_optimizer_defaults['critic_optimizer']['algo']
    args.critic_lr = args.critic_lr if args.critic_lr is not None else tree_optimizer_defaults['critic_optimizer']['lr']
    args.critic_beta_1 = args.critic_beta_1 if args.critic_beta_1 is not None else tree_optimizer_defaults['critic_optimizer'].get('beta_1', 0.9)
    args.critic_beta_2 = args.critic_beta_2 if args.critic_beta_2 is not None else tree_optimizer_defaults['critic_optimizer'].get('beta_2', 0.999)
    args.critic_eps = args.critic_eps if args.critic_eps is not None else tree_optimizer_defaults['critic_optimizer'].get('eps', 1.0e-5)
    # Tree struct params
    tree_struct_defaults = defaults['tree_struct']
    args.max_depth = args.max_depth if args.max_depth is not None else tree_struct_defaults['max_depth']
    args.n_bins = args.n_bins if args.n_bins is not None else tree_struct_defaults['n_bins']
    args.min_data_in_leaf = args.min_data_in_leaf if args.min_data_in_leaf is not None else tree_struct_defaults['min_data_in_leaf']
    args.par_th = args.par_th if args.par_th is not None else tree_struct_defaults['par_th']
    args.grow_policy = args.grow_policy if args.grow_policy is not None else tree_struct_defaults['grow_policy']                     
    # GBRL Params
    gbrl_param_defaults = defaults['gbrl_params']  
    args.control_variates = args.control_variates if args.control_variates is not None else gbrl_param_defaults['control_variates']                     
    args.split_score_func = args.split_score_func if args.split_score_func is not None else gbrl_param_defaults['split_score_func']                     
    args.generator_type = args.generator_type if args.generator_type is not None else gbrl_param_defaults['generator_type']                     
    args.shared_tree_struct = args.shared_tree_struct if args.shared_tree_struct is not None else gbrl_param_defaults['shared_tree_struct']                          
    # SAC GBRL Params
    sac_gbrl_defaults = defaults.get('sac_gbrl', {})
    args.ent_lr = args.ent_lr if args.ent_lr is not None else sac_gbrl_defaults.get('ent_lr', 1.0e-3)
    args.n_critics = args.n_critics if args.n_critics is not None else sac_gbrl_defaults.get('n_critics', 1)
    args.target_update_interval = args.target_update_interval if args.target_update_interval is not None else sac_gbrl_defaults.get('target_update_interval', 100)
    args.max_q_grad_norm = args.max_q_grad_norm if args.max_q_grad_norm is not None else sac_gbrl_defaults.get('max_q_grad_norm', 0.0)
    args.target_entropy = args.target_entropy if args.target_entropy is not None else sac_gbrl_defaults.get('target_entropy', "auto")
    args.q_func_type = args.q_func_type if args.q_func_type is not None else sac_gbrl_defaults.get('q_func_type', 'linear')
    # DQN Params
    dqn_gbrl_defaults = defaults.get('dqn_gbrl', {})
    args.exploration_fraction = args.exploration_fraction if args.exploration_fraction is not None else dqn_gbrl_defaults.get('exploration_fraction', 0.1)
    args.exploration_initial_eps = args.exploration_initial_eps if args.exploration_initial_eps is not None else dqn_gbrl_defaults.get('exploration_initial_eps', 1.0)
    args.exploration_final_eps = args.exploration_final_eps if args.exploration_final_eps is not None else dqn_gbrl_defaults.get('exploration_final_eps', 0.05)
    args.normalize_q_grads = args.normalize_q_grads if args.normalize_q_grads is not None else dqn_gbrl_defaults.get('normalize_q_grads', False)
    # Saving params
    args.save_every = args.save_every if args.save_every is not None else defaults['save']['save_every']
    args.save_name = args.save_name if args.save_name is not None else defaults['save']['save_name']
    args.save_path = args.save_path if args.save_path is not None else defaults['save']['save_path']
    
    # Distillation Params
    args.distil = args.distil if args.distil is not None else defaults['distillation']['distil']
    args.distil_kwargs = args.distil_kwargs if args.distil_kwargs is not None else defaults['distillation']['distil_kwargs']
    return args

def process_logging(args, callback_list):
    if not args.wandb:
        tb_name = f'n_{args.env_type}_{args.run_name}_{args.env_name}_seed_{args.seed}'
        tensorboard_log = f"runs/{tb_name}"
        return tensorboard_log

    print(f'args.wand: {args.wandb}')
    # run_name = args.project
    run = wandb.init(project=args.project, group=None if args.group_name is None else args.group_name + '_' + args.algo_type + '_' + args.env_type + '_' + args.env_name,
                     name=args.run_name + '_' + args.algo_type + '_' + args.env_type + '_' + args.env_name + '_seed_' + str(args.seed), mode="online",
                     config=args.__dict__, entity=args.entity, monitor_gym=True, sync_tensorboard=True, save_code=False
                    )

    callback_list.append(WandbCallback(
    gradient_save_freq=0,
    model_save_path=None,
    verbose=1)
)
    tb_name = ''
    if args.group_name is not None:
        tb_name += f'g_{args.group_name}_'
    tb_name += f'n_{args.env_type}_{args.run_name}_{args.env_name}_seed_{args.seed}'
    tensorboard_log = f"runs/{tb_name}"
    # args.wandb = wandb
    print('finished setting up wandb')
    return tensorboard_log

def process_policy_kwargs(args):
    if args.algo_type == 'ppo_gbrl':
        return { 
            "clip_range": args.clip_range,
            "clip_range_vf": args.clip_range_vf,
            "normalize_advantage": args.normalize_advantage,
            "target_kl": args.target_kl,
            "n_epochs": args.n_epochs,
            "max_policy_grad_norm": args.max_policy_grad_norm,
            "max_value_grad_norm": args.max_value_grad_norm,
            "is_categorical": True if args.env_type == 'minigrid' else False,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "n_steps": args.n_steps,
            "normalize_policy_grads": args.normalize_policy_grads,
            "batch_size": args.batch_size,
            "gae_lambda": args.gae_lambda,
            "gamma": args.gamma,
            "total_n_steps": args.total_n_steps,
            "policy_kwargs": args.policy_kwargs if args.policy_kwargs is not None else {
                "log_std_init": args.log_std_init,
                "squash": args.squash,
                "shared_tree_struct": args.shared_tree_struct,
                "tree_struct": {
                    "max_depth": args.max_depth,
                    "n_bins": args.n_bins,
                    "min_data_in_leaf": args.min_data_in_leaf,
                    "par_th": args.par_th,
                    "grow_policy": args.grow_policy,
                }, 
                "tree_optimizer": {
                    "gbrl_params": {
                        "split_score_func": args.split_score_func,
                        'control_variates': args.control_variates,
                        "generator_type": args.generator_type,
                    }, 
                    "policy_optimizer": {
                        "policy_algo": args.policy_algo,
                        "policy_lr": args.policy_lr,
                        # Adam optimizer
                        "policy_beta_1": args.policy_beta_1,
                        "policy_beta_2": args.policy_beta_2,
                        "policy_eps": args.policy_eps,
                        "policy_shrinkage": args.policy_shrinkage,
                    }, 
                    "value_optimizer": {
                        "value_algo": args.value_algo,
                        "value_lr": args.value_lr,
                        "value_beta_1": args.value_beta_1,
                        "value_beta_2": args.value_beta_2,
                        "value_eps": args.value_eps,
                        "value_shrinkage": args.value_shrinkage,
                    },
                },
            },
            "fixed_std": args.fixed_std,
            "log_std_lr": args.log_std_lr,
            "min_log_std_lr": args.min_log_std_lr,
            "policy_bound_loss_weight": args.policy_bound_loss_weight,
            "device": args.device,
            "seed": args.seed,
            "verbose": args.verbose,
        }
    elif args.algo_type == 'a2c_gbrl':
        return { 
            "normalize_advantage": args.normalize_advantage,
            "max_policy_grad_norm": args.max_policy_grad_norm,
            "max_value_grad_norm": args.max_value_grad_norm,
            "is_categorical": True if args.env_type == 'minigrid' else False,
            "ent_coef": args.ent_coef,
            "n_steps": args.n_steps,
            "vf_coef": args.vf_coef,
            "normalize_policy_grads": args.normalize_policy_grads,
            "gae_lambda": args.gae_lambda,
            "gamma": args.gamma,
            "total_n_steps": args.total_n_steps,
            "policy_kwargs": args.policy_kwargs if args.policy_kwargs is not None else {
                "log_std_init": args.log_std_init,
                "squash": args.squash,
                "shared_tree_struct": args.shared_tree_struct,
                "tree_struct": {
                    "max_depth": args.max_depth,
                    "n_bins": args.n_bins,
                    "min_data_in_leaf": args.min_data_in_leaf,
                    "par_th": args.par_th,
                    "grow_policy": args.grow_policy,
                }, 
                "tree_optimizer": {
                    "gbrl_params": {
                        "split_score_func": args.split_score_func,
                        'control_variates': args.control_variates,
                        "generator_type": args.generator_type,
                    }, 
                    "policy_optimizer": {
                        "policy_algo": args.policy_algo,
                        "policy_lr": args.policy_lr,
                        # Adam optimizer
                        "policy_beta_1": args.policy_beta_1,
                        "policy_beta_2": args.policy_beta_2,
                        "policy_eps": args.policy_eps,
                        "policy_shrinkage": args.policy_shrinkage,
                    }, 
                    "value_optimizer": {
                        "value_algo": args.value_algo,
                        "value_lr": args.value_lr,
                        "value_beta_1": args.value_beta_1,
                        "value_beta_2": args.value_beta_2,
                        "value_eps": args.value_eps,
                        "value_shrinkage": args.value_shrinkage,

                    },
                },
            },
            "fixed_std": args.fixed_std,
            "log_std_lr": args.log_std_lr,
            "min_log_std_lr": args.min_log_std_lr,
            "device": args.device,
            "seed": args.seed,
            "verbose": args.verbose,
        }
    elif args.algo_type == 'sac_gbrl':
        return {
            "train_freq": args.train_freq,
            "seed": args.seed,
            "buffer_size": args.buffer_size,
            "learning_starts": args.learning_starts,
            "ent_lr": args.ent_lr,
            "batch_size": args.batch_size,
            "tau": args.tau,
            "gamma": args.gamma,
            "ent_coef": args.ent_coef,
            "target_update_interval": args.target_update_interval,
            "gradient_steps": args.gradient_steps,
            "max_q_grad_norm": args.max_q_grad_norm,
            "max_policy_grad_norm": args.max_policy_grad_norm,
            "normalize_policy_grads": args.normalize_policy_grads,
            "verbose": args.verbose,
             "policy_kwargs": args.policy_kwargs if args.policy_kwargs is not None else {
                "shared_tree_struct": args.shared_tree_struct,
                "n_critics": args.n_critics,
                "tree_struct": {
                    "max_depth": args.max_depth,
                    "n_bins": args.n_bins,
                    "min_data_in_leaf": args.min_data_in_leaf,
                    "par_th": args.par_th,
                    "grow_policy": args.grow_policy,
                }, 
                "tree_optimizer": {
                    "gbrl_params": {
                        "split_score_func": args.split_score_func,
                        'control_variates': args.control_variates,
                        "generator_type": args.generator_type,
                    }, 
                    "actor_optimizer": {
                        "mu_optimizer": {
                            "mu_algo": args.mu_algo,
                            "mu_lr": args.mu_lr,
                            "mu_beta_1": args.mu_beta_1,
                            "mu_beta_2": args.mu_beta_2,
                            "mu_eps": args.mu_eps,
                            "mu_shrinkage": args.mu_shrinkage,
                        }, 
                        "std_optimizer": {
                            "std_algo": args.std_algo,
                            "std_lr": args.std_lr,
                            "std_beta_1": args.std_beta_1,
                            "std_beta_2": args.std_beta_2,
                            "std_eps": args.std_eps,
                            "std_shrinkage": args.std_shrinkage,
                        }, 
                    },
                    "critic_optimizer": {
                        "weights_optimizer": {
                            "weights_algo": args.weights_algo,
                            "weights_lr": args.weights_lr,
                            "weights_beta_1": args.weights_beta_1,
                            "weights_beta_2": args.weights_beta_2,
                            "weights_eps": args.weights_eps,
                            "weights_shrinkage": args.weights_shrinkage,
                        },
                        "bias_optimizer": {
                            "bias_algo": args.bias_algo,
                            "bias_lr": args.bias_lr,
                            "bias_beta_1": args.bias_beta_1,
                            "bias_beta_2": args.bias_beta_2,
                            "bias_eps": args.bias_eps,
                            "bias_shrinkage": args.bias_shrinkage,
                        },
                    },
                },
            },
            "device": args.device
        }
    elif args.algo_type == 'awr_gbrl':
        return {
            "normalize_advantage": args.normalize_advantage,
            "max_policy_grad_norm": args.max_policy_grad_norm,
            "max_value_grad_norm": args.max_value_grad_norm,
            "is_categorical": True if args.env_type == 'minigrid' else False,
            "ent_coef": args.ent_coef,
            "normalize_policy_grads": args.normalize_policy_grads,
            "batch_size": args.batch_size,
            "beta": args.beta,
            "buffer_size": args.buffer_size,
            "value_batch_size": args.value_batch_size,
            "reward_mode": args.reward_mode,
            "gradient_steps": args.gradient_steps,
            "learning_starts": args.learning_starts,
            "gae_lambda": args.gae_lambda,
            "gamma": args.gamma,
            "train_freq": args.train_freq,
            "weights_max": args.weights_max,
            "policy_kwargs": args.policy_kwargs if args.policy_kwargs is not None else {
                "log_std_init": args.log_std_init,
                "shared_tree_struct": args.shared_tree_struct,
                "squash": args.squash,
                "tree_struct": {
                    "max_depth": args.max_depth,
                    "n_bins": args.n_bins,
                    "min_data_in_leaf": args.min_data_in_leaf,
                    "par_th": args.par_th,
                    "grow_policy": args.grow_policy,
                }, 
                "tree_optimizer": {
                    "gbrl_params": {
                        "split_score_func": args.split_score_func,
                        'control_variates': args.control_variates,
                        "generator_type": args.generator_type,
                    }, 
                    "policy_optimizer": {
                        "policy_algo": args.policy_algo,
                        "policy_lr": args.policy_lr,
                        "policy_beta_1": args.policy_beta_1,
                        "policy_beta_2": args.policy_beta_2,
                        "policy_eps": args.policy_eps,
                        "policy_shrinkage": args.policy_shrinkage,
                    }, 
                    "value_optimizer": {
                        "value_algo": args.value_algo,
                        "value_lr": args.value_lr,
                        "value_beta_1": args.value_beta_1,
                        "value_beta_2": args.value_beta_2,
                        "value_eps": args.value_eps,
                        "value_shrinkage": args.value_shrinkage,
                    },
                },
            },
            "fixed_std": args.fixed_std,
            "log_std_lr": args.log_std_lr,
            "vf_coef": args.vf_coef,
            "min_log_std_lr": args.min_log_std_lr,
            "policy_bound_loss_weight": args.policy_bound_loss_weight,
            "device": args.device,
            "seed": args.seed,
            "verbose": args.verbose,
        }
    elif args.algo_type == 'dqn_gbrl':
        return {
            "max_q_grad_norm": args.max_q_grad_norm,
            "is_categorical": True if args.env_type == 'minigrid' else False,
            "normalize_q_grads": args.normalize_q_grads,
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "gradient_steps": args.gradient_steps,
            "learning_starts": args.learning_starts,
            "gamma": args.gamma,
            "train_freq": args.train_freq,
            "exploration_fraction": args.exploration_fraction,
            "exploration_initial_eps": args.exploration_initial_eps,
            "exploration_final_eps": args.exploration_final_eps,
            "target_update_interval": args.target_update_interval,
            "policy_kwargs": args.policy_kwargs if args.policy_kwargs is not None else {
                "tree_struct": {
                    "max_depth": args.max_depth,
                    "n_bins": args.n_bins,
                    "min_data_in_leaf": args.min_data_in_leaf,
                    "par_th": args.par_th,
                    "grow_policy": args.grow_policy,
                }, 
                "tree_optimizer": {
                    "gbrl_params": {
                        "split_score_func": args.split_score_func,
                        'control_variates': args.control_variates,
                        "generator_type": args.generator_type,
                    }, 
                    "critic_optimizer": {
                        "algo": args.critic_algo,
                        "lr": args.critic_lr,
                        "beta_1": args.critic_beta_1,
                        "beta_2": args.critic_beta_2,
                        "eps": args.critic_eps,
                        "shrinkage": args.critic_shrinkage,
                    }, 
                },
            },
            "device": args.device,
            "seed": args.seed,
            "verbose": args.verbose,
        }
    elif args.algo_type == 'ppo_nn':
        from stable_baselines3.common.policies import ActorCriticPolicy
        return {
            "policy": ActorCriticPolicy,
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
            "clip_range_vf": args.clip_range_vf,
            "normalize_advantage": args.normalize_advantage,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "max_grad_norm": args.max_grad_norm,
            "use_sde": args.use_sde,
            "sde_sample_freq": args.sde_sample_freq,
            "stats_window_size": args.stats_window_size,
            "policy_kwargs": args.policy_kwargs,
            "verbose": args.verbose,
            "seed": args.seed,
            "device": args.device,
            "_init_setup_model": True
        }
    elif args.algo_type == 'a2c_nn':
        from stable_baselines3.common.policies import ActorCriticPolicy
        return {
            "policy": ActorCriticPolicy,
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "normalize_advantage": args.normalize_advantage,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "max_grad_norm": args.max_grad_norm,
            "rms_prop_eps": args.rms_prop_eps,
            "use_rms_prop": args.use_rms_prop,
            "use_sde": args.use_sde,
            "sde_sample_freq": args.sde_sample_freq,
            "stats_window_size": args.stats_window_size,
            "policy_kwargs": args.policy_kwargs,
            "verbose": args.verbose,
            "seed": args.seed,
            "device": args.device,
            "_init_setup_model": True
        }
    elif args.algo_type == 'awr_nn':
        from policies.awr_nn_policy import AWRPolicy
        return {
            "policy": AWRPolicy,
            "learning_rate": args.learning_rate,
            "train_freq": args.train_freq,
            "gamma": args.gamma,
            "beta": args.beta,
            "gae_lambda": args.gae_lambda,
            "normalize_advantage": args.normalize_advantage,
            "weights_max": args.weights_max,
            "policy_bound_loss_weight": args.policy_bound_loss_weight,
            "ent_coef": args.ent_coef,
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "value_batch_size": args.value_batch_size,
            "reward_mode": args.reward_mode,
            "policy_kwargs":
            {
                "optimizer_kwargs": {
                    "actor_lr": args.actor_learning_rate,
                    "critic_lr": args.critic_learning_rate,
                },
                "log_std_init": args.log_std_init,
            },
            "policy_gradient_steps": args.policy_gradient_steps,
            "value_gradient_steps": args.value_gradient_steps,
            "learning_starts": args.learning_starts,
            "max_grad_norm": args.max_grad_norm,
            "verbose": args.verbose,
            "seed": args.seed,
            "device": args.device,
        }
    elif args.algo_type == 'dqn_nn':
        from stable_baselines3.dqn.policies import DQNPolicy
        return {
            'policy': DQNPolicy,
            "max_grad_norm": args.max_grad_norm,
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "gradient_steps": args.gradient_steps,
            "learning_starts": args.learning_starts,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "tau": args.tau,
            "train_freq": args.train_freq,
            "exploration_fraction": args.exploration_fraction,
            "exploration_initial_eps": args.exploration_initial_eps,
            "exploration_final_eps": args.exploration_final_eps,
            "target_update_interval": args.target_update_interval,
            "policy_kwargs": args.policy_kwargs, 
            "device": args.device,
            "seed": args.seed,
            "verbose": args.verbose,
        }





if __name__ == '__main__':
    args = parse_args()
    print('')