
##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
import io
import pathlib
import sys
import time
import copy
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib import MaskablePPO

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import (get_action_masks,
                                               is_masking_supported)
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import (GymEnv, MaybeCallback,
                                                   Schedule)
from stable_baselines3.common.utils import (explained_variance, obs_as_tensor, get_schedule_fn,
                                            safe_mean)
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as F

from algos.ppo import PPO_GBRL
from buffers.rollout_buffer import (CategoricalRolloutBuffer,
                                    MaskableCategoricalRolloutBuffer,
                                    MaskableRolloutBuffer)
from policies.actor_critic_policy import ActorCriticPolicy


class PPO_GBRL_SelfPlay(PPO_GBRL):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(self, env: Union[GymEnv, str],
                 clip_range: float = 0.2, 
                 clip_range_vf: float = None, 
                 policy: Type[BasePolicy]= ActorCriticPolicy,
                 normalize_advantage: bool = True, 
                 target_kl: float = None,
                 max_policy_grad_norm: float = None,
                 max_value_grad_norm: float = None,
                 vf_coef: float = 0.5,
                 ent_coef: float = 0.0,
                 batch_size: int = 256,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 n_epochs: int = 4,
                 n_steps: int = 512,
                 total_n_steps: int = 1e6,
                 learning_rate: float = 1e-3,
                 use_masking: bool = False,
                 policy_kwargs: Dict = {
                    'shared_tree_struct': True,
                    'tree_struct': {
                        'max_depth': 4,
                        'n_bins': 256,
                        'min_data_in_leaf': 0,
                        'par_th': 2, # parallelization threshold - min_number of samples for cpu parallelization
                    },
                    'tree_optimizer': {
                    'gbrl_params': {
                            'control_variates': False,
                            'split_score_func': 'cosine',
                            'generator_type': "Quantile",
                        },
                        'policy_optimizer': {
                            'policy_algo': 'SGD',
                            'policy_lr': 5.209483743331993e-03,
                            'policy_shrinkage': 0, # CPU only
                        }, 
                        'value_optimizer': {
                            'value_algo': 'SGD',
                            'value_lr': 0.009038987598789402,
                            'value_shrinkage': 0, # CPU only
                        },
                    }
                 },
                 fixed_std: bool = False,
                 log_std_lr: float = 3e-4,
                 min_log_std_lr: float = 3e-45,
                 policy_bound_loss_weight: float = None,
                 device: str = "cpu",
                 seed: Optional[int] = None,
                 verbose: int = 1,
                 tensorboard_log: str = None,
                 rollouts_player: int = 5,
                 _init_setup_model: bool = False):

        if use_masking:
            policy_kwargs['use_masking'] = True
        super().__init__(policy=policy,
        env=env,
        seed=seed,
        tensorboard_log=tensorboard_log,
        learning_rate=learning_rate, # not used
        vf_coef=vf_coef, # not used
        ent_coef=ent_coef,
        batch_size=batch_size,
        policy_bound_loss_weight=policy_bound_loss_weight,
        clip_range=clip_range, 
        clip_range_vf=clip_range_vf, 
        n_steps=n_steps,
        normalize_advantage=normalize_advantage,
        target_kl=target_kl,
        gamma=gamma,
        n_epochs=n_epochs,
        gae_lambda=gae_lambda,
        max_policy_grad_norm=max_policy_grad_norm,
        max_value_grad_norm=max_value_grad_norm,
        fixed_std=fixed_std,
        total_n_steps=total_n_steps,
        log_std_lr=log_std_lr,
        policy_kwargs=policy_kwargs,
        min_log_std_lr=min_log_std_lr,
        verbose=verbose,
        device=device,
        _init_setup_model=False
         )
        
        self.rollout_buffer_class = MaskableRolloutBuffer if use_masking else RolloutBuffer
        self.rollout_buffer_kwargs = {}

        self.use_masking = use_masking
        if self.is_categorical or self.is_mixed:
            self.rollout_buffer_class = MaskableCategoricalRolloutBuffer if use_masking else CategoricalRolloutBuffer 
            self.rollout_buffer_kwargs['is_mixed'] = self.is_mixed

        self.play_info = {'active_player': 'player_0',
                          'rollout': 0,
                          'rollouts_player': rollouts_player,
                          'num_trees': 1}
        
        self.num_players = len(self.env.envs[0].env.env.env.possible_agents)
        
        if _init_setup_model:
            self.ppo_setup_model()


    def update_play_info(self):
        if self.play_info['rollout'] >= self.play_info['rollouts_player'] and self.num_players > 1:
            if self.play_info['active_player'] == 'player_0':
                self.play_info['active_player'] = 'player_1'
            else:
                self.play_info['active_player'] = 'player_0'
            self.play_info['rollout'] = 0
            num_trees = self.policy.get_num_trees()
            if isinstance(num_trees, tuple):
                num_trees, _ = num_trees
            self.play_info['num_trees'] = num_trees


    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        if isinstance(self.policy.action_dist, DiagGaussianDistribution):
            self._update_learning_rate(self.policy.log_std_optimizer)
        if self.policy.nn_critic:
            self._update_learning_rate(self.policy.value_optimizer)
            self.logger.record("train/nn_critic", "True")
        else: 
            self.logger.record("train/nn_critic", "False")
        policy_lr, value_lr = self.policy.get_schedule_learning_rates()
        self.logger.record("train/policy_learning_rate", policy_lr)
        self.logger.record("train/value_learning_rate", value_lr)
        self.update_play_info()

        active_player = self.play_info['active_player']

        entropy_losses = []
        policy_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        theta_maxs, theta_mins = [], []
        theta_grad_maxs, theta_grad_mins = [], []
        values_maxs, values_mins = [], []
        values_grad_maxs, values_grad_mins = [], []
        log_std_s = []
        approx_kl_divs = []
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size, active_player):
                actions = rollout_data.actions
                action_masks = None if not self.use_masking else rollout_data.action_masks

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions, action_masks=action_masks)
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = 0.5*F.mse_loss(rollout_data.returns, values_pred)
            
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                loss = policy_loss + self.ent_coef*entropy_loss + self.vf_coef*value_loss
                if self.policy.nn_critic:
                    self.policy.value_optimizer.zero_grad()
                loss.backward()
                if self.policy.nn_critic:
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                # Entropy loss favor exploration
                entropy_losses.append(entropy_loss.item())
                    # Logging
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())


                if isinstance(self.policy.action_dist, DiagGaussianDistribution) and not self.fixed_std:
                    if self.max_policy_grad_norm is not None and self.max_policy_grad_norm > 0.0:
                        th.nn.utils.clip_grad_norm(self.policy.log_std, max_norm=self.max_policy_grad_norm, error_if_nonfinite=True)
                    self.policy.log_std_optimizer.step()
                    log_std_grad = self.policy.log_std.grad.clone().detach().cpu().numpy()
                    self.policy.log_std_optimizer.zero_grad()
                    assert ~np.isnan(log_std_grad).any(), "nan in assigned grads"
                    assert ~np.isinf(log_std_grad).any(), "infinity in assigned grads"
                    log_std_s.append(self.policy.log_std.detach().cpu().numpy())

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and self.target_kl > 0 and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                # Fit GBRL model on gradients - Optimization step
                self.policy.step(rollout_data.observations, self.max_policy_grad_norm, self.max_value_grad_norm)
                _, grads = self.policy.model.get_params()
                if isinstance(grads, tuple):
                    theta_grad, values_grad = grads
                    theta, values = self.policy.model.params
                    values_grad_maxs.append(values_grad.max().item())
                    values_grad_mins.append(values_grad.min().item())
                else:
                    theta_grad = grads
                    theta = self.policy.model.params
                values_maxs.append(values.max().item())
                values_mins.append(values.min().item())
  
                theta_maxs.append(theta.max().item())
                theta_mins.append(theta.min().item())
                theta_grad_maxs.append(theta_grad.max().item())
                theta_grad_mins.append(theta_grad.min().item())

                self._n_updates += 1
                # Logging
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

            if not continue_training:
                break
        
        if len(theta_maxs) > 0:

            self.rollout_cntr += 1
            explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
            
            iteration = self.policy.get_iteration()
            num_trees = self.policy.get_num_trees()
            value_iteration = 0
            
            if isinstance(iteration, tuple):
                iteration, value_iteration = iteration
            value_num_trees = 0
            if isinstance(num_trees, tuple):
                num_trees, value_num_trees = num_trees

            self.logger.record("train/entropy_loss", np.mean(entropy_losses))
            self.logger.record("train/policy_gradient_loss", np.mean(policy_losses))
            self.logger.record("train/value_loss", np.mean(value_losses))
            self.logger.record("param/theta_max", np.mean(theta_maxs))
            self.logger.record("param/theta_min", np.mean(theta_mins))
            self.logger.record("param/value_max", np.mean(values_maxs))
            self.logger.record("param/value_min", np.mean(values_mins))
            self.logger.record("param/theta_grad_max", np.mean(theta_grad_maxs))
            self.logger.record("param/theta_grad_min", np.mean(theta_grad_mins))
            if values_grad_maxs:
                self.logger.record("param/value_grad_max", np.mean(values_grad_maxs))
                self.logger.record("param/value_grad_min", np.mean(values_grad_mins))
            self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
            self.logger.record("train/clip_fraction", np.mean(clip_fractions))
            self.logger.record("train/explained_variance", explained_var)
            self.logger.record("train/total_boosting_iterations", self.policy.get_total_iterations())
            self.logger.record("train/policy_boosting_iterations", iteration)
            self.logger.record("train/value_boosting_iteration", value_iteration)
            self.logger.record("train/policy_num_trees", num_trees)
            self.logger.record("time/total_timesteps", self.num_timesteps)
            self.logger.record("train/value_num_trees", value_num_trees)

            if log_std_s:
                self.logger.record("param/std", np.mean(np.mean(np.exp(np.concatenate(log_std_s, axis=0)), axis=0)))
                self.logger.record("param/log_std", np.mean(np.mean(np.concatenate(log_std_s, axis=0), axis=0)))
            if hasattr(self.policy, "log_std"):
                self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        self.play_info['rollout'] += 1
        
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: Union[RolloutBuffer, CategoricalRolloutBuffer, MaskableRolloutBuffer],
        n_rollout_steps: int,
        use_masking: bool =False,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :param use_masking: Whether or not to use invalid action masks during training
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """

        if use_masking:
            assert isinstance(
            rollout_buffer, (MaskableRolloutBuffer, 
                            MaskableCategoricalRolloutBuffer)
        ), "RolloutBuffer doesn't support action masking"
        rollout_buffer.reset()
            
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0

        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)
        
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)
            active_player = env.envs[0].agent_selection
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = self._last_obs if self.is_categorical else obs_as_tensor(self._last_obs, self.device)
                action_masks = get_action_masks(env) if use_masking else None
                stop_idx = self.play_info['num_trees'] if self.play_info['active_player'] !=  active_player else None
                actions, values, log_probs = self.policy(obs_tensor, requires_grad=False, action_masks=action_masks, stop_idx=stop_idx)

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False
            if active_player == self.play_info['active_player']:
                self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = infos[idx]["terminal_observation"] if self.is_categorical else self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs, stop_idx=stop_idx)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            kwargs = {'player': active_player}
            if use_masking:
                kwargs['action_masks'] = action_masks
            # print(f'step: {n_steps}, active_player: {active_player}')
            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                **kwargs
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        active_player = env.envs[0].agent_selection
        with th.no_grad():
            # Compute value for the last timestep
            stop_idx = self.play_info['num_trees'] if self.play_info['active_player'] !=  active_player else None
            values = self.policy.predict_values(new_obs, requires_grad=False, stop_idx=stop_idx)  # type: ignore[arg-type] 

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones, final_player=active_player)

        callback.on_rollout_end()

        return True

    
    def learn(
        self: "PPO_GBRL",
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "GBRL",
        reset_num_timesteps: bool = True,
        use_masking: bool = False,
        progress_bar: bool = False,
    ) -> "PPO_GBRL":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps, use_masking=use_masking)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self



class PPO_SelfPlay(MaskablePPO):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version) with Invalid Action Masking.

    Based on the original Stable Baselines 3 implementation.

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html
    Background on Invalid Action Masking: https://arxiv.org/abs/2006.14171

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[MaskableActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        rollouts_player: int = 5,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
        )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()
        self.num_players = len(self.env.envs[0].env.env.env.possible_agents)

        self.play_info = {'active_player': 'player_0',
                            'rollout': 0,
                            'rollouts_player': rollouts_player,
                           }

    def update_play_info(self):
        if self.play_info['rollout'] >= self.play_info['rollouts_player'] and self.num_players > 1:
            if self.play_info['active_player'] == 'player_0':
                self.play_info['active_player'] = 'player_1'
            else:
                self.play_info['active_player'] = 'player_0'
            self.play_info['rollout'] = 0
            
            self.passive_policy = copy.deepcopy(self.active_policy)


    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = MaskableRolloutBuffer

        self.active_policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.active_policy = self.active_policy.to(self.device)

        self.passive_policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.passive_policy = self.passive_policy.to(self.device)


        if not isinstance(self.active_policy, MaskableActorCriticPolicy):
            raise ValueError("Policy must subclass MaskableActorCriticPolicy")

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)


    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        This method is largely identical to the implementation found in the parent class.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :param use_masking: Whether or not to use invalid action masks during training
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """

        assert isinstance(
            rollout_buffer, (MaskableRolloutBuffer)
        ), "RolloutBuffer doesn't support action masking"
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.active_policy.set_training_mode(False)
        n_steps = 0
        action_masks = None
        rollout_buffer.reset()

        if use_masking and not is_masking_supported(env):
            raise ValueError("Environment does not support action masking. Consider using ActionMasker wrapper")

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                active_player = env.envs[0].agent_selection
                # This is the only change related to invalid action masking
                if use_masking:
                    action_masks = get_action_masks(env)
                policy = self.active_policy if active_player == self.play_info['active_player'] else self.passive_policy
                actions, values, log_probs = policy(obs_tensor, action_masks=action_masks)

            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            if active_player == self.play_info['active_player']:
                self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.active_policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.active_policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                player=active_player,
                action_masks=action_masks,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        active_player = env.envs[0].agent_selection
        policy = self.active_policy if active_player == self.play_info['active_player'] else self.passive_policy

        with th.no_grad():
            # Compute value for the last timestep
            # Masking is not needed here, the choice of action doesn't matter.
            # We only want the value of the current observation.
            values = policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones, final_player=active_player)

        callback.on_rollout_end()

        return True

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        return self.active_policy.predict(observation, state, episode_start, deterministic, action_masks=action_masks)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.active_policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.active_policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        self.update_play_info()

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size, player=self.play_info['active_player']):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.active_policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.active_policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.active_policy.parameters(), self.max_grad_norm)
                self.active_policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


