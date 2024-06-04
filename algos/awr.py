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
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Type, Union

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import (GymEnv, TrainFreq,
                                                   TrainFrequencyUnit)
from stable_baselines3.common.utils import (explained_variance, get_linear_fn,
                                            get_schedule_fn, safe_mean)
from stable_baselines3.common.vec_env import VecNormalize
from torch.nn import functional as F

from buffers.replay_buffer import AWRReplayBuffer, CategoricalAWRReplayBuffer
from policies.actor_critic_policy import ActorCriticPolicy


class AWR_GBRL(OffPolicyAlgorithm):
    def __init__(self, env: Union[GymEnv, str],
                 train_freq: int = 2048,
                 beta: float = 1.0,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 ent_coef: float = 0.0,
                 weights_max: float = 20,
                 policy_gradient_steps: int = 1000,
                 value_gradient_steps: int = 200,
                 gradient_steps: int = 1,
                 normalize_advantage: bool = True,
                 policy_bound_loss_weight: float = None,
                 max_policy_grad_norm: float = None,
                 max_value_grad_norm: float = None,
                 normalize_policy_grads: bool = False,
                 batch_size: int = 256,
                 buffer_size: int = 1000000,
                 vf_coef: float = 0.56,
                 learning_starts: int = 100,
                 fixed_std: bool = False,
                 log_std_lr: float = 3e-4,
                 value_batch_size: int = 8192,
                 min_log_std_lr: float = 3e-45,
                 policy_kwargs: Dict = None,
                 seed: int = 0,
                 verbose: int = 1,
                 is_categorical: bool = False,
                 reward_mode: str = 'gae',
                 device: str = 'cpu',
                 tensorboard_log: str = None):

        assert policy_kwargs is not None, "Policy kwargs cannot be none!"
        self.shared_tree_struct = policy_kwargs.get("shared_tree_struct", True)
        if isinstance(log_std_lr, str):
            if 'lin_' in log_std_lr:
                log_std_lr = get_linear_fn(float(log_std_lr.replace('lin_' ,'')), min_log_std_lr, 1) 
            else:
                log_std_lr = float(log_std_lr)
        if is_categorical:
            policy_kwargs['is_categorical'] = True
        print("is_Categorical: ", is_categorical)
        self.beta = beta
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.weights_max = weights_max
        self.value_gradient_steps = value_gradient_steps
        self.policy_gradient_steps = policy_gradient_steps
        self.normalize_advantage = normalize_advantage
        self.policy_bound_loss_weight = policy_bound_loss_weight
        self.max_policy_grad_norm = max_policy_grad_norm
        self.max_value_grad_norm = max_value_grad_norm
        self.normalize_policy_grads = normalize_policy_grads
        self.fixed_std = fixed_std
        self.value_batch_size = value_batch_size
        buffer_kwargs = {'gae_lambda': gae_lambda, 'gamma': gamma, 'return_type': reward_mode}
        if not is_categorical:
            buffer_kwargs['env'] = env
        
        tr_freq = train_freq // env.num_envs
        policy_kwargs['tree_optimizer']['device'] = device
        super().__init__(policy=ActorCriticPolicy,
        env=env,
        replay_buffer_class=CategoricalAWRReplayBuffer if is_categorical else AWRReplayBuffer,
        support_multi_env=True,
        tensorboard_log=tensorboard_log,
        seed=seed,
        train_freq = tr_freq,
        verbose=verbose,
        device='cpu',
        learning_starts=learning_starts,
        learning_rate=log_std_lr,
        batch_size=batch_size,
        gradient_steps=gradient_steps,
        buffer_size=buffer_size,
        gamma=gamma,
        policy_kwargs=policy_kwargs,
        replay_buffer_kwargs=buffer_kwargs,
        )

        super()._setup_model()

        self.bound_min = self.get_action_bound_min()
        self.bound_max = self.get_action_bound_max()
        self.epochs = 0


    def get_values(self, observations: np.ndarray, next_observations: np.ndarray) -> np.ndarray:
        n_samples, n_envs = observations.shape[0], observations.shape[1]
        values = np.zeros((n_samples, n_envs), dtype=np.float32)
        next_values = np.zeros((n_samples, n_envs), dtype=np.float32)
        if self.value_batch_size >= n_samples:
            for env_idx in range(n_envs):
                torch_values = self.policy.critic(observations[:, env_idx])
                torch_next_values = self.policy.critic(next_observations[:, env_idx])
                values[:, env_idx] = torch_values.detach().cpu().numpy().squeeze()
                next_values[:, env_idx] = torch_next_values.detach().cpu().numpy().squeeze()
            return values, next_values
            
        for env_idx in range(n_envs):
            for i in range(0, n_samples, self.value_batch_size):
                torch_values = self.policy.critic(observations[i:i+self.value_batch_size, env_idx])
                torch_next_values = self.policy.critic(next_observations[i:i+self.value_batch_size, env_idx])
                
                values[i:i+self.value_batch_size, env_idx] = torch_values.detach().cpu().numpy().squeeze()
                next_values[i:i+self.value_batch_size, env_idx] = torch_next_values.detach().cpu().numpy().squeeze()
        return values, next_values

    def get_action_bound_min(self):
        if (isinstance(self.action_space, spaces.Box)):
            bound_min = self.action_space.low
        else:
            bound_min = -np.inf * np.ones(1)
        return th.tensor(bound_min, device=self.device)

    def get_action_bound_max(self):
        if (isinstance(self.action_space, spaces.Box)):
            bound_max = self.action_space.high
        else:
            bound_max = np.inf * np.ones(1)
        return th.tensor(bound_max, device=self.device)
  
    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Sample the replay buffer and do the updates
        (gradient descent and update target networks)
        """
         # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        if self.shared_tree_struct:
            policy_losses, value_losses, entropy_losses = self.shared_models_train(gradient_steps, batch_size)
        else: 
            policy_losses, value_losses, entropy_losses = self.separate_models_train(batch_size)

        self.epochs += 1
        iteration = self.policy.model.get_iteration()
        num_trees = self.policy.model.get_num_trees()
        value_iteration = 0
        
        if isinstance(iteration, tuple):
            iteration, value_iteration = iteration
        value_num_trees = 0
        if isinstance(num_trees, tuple):
            num_trees, value_num_trees = num_trees

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/value_loss", np.mean(value_losses))

        self.logger.record("train/replay_buffer_pos", self.replay_buffer.pos)
        self.logger.record("train/replay_buffer_size", self.replay_buffer.buffer_size)
        self.logger.record("train/replay_buffer_full", self.replay_buffer.full)
        self.logger.record("train/policy_loss", np.mean(policy_losses))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_boosting_iterations", iteration)
        self.logger.record("train/value_boosting_iteration", value_iteration)
        self.logger.record("train/policy_num_trees", num_trees)
        self.logger.record("time/total_timesteps", self.num_timesteps)
        self.logger.record("train/value_num_trees", value_num_trees)
    
    def shared_models_train(self, gradient_steps: int, batch_size: int) -> None:
        observations = self.replay_buffer._normalize_obs(self.replay_buffer.observations, self._vec_normalize_env)
        next_observations = self.replay_buffer._normalize_obs(self.replay_buffer.next_observations, self._vec_normalize_env)
        
        observations = observations[:self.replay_buffer.valid_pos]
        next_observations = next_observations[:self.replay_buffer.valid_pos]
        values, next_values = self.get_values(observations, next_observations)
        if self.replay_buffer.return_type == 'gae':
            self.replay_buffer.add_advantages_returns(values, next_values, env=self._vec_normalize_env)
        value_losses, policy_losses, entropy_losses = [], [], []
        theta_maxs, theta_mins, log_stds = [], [], []
        theta_grad_maxs, theta_grad_mins = [], []
        values_maxs, values_mins = [], []
        values_grad_maxs, values_grad_mins = [], []
        violations = []
        weights_max, weights_min = [], []

        for _ in range(gradient_steps):
            # replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            actions = replay_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()
            # get derivable parameters
            values, log_prob, entropy = self.policy.evaluate_actions(replay_data.observations, actions)
            # Normalize advantage
            advantages = replay_data.advantages
            if self.normalize_advantage and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            weights = th.exp(advantages / self.beta)
            weights = th.clamp(weights, max=self.weights_max)

            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            policy_loss = -(log_prob*weights).mean()
            value_loss = 0.5*F.mse_loss(values, replay_data.returns)

            if self.policy_bound_loss_weight is not None and self.policy_bound_loss_weight > 0 and isinstance(self.action_space, spaces.Box):
                val = self.model.distribution.mode()
                vio_min = th.clamp(val - self.bound_min, max=0)
                vio_max = th.clamp(val - self.bound_max, min=0)
                violation = vio_min.pow_(2).sum(axis=-1) + vio_max.pow_(2).sum(axis=-1)
                violation = 0.5 * self.policy_bound_loss_weight * violation.mean*()
                policy_loss += violation
                violations.append(violation.item())
            
            loss = (policy_loss + self.ent_coef*entropy_loss + self.vf_coef*value_loss)
            loss.backward()
    
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())

            grads = self.policy.step(replay_data.observations, self.max_policy_grad_norm, self.max_value_grad_norm)
            params = self.policy.model.params
            if isinstance(self.policy.action_dist, DiagGaussianDistribution) and not self.fixed_std:
                if self.max_policy_grad_norm is not None and self.max_policy_grad_norm > 0.0:
                    th.nn.utils.clip_grad_norm(self.policy.log_std, max_norm=self.max_policy_grad_norm, error_if_nonfinite=True)
                self.policy.log_std_optimizer.step()
                log_std_grad = self.policy.log_std.grad.clone().detach().cpu().numpy()
                self.policy.log_std_optimizer.zero_grad()
                assert ~np.isnan(log_std_grad).any(), "nan in assigned grads"
                assert ~np.isinf(log_std_grad).any(), "infinity in assigned grads"
                log_stds.append(self.policy.log_std.detach().cpu().numpy())
            
            theta_grad, values_grad = grads
            theta = params[0]
            values_maxs.append(values.max().item())
            values_mins.append(values.min().item())
            values_grad_maxs.append(values_grad.max().item())
            values_grad_mins.append(values_grad.min().item())
            theta_maxs.append(theta.max().item())
            theta_mins.append(theta.min().item())
            theta_grad_maxs.append(theta_grad.max().item())
            theta_grad_mins.append(theta_grad.min().item())
            weights_max.append(weights.max().item())
            weights_min.append(weights.min().item())

        self._n_updates += gradient_steps
        self.logger.record("param/theta_max", np.mean(theta_maxs))
        self.logger.record("param/theta_min", np.mean(theta_mins))
        self.logger.record("param/weights_max", np.mean(weights_max))
        self.logger.record("param/weights_min", np.mean(weights_min))
        self.logger.record("param/value_max", np.mean(values_maxs))
        self.logger.record("param/value_min", np.mean(values_mins))
        self.logger.record("param/theta_grad_max", np.mean(theta_grad_maxs))
        self.logger.record("param/theta_grad_min", np.mean(theta_grad_mins))
        self.logger.record("param/value_grad_max", np.mean(values_grad_maxs))
        self.logger.record("param/value_grad_min", np.mean(values_grad_mins))
        if log_stds:
            self.logger.record("param/std", np.mean(np.mean(np.exp(np.concatenate(log_stds, axis=0)), axis=0)))
            self.logger.record("param/log_std", np.mean(np.mean(np.concatenate(log_stds, axis=0), axis=0)))
            if hasattr(self.policy, "log_std"):
                self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        return policy_losses, value_losses, entropy_losses
        

    def separate_models_train(self, batch_size: int) -> None:
        env = self.env if isinstance(self.env, VecNormalize) else None 
        observations = self.replay_buffer._normalize_obs(self.replay_buffer.observations, env)
        next_observations = self.replay_buffer._normalize_obs(self.replay_buffer.next_observations, env)

        observations = observations[:self.replay_buffer.valid_pos]
        next_observations = next_observations[:self.replay_buffer.valid_pos]

        values, next_values = self.get_values(observations, next_observations)
        if self.replay_buffer.return_type == 'gae':
            self.replay_buffer.add_advantages_returns(values, next_values, env=self._vec_normalize_env)

        value_losses = []
        for _ in range(self.value_gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            values = self.policy.critic(replay_data.observations)

            value_loss = 0.5*F.mse_loss(values, replay_data.returns)
            value_loss.backward()

            value_losses.append(value_loss.item())
            self.policy.critic_step(replay_data.observations, self.max_value_grad_norm )

        policy_losses, entropy_losses = [], []
        values, next_values = self.get_values(observations, next_observations)
        if self.replay_buffer.return_type == 'gae':
            self.replay_buffer.add_advantages_returns(values, next_values, env=self._vec_normalize_env)
        self._n_updates += self.value_gradient_steps 

        for _ in range(self.policy_gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            actions = replay_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()
            advantages = replay_data.advantages

            if self.normalize_advantage:
                adv_mean, adv_std = advantages.mean(), advantages.std() 
                advantages = (advantages - adv_mean) / (adv_std + 1e-5)

            weights = th.exp(advantages / self.beta)
            weights = th.clamp(weights, max=self.weights_max)

            values, log_prob, entropy = self.policy.evaluate_actions(replay_data.observations, actions)

            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            policy_loss = -(log_prob*weights).mean() +  self.ent_coef*entropy_loss 

            if self.policy_bound_loss_weight is not None and self.policy_bound_loss_weight > 0 and isinstance(self.action_space, spaces.Box):
                val = self.model.distribution.mode()
                vio_min = th.clamp(val - self.bound_min, max=0)
                vio_max = th.clamp(val - self.bound_max, min=0)
                violation = vio_min.pow_(2).sum(axis=-1) + vio_max.pow_(2).sum(axis=-1)
                violation = 0.5 * self.policy_bound_loss_weight * violation.mean*()
                policy_loss += violation
            
            policy_loss.backward()


            self.policy.actor_step(replay_data.observations, self.max_policy_grad_norm)    
            policy_losses.append(policy_loss.item())
            entropy_losses.append(entropy_loss.item())

            if isinstance(self.policy.action_dist, DiagGaussianDistribution) and not self.fixed_std:
                if self.max_policy_grad_norm is not None and self.max_policy_grad_norm > 0.0:
                    th.nn.utils.clip_grad_norm(self.policy.log_std, max_norm=self.max_policy_grad_norm, error_if_nonfinite=True)
                self.policy.log_std_optimizer.step()
                log_std_grad = self.policy.log_std.grad.clone().detach().cpu().numpy()
                self.policy.log_std_optimizer.zero_grad()
                assert ~np.isnan(log_std_grad).any(), "nan in assigned grads"
                assert ~np.isinf(log_std_grad).any(), "infinity in assigned grads"
                # log_std_s.append(self.policy.log_std.detach().cpu().numpy())

        self._n_updates += self.policy_gradient_steps
        return policy_losses, value_losses, entropy_losses
        


    def learn(
        self,
        total_timesteps: int,
        callback= None,
        log_interval: int = 100,
        tb_log_name: str = "",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar
        )
    
    def save(self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        self.policy.model.save_model(path)
    
