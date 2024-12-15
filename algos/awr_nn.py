##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
from typing import Any, Dict, List, NamedTuple, Optional, Type, Union

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import (GymEnv, Schedule, TrainFreq,
                                                   TrainFrequencyUnit)
from stable_baselines3.common.vec_env import VecNormalize
from torch.nn import functional as F

from buffers.replay_buffer import AWRReplayBuffer
from policies.awr_nn_policy import (AWRPolicy, CnnPolicy, MlpPolicy,
                                    MultiInputPolicy)

ACTIVATION = {'relu': nn.ReLU, 'tanh': nn.Tanh}
OPTIMIZER = {'adam': th.optim.Adam, 'sgd': th.optim.SGD}

class AWR(OffPolicyAlgorithm):

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: AWRPolicy
    def __init__(self,
                 policy: Union[str, Type[AWRPolicy]],
                env: Union[GymEnv, str],
                learning_rate: Union[float, Schedule] = 3e-4,
                train_freq: int = 2048,
                batch_size: int = 64,
                gamma: float = 0.99,
                beta: float = 0.05,
                gae_lambda: float = 0.95,
                ent_coef: float = 0.0,
                weights_max: float = 20,
                learning_starts: int = 10000,
                max_grad_norm: float = None,
                buffer_size: int = 100000,
                policy_gradient_steps: int=1000,
                reward_mode: str = 'gae',
                value_gradient_steps: int=250,
                policy_bound_loss_weight: float = 0,
                tensorboard_log: Optional[str] = None,
                policy_kwargs: Optional[Dict[str, Any]] = None,
                verbose: int = 1,
                value_batch_size: int = 512,
                normalize_advantage: bool = False,
                seed: Optional[int] = None,
                device: Union[th.device, str] = "auto", 
                _init_setup_model: bool = True,
                 ):

        # policy_kwargs['activation_fn'] = ACTIVATION[policy_kwargs['activation_fn']]
        
        # optimizer_class = policy_kwargs.get('optimizer_class', None)
        # if optimizer_class is not None:
        #     policy_kwargs['optimizer_class'] = OPTIMIZER[optimizer_class]

        self.beta = beta
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.value_batch_size = value_batch_size
        self.weights_max = weights_max
        self.value_gradient_steps = value_gradient_steps
        self.policy_gradient_steps = policy_gradient_steps
        self.normalize_advantages = normalize_advantage
        self.policy_bound_loss_weight = policy_bound_loss_weight
        self.max_grad_norm = max_grad_norm


        # assert policy in ['MlpPolicy', 'CnnPolicy', 'MultiInputPolicy']
        
        tr_freq = train_freq // env.num_envs
        # tr_freq = TrainFreq(n_steps // config.env.num_envs, TrainFrequencyUnit.EPISODE)
        super().__init__(policy=policy,
        env=env,
        replay_buffer_class=AWRReplayBuffer,
        support_multi_env=True,
        tensorboard_log=tensorboard_log,
        seed=seed,
        train_freq = tr_freq,
        verbose=verbose,
        device=device,
        learning_starts=learning_starts,
        batch_size=batch_size,
        buffer_size=buffer_size,
        gamma=gamma,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        replay_buffer_kwargs={'gae_lambda': gae_lambda, 'gamma': gamma, 'return_type': reward_mode, 'env': env},
        
        )

        super()._setup_model()

        self.bound_min = self.get_action_bound_min()
        self.bound_max = self.get_action_bound_max()
        self.epochs = 0

    def get_values(self, observations: np.ndarray, next_observations: np.ndarray) -> np.ndarray:
        n_samples, n_envs = observations.shape[0], observations.shape[1]
        values = np.zeros((n_samples, n_envs), dtype=np.float32)
        next_values = np.zeros((n_samples, n_envs), dtype=np.float32)
        for env_idx in range(n_envs):
            for i in range(0, n_samples, self.value_batch_size):
                batch_obs = self.replay_buffer.to_torch(observations[i:i+self.value_batch_size, env_idx])
                batch_next_obs = self.replay_buffer.to_torch(next_observations[i:i+self.value_batch_size, env_idx])
                torch_values = self.policy.critic(batch_obs)
                torch_next_values = self.policy.critic(batch_next_obs)
                
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
        # print('training')
        env = self.env if isinstance(self.env, VecNormalize) else None 
        observations = self.replay_buffer._normalize_obs(self.replay_buffer.observations, env)
        # observations = self.replay_buffer.to_torch(observations)
        next_observations = self.replay_buffer._normalize_obs(self.replay_buffer.next_observations, env)
        # next_observations = self.replay_buffer.to_torch(next_observations)

        observations = observations[:self.replay_buffer.valid_pos]
        next_observations = next_observations[:self.replay_buffer.valid_pos]

        values, next_values = self.get_values(observations, next_observations)
        self.replay_buffer.add_advantages_returns(values, next_values, env=self._vec_normalize_env)

        value_losses = []
        for gradient_step in range(self.value_gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            pred_values = self.policy.critic(replay_data.observations)

            value_loss = 0.5*F.mse_loss(pred_values, replay_data.returns)
            self.policy.critic.optimizer.zero_grad()
            value_loss.backward()
            if self.max_grad_norm and self.max_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)

            self.policy.critic.optimizer.step()
            value_losses.append(value_loss.item())


        policy_losses = []
        values, next_values = self.get_values(observations, next_observations)
        self.replay_buffer.add_advantages_returns(values, next_values, env=self._vec_normalize_env)
        self._n_updates += self.value_gradient_steps 

        for gradient_step in range(self.policy_gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            actions = replay_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            advantages = replay_data.advantages

            if self.normalize_advantages:
                adv_mean, adv_std = advantages.mean(), advantages.std() 
                advantages = (advantages - adv_mean) / (adv_std + 1e-5)

            weights = th.exp(advantages / self.beta)
            weights = th.clamp(weights, max=self.weights_max)

            log_prob = self.policy.actor.action_log_prob(replay_data.observations, actions)

            entropy = -th.mean(-log_prob)

            policy_loss = -(log_prob*weights).mean() + self.ent_coef*entropy

            if self.policy_bound_loss_weight and self.policy_bound_loss_weight > 0 and isinstance(self.action_space, spaces.Box):
                distrib = self.policy.actor.get_distribution(replay_data.observations)
                val = distrib.mode()
                vio_min = th.clamp(val - self.bound_min, max=0)
                vio_max = th.clamp(val - self.bound_max, min=0)
                violation = vio_min.pow_(2).sum(axis=-1) + vio_max.pow_(2).sum(axis=-1)
                policy_loss += 0.5 * th.mean(violation)

            self.policy.actor.optimizer.zero_grad()
            policy_loss.backward()
            if self.max_grad_norm and self.max_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            self.policy.actor.optimizer.step()

            policy_losses.append(policy_loss.item())

        self._n_updates += self.policy_gradient_steps
        self.epochs += 1

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/replay_buffer_pos", self.replay_buffer.pos)
        self.logger.record("train/replay_buffer_size", self.replay_buffer.buffer_size)
        self.logger.record("train/replay_buffer_full", self.replay_buffer.full)
        self.logger.record("train/policy_loss", np.mean(policy_losses))


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
    

