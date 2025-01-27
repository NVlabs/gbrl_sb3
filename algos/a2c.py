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
from typing import (Any, Dict, Iterable, Optional, TypeVar,
                    Union)

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import (GymEnv, MaybeCallback,
                                                   Schedule)
from stable_baselines3.common.utils import (explained_variance, get_linear_fn,
                                            obs_as_tensor, safe_mean, update_learning_rate, get_schedule_fn)
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as F

from buffers.rollout_buffer import CategoricalRolloutBuffer
from policies.actor_critic_policy import ActorCriticPolicy

SelfA2C = TypeVar("SelfA2C", bound="A2C_GBRL")


class A2C_GBRL(OnPolicyAlgorithm):
    """
    Advantage Actor Critic (A2C)

    Paper: https://arxiv.org/abs/1602.01783
    Code: This implementation borrows code from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (https://github.com/hill-a/stable-baselines)

    Introduction to A2C: https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param rms_prop_eps: RMSProp epsilon. It stabilizes square root computation in denominator
        of RMSProp update
    :param use_rms_prop: Whether to use RMSprop (default) or Adam as optimizer
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param normalize_advantage: Whether to normalize or not the advantage
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        env: Union[GymEnv, str],
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        learning_rate: float = 3e-4,
        normalize_advantage: bool = False,
        stats_window_size: int = 100,
        max_policy_grad_norm: float = None,
        max_value_grad_norm: float = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        fixed_std: bool = False,
        log_std_lr: float = 3e-4,
        min_log_std_lr: float = 3e-45,
        verbose: int = 0,
        seed: Optional[int] = None,
        total_n_steps: int = 1e6,
        device: str = "cpu",
        _init_setup_model: bool = True,
    ):

        self.normalize_advantage = normalize_advantage
        self.max_policy_grad_norm = max_policy_grad_norm
        self.max_value_grad_norm = max_value_grad_norm
        num_rollouts = total_n_steps / (n_steps  * env.num_envs)
        total_num_updates = num_rollouts
        assert 'tree_optimizer' in policy_kwargs, "tree_optimizer must be a dictionary within policy_kwargs"
        assert 'gbrl_params' in policy_kwargs['tree_optimizer'], "gbrl_params must be a dictionary within policy_kwargs['tree_optimizer]"
        policy_kwargs['tree_optimizer']['policy_optimizer']['T'] = int(total_num_updates)
        policy_kwargs['tree_optimizer']['value_optimizer']['T'] = int(total_num_updates)
        policy_kwargs['tree_optimizer']['device'] = device
        self.fixed_std = fixed_std
        is_categorical = (hasattr(env, 'is_mixed') and env.is_mixed) or (hasattr(env, 'is_categorical') and env.is_categorical) 
        is_mixed = (hasattr(env, 'is_mixed') and env.is_mixed)
        if is_categorical:
            policy_kwargs['is_categorical'] = True
        self.is_categorical = is_categorical
        self.is_mixed = is_mixed

        if isinstance(log_std_lr, str):
            if 'lin_' in log_std_lr:
                log_std_lr = get_linear_fn(float(log_std_lr.replace('lin_' ,'')), min_log_std_lr, 1) 
            else:
                log_std_lr = float(log_std_lr)
        policy_kwargs['log_std_schedule'] = get_schedule_fn(log_std_lr)
        super().__init__(
            ActorCriticPolicy,
            env,
            learning_rate=learning_rate, #does nothing for categorical output spaces
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=1.0,
            use_sde=False,
            sde_sample_freq=-1,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam

        self.rollout_buffer_class =RolloutBuffer
        self.rollout_buffer_kwargs = {}
        if is_categorical or is_mixed:
            self.rollout_buffer_class = CategoricalRolloutBuffer 
            self.rollout_buffer_kwargs['is_mixed'] = is_mixed
        
        if _init_setup_model:
            self.a2c_setup_model()
    
    def a2c_setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = self.rollout_buffer_class(  # type: ignore[assignment]
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )

        # pytype:disable=not-instantiable
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        # pytype:enable=not-instantiable
        self.policy = self.policy.to(self.device)



    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        if isinstance(self.policy.action_dist, DiagGaussianDistribution):
            update_learning_rate(self.policy.log_std_optimizer, self.policy.log_std_schedule(self._current_progress_remaining))

        policy_losses, value_losses, entropy_losses = [], [], []
        log_std_s = []
        theta_maxs, theta_mins = [], []
        theta_grad_maxs, theta_grad_mins = [], []
        values_maxs, values_mins = [], []
        values_grad_maxs, values_grad_mins = [], []
        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            # get derivable parameters
            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)

            # Normalize advantage (not present in the original implementation)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            policy_loss = -(advantages * log_prob).mean()

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values.flatten())
            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)
            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
            loss.backward()

            # Entropy loss favor exploration
            entropy_losses.append(entropy_loss.item())
            # Logging
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

            self.policy.step(policy_grad_clip=self.max_policy_grad_norm, value_grad_clip=self.max_value_grad_norm)
            params, grads = self.policy.model.get_params()
            theta_grad, values_grad = grads
            theta = params[0]
            if isinstance(self.policy.action_dist, DiagGaussianDistribution) and not self.fixed_std:
                if self.max_policy_grad_norm is not None and self.max_policy_grad_norm > 0.0:
                    th.nn.utils.clip_grad_norm(self.policy.log_std, max_norm=self.max_policy_grad_norm, error_if_nonfinite=True)
                self.policy.log_std_optimizer.step()
                log_std_grad = self.policy.log_std.grad.clone().detach().cpu().numpy()
                self.policy.log_std_optimizer.zero_grad()
                assert ~np.isnan(log_std_grad).any(), "nan in assigned grads"
                assert ~np.isinf(log_std_grad).any(), "infinity in assigned grads"
                log_std_s.append(self.policy.log_std.detach().cpu().numpy())

            values_maxs.append(values.max().item())
            values_mins.append(values.min().item())

            values_grad_maxs.append(values_grad.max().item())
            values_grad_mins.append(values_grad.min().item())

            theta_maxs.append(theta.max().item())
            theta_mins.append(theta.min().item())
            theta_grad_maxs.append(theta_grad.max().item())
            theta_grad_mins.append(theta_grad.min().item())

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        
        iteration = self.policy.model.get_iteration()
        num_trees = self.policy.model.get_num_trees()
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
        self.logger.record("param/value_grad_max", np.mean(values_grad_maxs))
        self.logger.record("param/value_grad_min", np.mean(values_grad_mins))
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
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: Union[RolloutBuffer, CategoricalRolloutBuffer],
        n_rollout_steps: int,
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
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = self._last_obs if self.is_categorical else obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)

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
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(new_obs, requires_grad=False)  # type: ignore[arg-type] if self.is_categorical else self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True
    
    def learn(
        self: SelfA2C,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "A2C_GBRL",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfA2C:
        
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
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

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
    
    def save(self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        self.policy.model.save_model(path)
