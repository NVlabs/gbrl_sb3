import io
import pathlib
import sys
import time
import warnings
from typing import Any, Dict, Iterable, List, Optional, Type, Union

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import (explained_variance, get_linear_fn,
                                            get_schedule_fn, obs_as_tensor,
                                            safe_mean)
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as F

from buffers.rollout_buffer import CategoricalRolloutBuffer
from policies.actor_critic_policy import ActorCriticPolicy


class PPO_GBRL(OnPolicyAlgorithm):
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
                 is_categorical: bool = False, 
                 max_policy_grad_norm: float = None,
                 max_value_grad_norm: float = None,
                 vf_coef: float = 0.5,
                 ent_coef: float = 0.0,
                 normalize_policy_grads: bool = False,
                 batch_size: int = 256,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 n_epochs: int = 4,
                 n_steps: int = 512,
                 total_n_steps: int = 1e6,
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
                 tensorboard_log: str = None):
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        if self.target_kl is not None and self.target_kl == 0.0:
            self.target_kl = None
        self.max_policy_grad_norm = max_policy_grad_norm
        self.max_value_grad_norm = max_value_grad_norm
        self.normalize_policy_grads =normalize_policy_grads
        self.batch_size = batch_size
        num_envs = 1 if isinstance(env, str) else env.num_envs
        updates_per_rollout =  ((n_steps * num_envs) / batch_size) * n_epochs
        num_rollouts = total_n_steps / (n_steps  * env.num_envs)
        total_num_updates = updates_per_rollout*num_rollouts
        assert 'tree_optimizer' in policy_kwargs, "tree_optimizer must be a dictionary within policy_kwargs"
        assert 'gbrl_params' in policy_kwargs['tree_optimizer'], "gbrl_params must be a dictionary within policy_kwargs['tree_optimizer]"
        policy_kwargs['tree_optimizer']['gbrl_params']['T'] = int(total_num_updates)
        policy_kwargs['tree_optimizer']['device'] = device
        self.fixed_std = fixed_std

        if is_categorical:
            policy_kwargs['is_categorical'] = True

        if isinstance(log_std_lr, str):
            if 'lin_' in log_std_lr:
                log_std_lr = get_linear_fn(float(log_std_lr.replace('lin_' ,'')), min_log_std_lr, 1) 
            else:
                log_std_lr = float(log_std_lr)
        super().__init__(policy=policy,
        env=env,
        seed=seed,
        supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        tensorboard_log=tensorboard_log,
        learning_rate=log_std_lr, # not used
        vf_coef=vf_coef, # not used
        ent_coef=ent_coef,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        max_grad_norm=1.0, # not relevant,
        use_sde=False,
        sde_sample_freq=-1,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        device='cpu',
        _init_setup_model=not is_categorical,
         )
        self.env = env
        self.is_categorical = is_categorical

        self.log_data = None
        self.rollout_cntr = 0
        self.prev_timesteps = 0
        self.policy_bound_loss_weight = policy_bound_loss_weight

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if self.normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not self.normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )

        self.n_epochs = n_epochs

        self.bound_min = self.get_action_bound_min()
        self.bound_max = self.get_action_bound_max()

        if is_categorical:
            self._categorical_setup_model()
    
    def _categorical_setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = CategoricalRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.env.num_envs,
        )
        # pytype:disable=not-instantiable
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        # pytype:enable=not-instantiable
        self.policy = self.policy.to(self.device)
        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

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

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def load(self, load_name):
        self.policy.model.load_model(load_name)

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

        policy_lr, value_lr = self.policy.get_schedule_learning_rates()
        self.logger.record("train/policy_learning_rate", policy_lr)
        self.logger.record("train/vf_learning_rate", value_lr)
        if isinstance(self.policy.action_dist, DiagGaussianDistribution):
            self._update_learning_rate(self.policy.log_std_optimizer)

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
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
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
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
            
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                loss = policy_loss + self.ent_coef*entropy_loss + self.vf_coef*value_loss

                loss.backward()
  
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
                grads = self.policy.step(rollout_data.observations, self.max_policy_grad_norm, self.max_value_grad_norm)
                theta_grad, values_grad = grads
                theta, values = self.policy.model.params
                values_maxs.append(values.max().item())
                values_mins.append(values.min().item())
                values_grad_maxs.append(values_grad.max().item())
                values_grad_mins.append(values_grad.min().item())
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
                actions, values, log_probs = self.policy(obs_tensor, requires_grad=False)

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
        self: "PPO_GBRL",
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "GBRL",
        reset_num_timesteps: bool = True,
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
