##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
"""
PPO-Lagrangian with GBRL backbone.

Unlike Split-RL which uses multi-objective labels to route separate reward/cost
gradient streams through the tree, PPO-LAG combines advantages into a single
Lagrangian objective *before* computing the policy gradient:

    A_combined = (A_reward - lambda * A_cost) / (1 + lambda)

The Lagrangian multiplier lambda is updated per rollout based on whether
the mean episodic cost exceeds the cost_limit.
"""
import io
import pathlib
import sys
import time
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import (recursive_getattr,
                                                recursive_setattr)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import (check_for_correct_spaces,
                                            explained_variance, get_linear_fn,
                                            get_schedule_fn, get_system_info,
                                            obs_as_tensor, safe_mean,
                                            update_learning_rate)
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.patch_gym import _convert_space
from torch.nn import functional as F

from buffers.rollout_buffer import CostRolloutBuffer, CostCategoricalRolloutBuffer
from policies.cost_actor_critic import CostActorCriticPolicyGBRL
from utils.io_util import load_from_zip_file, save_to_zip_file


class PPOLagGBRL(OnPolicyAlgorithm):
    """
    PPO-Lagrangian with a GBRL (Gradient Boosted RL) backbone.

    Combines reward and cost advantages via a Lagrangian multiplier into a
    single policy gradient, then fits gradient-boosted trees on that signal.
    The value and cost critics are separate tree heads (or NN critics).

    Key difference from Split-RL:
        Split-RL uses GBRL's multi-objective label mechanism to route separate
        reward and cost gradient streams through the tree.  PPO-LAG instead
        merges them into one stream via the Lagrangian penalty *before* the
        tree step, so no labels are needed.

    :param env: The environment to learn from
    :param clip_range: PPO clipping parameter
    :param clip_range_vf: Clipping for value function (None = no clip)
    :param policy: Policy class (defaults to CostActorCriticPolicyGBRL)
    :param normalize_advantage: Whether to normalize the combined advantage
    :param target_kl: KL divergence limit for early stopping
    :param max_policy_grad_norm: Gradient clipping for policy
    :param max_value_grad_norm: Gradient clipping for value critic
    :param max_cost_grad_norm: Gradient clipping for cost critic
    :param vf_coef: Value function loss coefficient
    :param cf_coef: Cost function loss coefficient
    :param ent_coef: Entropy loss coefficient
    :param batch_size: Minibatch size
    :param gamma: Discount factor
    :param gae_lambda: GAE lambda
    :param n_epochs: Number of PPO epochs per rollout
    :param n_steps: Rollout length per env
    :param total_n_steps: Total training timesteps (for LR schedules)
    :param learning_rate: Not directly used by GBRL (tree optimizers have own LR)
    :param policy_kwargs: Dict with tree_struct, tree_optimizer, etc.
    :param cost_limit: Per-episode cost threshold for the Lagrangian constraint
    :param lagrangian_multiplier_init: Initial value for the Lagrangian multiplier
    :param lambda_lr: Learning rate for the Lagrangian multiplier optimizer
    :param lambda_optimizer: Torch optimizer name for the multiplier
    :param lagrangian_upper_bound: Optional upper bound on the multiplier
    :param fixed_std: Whether to fix the log_std (no learning)
    :param log_std_lr: Learning rate for log_std parameter
    :param min_log_std_lr: Minimum log_std LR (for linear schedule)
    :param device: Device string
    :param seed: Random seed
    :param verbose: Verbosity
    :param tensorboard_log: TB log directory
    :param _init_setup_model: Whether to call setup immediately
    """

    def __init__(
        self,
        env: Union[GymEnv, str],
        clip_range: float = 0.2,
        clip_range_vf: float = None,
        policy: Type[BasePolicy] = CostActorCriticPolicyGBRL,
        normalize_advantage: bool = True,
        target_kl: float = None,
        max_policy_grad_norm: float = None,
        max_value_grad_norm: float = None,
        max_cost_grad_norm: float = None,
        vf_coef: float = 0.5,
        cf_coef: float = 0.5,
        ent_coef: float = 0.0,
        batch_size: int = 256,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        n_epochs: int = 4,
        n_steps: int = 512,
        total_n_steps: int = 1e6,
        learning_rate: float = 1e-3,
        policy_kwargs: Dict = {
            'shared_tree_struct': True,
            'tree_struct': {
                'max_depth': 4,
                'n_bins': 256,
                'min_data_in_leaf': 0,
                'par_th': 2,
            },
            'tree_optimizer': {
                'params': {
                    'control_variates': False,
                    'split_score_func': 'cosine',
                    'generator_type': "Quantile",
                    'feature_weights': None,
                },
                'policy_optimizer': {
                    'policy_algo': 'SGD',
                    'policy_lr': 5e-3,
                    'policy_shrinkage': 0,
                },
                'value_optimizer': {
                    'value_algo': 'SGD',
                    'value_lr': 9e-3,
                    'value_shrinkage': 0,
                },
                'cost_optimizer': {
                    'cost_algo': 'SGD',
                    'cost_lr': 9e-3,
                    'cost_shrinkage': 0,
                },
            },
        },
        cost_limit: float = 25.0,
        lagrangian_multiplier_init: float = 0.001,
        lambda_lr: float = 0.035,
        lambda_optimizer: str = 'Adam',
        lagrangian_upper_bound: Optional[float] = None,
        fixed_std: bool = False,
        log_std_lr: float = 3e-4,
        min_log_std_lr: float = 3e-4,
        policy_bound_loss_weight: float = None,
        device: str = "cpu",
        seed: Optional[int] = None,
        verbose: int = 1,
        tensorboard_log: str = None,
        _init_setup_model: bool = False,
    ):
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        if self.target_kl is not None and self.target_kl == 0.0:
            self.target_kl = None
        self.max_policy_grad_norm = max_policy_grad_norm
        self.max_value_grad_norm = max_value_grad_norm
        self.max_cost_grad_norm = max_cost_grad_norm
        self.batch_size = batch_size
        self.cf_coef = cf_coef

        # --- Lagrangian multiplier ---
        self.cost_limit = cost_limit
        self.lambda_lr = lambda_lr
        self.lagrangian_upper_bound = lagrangian_upper_bound

        init_value = max(lagrangian_multiplier_init, 0.0)
        self.lagrangian_multiplier = th.nn.Parameter(
            th.as_tensor(init_value), requires_grad=True,
        )
        self.lambda_range_projection = th.nn.ReLU()
        assert hasattr(th.optim, lambda_optimizer), \
            f'Optimizer={lambda_optimizer} not found in torch.'
        torch_opt = getattr(th.optim, lambda_optimizer)
        self.lambda_optimizer = torch_opt([self.lagrangian_multiplier], lr=lambda_lr)
        self.ep_cost_mean = 0.0

        # --- GBRL schedule computations ---
        num_envs = 1 if isinstance(env, str) else env.num_envs
        updates_per_rollout = ((n_steps * num_envs) / batch_size) * n_epochs
        num_rollouts = total_n_steps / (n_steps * num_envs)
        total_num_updates = updates_per_rollout * num_rollouts

        assert 'tree_optimizer' in policy_kwargs, \
            "tree_optimizer must be a dictionary within policy_kwargs"
        assert 'params' in policy_kwargs['tree_optimizer'], \
            "params must be a dictionary within policy_kwargs['tree_optimizer']"

        policy_kwargs['tree_optimizer']['policy_optimizer']['T'] = int(total_num_updates)
        policy_kwargs['tree_optimizer']['value_optimizer']['T'] = int(total_num_updates)
        policy_kwargs['tree_optimizer']['cost_optimizer']['T'] = int(total_num_updates)
        policy_kwargs['tree_optimizer']['device'] = device

        self.fixed_std = fixed_std
        is_categorical = (hasattr(env, 'is_mixed') and env.is_mixed) or (
            hasattr(env, 'is_categorical') and env.is_categorical)
        is_mixed = (hasattr(env, 'is_mixed') and env.is_mixed)
        if is_categorical:
            policy_kwargs['is_categorical'] = True

        if isinstance(log_std_lr, str):
            if 'lin_' in log_std_lr:
                log_std_lr = get_linear_fn(float(log_std_lr.replace('lin_', '')), min_log_std_lr, 1)
            else:
                log_std_lr = float(log_std_lr)
        policy_kwargs['log_std_schedule'] = get_schedule_fn(log_std_lr)

        super().__init__(
            policy=policy,
            env=env,
            seed=seed,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
            tensorboard_log=tensorboard_log,
            learning_rate=learning_rate,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            max_grad_norm=max_value_grad_norm,
            use_sde=False,
            sde_sample_freq=-1,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            _init_setup_model=False,
        )
        self.env = env
        self.is_categorical = is_categorical
        self.is_mixed = is_mixed

        self.log_data = None
        self.rollout_cntr = 0
        self.prev_timesteps = 0
        self.policy_bound_loss_weight = policy_bound_loss_weight

        if self.normalize_advantage:
            assert batch_size > 1, \
                "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or not self.normalize_advantage, \
                (f"`n_steps * n_envs` must be greater than 1. "
                 f"Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}")
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

        self.rollout_buffer_class = CostRolloutBuffer
        self.rollout_buffer_kwargs = {}
        if self.is_categorical or self.is_mixed:
            self.rollout_buffer_class = CostCategoricalRolloutBuffer
            self.rollout_buffer_kwargs['is_mixed'] = self.is_mixed

        if _init_setup_model:
            self.ppo_setup_model()

    def ppo_setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )

        self.policy = self.policy_class(
            self.observation_space, self.action_space, self.lr_schedule,
            use_sde=self.use_sde, **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, \
                    "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def get_action_bound_min(self):
        if isinstance(self.action_space, spaces.Box):
            bound_min = self.action_space.low
        else:
            bound_min = -np.inf * np.ones(1)
        return th.tensor(bound_min, device=self.device)

    def get_action_bound_max(self):
        if isinstance(self.action_space, spaces.Box):
            bound_max = self.action_space.high
        else:
            bound_max = np.inf * np.ones(1)
        return th.tensor(bound_max, device=self.device)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.

        Lagrangian multiplier is updated first, then a standard PPO pass
        is performed with the combined advantage.
        """
        self.policy.set_training_mode(True)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        if isinstance(self.policy.action_dist, DiagGaussianDistribution):
            update_learning_rate(
                self.policy.log_std_optimizer,
                self.policy.log_std_schedule(self._current_progress_remaining),
            )
        if self.policy.nn_critic:
            self._update_learning_rate(self.policy.value_optimizer)

        policy_lr, value_lr, cost_lr = self.policy.get_schedule_learning_rates(
            lr_schedule=self.lr_schedule,
            progress_remaining=self._current_progress_remaining,
        )
        self.logger.record("train/policy_learning_rate", policy_lr)
        self.logger.record("train/value_learning_rate", value_lr)
        self.logger.record("train/cost_learning_rate", cost_lr)

        # --- Update Lagrangian multiplier ---
        self.lambda_optimizer.zero_grad()
        lambda_loss = -self.lagrangian_multiplier * (self.ep_cost_mean - self.cost_limit)
        lambda_loss.backward()
        self.lambda_optimizer.step()
        self.lagrangian_multiplier.data.clamp_(0.0, self.lagrangian_upper_bound)

        penalty = self.lagrangian_multiplier.item()
        self.logger.record("train/lagrangian_multiplier", penalty)
        self.logger.record("train/cost_limit", self.cost_limit)

        entropy_losses = []
        policy_losses, value_losses, cost_losses = [], [], []
        clip_fractions = []

        continue_training = True
        theta_maxs, theta_mins = [], []
        theta_grad_maxs, theta_grad_mins = [], []
        values_maxs, values_mins = [], []
        values_grad_maxs, values_grad_mins = [], []
        cost_maxs, cost_mins = [], []
        cost_grad_maxs, cost_grad_mins = [], []
        log_std_s = []
        approx_kl_divs = []

        for _ in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                costs, values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions,
                )
                values = values.flatten() if values.ndim > 1 else values
                costs = costs.flatten() if costs.ndim > 1 else costs

                # --- Combine advantages via Lagrangian penalty ---
                advantages_reward = rollout_data.advantages
                if self.normalize_advantage and len(advantages_reward) > 1:
                    advantages_reward = (advantages_reward - advantages_reward.mean()) / (advantages_reward.std() + 1e-8)

                advantages_costs = rollout_data.advantages_costs
                if self.normalize_advantage and len(advantages_costs) > 1:
                    advantages_costs = advantages_costs - advantages_costs.mean()

                advantages = (advantages_reward - penalty * advantages_costs) / (1 + penalty)

                # --- PPO clipped surrogate loss ---
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # --- Value critic loss ---
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf,
                    )
                value_loss = 0.5 * F.mse_loss(rollout_data.returns, values_pred)

                # --- Cost critic loss ---
                cost_loss = 0.5 * F.mse_loss(rollout_data.cost_returns, costs)

                # --- Entropy loss ---
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                loss = (policy_loss
                        + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss
                        + self.cf_coef * cost_loss)

                if self.policy.nn_critic:
                    self.policy.value_optimizer.zero_grad()
                    self.policy.cost_optimizer.zero_grad()

                loss.backward()

                if self.policy.nn_critic:
                    if self.max_grad_norm is not None:
                        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                # Logging
                entropy_losses.append(entropy_loss.item())
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                cost_losses.append(cost_loss.item())

                if isinstance(self.policy.action_dist, DiagGaussianDistribution) and not self.fixed_std:
                    if self.max_policy_grad_norm is not None and self.max_policy_grad_norm > 0.0:
                        th.nn.utils.clip_grad_norm_(
                            self.policy.log_std, max_norm=self.max_policy_grad_norm,
                            error_if_nonfinite=True,
                        )
                    self.policy.log_std_optimizer.step()
                    log_std_grad = self.policy.log_std.grad.clone().detach().cpu().numpy()
                    self.policy.log_std_optimizer.zero_grad()
                    assert ~np.isnan(log_std_grad).any(), "nan in assigned grads"
                    assert ~np.isinf(log_std_grad).any(), "infinity in assigned grads"
                    log_std_s.append(self.policy.log_std.detach().cpu().numpy())

                # KL divergence for early stopping
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and self.target_kl > 0 and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                # --- GBRL tree fitting step ---
                self.policy.step(
                    policy_grad_clip=self.max_policy_grad_norm,
                    value_grad_clip=self.max_value_grad_norm,
                    cost_grad_clip=self.max_cost_grad_norm,
                )

                params = self.policy.get_params()
                theta = params if not isinstance(params, tuple) else params[0]

                theta_maxs.append(theta.max().item())
                theta_mins.append(theta.min().item())

                values_maxs.append(values.max().item())
                values_mins.append(values.min().item())
                cost_maxs.append(costs.max().item())
                cost_mins.append(costs.min().item())

                if not self.policy.nn_critic:
                    grads = self.policy.get_grads()
                    if isinstance(grads, tuple) and len(grads) == 3:
                        policy_grad, value_grad, cost_grad = grads
                    elif isinstance(grads, tuple) and len(grads) == 2:
                        policy_grad, value_grad = grads
                        cost_grad = getattr(self.policy.model, 'cost_grads', None)
                    else:
                        policy_grad = grads
                        value_grad = None
                        cost_grad = None

                    if policy_grad is not None:
                        theta_grad_maxs.append(policy_grad.max().item())
                        theta_grad_mins.append(policy_grad.min().item())
                    if value_grad is not None:
                        values_grad_maxs.append(value_grad.max().item())
                        values_grad_mins.append(value_grad.min().item())
                    if cost_grad is not None:
                        cost_grad_maxs.append(cost_grad.max().item())
                        cost_grad_mins.append(cost_grad.min().item())

                self._n_updates += 1
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

            if not continue_training:
                break

        if len(theta_maxs) > 0:
            self.rollout_cntr += 1
            explained_var = explained_variance(
                self.rollout_buffer.values.flatten(),
                self.rollout_buffer.returns.flatten(),
            )

            iteration = self.policy.get_iteration()
            num_trees = self.policy.get_num_trees()
            value_iteration = 0
            if isinstance(iteration, tuple):
                iteration, value_iteration = iteration
            value_num_trees = 0
            cost_num_trees = 0
            if isinstance(num_trees, tuple):
                num_trees, value_num_trees, cost_num_trees = num_trees

            self.logger.record("loss/entropy_loss", np.mean(entropy_losses))
            self.logger.record("loss/policy_gradient_loss", np.mean(policy_losses))
            self.logger.record("loss/value_loss", np.mean(value_losses))
            self.logger.record("loss/cost_loss", np.mean(cost_losses))
            self.logger.record("param/theta_max", np.mean(theta_maxs))
            self.logger.record("param/theta_min", np.mean(theta_mins))
            self.logger.record("param/value_max", np.mean(values_maxs))
            self.logger.record("param/value_min", np.mean(values_mins))
            self.logger.record("param/cost_max", np.mean(cost_maxs))
            self.logger.record("param/cost_min", np.mean(cost_mins))
            if theta_grad_maxs:
                self.logger.record("grad/theta_grad_max", np.mean(theta_grad_maxs))
                self.logger.record("grad/theta_grad_min", np.mean(theta_grad_mins))
            if values_grad_maxs:
                self.logger.record("grad/value_grad_max", np.mean(values_grad_maxs))
                self.logger.record("grad/value_grad_min", np.mean(values_grad_mins))
            if cost_grad_maxs:
                self.logger.record("grad/cost_grad_max", np.mean(cost_grad_maxs))
                self.logger.record("grad/cost_grad_min", np.mean(cost_grad_mins))
            self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
            self.logger.record("train/clip_fraction", np.mean(clip_fractions))
            self.logger.record("train/explained_variance", explained_var)
            self.logger.record("train/total_boosting_iterations", self.policy.get_total_iterations())
            self.logger.record("train/policy_boosting_iterations", iteration)
            self.logger.record("train/value_boosting_iteration", value_iteration)
            self.logger.record("train/policy_num_trees", num_trees)
            self.logger.record("train/value_num_trees", value_num_trees)
            self.logger.record("train/cost_num_trees", cost_num_trees)
            self.logger.record("time/total_timesteps", self.num_timesteps)

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
        rollout_buffer: CostRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = self._last_obs if self.is_categorical else obs_as_tensor(self._last_obs, self.device)
                actions, value_costs, values, log_probs = self.policy(obs_tensor, requires_grad=False)

            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = (
                        infos[idx]["terminal_observation"] if self.is_categorical
                        else self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    )
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            costs = th.tensor([info.get('cost', 0.0) for info in infos])

            rollout_buffer.add(
                obs=self._last_obs,
                action=actions,
                reward=rewards,
                episode_start=self._last_episode_starts,
                value=values,
                value_cost=value_costs,
                log_prob=log_probs,
                cost=costs,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(new_obs, requires_grad=False)
            value_costs = self.policy.predict_costs(new_obs, requires_grad=False)

        rollout_buffer.compute_returns_and_advantage(
            last_values=values, last_value_costs=value_costs, dones=dones,
        )

        callback.on_rollout_end()
        return True

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPOLagGBRL",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "PPOLagGBRL":
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
            continue_training = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps,
            )
            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Update ep_cost_mean from rollout buffer costs each iteration
            # (used by the Lagrangian update regardless of logging)
            self.ep_cost_mean = float(self.rollout_buffer.costs.sum() / max(
                np.sum(self.rollout_buffer.episode_starts), 1.0))

            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    # Use "c" from ep_info if available (safety-gym convention),
                    # otherwise fall back to rollout buffer estimate
                    if "c" in self.ep_info_buffer[0]:
                        self.ep_cost_mean = safe_mean([ep_info["c"] for ep_info in self.ep_info_buffer])
                    self.logger.record("rollout/ep_rew_mean",
                                       safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_cost_mean", self.ep_cost_mean)
                    self.logger.record("rollout/ep_len_mean",
                                       safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                    if "s" in self.ep_info_buffer[0]:
                        self.logger.record("rollout/ep_scalarization_mean",
                                           safe_mean([ep_info["s"] for ep_info in self.ep_info_buffer]))
                    from utils.helpers import log_ep_info_metrics
                    log_ep_info_metrics(self.logger, self.ep_info_buffer)
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()
        return self

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        print(f"saving model to: {path}")
        self.policy.model.save_learner(path.replace('.zip', ''))
        data = self.__dict__.copy()

        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            var_name = torch_var.split(".")[0]
            exclude.add(var_name)

        for param_name in exclude:
            data.pop(param_name, None)

        pytorch_variables = None
        params_to_save = {}
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        for name in state_dicts_names:
            attr = recursive_getattr(self, name)
            params_to_save[name] = attr.state_dict()

        data['gbrl'] = True
        data['nn_critic'] = self.policy.nn_critic
        data['shared_tree_struct'] = self.policy.shared_tree_struct

        save_to_zip_file(path, data=data, params=params_to_save, pytorch_variables=pytorch_variables)

    @classmethod
    def load(
        cls,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> "PPOLagGBRL":
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables, gbrl_model = load_from_zip_file(
            path, device=device, custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            data['policy_kwargs']['tree_optimizer']['device'] = device
            if "net_arch" in data["policy_kwargs"] and len(data["policy_kwargs"]["net_arch"]) > 0:
                saved_net_arch = data["policy_kwargs"]["net_arch"]
                if isinstance(saved_net_arch, list) and isinstance(saved_net_arch[0], dict):
                    data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])

        if env is not None:
            env = cls._wrap_env(env, data["verbose"])
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            if force_reset and data is not None:
                data["_last_obs"] = None
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            if "env" in data:
                env = data["env"]

        model = cls(
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,
        )

        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    f"(see https://github.com/DLR-RM/stable-baselines3/issues/1233). "
                    f"Original error: {e} \n"
                )
            else:
                raise e

        if pytorch_variables is not None:
            for name in pytorch_variables:
                if pytorch_variables[name] is None:
                    continue
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        if model.use_sde:
            model.policy.reset_noise()
        model.policy.model = gbrl_model
        return model

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = []
        if self.policy.nn_critic and self.policy.value_net is not None:
            state_dicts = ["policy", "policy.value_optimizer", "policy.cost_optimizer"]
            if self.policy.log_std_optimizer is not None:
                state_dicts.append("policy.log_std_optimizer")
        return state_dicts, []
