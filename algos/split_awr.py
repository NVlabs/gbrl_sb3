##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
"""Split-RL AWR GBRL: Multi-objective AWR with expert demonstrations.

Single unified replay buffer with per-transition labels:
  Label 0 = online data (AWR gradient)
  Labels 1..K = expert data (BC gradient)

Training loop:
  1. Sample a batch from the unified buffer (mix of online + expert)
  2. Compute AWR for ALL samples (advantages, weights, log-probs)
  3. Compute BC grads for ALL samples (predicted_probs - one_hot)
  4. Mask: label=0 keeps AWR grads, label=k keeps BC grads for expert k
  5. Pass (n_objs, n_samples, policy_dim+1) gradient tensor to GBRL multi-objective step
"""
import io
import pathlib
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium import spaces
from minigrid.core.constants import IDX_TO_COLOR, STATE_TO_IDX
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import (get_linear_fn, get_schedule_fn,
                                            update_learning_rate)
from stable_baselines3.common.vec_env import VecNormalize
from torch.nn import functional as F

from buffers.replay_buffer import SplitCategoricalAWRReplayBuffer
from env.safety.utils import IDX_TO_OBJECT
from gbrl.common.utils import categorical_dtype
from policies.actor_critic_policy import ActorCriticPolicy

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}

# Expert label mapping (consistent ordering)
EXPERT_LABEL_MAP = {
    'moveball': 1,
    'keydoor': 2,
    'boxkey': 3,
}


def raw_obs_to_categorical(image: np.ndarray, direction: int, mission: str = "") -> np.ndarray:
    """Convert raw MiniGrid observation (image + direction) to GBRL categorical format.

    Returns (51,) S128 array matching MiniGridCategoricalObservationWrapper output.
    """
    h, w = image.shape[0], image.shape[1]
    flattened_shape = h * w + 2  # image cells + direction + mission
    cat = np.empty(flattened_shape, dtype=categorical_dtype)
    for i in range(h):
        for j in range(w):
            obj_idx, color_idx, state_idx = image[i, j]
            category = (f"{IDX_TO_OBJECT[obj_idx]},"
                        f"{IDX_TO_COLOR[color_idx]},"
                        f"{IDX_TO_STATE[state_idx]}")
            cat[i * w + j] = category.encode('utf-8')
    cat[h * w] = str(direction).encode('utf-8')
    cat[h * w + 1] = mission.encode('utf-8') if mission else b""
    return cat


def load_expert_dataset(npz_path: str, label: int, mission: str = "",
                        max_transitions: int = 0) -> Dict[str, np.ndarray]:
    """Load expert .npz and convert to categorical observations.

    Args:
        npz_path: path to .npz file
        label: objective label (1, 2, 3, ...)
        mission: mission string for categorical encoding
        max_transitions: max transitions to load (0 = all)

    Returns dict with observations, next_observations, actions, rewards, dones, labels.
    """
    data = np.load(npz_path)
    images = data['observations_image']       # (N, 7, 7, 3)
    directions = data['observations_direction']  # (N,)
    actions = data['actions']                 # (N,)
    rewards = data['rewards']                 # (N,)
    ep_terminals = data['episode_terminals']  # (n_eps,) cumulative end indices

    n_total = len(actions)
    # Truncate to max_transitions if needed (at episode boundaries)
    if max_transitions > 0 and max_transitions < n_total:
        # Find last complete episode that fits
        valid_eps = ep_terminals[ep_terminals <= max_transitions]
        if len(valid_eps) > 0:
            n = int(valid_eps[-1])
            ep_terminals = ep_terminals[:len(valid_eps)]
        else:
            n = n_total
    else:
        n = n_total

    obs_cat = np.empty((n, images.shape[1] * images.shape[2] + 2), dtype=categorical_dtype)
    for i in range(n):
        obs_cat[i] = raw_obs_to_categorical(images[i], directions[i], mission)

    # Build next_observations (shift by 1 within episodes)
    next_obs_cat = np.empty_like(obs_cat)
    ep_starts = np.concatenate([[0], ep_terminals[:-1]])
    ep_ends = ep_terminals
    for ep_idx in range(len(ep_terminals)):
        start, end = int(ep_starts[ep_idx]), int(ep_ends[ep_idx])
        if end - start > 1:
            next_obs_cat[start:end - 1] = obs_cat[start + 1:end]
        next_obs_cat[end - 1] = obs_cat[end - 1]

    # Build dones
    dones = np.zeros(n, dtype=np.float32)
    for end_idx in ep_terminals:
        dones[int(end_idx) - 1] = 1.0

    return {
        'observations': obs_cat,
        'next_observations': next_obs_cat,
        'actions': actions[:n].reshape(-1, 1).astype(np.float32),
        'rewards': rewards[:n].reshape(-1, 1).astype(np.float32),
        'dones': dones.reshape(-1, 1),
        'labels': np.full(n, label, dtype=np.int32),
    }


class SPLIT_AWR_GBRL(OffPolicyAlgorithm):
    """Split-RL AWR with GBRL shared tree structure.

    Single replay buffer with labels. Expert data pre-filled (protected from
    overwrite). Online data fills the rest. Each training step samples a mixed
    batch, computes AWR grads for label=0, BC grads for labels 1..K, then
    passes the (n_objs, batch_size, policy_dim+1) gradient matrix to GBRL.
    """

    def __init__(self, env: Union[GymEnv, str],
                 train_freq: int = 2048,
                 beta: float = 1.0,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 ent_coef: float = 0.0,
                 weights_max: float = 20,
                 learning_rate: float = 3e-4,
                 gradient_steps: int = 1,
                 normalize_advantage: bool = True,
                 max_policy_grad_norm: float = None,
                 max_value_grad_norm: float = None,
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
                 reward_mode: str = 'gae',
                 device: str = 'cuda',
                 tensorboard_log: str = None,
                 expert_datasets: Optional[Dict[str, str]] = None,
                 expert_buffer_per_label: int = 0,
                 n_objs: int = 4,
                 _init_setup_model: bool = True,
                 ):
        assert policy_kwargs is not None, "Policy kwargs cannot be none!"
        self.shared_tree_struct = policy_kwargs.get("shared_tree_struct", True)
        assert self.shared_tree_struct, "SPLIT_AWR_GBRL only supports shared_tree_struct=True"

        if isinstance(log_std_lr, str):
            if 'lin_' in log_std_lr:
                log_std_lr = get_linear_fn(float(log_std_lr.replace('lin_', '')), min_log_std_lr, 1)
            else:
                log_std_lr = float(log_std_lr)

        is_categorical = (hasattr(env, 'is_mixed') and env.is_mixed) or (
            hasattr(env, 'is_categorical') and env.is_categorical)
        is_mixed = (hasattr(env, 'is_mixed') and env.is_mixed)
        self.is_categorical = is_categorical
        self.is_mixed = is_mixed
        if is_categorical:
            policy_kwargs['is_categorical'] = True

        self.beta = beta
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.weights_max = weights_max
        self.normalize_advantage = normalize_advantage
        self.max_policy_grad_norm = max_policy_grad_norm
        self.max_value_grad_norm = max_value_grad_norm
        self.fixed_std = fixed_std
        self.value_batch_size = value_batch_size
        self.n_objs = n_objs

        buffer_kwargs = {'gae_lambda': gae_lambda, 'gamma': gamma, 'return_type': reward_mode}
        if is_mixed:
            buffer_kwargs['is_mixed'] = True

        tr_freq = train_freq // env.num_envs
        policy_kwargs['tree_optimizer']['device'] = device
        policy_kwargs['log_std_schedule'] = get_schedule_fn(log_std_lr)

        # Set n_objs in tree params for multi-objective GBRL
        if 'tree_optimizer' in policy_kwargs and 'params' in policy_kwargs['tree_optimizer']:
            policy_kwargs['tree_optimizer']['params']['n_objs'] = n_objs

        super().__init__(policy=ActorCriticPolicy,
                         env=env,
                         replay_buffer_class=SplitCategoricalAWRReplayBuffer,
                         support_multi_env=True,
                         tensorboard_log=tensorboard_log,
                         seed=seed,
                         train_freq=tr_freq,
                         verbose=verbose,
                         device=device,
                         learning_starts=learning_starts,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         gradient_steps=gradient_steps,
                         buffer_size=buffer_size,
                         gamma=gamma,
                         policy_kwargs=policy_kwargs,
                         replay_buffer_kwargs=buffer_kwargs,
                         )

        super()._setup_model()

        self.bound_min = self._get_action_bound_min()
        self.bound_max = self._get_action_bound_max()
        self.epochs = 0

        # Pre-fill replay buffer with expert data
        self._expert_datasets_config = expert_datasets or {}
        self._expert_buffer_per_label = expert_buffer_per_label
        if self._expert_datasets_config:
            self._prefill_expert_data()

    def _prefill_expert_data(self):
        """Load expert datasets and pre-fill the replay buffer."""
        for name, path in self._expert_datasets_config.items():
            label = EXPERT_LABEL_MAP.get(name)
            if label is None:
                raise ValueError(f"Unknown expert '{name}'. Must be one of {list(EXPERT_LABEL_MAP.keys())}")

            print(f"Loading expert: {name} (label={label}) from {path}")
            expert = load_expert_dataset(
                path, label=label, max_transitions=self._expert_buffer_per_label)

            self.replay_buffer.prefill_expert(
                observations=expert['observations'],
                next_observations=expert['next_observations'],
                actions=expert['actions'],
                rewards=expert['rewards'],
                dones=expert['dones'],
                label=label,
            )
            n = len(expert['actions'])
            print(f"  Loaded {n} transitions (buffer pos now {self.replay_buffer.pos})")

        expert_total = self.replay_buffer._expert_boundary
        online_capacity = self.replay_buffer.buffer_size - expert_total
        print(f"Buffer: {expert_total} expert + {online_capacity} online capacity "
              f"(total {self.replay_buffer.buffer_size})")

    def get_values(self, observations: np.ndarray, next_observations: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray]:
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
                end = min(i + self.value_batch_size, n_samples)
                torch_values = self.policy.critic(observations[i:end, env_idx])
                torch_next_values = self.policy.critic(next_observations[i:end, env_idx])
                values[i:end, env_idx] = torch_values.detach().cpu().numpy().squeeze()
                next_values[i:end, env_idx] = torch_next_values.detach().cpu().numpy().squeeze()
        return values, next_values

    def _get_action_bound_min(self):
        if isinstance(self.action_space, spaces.Box):
            return th.tensor(self.action_space.low, device=self.device)
        return th.tensor(-np.inf * np.ones(1), device=self.device)

    def _get_action_bound_max(self):
        if isinstance(self.action_space, spaces.Box):
            return th.tensor(self.action_space.high, device=self.device)
        return th.tensor(np.inf * np.ones(1), device=self.device)

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """Train with multi-objective gradients: AWR for label=0, BC for labels 1..K.

        For each gradient step:
        1. Sample mixed batch from unified buffer (online + expert)
        2. AWR: forward pass → loss → backward → extract per-sample policy & value grads
        3. BC: forward pass → probs → analytical grad (probs - one_hot) for ALL samples
        4. Provide (n_objs, batch, policy_dim+1): obj 0 = AWR, objs 1..K = BC (same tensor)
        5. obj_labels routes each sample to pick gradient from the correct objective
        """
        self.policy.set_training_mode(True)

        if isinstance(self.policy.action_dist, DiagGaussianDistribution):
            update_learning_rate(self.policy.log_std_optimizer, self.policy.log_std_schedule(
                self._current_progress_remaining))

        # Compute advantages/returns for entire buffer
        observations = self.replay_buffer._normalize_obs(
            self.replay_buffer.observations, self._vec_normalize_env)
        next_observations = self.replay_buffer._normalize_obs(
            self.replay_buffer.next_observations, self._vec_normalize_env)
        observations = observations[:self.replay_buffer.valid_pos]
        next_observations = next_observations[:self.replay_buffer.valid_pos]
        values, next_values = self.get_values(observations, next_observations)
        if self.replay_buffer.return_type == 'gae':
            self.replay_buffer.add_advantages_returns(
                values, next_values, env=self._vec_normalize_env)

        policy_losses, value_losses, entropy_losses, bc_losses = [], [], [], []
        weights_max_log, weights_min_log = [], []

        for _ in range(gradient_steps):
            # 1. Sample mixed batch
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            actions = replay_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.long().flatten()
            labels = replay_data.labels  # numpy int32 (batch,)

            # 2. AWR: forward pass for ALL samples → backward → extract grads
            values_pred, log_prob, entropy = self.policy.evaluate_actions(
                replay_data.observations, actions)

            advantages = replay_data.advantages
            if self.normalize_advantage and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            weights = th.clamp(th.exp(advantages / self.beta), max=self.weights_max)

            if entropy is None:
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            policy_loss = -(log_prob * weights).mean()
            value_loss = 0.5 * F.mse_loss(values_pred, replay_data.returns)
            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
            loss.backward()

            # Extract per-sample AWR grads (obj 0)
            policy_grad, value_grad = self.policy.model.extract_grads()
            n_samples = len(replay_data.observations)
            policy_grad = policy_grad * n_samples  # (batch, policy_dim)
            value_grad = value_grad * n_samples    # (batch, 1)

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())
            weights_max_log.append(weights.max().item())
            weights_min_log.append(weights.min().item())

            # 3. BC: analytical grad for ALL samples (objs 1..K)
            dist = self.policy.get_distribution(obs=replay_data.observations)
            probs = dist.distribution.probs  # (batch, n_actions)
            one_hot = th.zeros_like(probs)
            one_hot.scatter_(1, actions.unsqueeze(1), 1.0)
            bc_grads = probs - one_hot  # (batch, policy_dim)

            # Log BC loss for expert samples only
            expert_mask_any = labels > 0
            if expert_mask_any.any():
                with th.no_grad():
                    expert_logits = dist.distribution.logits[expert_mask_any]
                    expert_acts = actions[expert_mask_any]
                    bc_loss_val = F.cross_entropy(expert_logits, expert_acts)
                    bc_losses.append(bc_loss_val.item())

            # 4. Build gradient tuple: obj 0 = AWR, objs 1..K = BC (same tensor)
            #    obj_labels routes each sample to the correct objective's gradient
            policy_grads = (policy_grad,) + (bc_grads,) * (self.n_objs - 1)

            # 5. Multi-objective GBRL step
            self.policy.step(
                policy_grad_clip=self.max_policy_grad_norm,
                value_grad_clip=self.max_value_grad_norm,
                policy_grads=policy_grads,
                value_grads=value_grad,
                observations=replay_data.observations,
                obj_labels=labels,
            )

            # Update log_std if continuous actions
            if isinstance(self.policy.action_dist, DiagGaussianDistribution) and not self.fixed_std:
                if self.max_policy_grad_norm is not None and self.max_policy_grad_norm > 0.0:
                    th.nn.utils.clip_grad_norm_(self.policy.log_std,
                                                max_norm=self.max_policy_grad_norm,
                                                error_if_nonfinite=True)
                self.policy.log_std_optimizer.step()
                self.policy.log_std_optimizer.zero_grad()

        self._n_updates += gradient_steps
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
        self.logger.record("train/policy_loss", np.mean(policy_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        if bc_losses:
            self.logger.record("train/expert_bc_loss", np.mean(bc_losses))
        self.logger.record("train/replay_buffer_pos", self.replay_buffer.pos)
        self.logger.record("train/replay_buffer_full", self.replay_buffer.full)
        self.logger.record("train/policy_boosting_iterations", iteration)
        self.logger.record("train/value_boosting_iteration", value_iteration)
        self.logger.record("train/policy_num_trees", num_trees)
        self.logger.record("train/value_num_trees", value_num_trees)
        self.logger.record("param/weights_max", np.mean(weights_max_log))
        self.logger.record("param/weights_min", np.mean(weights_min_log))
        self.logger.record("time/total_timesteps", self.num_timesteps)

    def dump_logs(self) -> None:
        from utils.helpers import log_ep_info_metrics
        super().dump_logs()
        log_ep_info_metrics(self.logger, self.ep_info_buffer)

    def learn(self, total_timesteps: int, callback=None, log_interval: int = 100,
              tb_log_name: str = "", reset_num_timesteps: bool = True,
              progress_bar: bool = False):
        return super().learn(
            total_timesteps=total_timesteps, callback=callback,
            log_interval=log_interval, tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps, progress_bar=progress_bar)

    def save(self, path: Union[str, pathlib.Path, io.BufferedIOBase],
             exclude: Optional[Iterable[str]] = None,
             include: Optional[Iterable[str]] = None) -> None:
        self.policy.model.save_learner(path)
