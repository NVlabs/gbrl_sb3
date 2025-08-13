##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch as th
from gbrl.models.actor import ParametricActor
from gbrl.models.actor_critic import ActorCritic
from gymnasium import spaces
from sb3_contrib.common.maskable.distributions import (
    MaskableCategoricalDistribution, make_masked_proba_distribution)
from stable_baselines3.common.distributions import (
    BernoulliDistribution, CategoricalDistribution, DiagGaussianDistribution,
    Distribution, MultiCategoricalDistribution,
    SquashedDiagGaussianDistribution, StateDependentNoiseDistribution,
    make_proba_distribution)
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (BaseFeaturesExtractor,
                                                   FlattenExtractor,
                                                   create_mlp)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import is_vectorized_observation
from torch import nn

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class ActorCriticPolicy(BasePolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        tree_struct: Dict = None,
        tree_optimizer: Dict = None,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        squash: bool = False,
        log_std_init: float = -2.0,
        shared_tree_struct: bool = True,
        log_std_schedule: Schedule = None,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        use_masking: bool = False,
        squash_output: bool = False,
        nn_critic: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        is_categorical: bool = False
    ):

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        # Default network architecture, from stable-baselines
        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init
        self.is_categorical = is_categorical
        self.distribution = None
        self.nn_critic = nn_critic
        assert (nn_critic and not shared_tree_struct) or (not nn_critic), \
            "Cannot use shared tree structure with NN critic"
        self.shared_tree_struct = shared_tree_struct
        self.value_net = None

        self.logits_dim = get_action_dim(action_space)
        if isinstance(action_space, gym.spaces.Discrete):
            self.logits_dim = self.logits_dim*action_space.n
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            self.logits_dim = action_space.nvec.sum()

        self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim
        self.value_optimizer = None
        self.log_std_schedule = log_std_schedule

        self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        self.log_std_optimizer = None
        self.use_masking = use_masking

        # Action distribution
        if use_masking:
            self.action_dist = make_masked_proba_distribution(action_space)
        else:
            self.action_dist = SquashedDiagGaussianDistribution(get_action_dim(action_space)) \
                if isinstance(action_space, spaces.Box) and squash else make_proba_distribution(action_space)
        self._build(tree_struct, tree_optimizer, lr_schedule, log_std_schedule)

    def _build(self, tree_struct: Dict, tree_optimizer: Dict, lr_schedule: Schedule,
               log_std_schedule: Schedule) -> None:

        policy_optimizer = tree_optimizer['policy_optimizer']
        policy_optimizer['start_idx'] = 0
        policy_optimizer['stop_idx'] = self.logits_dim
        value_optimizer = tree_optimizer.get('value_optimizer', None)
        if value_optimizer is not None:
            value_start_idx = self.logits_dim if self.shared_tree_struct else 0
            value_optimizer['start_idx'] = value_start_idx
            value_optimizer['stop_idx'] = value_start_idx + 1
        else:
            policy_optimizer['stop_idx'] = self.logits_dim + 1

        if isinstance(self.action_dist, DiagGaussianDistribution) or isinstance(self.action_dist,
                                                                                SquashedDiagGaussianDistribution):
            self.log_std = nn.Parameter(th.ones(self.action_dist.action_dim) * self.log_std_init, requires_grad=True)
            assert log_std_schedule is not None, "log_std_schedule is None"
            self.log_std_optimizer = th.optim.Adam([self.log_std], lr=log_std_schedule(1))

        if self.nn_critic:
            self.value_net = nn.Sequential(*create_mlp(input_dim=self.features_dim,
                                           output_dim=1,
                                           net_arch=self.net_arch,
                                           activation_fn=self.activation_fn)
                                           ).to(self.device)
            self.model = ParametricActor(tree_struct=tree_struct,
                                         input_dim=self.features_dim,
                                         output_dim=self.logits_dim,
                                         policy_optimizer=policy_optimizer,
                                         params=tree_optimizer['params'],
                                         device=tree_optimizer.get('device', 'cpu'))
            # Setup optimizer with initial learning rate
            self.value_optimizer = self.optimizer_class(self.value_net.parameters(),
                                                        lr=lr_schedule(1), **self.optimizer_kwargs)
        else:
            self.model = ActorCritic(tree_struct=tree_struct,
                                     input_dim=self.features_dim,
                                     output_dim=self.logits_dim + 1,
                                     shared_tree_struct=self.shared_tree_struct,
                                     policy_optimizer=policy_optimizer,
                                     value_optimizer=value_optimizer,
                                     params=tree_optimizer['params'],
                                     device=tree_optimizer.get('device', 'cpu'))

    def forward(self, obs: Union[th.Tensor, np.ndarray], deterministic: bool = False,
                requires_grad: bool = False, action_masks: Optional[np.ndarray] = None,
                stop_idx: int = None) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        distribution, values = self._get_action_dist_from_obs(obs, requires_grad, stop_idx)
        if action_masks is not None and self.use_masking:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_action_dist_from_obs(self, obs: Union[th.Tensor, np.ndarray],
                                  requires_grad: bool = True, stop_idx: int = None,
                                  policy_only: bool = False) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        if self.nn_critic:
            mean_actions = self.model(obs, requires_grad, tensor=True, stop_idx=stop_idx)
            values = self.value_net(obs)
        else:
            if policy_only:
                mean_actions = self.model.predict_policy(obs, requires_grad, tensor=True)
                values = None
            else:
                mean_actions, values = self.model(obs, requires_grad, tensor=True)
        if self.logits_dim == 1 and mean_actions.ndim == 1:
            mean_actions = mean_actions.reshape((len(mean_actions), self.logits_dim))

        self.mean_actions = mean_actions

        # if obs[:, 0].any() == 0 and obs[:, 2].any() == 0:
        #     idx = (obs[:, 0] == 0) & (obs[:, 2] == 0)
        #     print(f'y = {obs[idx, 1]} and action: {mean_actions[idx]}')
        if isinstance(self.action_dist, SquashedDiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std), values
        elif isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std), values
        elif isinstance(self.action_dist, CategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions), values
        elif isinstance(self.action_dist, MaskableCategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions), values
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions), values
        elif isinstance(self.action_dist, BernoulliDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions), values
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            raise NotImplementedError
            # return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
            # return self.action_dist.proba_distribution(mean_actions, self.log_std, mean_actions)
        else:
            raise ValueError("Invalid action distribution")

    def get_schedule_learning_rates(self):
        lrs = self.model.get_schedule_learning_rates()
        # lrs is a numpy array for each optimizer
        if len(lrs) == 1:
            policy_lr = lrs[0]
            if self.nn_critic:
                value_lr = self.lr_schedule(self._current_progress_remaining)
            else:
                value_lr = 0.0
        else:
            policy_lr, value_lr = lrs
        return policy_lr, value_lr

    def _predict(self, observation: Union[th.Tensor, np.ndarray], deterministic: bool = False,
                 action_masks: Optional[np.ndarray] = None, requires_grad: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_distribution(obs=observation, action_masks=action_masks,
                                     requires_grad=requires_grad).get_actions(deterministic=deterministic)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
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
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        vectorized_env = None
        if not self.is_categorical:
            observation, vectorized_env = self.obs_to_tensor(observation)
        vectorized_env = is_vectorized_observation(observation, self.observation_space)

        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic, requires_grad=False)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, state

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor, requires_grad: bool = True,
                         action_masks: Optional[np.ndarray] = None,
                         policy_only: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        distribution, values = self._get_action_dist_from_obs(obs, requires_grad=requires_grad, policy_only=policy_only)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        self.distribution = distribution
        return values, log_prob, distribution.entropy()

    def get_distribution(self, obs: Union[th.Tensor, np.ndarray], requires_grad: bool = False,
                         action_masks: Optional[np.ndarray] = None) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        distribution, _ = self._get_action_dist_from_obs(obs, requires_grad=requires_grad)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def get_iteration(self):
        return self.model.get_iteration()

    def get_total_iterations(self):
        return self.model.get_total_iterations()

    def get_num_trees(self):
        return self.model.get_num_trees()

    def predict_values(self, obs: Union[th.Tensor, np.ndarray], requires_grad: bool = True,
                       stop_idx: int = None) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs:
        :return: the estimated values.
        """
        if self.nn_critic:
            if not isinstance(obs, th.Tensor):
                obs = th.tensor(obs, device=self.device)
            return self.value_net(obs)
        return self.model.predict_values(obs, requires_grad, tensor=True, stop_idx=stop_idx)

    def critic(self, obs: Union[th.Tensor, np.ndarray]) -> th.Tensor:
        return self.predict_values(obs)

    def get_params(self):
        return self.model.get_params()

    def step(self, observations: Optional[Union[np.ndarray, th.Tensor]] = None, policy_grad_clip: float = None,
             value_grad_clip: float = None,
             compliance: Optional[Union[np.ndarray, th.Tensor]] = None,
             user_actions: Optional[Union[np.ndarray, th.Tensor]] = None,
             ) -> None:
        # if user_actions is not None:
        #     user_actions = (user_actions.reshape(self.mean_actions.shape) - self.mean_actions.detach()).flatten()

        if self.nn_critic:
            self.value_optimizer.step()
            return self.model.step(observations=observations, policy_grad_clip=policy_grad_clip, compliance=compliance, user_actions=user_actions)
        return self.model.step(observations=observations, policy_grad_clip=policy_grad_clip,
                               value_grad_clip=value_grad_clip, compliance=compliance, user_actions=user_actions)

    def actor_step(self, observations: Optional[Union[th.Tensor, np.ndarray]] = None,
                   policy_grad_clip: float = None,
                   compliance: Optional[Union[np.ndarray, th.Tensor]] = None,
                   user_actions: Optional[Union[np.ndarray, th.Tensor]] = None) -> None:
        self.model.actor_step(observations=observations, policy_grad_clip=policy_grad_clip, compliance=compliance, user_actions=user_actions)

    def critic_step(self, observations: Optional[Union[th.Tensor, np.ndarray]] = None,
                    value_grad_clip: float = None,
                    compliance: Optional[Union[np.ndarray, th.Tensor]] = None,
                    user_actions: Optional[Union[np.ndarray, th.Tensor]] = None,
                    ) -> None:
        self.model.critic_step(observations=observations, value_grad_clip=value_grad_clip, compliance=compliance, user_actions=user_actions)

    def update_learning_rate(self, policy_learning_rate, value_learning_rate):
        self.model.adjust_learning_rates(policy_learning_rate, value_learning_rate)
