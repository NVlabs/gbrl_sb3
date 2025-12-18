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
import collections
from gbrl.models.actor import ParametricActor
from gbrl.models.actor_critic import CostActorCritic
from functools import partial

import warnings
from gymnasium import spaces
from sb3_contrib.common.maskable.distributions import (
    MaskableCategoricalDistribution, make_masked_proba_distribution)
from stable_baselines3.common.distributions import (
    BernoulliDistribution, CategoricalDistribution, DiagGaussianDistribution,
    Distribution, MultiCategoricalDistribution,
    SquashedDiagGaussianDistribution, StateDependentNoiseDistribution,
    make_proba_distribution)
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (BaseFeaturesExtractor,
                                                   FlattenExtractor,
                                                   NatureCNN,
                                                   create_mlp)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import is_vectorized_observation
from torch import nn
from stable_baselines3.common.utils import get_device

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class CostActorCriticPolicyGBRL(BasePolicy):
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
        self.policy_device = tree_optimizer.get('device', 'cpu')
        self.lr_schedule = lr_schedule

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

        cost_optimizer = tree_optimizer.get('cost_optimizer', None)
        if cost_optimizer is not None:
            cost_start_idx = self.logits_dim + 1 if self.shared_tree_struct else 0
            cost_optimizer['start_idx'] = cost_start_idx
            cost_optimizer['stop_idx'] = cost_start_idx + 1
        else:
            policy_optimizer['stop_idx'] = self.logits_dim + 2

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
            self.cost_net = nn.Sequential(*create_mlp(input_dim=self.features_dim,
                                           output_dim=1,
                                           net_arch=self.net_arch,
                                           activation_fn=self.activation_fn)
                                           ).to(self.device)
            self.model = ParametricActor(tree_struct=tree_struct,
                                         input_dim=self.features_dim,
                                         output_dim=self.logits_dim,
                                         policy_optimizer=policy_optimizer,
                                         params=tree_optimizer['params'],
                                         device=self.device)
            # Setup optimizer with initial learning rate
            self.value_optimizer = self.optimizer_class(self.value_net.parameters(),
                                                        lr=lr_schedule(1), **self.optimizer_kwargs)
            self.cost_optimizer = self.optimizer_class(self.cost_net.parameters(),
                                                        lr=lr_schedule(1), **self.optimizer_kwargs)
        else:
            self.model = CostActorCritic(tree_struct=tree_struct,
                                     input_dim=self.features_dim,
                                     output_dim=self.logits_dim + 2,
                                     shared_tree_struct=self.shared_tree_struct,
                                     policy_optimizer=policy_optimizer,
                                     value_optimizer=value_optimizer,
                                     cost_optimizer=cost_optimizer,
                                     params=tree_optimizer['params'],
                                     device=self.policy_device)

    def forward(self, obs: Union[th.Tensor, np.ndarray], deterministic: bool = False,
                requires_grad: bool = False, action_masks: Optional[np.ndarray] = None,
                stop_idx: int = None) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        distribution, values, costs = self._get_action_dist_from_obs(obs, requires_grad, stop_idx)
        if action_masks is not None and self.use_masking:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, costs, values, log_prob

    def _get_action_dist_from_obs(self, obs: Union[th.Tensor, np.ndarray],
                                  requires_grad: bool = True, stop_idx: int = None,
                                  policy_only: bool = False) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        costs = None
        if self.nn_critic:
            mean_actions = self.model(obs, requires_grad, tensor=True, stop_idx=stop_idx)
            values = self.value_net(obs.float())
            costs = self.cost_net(obs.float())
        else:
            if policy_only:
                mean_actions = self.model.predict_policy(obs, requires_grad, tensor=True)
                values = None
                costs = None
            else:
                mean_actions, values, costs = self.model(obs, requires_grad, tensor=True)
        if self.logits_dim == 1 and mean_actions.ndim == 1:
            mean_actions = mean_actions.reshape((len(mean_actions), self.logits_dim))
        self.mean_actions = mean_actions

        if isinstance(self.action_dist, SquashedDiagGaussianDistribution):
            dist = self.action_dist.proba_distribution(mean_actions, self.log_std), values, costs
        elif isinstance(self.action_dist, DiagGaussianDistribution):
            dist = self.action_dist.proba_distribution(mean_actions, self.log_std), values, costs
        elif isinstance(self.action_dist, CategoricalDistribution):
            dist = self.action_dist.proba_distribution(action_logits=mean_actions), values, costs
        elif isinstance(self.action_dist, MaskableCategoricalDistribution):
            dist = self.action_dist.proba_distribution(action_logits=mean_actions), values, costs
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            dist = self.action_dist.proba_distribution(action_logits=mean_actions), values, costs
        elif isinstance(self.action_dist, BernoulliDistribution):
            dist = self.action_dist.proba_distribution(action_logits=mean_actions), values, costs
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            raise NotImplementedError
            # return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
            # return self.action_dist.proba_distribution(mean_actions, self.log_std, mean_actions)
        else:
            raise ValueError("Invalid action distribution")
        self.distribution = dist
        return dist

    def get_schedule_learning_rates(self, lr_schedule: Schedule = None,
                                    progress_remaining: float = None) -> Tuple[float, float, float]:
        lrs = self.model.get_schedule_learning_rates()
        # lrs is a numpy array for each optimizer
        if len(lrs) == 1:
            policy_lr = lrs[0]
            if self.nn_critic:
                value_lr = lr_schedule(progress_remaining)
                cost_lr = lr_schedule(progress_remaining)
            else:
                value_lr = 0.0
                cost_lr = 0.0
        else:
            policy_lr, value_lr, cost_lr = lrs
        return policy_lr, value_lr, cost_lr

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
                         policy_only: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        distribution, values, costs = self._get_action_dist_from_obs(obs, requires_grad=requires_grad, policy_only=policy_only)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        self.distribution = distribution
        return costs, values, log_prob, distribution.entropy()

    def get_distribution(self, obs: Union[th.Tensor, np.ndarray], requires_grad: bool = False,
                         action_masks: Optional[np.ndarray] = None) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        distribution, _, _ = self._get_action_dist_from_obs(obs, requires_grad=requires_grad)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def get_iteration(self):
        return self.model.get_iteration()

    def get_total_iterations(self):
        return self.model.get_total_iterations()

    def get_num_trees(self):
        return self.model.get_num_trees()

    def predict_costs(self, obs: Union[th.Tensor, np.ndarray], requires_grad: bool = True,
                       stop_idx: int = None) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs:
        :return: the estimated values.
        """
        if self.nn_critic:
            if not isinstance(obs, th.Tensor):
                obs = th.tensor(obs, device=self.device)
            return self.cost_net(obs.float())
        return self.model.predict_cost(obs, requires_grad, tensor=True, stop_idx=stop_idx)

    def predict_values(self, obs: Union[th.Tensor, np.ndarray], requires_grad: bool = True,
                       stop_idx: Optional[int] = None) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs:
        :return: the estimated values.
        """
        if self.nn_critic:
            if not isinstance(obs, th.Tensor):
                obs = th.tensor(obs, device=self.device)
            return self.value_net(obs.float())
        return self.model.predict_values(obs, requires_grad, tensor=True, stop_idx=stop_idx)

    def critic(self, obs: Union[th.Tensor, np.ndarray]) -> Tuple[th.Tensor, th.Tensor]:
        return self.predict_values(obs), self.predict_cost(obs)

    def get_params(self):
        return self.model.get_params()

    def get_grads(self):    
        return self.model.get_grads()

    def step(self,
             observations: Optional[Union[np.ndarray, th.Tensor]] = None,
             policy_grad_clip: float = None,
             value_grad_clip: float = None,
             cost_grad_clip: float = None,
             safety_labels: Optional[Union[np.ndarray, th.Tensor]] = None,
             cost_grads: Optional[Union[np.ndarray, th.Tensor]] = None,
             policy_grads: Optional[Union[np.ndarray, th.Tensor]] = None,
             value_grads: Optional[Union[np.ndarray, th.Tensor]] = None,
             ) -> None:

        if self.nn_critic:
            self.value_optimizer.step()
            self.cost_optimizer.step()
            return self.model.step(observations=observations, policy_grad_clip=policy_grad_clip)
        return self.model.step(observations=observations, policy_grad_clip=policy_grad_clip,
                               value_grad_clip=value_grad_clip, cost_grad_clip=cost_grad_clip,
                               obj_labels=safety_labels,
                               policy_grads=policy_grads, value_grads=value_grads, cost_grads=cost_grads)

    def actor_step(self, observations: Optional[Union[th.Tensor, np.ndarray]] = None,
                   policy_grad_clip: float = None,
                   safety_labels: Optional[Union[np.ndarray, th.Tensor]] = None) -> None:
        self.model.actor_step(observations=observations, policy_grad_clip=policy_grad_clip,
                              obj_labels=safety_labels)

    def critic_step(self,
                    observations: Optional[Union[th.Tensor, np.ndarray]] = None,
                    value_grad_clip: float = None,
                    ) -> None:
        self.model.critic_step(observations=observations, value_grad_clip=value_grad_clip)

    def cost_critic_step(self,
                    observations: Optional[Union[th.Tensor, np.ndarray]] = None,
                    cost_grad_clip: float = None,
                    ) -> None:
        self.model.cost_critic_step(observations=observations, cost_grad_clip=cost_grad_clip)

    def update_learning_rate(self, policy_learning_rate, value_learning_rate, cost_learning_rate):
        self.model.adjust_learning_rates(policy_learning_rate, value_learning_rate, cost_learning_rate)
        
class MlpExtractor(nn.Module):
    """
    Constructs an MLP that receives the output from a previous features extractor (i.e. a CNN) or directly
    the observations (if no features extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
    It can be in either of the following forms:
    1. ``dict(vf=[<list of layer sizes>], pi=[<list of layer sizes>])``: to specify the amount and size of the layers in the
        policy and value nets individually. If it is missing any of the keys (pi or vf),
        zero layers will be considered for that key.
    2. ``[<list of layer sizes>]``: "shortcut" in case the amount and size of the layers
        in the policy and value nets are the same. Same as ``dict(vf=int_list, pi=int_list)``
        where int_list is the same for the actor and critic.

    .. note::
        If a key is not specified or an empty list is passed ``[]``, a linear network will be used.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device: PyTorch device.
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        cost_net: List[nn.Module] = []
        last_layer_dim_pi = feature_dim
        last_layer_dim_vf = feature_dim
        last_layer_dim_cf = feature_dim

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specificed, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
            cf_layers_dims = net_arch.get("cf", [])  # Layer sizes of the cost network
        else:
            pi_layers_dims = vf_layers_dims = cf_layers_dims = net_arch
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim
        # Iterate through the cost layers and build the cost net
        for curr_layer_dim in cf_layers_dims:
            cost_net.append(nn.Linear(last_layer_dim_cf, curr_layer_dim))
            cost_net.append(activation_fn())
            last_layer_dim_cf = curr_layer_dim
        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.latent_dim_cf = last_layer_dim_cf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)
        self.cost_net = nn.Sequential(*cost_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features), self.forward_cost_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)
    
    def forward_cost_critic(self, features: th.Tensor) -> th.Tensor:
        return self.cost_net(features)
    
class CostActorCriticPolicy(ActorCriticPolicy):
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
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        
        BasePolicy.__init__(
            self,
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )
        
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        if isinstance(net_arch, list) and len(net_arch) > 0 and isinstance(net_arch[0], dict):
            warnings.warn(
                (
                    "As shared layers in the mlp_extractor are removed since SB3 v1.8.0, "
                    "you should now pass directly a dictionary and not a list "
                    "(net_arch=dict(pi=..., vf=...) instead of net_arch=[dict(pi=..., vf=...)])"
                ),
            )
            net_arch = net_arch[0]

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = dict(pi=[64, 64], vf=[64, 64], cf=[64, 64])

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.share_features_extractor = share_features_extractor
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        if self.share_features_extractor:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor
            self.cf_features_extractor = self.features_extractor
        else:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.make_features_extractor()
            self.cf_features_extractor = self.make_features_extractor()

        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs:
        """
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.cost_net = nn.Linear(self.mlp_extractor.latent_dim_cf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
                self.cost_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
                module_gains[self.cf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf, latent_cf = self.mlp_extractor(features)
        else:
            pi_features, vf_features, cf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
            latent_cf = self.mlp_extractor.forward_cost_critic(cf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        costs = self.cost_net(latent_cf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, costs, log_prob

    def extract_features(self, obs: th.Tensor) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor, th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :return: the output of the features extractor(s)
        """
        if self.share_features_extractor:
            return BasePolicy.extract_features(self, obs, self.features_extractor)
        else:
            pi_features = BasePolicy.extract_features(self, obs, self.pi_features_extractor)
            vf_features = BasePolicy.extract_features(self, obs, self.vf_features_extractor)
            cf_features = BasePolicy.extract_features(self, obs, self.cf_features_extractor)
            return pi_features, vf_features, cf_features

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf, latent_cf = self.mlp_extractor(features)
        else:
            pi_features, vf_features, cf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
            latent_cf = self.mlp_extractor.forward_cost_critic(cf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        costs = self.cost_net(latent_cf)
        entropy = distribution.entropy()
        return values, costs, log_prob, entropy

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = BasePolicy.extract_features(self, obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = BasePolicy.extract_features(self, obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)
    
    def predict_costs(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = BasePolicy.extract_features(self, obs, self.cf_features_extractor)
        latent_cf = self.mlp_extractor.forward_cost_critic(features)
        return self.cost_net(latent_cf)