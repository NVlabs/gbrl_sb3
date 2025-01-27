##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import \
    SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import BaseModel, BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (BaseFeaturesExtractor,
                                                   FlattenExtractor,
                                                   create_mlp)
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from gbrl.ac_gbrl import GaussianActor
from gbrl.ac_gbrl import ContinuousCritic as GBRLContinuousCritic

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20

import numpy as np


class ContinuousCritic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    Adaptation of stable_baselines3 class to incorporate GBRL.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        tree_struct: Dict = None, 
        critic_optimizer: Dict = None,
        normalize_images: bool = True,
        n_critics: int = 2,
        net_arch: List[int] = None,
        features_dim: int = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        share_features_extractor: bool = True,
        q_func_type: str = 'linear',
        gbrl_params: Dict=dict(),
        target_update_interval: int = 100,
        device: str = 'cpu',
        verbose: int = 0,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.critic_optimizer = critic_optimizer
        self.tree_struct = tree_struct
        action_dim = get_action_dim(self.action_space)
        self.action_dim = action_dim
        self.q_func_type = q_func_type
        self.share_features_extractor = share_features_extractor
        self.optimizer = None

        self.n_critics = n_critics
        self.q_models = []
        if self.q_func_type == 'nn':
            for idx in range(n_critics):
                q_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
                q_net = nn.Sequential(*q_net)
                self.add_module(f"qf{idx}", q_net)
                self.q_models.append(q_net)
        else:
                
            self.theta_dim = 1 if q_func_type != 'quadratic' else 2
            weights_optimizer = critic_optimizer['weights_optimizer']
            weights_optimizer['start_idx'] = 0
            bias_optimizer = critic_optimizer.get('bias_optimizer', None)
            if bias_optimizer is not None:
                bias_optimizer['start_idx'] = action_dim
                bias_optimizer['stop_idx'] = action_dim + self.theta_dim
            weights_optimizer['stop_idx'] = action_dim
            for _ in range(n_critics):
                bias = np.random.randn(action_dim + self.theta_dim) * np.sqrt(2.0 / action_dim)
                bias[-self.theta_dim:] = 0
                q_model = GBRLContinuousCritic(tree_struct=tree_struct, 
                                        output_dim=action_dim + self.theta_dim, 
                                        weights_optimizer=weights_optimizer,
                                        bias_optimizer=bias_optimizer,
                                        gbrl_params=gbrl_params,
                                        target_update_interval=target_update_interval,
                                        bias=bias,
                                        verbose=verbose,
                                        device=device
                                            )
                self.q_models.append(q_model)

    def forward(self, obs: th.Tensor, actions: th.Tensor, target: bool, requires_grad: bool = False, tensor: bool = True) -> Tuple[th.Tensor, ...]:
        if self.optimizer:
            self.optimizer.zero_grad()
        if self.q_func_type == 'nn':
            with th.set_grad_enabled(not self.share_features_extractor):
                features = self.extract_features(obs, self.features_extractor)
                qvalue_input = th.cat([features, actions], dim=1)
                return tuple(q_net(qvalue_input) for q_net in self.q_models)
        q_s = []
        for q_net in self.q_models:
            weights, bias = q_net(obs, requires_grad, target, tensor=True) 

            dot = (weights * actions).sum(dim=1)
            if self.q_func_type == 'linear':
                q = (dot + bias.squeeze())
            elif self.q_func_type == 'tanh':
                q = (bias.squeeze()*th.tanh(dot))
            elif self.q_func_type == 'quadratic':
                q = -((dot - bias[:, 0])**2) + bias[:, 1]
            q_s.append(q.unsqueeze(-1))
        return tuple(q_s)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        return self.q_models[0](th.cat([features, actions], dim=1))
    
    def step(self, observations: Optional[Union[np.array, th.Tensor]] = None, q_grad_clip: float=None) -> None:
        if self.q_func_type == 'nn':
             self.optimizer.step()
        else:
            for idx in range(self.n_critics):
                self.q_models[idx].step(observations=observations, q_grad_clip=q_grad_clip)

    def get_num_trees(self):
        return self.q_models[0].get_num_trees()

    def get_iteration(self):
        return self.q_models[0].get_iteration()
    
    def copy(self):
        return self.__copy__()
    
    def __copy__(self):
        copy_ = ContinuousCritic(self.observation_space, self.action_space, self.features_extractor, self.tree_struct, self.critic_optimizer, self.normalize_images, self.n_critics, self.q_func_type)
        for idx in range(self.n_critics):
            copy_.q_models[idx] = self.q_models[idx].copy()
        return copy_

    def save_model(self, name: str, env_name: str):
        for i in range(self.n_critics):
            self.q_models[i].save_model(name + f'_critic_{i+1}', env_name)

    def load_model(self, name: str, env_name: str):
        for i in range(self.n_critics):
            self.q_models[i] = ContinuousCritic.load_model(name + f'_critic_{i+1}', env_name)

class Actor(BasePolicy):
    """
    Actor network (policy) for SAC.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    action_space: spaces.Box

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: nn.Module,
        tree_struct: Dict = None, 
        actor_optimizer: Dict = None,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        gbrl_params: Dict=dict(),
        normalize_images: bool = True,
        verbose: int = 0,
        device: str = 'cpu',
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.log_std_init = log_std_init
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        action_dim = get_action_dim(self.action_space)
        self.action_dim = action_dim
        mu_optimizer = actor_optimizer.get('mu_optimizer', None)
        std_optimizer = actor_optimizer.get('std_optimizer', None)
        mu_optimizer['start_idx'] = 0
        stop_idx = action_dim*2
        if std_optimizer is not None:
            stop_idx = action_dim
            std_optimizer['start_idx'] = stop_idx
            std_optimizer['stop_idx'] = stop_idx*2
        mu_optimizer['stop_idx'] = stop_idx

        self.model = GaussianActor(tree_struct=tree_struct, 
                            output_dim=action_dim*2, 
                            mu_optimizer=mu_optimizer,
                            std_optimizer=std_optimizer,
                            log_std_init=log_std_init,
                            gbrl_params=gbrl_params,
                            verbose=verbose,
                            device=device)
        self.action_dist = SquashedDiagGaussianDistribution(action_dim)  # type: ignore[assignment]

    def get_action_dist_params(self, obs: th.Tensor, requires_grad: bool = False) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        mean_actions, log_std = self.model(obs, requires_grad, tensor=True)
        # Unstructured exploration (Original implementation)
        # Original Implementation to cap the standard deviation
        return mean_actions, th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX), {}

    def forward(self, obs: th.Tensor, deterministic: bool = False, requires_grad: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs, requires_grad)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: th.Tensor, requires_grad: bool=False) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs, requires_grad)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)

    def step(self, observations: Optional[th.Tensor] = None, mu_grad_clip: Optional[float] = None, log_std_grad_clip: Optional[float] = None) -> None:
        return self.model.step(observations=observations, mu_grad_clip=mu_grad_clip, log_std_grad_clip=log_std_grad_clip)

    def get_num_trees(self):
        return self.model.get_num_trees()

    def get_iteration(self):
        return self.model.get_iteration()



class SACPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    actor: Actor
    critic: ContinuousCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        tree_optimizer: Dict, 
        tree_struct: Dict, 
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        clip_mean: float = 2.0,
        q_func_type: str = 'linear',
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [256, 256]

        self.nn_critic = q_func_type == 'nn'
        self.critic_target = None

        self.actor_kwargs = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "features_extractor": self.make_features_extractor(),
            "tree_struct": tree_struct,
            "actor_optimizer": tree_optimizer['actor_optimizer'],
            "device": tree_optimizer.get('device', 'cpu'),
            "verbose": tree_optimizer.get('verbose', 1),
            "log_std_init": log_std_init,
            "normalize_images": normalize_images,
            "gbrl_params": tree_optimizer.get("gbrl_params", dict())
        }
        if self.nn_critic:
            self.net_args = {
                "observation_space": self.observation_space,
                "action_space": self.action_space,
                "net_arch": net_arch,
                "activation_fn": activation_fn,
                "normalize_images": normalize_images,
                "q_func_type": "nn",
            }
            self.net_arch = net_arch
            self.activation_fn = activation_fn
            self.critic_kwargs = self.net_args.copy()
            self.critic_kwargs.update(
                {
                    "n_critics": n_critics,
                    "share_features_extractor": False,
                }
            )

            self.share_features_extractor = False
        else:
            self.critic_kwargs =  {
                    "observation_space": self.observation_space,
                    "action_space": self.action_space,
                    "features_extractor": self.make_features_extractor(),
                    "tree_struct": tree_struct,
                    "critic_optimizer": tree_optimizer['critic_optimizer'],
                    "device": tree_optimizer.get('device', 'cpu'),
                    "target_update_interval": tree_optimizer.get('target_update_interval', 100),
                    "verbose": tree_optimizer.get('verbose', 1),
                    "n_critics": n_critics,
                    "q_func_type": q_func_type,
                    "gbrl_params": tree_optimizer.get("gbrl_params", dict())
                }   

        self._build(lr_schedule)
        

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.critic = self.make_critic(features_extractor=None)
        
        if self.nn_critic:
            critic_parameters = list(self.critic.parameters())
            # Critic target should not share the features extractor with critic
            self.critic_target = self.make_critic(features_extractor=None)
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.critic.optimizer = self.optimizer_class(
            critic_parameters,
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )
            # Target networks should always be in eval mode
            self.critic_target.set_training_mode(False)

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        self.actor.reset_noise(batch_size=batch_size)

    def make_actor(self) -> Actor:
        return Actor(**self.actor_kwargs)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Union[ContinuousCritic, ContinuousCritic]:
        critic_kwargs = self.critic_kwargs
        if self.nn_critic:
            critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic)
    
    def save_model(self, name: str, env_name: str):
        self.actor.model.save_model(name + '_actor', env_name)
        self.critic.save_model(name, env_name)

    def load_model(self, name:str, env_name: str):
        self.actor.model.load_model(name + '_actor', env_name)
        self.critic.load_model(name, env_name)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        if self.nn_critic:
            self.critic.set_training_mode(mode)
        self.training = mode

    def predict_critic(self,  obs: th.Tensor, actions: th.Tensor, target: bool = False, requires_grad: bool = False):
        if self.critic_target is not None and target:
            kwargs = {}
            return self.critic_target(obs, actions, target)
        return self.critic(obs, actions, target, requires_grad=requires_grad)


