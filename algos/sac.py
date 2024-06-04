##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import (GymEnv, MaybeCallback,
                                                   Schedule, TrainFreq,
                                                   TrainFrequencyUnit)
from stable_baselines3.common.utils import (get_parameters_by_name,
                                            polyak_update)
from torch.nn import functional as F

from policies.sac_policy import ContinuousCritic, SACPolicy

SelfSAC = TypeVar("SelfSAC", bound="SAC_GBRL")

class SAC_GBRL(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
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
            self, env, 
            train_freq: int,
            seed: int = 10,
            buffer_size: int = 100000,
            learning_starts: int = 100,
            ent_lr: float = 3e-4, 
            batch_size: int = 64,
            tau: float = 0.005,
            gamma: float = 0.99,
            target_update_interval: int = 100,
            ent_coef: Union[str, float] = "auto",
            gradient_steps: int = 1,
            max_q_grad_norm: float = None,
            max_policy_grad_norm: float = None,
            normalize_policy_grads: bool = True,
            policy_kwargs: Dict = None,
            tensorboard_log: str = None, 
            verbose: int = 1,
            device: str = 'cpu',
    ):
        
        tr_freq = TrainFreq(train_freq, TrainFrequencyUnit.STEP)
        policy_kwargs['tree_optimizer']['device'] = device
        policy_kwargs['tree_optimizer']['target_update_interval'] = target_update_interval
        policy_kwargs['tree_optimizer']['verbose'] = verbose
        super().__init__(
            SACPolicy,
            env,
            learning_rate=ent_lr,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=tr_freq,
            gradient_steps=gradient_steps,
            action_noise=None,
            replay_buffer_class=None,
            replay_buffer_kwargs=None,
            policy_kwargs=policy_kwargs,
            stats_window_size=100,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device='cpu',
            seed=seed,
            use_sde=False,
            sde_sample_freq=-1,
            use_sde_at_warmup=False,
            optimize_memory_usage=False,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        # self.target_entropy = self.config.training.target_entropy
        self.target_entropy = "auto"
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.max_q_grad_norm = max_q_grad_norm
        self.max_policy_grad_norm = max_policy_grad_norm
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None
        self.reg_coef = 1
        self.accum_gradient_steps = 0
        self.prev_timesteps = 0
        self.normalize_policy_grads = normalize_policy_grads

        self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # # Running mean and running var
        # self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        # self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)
            self.log_ent_coef = th.log(self.ent_coef_tensor)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        # optimizers = [self.actor.optimizer, self.critic.optimizer]
        # if self.ent_coef_optimizer is not None:
        #     optimizers += [self.ent_coef_optimizer]

        ent_coef_losses, ent_coefs, log_ent_coefs = [], [], []
        actor_losses, critic_losses = [], []
        mu_maxs, mu_mins, log_std_maxs, log_std_mins = [], [], [], []
        mu_grads_maxs, mu_grads_mins, log_std_grads_maxs, log_std_grads_mins = [], [], [], []
        q_s_max, q_s_min, target_q_s_max, target_q_s_min = [], [], [], []
        mu_penalties = []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            
            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations, requires_grad=True)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()          
                ent_coef_losses.append(ent_coef_loss.item())    
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())
            log_ent_coefs.append(self.log_ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()


            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                # if self.use_target:
                next_q_values = th.cat(self.critic.predict_target(replay_data.next_observations, next_actions), dim=1) # concatenation already done within critic
                # else:
                    # next_q_values = th.cat(self.critic(replay_data.next_observations, next_actions, train_mode=False), dim=1) # concatenation already done within critic
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                target_q_s_max.append(target_q_values.max().item())
                target_q_s_min.append(target_q_values.min().item())
                assert ~th.isinf(target_q_values).any(), "target_q_values has infinite values"

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions, requires_grad=True)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            critic_loss.backward()
            self.critic.step(replay_data.observations, self.max_q_grad_norm)

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)

            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())
            # type: ignore[union-attr]
            actor_loss.backward()

            q_s_max.append(min_qf_pi.max().item())
            q_s_min.append(min_qf_pi.min().item())

            actor_grads = self.actor.step(replay_data.observations, self.max_policy_grad_norm)
            actor_params = self.actor.model.params
            mu, log_std = actor_params 
            mu_grad, log_std_grad = actor_grads

            mu_maxs.append(mu.max().item())
            mu_mins.append(mu.min().item())
            log_std_maxs.append(log_std.max().item())
            log_std_mins.append(log_std.min().item())
            mu_grads_maxs.append(np.max(mu_grad))
            mu_grads_mins.append(np.min(mu_grad))
            log_std_grads_maxs.append(np.max(log_std_grad))
            log_std_grads_mins.append(np.min(log_std_grad))


                
        self._n_updates += gradient_steps
        
        actor_iteration = self.actor.get_iteration()
        actor_num_trees = self.actor.get_num_trees()
        critic_iteration = self.critic.get_iteration()
        critic_num_trees = self.critic.get_num_trees()

        # if self.save_every > 0 and self.num_timesteps - self.prev_timesteps > self.save_every:
        #     self.policy.save_model(self.save_name + f'_{self.num_timesteps}_steps', self.config.env.env_name)
        #     self.prev_timesteps = self.num_timesteps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/log_ent_coef", np.mean(log_ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/actor_num_trees", actor_num_trees)
        self.logger.record("train/actor_boosting_iteration", actor_iteration)
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/critic_num_trees", critic_iteration)
        self.logger.record("train/critic_boosting_iteration", critic_num_trees)
        self.logger.record("buffer/replay_buffer_pos", self.replay_buffer.pos)
        self.logger.record("buffer/replay_buffer_size", self.replay_buffer.buffer_size)
        self.logger.record("buffer/replay_buffer_full", self.replay_buffer.full)
        self.logger.record("param/qs_max", np.mean(q_s_max))
        self.logger.record("param/qs_min", np.mean(q_s_min))
        self.logger.record("param/target_qs_max", np.mean(target_q_s_max))
        self.logger.record("param/target_qs_min", np.mean(target_q_s_min))
        self.logger.record("param/mu_max", np.mean(mu_maxs))
        self.logger.record("param/mu_min", np.mean(mu_mins))
        self.logger.record("param/mu_grads_max", np.mean(mu_grads_maxs))
        self.logger.record("param/mu_grads_min", np.mean(mu_grads_mins))
        self.logger.record("param/log_std_max", np.mean(log_std_maxs))
        self.logger.record("param/log_std_min", np.mean(log_std_mins))
        self.logger.record("param/log_std_grads_max", np.mean(log_std_grads_maxs))
        self.logger.record("param/log_std_grads_min", np.mean(log_std_grads_mins))
        if mu_penalties:
            self.logger.record("train/mu_penalties", np.mean(log_std_grads_mins))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self: SelfSAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfSAC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
