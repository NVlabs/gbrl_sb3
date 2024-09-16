##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
from collections import deque

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]

from typing import Any, Dict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import (
    CategoricalDistribution, DiagGaussianDistribution,
    SquashedDiagGaussianDistribution, 
)



class GradientHistogramCallback(BaseCallback):
    def __init__(self, log_dir: str, awr: bool = True, save_freq: int=-1):
        super().__init__()
        self.log_dir = log_dir
        self.awr = awr
        self.save_freq = save_freq
        if log_dir:
            self.writer = SummaryWriter(log_dir=log_dir)

    def _on_step(self) -> bool:
        # only log histograms every `save_freq` number of steps
        return self.num_timesteps % self.save_freq == 0

    def _on_log(self) -> None:
        if self.log_dir and self.awr:
            for name, params in self.model.policy.actor.named_parameters():
                if params.grad is not None:
                    self.writer.add_histogram(
                        tag=f'gradients/actor_{name}',
                        values=params.grad,
                        global_step=self.num_timesteps
                    )
            for name, params in self.model.policy.critic.named_parameters():
                if params.grad is not None:
                    self.writer.add_histogram(
                        tag=f'gradients/critic_{name}',
                        values=params.grad,
                        global_step=self.num_timesteps
                    )

        
    def close(self):
        self.writer.close()

class OnPolicyDistillationCallback(BaseCallback):
    def __init__(self, params: Dict, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.params = params
        self.reset()


    def reset(self):
        self.distil_data = deque(maxlen=self.params['capacity'])

    def update_locals(self, locals_: Dict[str, Any]) -> None:
        obs = locals_.get('obs_tensor')
        if obs is not None:
            obs = obs.detach().cpu().numpy()
            self.distil_data.append(obs)
        super().update_locals(locals_)

    def _on_rollout_start(self) -> None:
        if self.model.policy.model.shared_tree_struct:
            num_trees = self.model.policy.model.get_num_trees()
            if num_trees >= self.params['max_steps']:
                obs = np.concatenate(self.distil_data, axis=0)
                self.logger.record("distill/num_samples", len(obs))
                pg_targets, vf_targets = self.model.policy.model.predict(obs)
                distil_loss, self.params = self.model.policy.model.model.distil(obs, pg_targets, vf_targets, self.params, self.verbose)
                self.logger.record("distill/distilation_loss", distil_loss)
                self.logger.record("distill/max_steps", self.params['max_steps'])
                self.logger.record("distill/min_steps", self.params['min_steps'])
        else:
            policy_num_trees, value_num_trees = self.model.policy.model.get_num_trees()
            if policy_num_trees > self.distil_steps or value_num_trees > self.distil_steps:
                obs = np.concatenate(self.distil_data, axis=0).astype(np.single)
                pg_targets, vf_targets = self.model.policy.model.predict(obs)
                self.logger.record("distill/num_samples", len(obs))
                if policy_num_trees > self.distil_steps:
                    distil_loss, self.params = self.model.policy.model.model.distil_policy(obs, pg_targets, self.params)
                    self.logger.record("distill/pg_distilation_loss", distil_loss)
                else:
                    distil_loss, self.params = self.model.policy.model.model.distil_value(obs, vf_targets, self.params)
                    self.logger.record("distill/vf_distilation_loss", distil_loss)
                self.logger.record("distill/max_steps", self.params['max_steps'])
                self.logger.record("distill/min_steps", self.params['min_steps'])

    def _on_step(self) -> bool:
        return True

class OffPolicyDistillationCallback(BaseCallback):
    def __init__(self, params: Dict, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.params = params

    def _on_rollout_start(self) -> None:
        if self.model.shared_tree_struct:
            num_trees = self.model.policy.model.get_num_trees()
            if num_trees >= self.params['max_steps']:
                valid_pos = self.model.replay_buffer.buffer_size if self.model.replay_buffer.full else self.model.replay_buffer.pos
                obs = [self.model.replay_buffer.observations[:valid_pos, env_idx] for env_idx in range(self.model.env.num_envs)]
                obs = np.concatenate(obs, axis=0)
                self.logger.record("distill/num_samples", len(obs))
                pg_targets, vf_targets = self.model.policy.model.predict(obs)
                distil_loss, self.params = self.model.policy.model.model.distil(obs, pg_targets, vf_targets, self.params, self.verbose)
                self.logger.record("distill/distilation_loss", distil_loss)
                self.logger.record("distill/max_steps", self.params['max_steps'])
                self.logger.record("distill/min_steps", self.params['min_steps'])
        
    def _on_step(self) -> bool:
        return True


class StopTrainingOnNoImprovementInTraining(BaseCallback):
    """
    Callback for stopping the training when there is no improvement in training reward
    using the mean reward from ep_info_buffer.
    """

    def __init__(self, improvement_threshold: float, check_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.improvement_threshold = improvement_threshold
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf
        self.last_check_step = 0

    def _on_step(self) -> bool:
        # Check if it's time to evaluate the training performance
        if self.num_timesteps - self.last_check_step >= self.check_freq or self.best_mean_reward == -np.inf:
            # Compute mean reward using the ep_info_buffer
            ep_info_buffer = self.model.ep_info_buffer if self.model is not None else []
            if len(ep_info_buffer) > 0:
                current_mean_reward = np.mean([ep_info["r"] for ep_info in ep_info_buffer])

                if self.verbose > 0:
                    print(f"Current mean reward: {current_mean_reward}, Best mean reward: {self.best_mean_reward}")

                # Check for improvement
                if current_mean_reward <= self.best_mean_reward + self.improvement_threshold:
                    if self.verbose > 0:
                        print("No improvement in training reward. Stopping training.")
                    return False  # Stop training
                else:
                    self.best_mean_reward = current_mean_reward

            self.last_check_step = self.num_timesteps

        return True  # Continue training


class ActorCriticCompressionCallback(BaseCallback):
    def __init__(self, params: Dict, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.max_steps = params['max_steps']
        self.capacity = params['capacity']
        params['k'] = params['max_steps'] - params['trees_to_keep']
        print('capacity', self.capacity)
        del params['max_steps']
        del params['capacity']
        del params['trees_to_keep']
        self.params = params
        
        self.dist_type = None 
       
        self.reset()


    def reset(self):
        self.compression_data = deque(maxlen=self.capacity)

    def update_locals(self, locals_: Dict[str, Any]) -> None:
        obs = locals_.get('obs_tensor')
        if obs is not None:
            obs = obs.detach().cpu().numpy()
            self.compression_data.append(obs)
        super().update_locals(locals_)

    def _on_rollout_start(self) -> None:
        if self.dist_type is None:
            self.dist_type = 'deterministic'
            if isinstance(self.model.policy.action_dist, DiagGaussianDistribution) or isinstance(self.model.policy.action_dist, SquashedDiagGaussianDistribution):
                self.dist_type = 'gaussain'
            elif isinstance(self.model.policy.action_dist, CategoricalDistribution):
                self.dist_type = 'categorical'
        num_trees = self.model.policy.model.get_num_trees()
        if not self.model.policy.model.shared_tree_struct:
            num_trees = num_trees[0]
        if num_trees >= self.max_steps:
            print(f"Compressing with: {len(self.compression_data)}")
            obs = np.concatenate(list(self.compression_data), axis=0)
            self.logger.record("compression/num_samples", len(obs))
            actions = self.model.policy._predict(obs, deterministic=False)
            log_std = None if self.dist_type != 'gaussian' else self.model.policy.log_std.detach().cpu().numpy()
            compression_params = {**self.params, 'features': obs, 'actions': actions.detach().cpu(), 'log_std': log_std, 'dist_type': self.dist_type}
            self.model.policy.model.compress(**compression_params)
        # else:
        #     policy_num_trees, value_num_trees = self.model.policy.model.get_num_trees()
        #     if policy_num_trees > self.distil_steps or value_num_trees > self.distil_steps:
        #         obs = np.concatenate(self.distil_data, axis=0).astype(np.single)
        #         pg_targets, vf_targets = self.model.policy.model.predict(obs)
        #         self.logger.record("distill/num_samples", len(obs))
        #         if policy_num_trees > self.distil_steps:
        #             distil_loss, self.params = self.model.policy.model.model.distil_policy(obs, pg_targets, self.params)
        #             self.logger.record("distill/pg_distilation_loss", distil_loss)
        #         else:
        #             distil_loss, self.params = self.model.policy.model.model.distil_value(obs, vf_targets, self.params)
        #             self.logger.record("distill/vf_distilation_loss", distil_loss)
        #         self.logger.record("distill/max_steps", self.params['max_steps'])
        #         self.logger.record("distill/min_steps", self.params['min_steps'])

    def _on_step(self) -> bool:
        return True