##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
import os
from collections import deque

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]

from typing import Any, Dict, Optional, Union

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import (VecEnv, sync_envs_normalization)

from utils.helpers import evaluate_policy_and_obs, evaluate_policy_with_noise


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
                if current_mean_reward < self.best_mean_reward + self.improvement_threshold:
                    if self.verbose > 0:
                        print("No improvement in training reward. Stopping training.")
                    return False  # Stop training
                else:
                    self.best_mean_reward = current_mean_reward

            self.last_check_step = self.num_timesteps

        return True  # Continue training

    
class MultiEvalCallback(EvalCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        env_name: str,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(eval_env, callback_on_new_best, callback_after_eval,
                         n_eval_episodes, eval_freq, log_path, best_model_save_path, 
                         deterministic, render, verbose, warn)
        self.env_name = env_name

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(f"{self.env_name}: Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"{self.env_name} Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record(f"eval/{self.env_name}_mean_reward", float(mean_reward))
            self.logger.record(f"eval/{self.env_name}_mean_ep_length", mean_ep_length)
            self.logger.record(f"eval/{self.env_name}_std_reward", float(std_reward))
            self.logger.record(f"eval/{self.env_name}_std_ep_length", std_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(f"time/{self.env_name}_total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

class ChangeEnvCallback(BaseCallback):
    """
    A callback that changes a value in the environment
    at a specific frequency during training.

    :param change_freq: The frequency (in number of calls) at which to change the environment value.
    :param change_function: A function to modify the environment.
        It should take the environment as input and perform the desired changes.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages.
    """
    def __init__(self, change_freq: int, change_function: callable, change_function_args: tuple = (), 
                 change_function_kwargs: dict = None, warmup_time: int = 0, n_envs: int = 1, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.change_freq = change_freq
        self.change_function = change_function
        self.change_function_args = change_function_args
        self.change_function_kwargs = change_function_kwargs or {}
        self.warmup_time = warmup_time
        self.n_envs = n_envs

    def _on_step(self) -> bool:
        # Check if the current step matches the change frequency
        if (self.n_calls * self.n_envs) % self.change_freq == 0:
            if self.verbose >= 1:
                print(f"Step {self.n_calls}: Changing environment configuration.")
            # Apply the change function to the environment
            if 'n_steps' in self.change_function_kwargs.keys(): 
                self.change_function_kwargs['n_steps'] = self.n_calls * self.n_envs
            if self.training_env is not None:
                if isinstance(self.training_env, VecEnv):
                    for env_idx in range(self.training_env.num_envs):
                        self.change_function(self.training_env.envs[env_idx],
                                             *self.change_function_args,
                                             **self.change_function_kwargs)
                else:
                    self.change_function(self.training_env,
                                         *self.change_function_args,
                                         **self.change_function_kwargs)
        return True  # Continue training
    

class MultiEvalWithObsCallback(EvalCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        env_name: str,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        min_values: np.array = None,
        max_values: np.array =None
    ):
        super().__init__(eval_env, callback_on_new_best, callback_after_eval,
                         n_eval_episodes, eval_freq, log_path, best_model_save_path, 
                         deterministic, render, verbose, warn)
        self.env_name = env_name
        self.min_values = min_values 
        self.max_values = max_values

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths, min_obs, max_obs = evaluate_policy_and_obs(
                self.model,
                self.eval_env,
                min_values=self.min_values, 
                max_values=self.max_values,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(f"{self.env_name}: Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"{self.env_name} Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record(f"eval/{self.env_name}_mean_reward", float(mean_reward))
            self.logger.record(f"eval/{self.env_name}_mean_ep_length", mean_ep_length)
            for i in range(len(min_obs)):
                self.logger.record(f"obs/{self.env_name}_min_{i}", float(min_obs[i]))
                self.logger.record(f"obs/{self.env_name}_max_{i}", float(max_obs[i]))

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(f"time/{self.env_name}_total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


class NoisyEvalCallback(EvalCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        env_name: str = '',
        noise_std: float = 0.05,
        warn: bool = True,
    ):
        super().__init__(eval_env, callback_on_new_best, callback_after_eval,
                         n_eval_episodes, eval_freq, log_path, best_model_save_path, 
                         deterministic, render, verbose, warn)
        self.env_name = env_name
        self.noise_std = noise_std
    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy_with_noise(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                noise_std=self.noise_std,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(f"{self.env_name}: Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"{self.env_name} Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record(f"eval/{self.env_name}_mean_reward", float(mean_reward))
            self.logger.record(f"eval/{self.env_name}_mean_ep_length", mean_ep_length)
            # self.logger.record(f"eval/{self.env_name}_std_reward", float(std_reward))
            # self.logger.record(f"eval/{self.env_name}_std_ep_length", std_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(f"time/{self.env_name}_total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training