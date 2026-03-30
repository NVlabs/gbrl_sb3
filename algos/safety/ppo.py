##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
from typing import Any, Dict, Optional, Type, TypeVar, Union

import torch as th
import sys
import time


from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import safe_mean

from stable_baselines3.ppo.ppo import PPO as SB3_PPO

SelfPPO = TypeVar("SelfPPO", bound="VanillaPPO")

class VanillaPPO(SB3_PPO):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
    

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
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
                    if "c" in self.ep_info_buffer[0]:
                        self.logger.record("rollout/ep_cost_mean",
                            safe_mean([ep_info["c"] for ep_info in self.ep_info_buffer]))
                    if "max_queued" in self.ep_info_buffer[0]:
                        self.logger.record("rollout/ep_max_queued_mean",
                            safe_mean([ep_info["max_queued"] for ep_info in self.ep_info_buffer]))
                    if "max_wait" in self.ep_info_buffer[0]:
                        self.logger.record("rollout/ep_max_wait_mean",
                            safe_mean([ep_info["max_wait"] for ep_info in self.ep_info_buffer]))
                    if "cost_queue" in self.ep_info_buffer[0]:
                        self.logger.record("rollout/ep_cost_queue_mean",
                            safe_mean([ep_info["cost_queue"] for ep_info in self.ep_info_buffer]))
                    if "cost_wait" in self.ep_info_buffer[0]:
                        self.logger.record("rollout/ep_cost_wait_mean",
                            safe_mean([ep_info["cost_wait"] for ep_info in self.ep_info_buffer]))
                    if "cost_saturation" in self.ep_info_buffer[0]:
                        self.logger.record("rollout/ep_cost_saturation_mean",
                            safe_mean([ep_info["cost_saturation"] for ep_info in self.ep_info_buffer]))
                    if "normalized_score" in self.ep_info_buffer[0]:
                        self.logger.record("rollout/normalized_score",
                            safe_mean([ep_info["normalized_score"] for ep_info in self.ep_info_buffer]))
                    if "completion_rate" in self.ep_info_buffer[0]:
                        self.logger.record("rollout/completion_rate",
                            safe_mean([ep_info["completion_rate"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self