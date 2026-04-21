"""SQIL baseline: Soft Q Imitation Learning.

Wraps imitation library's SQIL which internally uses SB3's DQN.
Expert transitions get reward=1, online transitions get reward=0.
"""
import json
from typing import Dict, Optional, Union

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.dqn import DQN

from algos.baselines.data_utils import npz_to_transitions


class SQILBaseline:
    """SQIL baseline that integrates with our training pipeline.

    Uses imitation library's SQIL implementation (wraps SB3 DQN).
    """

    def __init__(
        self,
        env: GymEnv,
        expert_datasets: Union[str, Dict[str, str]],
        learning_rate: float = 1e-4,
        buffer_size: int = 100000,
        batch_size: int = 256,
        learning_starts: int = 1000,
        gamma: float = 0.99,
        target_update_interval: int = 1000,
        exploration_fraction: float = 0.1,
        exploration_final_eps: float = 0.05,
        train_freq: int = 4,
        gradient_steps: int = 1,
        seed: int = 0,
        device: str = "cuda",
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
    ):
        from imitation.algorithms.sqil import SQIL

        if isinstance(expert_datasets, str):
            expert_datasets = json.loads(expert_datasets)

        # imitation's SQILReplayBuffer creates expert_buffer with n_envs=1
        # but inherits the main buffer's n_envs from the VecEnv.
        # This causes shape mismatches during sample() if n_envs > 1.
        n_envs = env.num_envs if hasattr(env, 'num_envs') else 1
        if n_envs > 1:
            raise ValueError(
                f"SQIL requires num_envs=1 (got {n_envs}). "
                "Add 'num_envs: 1' to your sweep YAML."
            )

        # Get a reference env for observation conversion
        # env is a VecEnv, get the first underlying env
        ref_env = env.envs[0] if hasattr(env, 'envs') else env

        # Convert our .npz expert data to imitation Transitions
        demonstrations = npz_to_transitions(expert_datasets, ref_env)
        print(f"SQIL: Loaded {len(demonstrations)} expert transitions")

        rl_kwargs = {
            "learning_rate": learning_rate,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "learning_starts": learning_starts,
            "gamma": gamma,
            "target_update_interval": target_update_interval,
            "exploration_fraction": exploration_fraction,
            "exploration_final_eps": exploration_final_eps,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            "seed": seed,
            "device": device,
            "verbose": verbose,
            "tensorboard_log": tensorboard_log,
        }

        self.sqil = SQIL(
            venv=env,
            demonstrations=demonstrations,
            policy="MlpPolicy",
            rl_algo_class=DQN,
            rl_kwargs=rl_kwargs,
        )

        # Fix imitation 1.0.0 bug: expert_buffer is created without a device
        # arg, so it defaults to "auto" (→ CUDA if available) while the main
        # buffer uses the device from rl_kwargs. This causes device mismatch
        # in SQILReplayBuffer.sample() → th.cat().
        sqil_buf = self.sqil.rl_algo.replay_buffer
        if hasattr(sqil_buf, 'expert_buffer'):
            import torch as th
            target_device = sqil_buf.device
            sqil_buf.expert_buffer.device = target_device

        # Expose the underlying DQN for compatibility
        self.policy = self.sqil.policy

    def learn(self, total_timesteps: int, callback=None, log_interval: int = 4,
              progress_bar: bool = False, **kwargs):
        """Train SQIL agent."""
        self.sqil.train(total_timesteps=total_timesteps, callback=callback)
        return self

    def save(self, path: str):
        """Save the underlying DQN model."""
        self.sqil.policy.save(path)

    @classmethod
    def load(cls, path: str, env=None):
        return DQN.load(path, env=env)
