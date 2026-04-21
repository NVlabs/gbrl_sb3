"""RLPD baseline: RL with Prior Data.

Simplest possible demo-augmented RL: add expert demonstrations directly
to the DQN replay buffer, then train with a high update-to-data (UTD) ratio.

Reference: Ball et al., "Efficient Online Reinforcement Learning with
Offline Data", ICLR 2023.

This does NOT require the imitation library — it uses SB3's DQN directly.
"""
import json
from typing import Dict, Optional, Union

import numpy as np
from stable_baselines3.dqn import DQN
from stable_baselines3.common.type_aliases import GymEnv

from algos.baselines.data_utils import npz_to_flat_observations


class RLPDBaseline:
    """RLPD: Just add demonstrations to DQN's replay buffer.

    1. Create DQN agent
    2. Pre-fill replay buffer with expert demonstrations
    3. Train normally (expert data gets sampled alongside online data)
    """

    def __init__(
        self,
        env: GymEnv,
        expert_datasets: Union[str, Dict[str, str]],
        learning_rate: float = 1e-4,
        buffer_size: int = 100000,
        batch_size: int = 256,
        learning_starts: int = 0,  # Start immediately (we have expert data)
        gamma: float = 0.99,
        target_update_interval: int = 1000,
        exploration_fraction: float = 0.1,
        exploration_final_eps: float = 0.05,
        train_freq: int = 4,
        gradient_steps: int = 4,  # Higher UTD ratio (key for RLPD)
        seed: int = 0,
        device: str = "cuda",
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
        _init_setup_model: bool = True,
    ):
        if isinstance(expert_datasets, str):
            expert_datasets = json.loads(expert_datasets)

        self.expert_datasets = expert_datasets
        self.env = env

        self.dqn = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            learning_starts=learning_starts,
            gamma=gamma,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            seed=seed,
            device=device,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
        )

        # Pre-fill the replay buffer with expert data
        self._load_expert_data()

    def _load_expert_data(self):
        """Add expert demonstrations to DQN's replay buffer."""
        ref_env = self.env.envs[0] if hasattr(self.env, 'envs') else self.env

        # replay_buffer.add() reshapes arrays to (n_envs, ...).
        # We add one transition at a time, so temporarily set n_envs=1.
        buf = self.dqn.replay_buffer
        orig_n_envs = buf.n_envs
        buf.n_envs = 1

        total_added = 0
        for name, path in self.expert_datasets.items():
            data = npz_to_flat_observations(path, ref_env)
            obs = data['observations']
            acts = data['actions']
            rewards = data['rewards']
            dones = data['dones']

            n = len(acts)

            # Build next_obs
            next_obs = np.zeros_like(obs)
            next_obs[:-1] = obs[1:]
            next_obs[-1] = obs[-1]

            # Fix episode boundaries
            done_indices = np.where(dones)[0]
            for idx in done_indices:
                next_obs[idx] = obs[idx]

            # Add to replay buffer one transition at a time
            for i in range(n):
                buf.add(
                    obs=obs[i],
                    next_obs=next_obs[i],
                    action=np.array([acts[i]]),
                    reward=np.array([rewards[i]]),
                    done=np.array([dones[i]]),
                    infos=[{}],
                )

            total_added += n
            print(f"RLPD: Added {n} transitions from {name}")

        buf.n_envs = orig_n_envs
        print(f"RLPD: Total {total_added} expert transitions in replay buffer "
              f"(buffer pos: {buf.pos})")

    def learn(self, total_timesteps: int, callback=None, log_interval: int = 4,
              progress_bar: bool = False, **kwargs):
        """Train DQN with expert data already in buffer."""
        self.dqn.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            progress_bar=progress_bar,
        )
        return self

    @property
    def policy(self):
        return self.dqn.policy

    def save(self, path: str):
        self.dqn.save(path)

    @classmethod
    def load(cls, path: str, env=None):
        return DQN.load(path, env=env)
