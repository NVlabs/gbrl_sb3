"""AWR NN + Expert Data baseline (neural network ablation).

Same as AWR (neural network) but with expert demonstrations pre-filled
into the AWRReplayBuffer. All data (online + expert) is treated
identically — single-objective AWR.

This baseline tests: "Does GBRL matter vs neural networks when
mixing expert data into AWR?"
"""
import json
from typing import Any, Dict, Optional, Type, Union

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.type_aliases import GymEnv

from algos.awr_nn import AWR
from algos.baselines.data_utils import npz_to_flat_observations


class AWR_NN_Expert(AWR):
    """AWR NN with expert demonstrations mixed into the replay buffer.

    Single-objective: expert data gets the same AWR treatment as online data.
    Expert data protected from overwrite.
    """

    def __init__(self,
                 policy: Union[str, Type],
                 env: Union[GymEnv, str],
                 expert_datasets: Optional[Dict[str, str]] = None,
                 **kwargs):
        self._expert_datasets_config = expert_datasets or {}

        super().__init__(policy=policy, env=env, **kwargs)

        if self._expert_datasets_config:
            self._prefill_expert_data()

    def _prefill_expert_data(self):
        """Load expert datasets and pre-fill the AWRReplayBuffer with flat obs."""
        # Get reference env for FlatObsWrapper conversion
        ref_env = self.env.envs[0] if hasattr(self.env, 'envs') else self.env

        total_added = 0
        for name, path in self._expert_datasets_config.items():
            print(f"Loading expert: {name} from {path}")
            data = npz_to_flat_observations(path, ref_env)

            obs = data['observations']
            acts = data['actions']
            rewards = data['rewards']
            dones = data['dones']

            n = len(acts)
            end_pos = self.replay_buffer.pos + n
            if end_pos > self.replay_buffer.buffer_size:
                print(f"  Warning: Expert data ({n}) exceeds buffer. Truncating.")
                n = self.replay_buffer.buffer_size - self.replay_buffer.pos
                if n <= 0:
                    break

            # Build next_obs
            next_obs = np.zeros_like(obs[:n])
            next_obs[:-1] = obs[1:n]
            next_obs[-1] = obs[n - 1]
            done_indices = np.where(dones[:n])[0]
            for idx in done_indices:
                next_obs[idx] = obs[idx]

            sl = slice(self.replay_buffer.pos, self.replay_buffer.pos + n)
            self.replay_buffer.observations[sl, 0] = obs[:n]
            self.replay_buffer.next_observations[sl, 0] = next_obs
            self.replay_buffer.actions[sl, 0] = acts[:n].reshape(-1, 1).astype(np.float32)
            self.replay_buffer.rewards[sl, 0] = rewards[:n]
            self.replay_buffer.dones[sl, 0] = dones[:n].astype(np.float32)
            self.replay_buffer.timeouts[sl, 0] = 0.0

            self.replay_buffer.pos = self.replay_buffer.pos + n
            self.replay_buffer.valid_pos = self.replay_buffer.pos

            total_added += n
            print(f"  Loaded {n} transitions (buffer pos now {self.replay_buffer.pos})")

        self._expert_boundary = self.replay_buffer.pos
        online_capacity = self.replay_buffer.buffer_size - self._expert_boundary
        print(f"Buffer: {total_added} expert + {online_capacity} online capacity "
              f"(total {self.replay_buffer.buffer_size})")

    def _store_transition(self, *args, **kwargs):
        """Override to protect expert data region from being overwritten."""
        super()._store_transition(*args, **kwargs)
        if hasattr(self, '_expert_boundary') and self._expert_boundary > 0:
            if self.replay_buffer.pos < self._expert_boundary:
                self.replay_buffer.pos = self._expert_boundary
