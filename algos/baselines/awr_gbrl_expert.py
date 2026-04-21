"""AWR GBRL + Expert Data baseline (single-objective ablation).

Same as AWR_GBRL but with expert demonstrations pre-filled into the
CategoricalAWRReplayBuffer. All data (online + expert) is treated
identically — no label split, no multi-objective gradients.

This baseline tests: "Does the multi-objective label split in
Split-AWR actually help, or is mixing expert data enough?"
"""
import json
from typing import Dict, Optional, Union

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.type_aliases import GymEnv

from algos.awr import AWR_GBRL
from algos.split_awr import load_expert_dataset, EXPERT_LABEL_MAP


class AWR_GBRL_Expert(AWR_GBRL):
    """AWR GBRL with expert demonstrations mixed into the replay buffer.

    Single-objective: expert data gets the same AWR treatment as online data.
    No BC loss, no label split. Expert data protected from overwrite.
    """

    def __init__(self, env: Union[GymEnv, str],
                 expert_datasets: Optional[Dict[str, str]] = None,
                 expert_buffer_per_label: int = 0,
                 **kwargs):
        # Store expert config before super().__init__ (which calls _setup_model)
        self._expert_datasets_config = expert_datasets or {}
        self._expert_buffer_per_label = expert_buffer_per_label

        super().__init__(env=env, **kwargs)

        # Pre-fill expert data into the buffer
        if self._expert_datasets_config:
            self._prefill_expert_data()

    def _prefill_expert_data(self):
        """Load expert datasets and pre-fill the CategoricalAWRReplayBuffer."""
        total_added = 0
        for name, path in self._expert_datasets_config.items():
            label = EXPERT_LABEL_MAP.get(name)
            if label is None:
                print(f"Warning: Unknown expert '{name}', skipping label map. Using label=1.")
                label = 1

            print(f"Loading expert: {name} from {path}")
            expert = load_expert_dataset(
                path, label=label,
                max_transitions=self._expert_buffer_per_label)

            n = len(expert['actions'])
            end_pos = self.replay_buffer.pos + n
            if end_pos > self.replay_buffer.buffer_size:
                print(f"  Warning: Expert data ({n}) exceeds buffer capacity. Truncating.")
                n = self.replay_buffer.buffer_size - self.replay_buffer.pos
                if n <= 0:
                    break

            sl = slice(self.replay_buffer.pos, self.replay_buffer.pos + n)
            self.replay_buffer.observations[sl, 0] = expert['observations'][:n]
            self.replay_buffer.next_observations[sl, 0] = expert['next_observations'][:n]
            self.replay_buffer.actions[sl, 0] = expert['actions'][:n]
            self.replay_buffer.rewards[sl, 0] = expert['rewards'][:n].flatten()
            self.replay_buffer.dones[sl, 0] = expert['dones'][:n].flatten()
            self.replay_buffer.timeouts[sl, 0] = 0.0

            self.replay_buffer.pos = self.replay_buffer.pos + n
            self.replay_buffer.valid_pos = self.replay_buffer.pos

            total_added += n
            print(f"  Loaded {n} transitions (buffer pos now {self.replay_buffer.pos})")

        # Store expert boundary to protect from overwrite
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
