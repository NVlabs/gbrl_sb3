"""BC baseline: Behavioral Cloning, optionally followed by PPO fine-tuning.

Uses imitation library's BC for pre-training, then SB3 PPO for fine-tuning.
BC+finetune is the universal IL+RL baseline that reviewers always expect.
"""
import json
from typing import Dict, Optional, Union

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.type_aliases import GymEnv

from algos.baselines.data_utils import npz_to_transitions


class BCBaseline:
    """Pure Behavioral Cloning baseline.

    Pre-trains a policy on expert demonstrations via supervised learning.
    No environment interaction during training.
    """

    def __init__(
        self,
        env: GymEnv,
        expert_datasets: Union[str, Dict[str, str]],
        batch_size: int = 256,
        n_epochs: int = 50,
        lr: float = 1e-3,
        ent_weight: float = 1e-3,
        l2_weight: float = 0.0,
        seed: int = 0,
        device: str = "cuda",
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
    ):
        from imitation.algorithms.bc import BC

        if isinstance(expert_datasets, str):
            expert_datasets = json.loads(expert_datasets)

        ref_env = env.envs[0] if hasattr(env, 'envs') else env
        demonstrations = npz_to_transitions(expert_datasets, ref_env)
        print(f"BC: Loaded {len(demonstrations)} expert transitions")

        rng = np.random.default_rng(seed)

        self.bc = BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=demonstrations,
            batch_size=batch_size,
            rng=rng,
            ent_weight=ent_weight,
            l2_weight=l2_weight,
            device=device,
        )

        self.n_epochs = n_epochs
        self.env = env
        self.verbose = verbose

    def learn(self, total_timesteps: int = 0, callback=None,
              log_interval: int = 4, progress_bar: bool = False, **kwargs):
        """Train BC (ignores total_timesteps, uses n_epochs)."""
        if self.verbose:
            print(f"BC: Training for {self.n_epochs} epochs")
        self.bc.train(n_epochs=self.n_epochs, progress_bar=progress_bar)
        return self

    @property
    def policy(self):
        return self.bc.policy

    def save(self, path: str):
        self.bc.save_policy(path)


class BCPPOFinetuneBaseline:
    """BC pre-training followed by PPO fine-tuning.

    Phase 1: Train BC on expert demonstrations (n_epochs)
    Phase 2: Fine-tune the BC policy with PPO on the environment
    """

    def __init__(
        self,
        env: GymEnv,
        expert_datasets: Union[str, Dict[str, str]],
        # BC params
        bc_epochs: int = 50,
        bc_batch_size: int = 256,
        bc_lr: float = 1e-3,
        bc_ent_weight: float = 1e-3,
        # PPO params
        ppo_lr: float = 3e-4,
        ppo_n_steps: int = 2048,
        ppo_batch_size: int = 64,
        ppo_n_epochs: int = 10,
        ppo_gamma: float = 0.99,
        ppo_ent_coef: float = 0.01,
        ppo_clip_range: float = 0.2,
        seed: int = 0,
        device: str = "cuda",
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
    ):
        from imitation.algorithms.bc import BC
        from stable_baselines3 import PPO

        if isinstance(expert_datasets, str):
            expert_datasets = json.loads(expert_datasets)

        ref_env = env.envs[0] if hasattr(env, 'envs') else env
        demonstrations = npz_to_transitions(expert_datasets, ref_env)
        print(f"BC+PPO: Loaded {len(demonstrations)} expert transitions")

        rng = np.random.default_rng(seed)

        self.bc = BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=demonstrations,
            batch_size=bc_batch_size,
            rng=rng,
            ent_weight=bc_ent_weight,
            device=device,
        )

        self.bc_epochs = bc_epochs
        self.env = env
        self.verbose = verbose
        self.seed = seed
        self.device = device
        self.tensorboard_log = tensorboard_log

        # PPO kwargs saved for Phase 2
        self.ppo_kwargs = {
            "learning_rate": ppo_lr,
            "n_steps": ppo_n_steps,
            "batch_size": ppo_batch_size,
            "n_epochs": ppo_n_epochs,
            "gamma": ppo_gamma,
            "ent_coef": ppo_ent_coef,
            "clip_range": ppo_clip_range,
            "seed": seed,
            "device": device,
            "verbose": verbose,
            "tensorboard_log": tensorboard_log,
        }

        self._ppo = None

    def learn(self, total_timesteps: int, callback=None,
              log_interval: int = 4, progress_bar: bool = False, **kwargs):
        """Phase 1: BC pre-training, Phase 2: PPO fine-tuning."""
        from stable_baselines3 import PPO

        # Phase 1: BC
        if self.verbose:
            print(f"BC+PPO Phase 1: BC pre-training for {self.bc_epochs} epochs")
        self.bc.train(n_epochs=self.bc_epochs, progress_bar=progress_bar)

        # Phase 2: PPO with the BC-initialized policy
        if self.verbose:
            print(f"BC+PPO Phase 2: PPO fine-tuning for {total_timesteps} steps")

        self._ppo = PPO(
            policy="MlpPolicy",
            env=self.env,
            **self.ppo_kwargs,
        )

        # Copy BC weights into PPO's policy network
        bc_state = self.bc.policy.state_dict()
        ppo_state = self._ppo.policy.state_dict()
        # Only copy matching keys
        matched = {k: v for k, v in bc_state.items() if k in ppo_state and v.shape == ppo_state[k].shape}
        if self.verbose:
            print(f"  Copied {len(matched)}/{len(ppo_state)} weight tensors from BC to PPO")
        ppo_state.update(matched)
        self._ppo.policy.load_state_dict(ppo_state)

        self._ppo.learn(total_timesteps=total_timesteps, callback=callback,
                        log_interval=log_interval, progress_bar=progress_bar)
        return self

    @property
    def policy(self):
        if self._ppo is not None:
            return self._ppo.policy
        return self.bc.policy

    def save(self, path: str):
        if self._ppo is not None:
            self._ppo.save(path)
        else:
            self.bc.save_policy(path)
