"""Append the one-hot guidance label to the observation.

This is the cheapest label-aware control for the constrained-RL baselines: it
hands PPO-Lag / IPO / CPO / CUP the same guidance label Split-RL routes on,
without touching any algorithm internals.  Because the label is a deterministic
function of the observation in these environments, the expected outcome is a
no-op -- and that no-op is the point.  It is the empirical form of the claim
that the label carries no information the baselines did not already have.

``info['safety_label']`` describes the observation returned alongside it, so the
one-hot is simply concatenated onto that observation.

Two observation layouts are supported:

* **Numeric** (flat float ``Box``): the one-hot is concatenated and the array
  stays float32.  This is the MiniGrid FlatObs path, SUMO, and CityLearn.
* **Categorical** (MiniGrid's ``MiniGridCategoricalObservationWrapper``, whose
  columns are byte-string categories): appending numeric columns makes the
  array *mixed*, so it is rebuilt as ``dtype=object`` and ``is_mixed`` is set.
  GBRL inspects element dtype per column in ``process_array()``, so the label
  columns get numerical threshold splits while the grid columns keep equality
  splits.  Without ``is_mixed`` the downstream buffer would allocate
  ``categorical_dtype`` and silently stringify the one-hot.
"""
from typing import Any, Dict, List

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper

LABEL_KEY = "safety_label"


class VecLabelObsWrapper(VecEnvWrapper):
    """Concatenate a one-hot guidance label onto every observation.

    :param venv: the vectorised env to wrap
    :param n_label_values: label alphabet size (2, or 3 when blending is used)
    """

    def __init__(self, venv: VecEnv, n_label_values: int = 2):
        obs_space = venv.observation_space
        if not isinstance(obs_space, spaces.Box) or len(obs_space.shape) != 1:
            raise ValueError(
                "VecLabelObsWrapper expects a flat Box observation space, got "
                f"{obs_space}. Image/dict observations are not supported."
            )

        # Categorical envs declare a Box space but emit byte-string columns, so
        # the layout has to be read off the flag rather than the space.
        self._categorical = bool(getattr(venv, 'is_categorical', False)
                                 or getattr(venv, 'is_mixed', False))

        low = np.concatenate([obs_space.low, np.zeros(n_label_values, dtype=obs_space.dtype)])
        high = np.concatenate([obs_space.high, np.ones(n_label_values, dtype=obs_space.dtype)])
        super().__init__(venv, observation_space=spaces.Box(low=low, high=high, dtype=obs_space.dtype))

        self.n_label_values = n_label_values
        if self._categorical:
            # Read by the algorithms (e.g. algos/safety/ppo_lag_gbrl.py:224) to
            # pick an object-dtype rollout buffer. Set on the wrapper itself so
            # VecEnvWrapper.__getattr__ does not resolve them to the inner env's
            # is_mixed=False, which is stale once the one-hot is appended.
            self.is_categorical = True
            self.is_mixed = True

    def _one_hot(self, labels: np.ndarray) -> np.ndarray:
        labels = np.clip(np.asarray(labels, dtype=np.int64), 0, self.n_label_values - 1)
        one_hot = np.zeros((labels.shape[0], self.n_label_values), dtype=np.float32)
        one_hot[np.arange(labels.shape[0]), labels] = 1.0
        return one_hot

    def _labels_from_infos(self, infos: List[Dict[str, Any]]) -> np.ndarray:
        return np.array([int(info.get(LABEL_KEY, 0)) for info in infos], dtype=np.int64)

    def _augment(self, obs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        one_hot = self._one_hot(labels)
        if not self._categorical:
            return np.concatenate([np.asarray(obs, dtype=np.float32), one_hot], axis=1)

        # Mixed layout: keep the categorical columns as they came (casting them
        # would destroy the category strings) and append the one-hot as numeric
        # columns in the same object array.
        obs = np.asarray(obs)
        n_obs_cols = obs.shape[1]
        mixed = np.empty((obs.shape[0], n_obs_cols + self.n_label_values), dtype=object)
        mixed[:, :n_obs_cols] = obs
        mixed[:, n_obs_cols:] = one_hot.astype(np.float64)
        return np.ascontiguousarray(mixed)

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        # No infos are returned by VecEnv.reset, so the initial state is labelled
        # reward-only. One transition out of a rollout is immaterial here.
        return self._augment(obs, np.zeros(self.num_envs, dtype=np.int64))

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        obs = self._augment(obs, self._labels_from_infos(infos))
        # Terminal observations are consumed for value bootstrapping, so they
        # need the same dimensionality as everything else.
        for info in infos:
            terminal = info.get("terminal_observation")
            if terminal is not None:
                dtype = object if self._categorical else np.float32
                info["terminal_observation"] = self._augment(
                    np.asarray(terminal, dtype=dtype).reshape(1, -1),
                    np.array([int(info.get(LABEL_KEY, 0))], dtype=np.int64),
                )[0]
        return obs, rewards, dones, infos
