##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
"""Highway-env integration for constrained SPLIT-RL experiments.

This wrapper keeps the setup simple and honest:
  - reward = native highway-env scalar reward
  - cost   = hard failure indicator from native info (crash, optionally off-road)
  - label  = static observation-derived risk label from relative kinematics

The label is intentionally local and uses only relative kinematics features.
It does not change during training.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from utils.helpers import make_cost_vec_env


DEFAULT_HIGHWAY_CONFIG: Dict[str, Any] = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 8,
        "features": ["presence", "x", "y", "vx", "vy"],
        "features_range": {
            "x": [-60.0, 60.0],
            "y": [-12.0, 12.0],
            "vx": [-15.0, 15.0],
            "vy": [-15.0, 15.0],
        },
        "absolute": False,
        "order": "sorted",
        "normalize": True,
        "clip": True,
        "see_behind": True,
    }
}


def _deep_update(base: Dict[str, Any], updates: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not updates:
        return base
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


class HighwaySafetyWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        cost_on_crash: bool = True,
        cost_on_offroad: bool = False,
        offroad_cost: float = 1.0,
        frontal_gap_thresh: float = 0.25,
        immediate_gap_thresh: float = 0.12,
        rear_gap_thresh: float = 0.08,
        same_lane_y_thresh: float = 0.10,
        merge_lane_y_thresh: float = 0.35,
        closing_speed_thresh: float = 0.05,
        lateral_speed_thresh: float = 0.02,
    ):
        super().__init__(env)
        self._cost_on_crash = cost_on_crash
        self._cost_on_offroad = cost_on_offroad
        self._offroad_cost = offroad_cost

        self._frontal_gap_thresh = frontal_gap_thresh
        self._immediate_gap_thresh = immediate_gap_thresh
        self._rear_gap_thresh = rear_gap_thresh
        self._same_lane_y_thresh = same_lane_y_thresh
        self._merge_lane_y_thresh = merge_lane_y_thresh
        self._closing_speed_thresh = closing_speed_thresh
        self._lateral_speed_thresh = lateral_speed_thresh

        obs_type = getattr(self.unwrapped, "observation_type", None)
        if obs_type is None or not hasattr(obs_type, "features"):
            raise ValueError("HighwaySafetyWrapper requires a Kinematics observation.")
        if getattr(obs_type, "absolute", True):
            raise ValueError("HighwaySafetyWrapper requires relative coordinates (absolute=False).")

        self._features = list(obs_type.features)
        self._feature_index = {name: idx for idx, name in enumerate(self._features)}
        required = {"x", "y", "vx", "vy"}
        missing = required.difference(self._feature_index)
        if missing:
            raise ValueError(f"HighwaySafetyWrapper missing required kinematics features: {sorted(missing)}")

    def _reshape_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 2:
            return obs
        if obs.ndim == 1:
            num_features = len(self._features)
            if obs.size % num_features != 0:
                raise ValueError(f"Unexpected flattened highway observation shape: {obs.shape}")
            return obs.reshape(obs.size // num_features, num_features)
        raise ValueError(f"Unsupported highway observation rank: {obs.ndim}")

    def _compute_cost(self, info: Dict[str, Any]) -> float:
        cost = 0.0
        if self._cost_on_crash and bool(info.get("crashed", False)):
            cost += 1.0

        if self._cost_on_offroad:
            rewards = info.get("rewards", {})
            on_road_reward = float(rewards.get("on_road_reward", 1.0))
            if on_road_reward <= 0.0:
                cost += self._offroad_cost
        return cost

    def _compute_safety_label(self, obs: np.ndarray, info: Dict[str, Any]) -> int:
        if bool(info.get("crashed", False)):
            return 1

        data = self._reshape_obs(obs)
        if data.shape[0] <= 1:
            return 0

        other = data[1:]
        if "presence" in self._feature_index:
            present = other[:, self._feature_index["presence"]] > 0.5
        else:
            present = np.ones(other.shape[0], dtype=bool)

        x = other[:, self._feature_index["x"]]
        y = np.abs(other[:, self._feature_index["y"]])
        vx = other[:, self._feature_index["vx"]]
        vy = np.abs(other[:, self._feature_index["vy"]])

        immediate_proximity = present & (y <= self._same_lane_y_thresh) & (np.abs(x) <= self._immediate_gap_thresh)
        frontal_conflict = (
            present
            & (y <= self._same_lane_y_thresh)
            & (x >= -self._rear_gap_thresh)
            & (x <= self._frontal_gap_thresh)
            & (vx < -self._closing_speed_thresh)
        )
        lateral_merge_conflict = (
            present
            & (y > self._same_lane_y_thresh)
            & (y <= self._merge_lane_y_thresh)
            & (x >= -self._rear_gap_thresh)
            & (x <= self._frontal_gap_thresh)
            & (vy > self._lateral_speed_thresh)
        )

        return int(np.any(immediate_proximity | frontal_conflict | lateral_merge_conflict))

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info["cost"] = 0.0
        info["original_reward"] = 0.0
        info["safety_label"] = self._compute_safety_label(obs, info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["original_reward"] = float(reward)
        info["cost"] = self._compute_cost(info)
        info["safety_label"] = self._compute_safety_label(obs, info)
        if "rewards" in info and isinstance(info["rewards"], dict):
            info["reward_terms"] = dict(info["rewards"])
        return obs, float(reward), terminated, truncated, info


def make_highway_raw_env(
    env_name: str = "merge-v0",
    config: Optional[Dict[str, Any]] = None,
    **wrapper_kwargs,
) -> gym.Env:
    import highway_env  # noqa: F401  Registers gymnasium envs.

    env_config = copy.deepcopy(DEFAULT_HIGHWAY_CONFIG)
    _deep_update(env_config, config)
    env = gym.make(env_name, render_mode="rgb_array", config=env_config)
    return HighwaySafetyWrapper(env, **wrapper_kwargs)


def make_highway_vec_env(
    env_name: str = "merge-v0",
    n_envs: int = 1,
    seed: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    **wrapper_kwargs,
):
    env_kwargs = {
        "env_name": env_name,
        "config": config,
        **wrapper_kwargs,
    }
    return make_cost_vec_env(
        make_highway_raw_env,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
    )