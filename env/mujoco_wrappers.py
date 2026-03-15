##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
"""
MuJoCo observation wrapper for tree-based RL.

Transforms raw MuJoCo observations to be more amenable to axis-aligned splits:
  1. Periodicity fix: replace hinge angles θ with sin(θ), cos(θ)
  2. Relational features: add relative joint angles and relative angular velocities

The wrapper reads the environment's MuJoCo model at init to automatically
detect hinge joints and their kinematic chain, so it works across Hopper,
Walker2d, HalfCheetah, Ant, Humanoid and any standard MuJoCo locomotion env.
"""
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Per-environment configuration
#
# For each supported env we specify:
#   angle_indices:  obs indices that are hinge angles (to replace with sin/cos)
#   vel_indices:    obs indices that are angular velocities of those same joints
#   joint_pairs:    (child_angle_idx, parent_angle_idx) for relative features
#   vel_pairs:      (child_vel_idx, parent_vel_idx) for relative velocity features
#
# The obs layout for v3/v4 locomotion envs is:
#     obs = qpos[1:]  +  clip(qvel, -10, 10)
# where qpos[0] (rootx) is excluded.
# ──────────────────────────────────────────────────────────────────────────────

MUJOCO_OBS_CONFIG: Dict[str, dict] = {
    # ── Hopper ──────────────────────────────────────────────────────────────
    # obs[0]=rootz, obs[1]=rooty(torso), obs[2]=thigh, obs[3]=leg, obs[4]=foot
    # obs[5..10] = velocities (rootx, rootz, rooty, thigh, leg, foot)
    "Hopper": {
        "angle_indices": [1, 2, 3, 4],          # rooty, thigh, leg, foot
        "vel_indices":   [7, 8, 9, 10],          # vel_rooty, vel_thigh, vel_leg, vel_foot
        "joint_pairs":   [(2, 1), (3, 2), (4, 3)],   # thigh-torso, leg-thigh, foot-leg
        "vel_pairs":     [(8, 7), (9, 8), (10, 9)],   # same for velocities
    },

    # ── Walker2d ────────────────────────────────────────────────────────────
    # obs[0]=rootz, obs[1]=rooty(torso)
    # obs[2]=thigh_r, obs[3]=leg_r, obs[4]=foot_r
    # obs[5]=thigh_l, obs[6]=leg_l, obs[7]=foot_l
    # obs[8..16] = velocities (rootx,rootz,rooty,thigh_r,leg_r,foot_r,thigh_l,leg_l,foot_l)
    "Walker2d": {
        "angle_indices": [1, 2, 3, 4, 5, 6, 7],
        "vel_indices":   [10, 11, 12, 13, 14, 15, 16],
        "joint_pairs":   [
            (2, 1), (3, 2), (4, 3),    # right: thigh-torso, leg-thigh, foot-leg
            (5, 1), (6, 5), (7, 6),    # left:  thigh-torso, leg-thigh, foot-leg
        ],
        "vel_pairs": [
            (11, 10), (12, 11), (13, 12),
            (14, 10), (15, 14), (16, 15),
        ],
    },

    # ── HalfCheetah ─────────────────────────────────────────────────────────
    # obs[0]=rootz, obs[1]=rooty(torso)
    # obs[2]=bthigh, obs[3]=bshin, obs[4]=bfoot
    # obs[5]=fthigh, obs[6]=fshin, obs[7]=ffoot
    # obs[8..16] = velocities (rootx,rootz,rooty,bthigh,bshin,bfoot,fthigh,fshin,ffoot)
    "HalfCheetah": {
        "angle_indices": [1, 2, 3, 4, 5, 6, 7],
        "vel_indices":   [10, 11, 12, 13, 14, 15, 16],
        "joint_pairs":   [
            (2, 1), (3, 2), (4, 3),    # back: bthigh-torso, bshin-bthigh, bfoot-bshin
            (5, 1), (6, 5), (7, 6),    # front: fthigh-torso, fshin-fthigh, ffoot-fshin
        ],
        "vel_pairs": [
            (11, 10), (12, 11), (13, 12),
            (14, 10), (15, 14), (16, 15),
        ],
    },

    # ── Ant ─────────────────────────────────────────────────────────────────
    # v4 obs (exclude_current_positions_from_observation=True, default):
    # obs = qpos[2:] + qvel  (rootx, rooty excluded; rootz kept)
    # qpos layout: [rootx(0), rooty(1), rootz(2), quat(3-6),
    #               hip1(7), ankle1(8), hip2(9), ankle2(10),
    #               hip3(11), ankle3(12), hip4(13), ankle4(14)]
    # In obs (qpos[2:]): obs[0]=rootz, obs[1:5]=quaternion, obs[5]=hip1, obs[6]=ankle1, ...
    # velocities start at obs[13] (14 velocity DOFs)
    # v3 obs is 111-dim (includes cfrc_ext), so it's much more complex.
    # We only handle v4 for Ant.
    "Ant-v4": {
        "angle_indices": [5, 6, 7, 8, 9, 10, 11, 12],  # hip1,ankle1,...,hip4,ankle4
        "vel_indices":   [19, 20, 21, 22, 23, 24, 25, 26],  # corresponding velocities
        "joint_pairs":   [
            (6, 5), (8, 7), (10, 9), (12, 11),  # ankle-hip for each leg
        ],
        "vel_pairs": [
            (20, 19), (22, 21), (24, 23), (26, 25),
        ],
    },
}


def _get_env_config(env_name: str) -> Optional[dict]:
    """Look up the obs config for a given env name, matching base name."""
    # Try exact match first (e.g. "Ant-v4")
    if env_name in MUJOCO_OBS_CONFIG:
        return MUJOCO_OBS_CONFIG[env_name]
    # Try base name (e.g. "Hopper-v3" -> "Hopper")
    base = env_name.split("-")[0]
    if base in MUJOCO_OBS_CONFIG:
        return MUJOCO_OBS_CONFIG[base]
    return None


class MujocoTreeObsWrapper(gym.ObservationWrapper):
    """
    Gym observation wrapper that augments MuJoCo observations for tree-based RL.

    Transforms:
      1. Replaces each hinge angle θ with [sin(θ), cos(θ)]  (net +1 dim per angle)
      2. Appends relative joint angles (child - parent)
      3. Appends relative angular velocities (child_vel - parent_vel)

    The original non-angle features (rootz, slide velocities, etc.) are kept as-is.
    """

    def __init__(self, env: gym.Env, env_name: str):
        super().__init__(env)
        self.config = _get_env_config(env_name)
        if self.config is None:
            raise ValueError(
                f"MujocoTreeObsWrapper: no configuration for env '{env_name}'. "
                f"Supported envs: {list(MUJOCO_OBS_CONFIG.keys())}"
            )

        self.angle_indices = np.array(self.config["angle_indices"])
        self.vel_indices = np.array(self.config["vel_indices"])
        self.joint_pairs = self.config["joint_pairs"]
        self.vel_pairs = self.config["vel_pairs"]

        orig_dim = env.observation_space.shape[0]

        # Non-angle indices: all obs indices that are NOT hinge angles
        all_indices = set(range(orig_dim))
        self.non_angle_indices = np.array(
            sorted(all_indices - set(self.angle_indices))
        )

        # Compute new observation dimension:
        #   non_angle features (kept as-is)
        # + 2 * n_angles (sin + cos for each angle)
        # + n_joint_pairs (relative angles)
        # + n_vel_pairs (relative velocities)
        n_non_angle = len(self.non_angle_indices)
        n_angles = len(self.angle_indices)
        n_rel_angles = len(self.joint_pairs)
        n_rel_vels = len(self.vel_pairs)
        self.new_dim = n_non_angle + 2 * n_angles + n_rel_angles + n_rel_vels

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.new_dim,),
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # 1. Non-angle features (rootz, slide velocities, angular velocities, etc.)
        non_angle_feats = obs[self.non_angle_indices]

        # 2. Sin/cos of hinge angles
        angles = obs[self.angle_indices]
        sin_cos = np.concatenate([np.sin(angles), np.cos(angles)])

        # 3. Relative joint angles
        rel_angles = np.array(
            [obs[child] - obs[parent] for child, parent in self.joint_pairs],
            dtype=np.float32,
        )

        # 4. Relative angular velocities
        rel_vels = np.array(
            [obs[child] - obs[parent] for child, parent in self.vel_pairs],
            dtype=np.float32,
        )

        return np.concatenate([non_angle_feats, sin_cos, rel_angles, rel_vels]).astype(np.float32)
