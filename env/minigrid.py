##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.world_object import Ball, Box, Key, Goal
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from gymnasium.core import ActType, ObsType

from typing import Any
from gymnasium.envs.registration import register


import numpy as np

class OODFetchEnv(MiniGridEnv):
    def __init__(self, size=8, numObjs=3, max_steps: int | None = None, undersampling_rate: float = None, **kwargs):
        self.numObjs = 3
        self.size = 8
        self.obj_types = ["ball"]
        self.colors = {
            "red": np.array([255, 0, 0]),
            "green": np.array([0, 255, 0]),
            "blue": np.array([0, 0, 255]),
        }

        self.undersampling_rate = undersampling_rate
        self.color_names = sorted(list(self.colors.keys()))

        MISSION_SYNTAX = [
            "get a"
        ]
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[MISSION_SYNTAX, self.color_names, self.obj_types],
        )

        if max_steps is None:
            max_steps = 5 * self.size**2

        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    def _calculate_probabilities(self):
        """Calculate probabilities for undersampling one color."""
        num_colors = len(self.color_names)
        if self.undersampling_rate is None or self.undersampling_rate <= 0:
            return [1 / float(num_colors)] * num_colors  # Uniform distribution

        # Solve for x based on the equation 2x + x / undersampling_rate = 1
        y = float(self.undersampling_rate)
        x = y / (float(num_colors - 1) * y + 1)
        probabilities = [x, x, x / y]
        return probabilities

    def _rand_color(self):
        """Sample a color with custom probabilities."""
        probabilities = self._calculate_probabilities()
        return np.random.choice(self.color_names, p=probabilities)
    @staticmethod
    def _gen_mission(syntax: str, color: str, obj_type: str):
        return f"{syntax} {color} {obj_type}"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        objs = []

        # For each object to be generated
        while len(objs) < self.numObjs:
            objColor = self._rand_color()
            obj = Ball(objColor)

            self.place_obj(obj)
            objs.append(obj)

        # Randomize the player start position and orientation
        self.place_agent()
        # Choose a random object to be picked up
        target = objs[self._rand_int(0, len(objs))]
        self.targetType = target.type
        self.targetColor = target.color

        descStr = f"{self.targetColor} {self.targetType}"

        # Generate the mission string
        self.mission = "get a %s" % descStr
        assert hasattr(self, "mission")

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if self.carrying:
            if (
                self.carrying.color == self.targetColor
                and self.carrying.type == self.targetType
            ):
                reward = self._reward()
                terminated = True
            else:
                reward = 0
                terminated = True

        return obs, reward, terminated, truncated, info



def register_minigrid_tests():
    # PutNear
    # ----------------------------------------

    register(
        id="MiniGrid-OODFetch-8x8-N3-6x6-N2-v0",
        entry_point="env.minigrid:OODFetchEnv",
    )
