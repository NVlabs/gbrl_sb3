##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from gymnasium.envs.registration import register
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Goal, Key
from minigrid.minigrid_env import MiniGridEnv


class OODFetchEnv(MiniGridEnv):
    def __init__(self, size=8, numObjs=3, max_steps: int | None = None, undersampling_rate: float = None, test_box_idx: int = None, **kwargs):
        self.numObjs = 3
        self.size = 8
        self.obj_types = ["ball"]
        self.colors = {
            "red": np.array([255, 0, 0]),
            "green": np.array([0, 255, 0]),
            "blue": np.array([0, 0, 255]),
        }
        if test_box_idx is not None:
            assert test_box_idx >= 0 and test_box_idx < 3
        self.test_box_idx = test_box_idx
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
    
    def _rand_obj(self):
        """Sample a color with custom probabilities."""
        probabilities = self._calculate_probabilities()
        return np.random.choice([0, 1, 2], p=probabilities)
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
        obs_red = Ball('red')
        obs_green = Ball('green')
        obs_blue = Ball('blue')
        objs = [obs_red, obs_green, obs_blue]
        self.place_obj(obs_red)
        self.place_obj(obs_green)
        self.place_obj(obs_blue)
        #     objs.append(obj)

        # Randomize the player start position and orientation
        self.place_agent()
        # Choose a random object to be picked up
        if self.test_box_idx is None:
            target = objs[self._rand_obj()]
        else:
            target = objs[self.test_box_idx]
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
        id="MiniGrid-OODFetch-8x8-N3-v0",
        entry_point="env.minigrid:OODFetchEnv",
    )

class DistanceFetchEnv(MiniGridEnv):
    def __init__(self, size=8, numObjs=3, max_steps: int | None = None, undersampling_rate: float = None, test_box_idx: int = None, **kwargs):
        self.numObjs = 3
        self.size = 8
        self.obj_types = ["ball"]
        self.colors = {
            "red": np.array([255, 0, 0]),
            "green": np.array([0, 255, 0]),
            "blue": np.array([0, 0, 255]),
        }
        if test_box_idx is not None:
            assert test_box_idx >= 0 and test_box_idx < 3
        self.test_box_idx = test_box_idx
        self.undersampling_rate = undersampling_rate
        self.color_names = sorted(list(self.colors.keys()))

        MISSION_SYNTAX = [
            "get closest ball",
            "get farthest ball",
        ]
        self.mission_syntax = MISSION_SYNTAX
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[MISSION_SYNTAX],
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
        image_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {
                "image": image_observation_space,
                "direction": spaces.Discrete(4),
                "mission": mission_space,
                "distances": spaces.Box(
                            low=0,
                            high=np.inf,
                            shape=(len(self.colors),),
                            dtype=float,
                        )
                }
        )

    def _calculate_probabilities(self):
        """Calculate probabilities for undersampling one color."""
        num_missions = len(self.mission_syntax)
        if self.undersampling_rate is None or self.undersampling_rate <= 0:
            return [1 / float(num_missions)] * num_missions  # Uniform distribution

        # Solve for x based on the equation 2x + x / undersampling_rate = 1
        y = float(self.undersampling_rate)
        x = y / (float(num_missions - 1) * y + 1)
        probabilities = [x, x / y]
        return probabilities

    def _rand_mission(self):
        """Sample a color with custom probabilities."""
        probabilities = self._calculate_probabilities()
        return np.random.choice(self.mission_syntax, p=probabilities)
    
    @staticmethod
    def _gen_mission(syntax: str):
        return f"{syntax}"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        objs = []
        obs_red = Ball('red')
        obs_green = Ball('green')
        obs_blue = Ball('blue')
        objs = [obs_red, obs_green, obs_blue]
        red_pos = self.place_obj(obs_red)
        green_pos = self.place_obj(obs_green)
        blue_pos = self.place_obj(obs_blue)
        
        self.obj_pos = np.array([red_pos, green_pos, blue_pos])

        # Randomize the player start position and orientation
        agent_pos = self.place_agent()
        agent_pos = np.array(agent_pos)
        self.distances = np.linalg.norm((self.obj_pos - agent_pos), axis=1)
        # Generate the mission string
        self.mission =  self._rand_mission()
        if 'farthest' in self.mission:
            target_idx = np.argmax(self.distances)
        else:
            target_idx = np.argmin(self.distances)
        target = objs[target_idx]
        self.targetType = target.type
        self.targetColor = target.color
        # print(self.mission)
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
        obs['distances'] = self.distances
        return obs, reward, terminated, truncated, info
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        obs, info = super().reset(seed=seed, options=options)
        obs['distances'] = self.distances
        return obs, info



def register_minigrid_tests():
    # PutNear
    # ----------------------------------------

    register(
        id="MiniGrid-OODFetch-8x8-N3-v0",
        entry_point="env.minigrid:OODFetchEnv",
    )

    register(
        id="MiniGrid-DistanceFetch-8x8-N3-v0",
        entry_point="env.minigrid:DistanceFetchEnv",
    )
