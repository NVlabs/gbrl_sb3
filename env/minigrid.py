##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
from __future__ import annotations

from typing import Any, List

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
    def __init__(self, size=8, numObjs=3, max_steps: int | None = None, distances_type: str = 'raw', mission_probs: List | None = None, test_box_idx: int = None, **kwargs):
        self.numObjs = 3
        self.size = 8
        self.obj_types = ["ball"]
        self.colors = {
            "red": np.array([255, 0, 0]),
            "green": np.array([0, 255, 0]),
            "blue": np.array([0, 0, 255]),
        }
        assert distances_type in ['raw', 'rank', 'categorical']
        if test_box_idx is not None:
            assert test_box_idx >= 0 and test_box_idx < 3
        self.test_box_idx = test_box_idx
        self.mission_probs = [1.0, 0, 0, 0] if mission_probs is None else mission_probs
        self.mission_probs = [float(prob) / sum(self.mission_probs) for prob in self.mission_probs]
        self.color_names = sorted(list(self.colors.keys()))
        self.distances_type = distances_type

        MISSION_SYNTAX = [
            "get closest ball",
            "get closest ball ignoring the red ball",
            "get closest ball ignoring the",
            "get farthest ball",
        ]

        self.mission_syntax = MISSION_SYNTAX
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[MISSION_SYNTAX, self.color_names],
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
                }
        )
        self.observation_space['distances'] = spaces.Box(
                        low=0,
                        high=np.inf,
                        shape=(len(self.colors),),
                        dtype=float,
                    )

    def _rand_mission(self):
        """Sample a color with custom probabilities."""
        mission = np.random.choice(self.mission_syntax, p=self.mission_probs)
        if mission == 'get closest ball ignoring the':
            mission += ' ' + np.random.choice(self.color_names) + ' ball'
        return mission
    
    @staticmethod
    def _gen_mission(syntax: str, color: str = None):
        if color is None:
            return f"{syntax}"
        return f"{syntax} {color} ball"

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
        distances = np.linalg.norm((self.obj_pos - agent_pos), axis=1)
        sorted_pairs = sorted(zip(distances, objs), key=lambda x: x[0])
        # Extract the sorted distances and objects
        # sorted_distances = [pair[0] for pair in sorted_pairs]
        sorted_objs = [pair[1] for pair in sorted_pairs]
        sorted_colors = [obj.color for obj in sorted_objs]
        # Generate the mission string
        self.mission =  self._rand_mission()
        if self.mission == 'get closest ball':
            target_idx = 0
        elif self.mission == 'get closest ball ignoring the red ball':
            target_idx = 0 if sorted_objs[0].color != 'red' else 1
        elif self.mission == 'get farthest ball':
            target_idx = -1
        else: 
            color = self.mission.split("the")[-1].split("ball")[0].replace(" ", "")
            target_idx = [idx for idx, obj in enumerate(sorted_objs) if color != obj.color][0]
        target = sorted_objs[target_idx]
        if self.distances_type == 'raw':
            self.distances = distances 
        elif self.distances_type == 'rank':
            self.distances = np.zeros_like(distances)
            colors_to_idx = {'red': 0, 'green': 1, 'blue': 2}
            for i in range(len(sorted_objs)):
                self.distances[colors_to_idx[sorted_objs[i].color]] = i + 1
        self.targetType = target.type
        self.targetColor = target.color
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

class SpuriousFetchEnv(MiniGridEnv):
    def __init__(self, size=8, numObjs=3, max_steps: int | None = None, train: bool = True, randomize: bool = False, mission_based: bool = False, test_box_idx: int = None, **kwargs):
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
        self.color_names = sorted(list(self.colors.keys()))
        self.randomize = randomize
        self.train = train
        self.mission_based = mission_based
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

    def _rand_color(self):
        """Sample a color with custom probabilities."""
        return np.random.choice(self.color_names)
    
    def _rand_obj(self):
        """Sample a color with custom probabilities."""
        return np.random.choice([0, 1, 2])
    @staticmethod
    def _gen_mission(syntax: str, color: str, obj_type: str):
        return f"{syntax} {color} {obj_type}"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        target_idx = np.random.choice([0, 1, 2])
        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)
        # Place a goal square in the bottom-right corner
        if self.train:
            if self.randomize:
                self.place_obj(Box('red'))
            else:
                place = np.random.choice([True, False])
                if place:
                    if self.mission_based:
                        if target_idx == 0:
                            self.put_obj(Box('red'), width - 2, height - 2)
                        elif target_idx == 1:
                            self.put_obj(Box('red'), width - 2, 1)
                    else:
                        self.put_obj(Box('red'), width - 2, height - 2)

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
        target = objs[target_idx]

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
        kwargs={"size": 8, "numObjs": 3},
    )

    register(
        id="MiniGrid-DistanceFetch-8x8-N3-v0",
        entry_point="env.minigrid:DistanceFetchEnv",
        kwargs={"size": 8, "numObjs": 3},
    )
    register(
        id="MiniGrid-DistanceFetch-8x8-N3-raw-v0",
        entry_point="env.minigrid:DistanceFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "distances_type": "raw", "mission_probs": [1, 0, 0, 0]},
    )
    register(
        id="MiniGrid-DistanceFetch-8x8-N3-rank-v0",
        entry_point="env.minigrid:DistanceFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "distances_type": "rank", "mission_probs": [1, 0, 0, 0]},
    )
    register(
        id="MiniGrid-DistanceFetch-8x8-N3-raw-v1",
        entry_point="env.minigrid:DistanceFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "distances_type": "raw", "mission_probs": [1, 1, 0, 0]},
    )
    register(
        id="MiniGrid-DistanceFetch-8x8-N3-rank-v1",
        entry_point="env.minigrid:DistanceFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "distances_type": "rank", "mission_probs": [1, 1, 0, 0]},
    )
    register(
        id="MiniGrid-DistanceFetch-8x8-N3-raw-v2",
        entry_point="env.minigrid:DistanceFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "distances_type": "raw", "mission_probs": [1, 0, 1, 0]},
    )
    register(
        id="MiniGrid-DistanceFetch-8x8-N3-rank-v2",
        entry_point="env.minigrid:DistanceFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "distances_type": "rank", "mission_probs": [1, 0, 1, 0]},
    )
    register(
        id="MiniGrid-DistanceFetch-8x8-N3-raw-v3",
        entry_point="env.minigrid:DistanceFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "distances_type": "raw", "mission_probs": [1, 0, 0, 1]},
    )
    register(
        id="MiniGrid-DistanceFetch-8x8-N3-rank-v3",
        entry_point="env.minigrid:DistanceFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "distances_type": "rank", "mission_probs": [1, 0, 0, 1]},
    )
    register(
        id="MiniGrid-SpuriousFetch-8x8-N3-v0",
        entry_point="env.minigrid:SpuriousFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "randomize": False},
    )
    register(
        id="MiniGrid-SpuriousFetch-8x8-N3-v1",
        entry_point="env.minigrid:SpuriousFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "randomize": True},
    )
    register(
        id="MiniGrid-SpuriousFetch-8x8-N3-v2",
        entry_point="env.minigrid:SpuriousFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "randomize": False, "mission_based": True},
    )
