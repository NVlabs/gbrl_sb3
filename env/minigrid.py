##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
from __future__ import annotations


import numpy as np
from gymnasium.envs.registration import register
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box
from minigrid.minigrid_env import MiniGridEnv


class SpuriousFetchEnv(MiniGridEnv):
    def __init__(self, size=8, numObjs=3, max_steps: int | None = None, use_box: bool = True, randomize: bool = False, purple_ball: bool = False, purple_box: bool = False, add_red_ball: bool = False, grey_ball: bool = False, mission_based: bool = True, test_box_idx: int = None, **kwargs):
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
        self.purple_ball = purple_ball
        self.grey_ball = grey_ball
        self.use_box = use_box
        self.mission_based = mission_based
        self.purple_box = purple_box
        self.add_red_ball = add_red_ball 
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
        self.agent_pov = True
        # self.agent_pov = False
        self.metadata['render_fps'] = 2
        # self.metadata['render_fps'] = 10
    
    def _rand_obj(self):
        """Sample a color with custom probabilities."""
        return np.random.choice([0, 1, 2])
    
    @staticmethod
    def _gen_mission(syntax: str, color: str, obj_type: str):
        return f"{syntax} {color} {obj_type}"
    
    def place_next_to(self, obj, target_obj):
        target_pos = target_obj.cur_pos
        self.place_obj(obj, top=(target_pos[0]-1, target_pos[1] -1), size=(3, 3))

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        target_idx = np.random.choice([0, 1, 2])
        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)
        # Place a goal square in the bottom-right corner
        objs = []
        obs_red = Ball('red')
        obs_green = Ball('green')
        obs_blue = Ball('blue')
        # Randomize the player start position and orientation
        self.place_agent()
        red_pos = self.place_obj(obs_red)

        objPos = [red_pos]
        def near_obj(env, p1):
            for p2 in objPos:
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                if abs(dx) <= 1 and abs(dy) <= 1:
                    return True
            return False
        
        green_pos = self.place_obj(obs_green, reject_fn=near_obj)
        objPos.append(green_pos)
        blue_pos = self.place_obj(obs_blue, reject_fn=near_obj)
        objPos.append(blue_pos)
        objs = [obs_red, obs_green, obs_blue]
        # Choose a random object to be picked up
        target = objs[target_idx]
        
        if self.use_box:
            box_obj = Box('red') if not self.purple_box else Box('purple')
            if self.randomize:
                self.place_obj(box_obj, reject_fn=near_obj)
            elif self.mission_based:
                self.place_next_to(box_obj, target)
            else:
                obj_idx = np.random.choice([i for i in range(2) if i != target_idx])
                self.place_next_to(box_obj, objs[obj_idx])
        if self.purple_ball:
            purple_ball = Ball('purple')
            self.place_obj(purple_ball, reject_fn=near_obj)
        if self.grey_ball:
            grey_ball = Ball('grey')
            self.place_obj(grey_ball, reject_fn=near_obj)
        self.targetType = target.type
        self.targetColor = target.color
        descStr = f"{self.targetColor} {self.targetType}"
        if self.add_red_ball:
            self.place_obj(Ball('red'), reject_fn=near_obj)

        # Generate the mission string
        self.mission = "get a %s" % descStr
        assert hasattr(self, "mission")
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

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
    register(
        id="MiniGrid-SpuriousFetch-8x8-N3-v0",
        entry_point="env.minigrid:SpuriousFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "use_box": True, "mission_based": True, "randomize": False},
    )
    register(
        id="MiniGrid-SpuriousFetch-8x8-N3-v1",
        entry_point="env.minigrid:SpuriousFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "use_box": True, "mission_based": False, "randomize": False},
    )
    register(
        id="MiniGrid-SpuriousFetch-8x8-N3-v2",
        entry_point="env.minigrid:SpuriousFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "use_box": True, "mission_based": False, "randomize": True},
    )
    register(
        id="MiniGrid-SpuriousFetch-8x8-N3-v3",
        entry_point="env.minigrid:SpuriousFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "use_box": False},
    )
    register(
        id="MiniGrid-SpuriousFetch-8x8-N3-v4",
        entry_point="env.minigrid:SpuriousFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "use_box": False, "purple_ball": True},
    )
    register(
        id="MiniGrid-SpuriousFetch-8x8-N3-v5",
        entry_point="env.minigrid:SpuriousFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "use_box": False, "purple_ball": True, 'grey_ball': True},
    )
    register(
        id="MiniGrid-SpuriousFetch-8x8-N3-v6",
        entry_point="env.minigrid:SpuriousFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "use_box": True, "purple_box": True, 'randomize': True},
    )
    register(
        id="MiniGrid-SpuriousFetch-8x8-N3-v7",
        entry_point="env.minigrid:SpuriousFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "use_box": False, "add_red_ball": True},
    )
