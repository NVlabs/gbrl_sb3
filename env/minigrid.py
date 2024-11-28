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

from gymnasium.envs.registration import register


import numpy as np
class SequentialPutNearEnv(MiniGridEnv):
    def __init__(self, size=6, numObjs=2, max_steps: int | None = None, **kwargs):
        self.size = size
        self.numObjs = numObjs
        self.obj_types = ["key", "ball", "box"]
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[
                COLOR_NAMES,
                self.obj_types,
                COLOR_NAMES,
                self.obj_types,
            ],
        )

        if max_steps is None:
            max_steps = 5 * size

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(
        move_color: str, move_type: str, target_color: str, target_type: str
    ):
        return f"put the {move_color} {move_type} near the {target_color} {target_type}"
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # Types and colors of objects we can generate
        types = ["key", "ball", "box"]

        objs = []
        objPos = []

        def near_obj(env, p1):
            for p2 in objPos:
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                if abs(dx) <= 1 and abs(dy) <= 1:
                    return True
            return False

        # Until we have generated all the objects
        while len(objs) < self.numObjs:
            objType = self._rand_elem(types)
            objColor = self._rand_elem(COLOR_NAMES)

            # If this object already exists, try again
            if (objType, objColor) in objs:
                continue

            if objType == "key":
                obj = Key(objColor)
            elif objType == "ball":
                obj = Ball(objColor)
            elif objType == "box":
                obj = Box(objColor)
            else:
                raise ValueError(
                    "{} object type given. Object type can only be of values key, ball and box.".format(
                        objType
                    )
                )

            pos = self.place_obj(obj, reject_fn=near_obj)

            objs.append((objType, objColor))
            objPos.append(pos)

        self.goal_pos = self.place_obj(Goal(), reject_fn=near_obj)

        # Randomize the agent start position and orientation
        self.place_agent()

        # Choose a random object to be moved
        objIdx = self._rand_int(0, len(objs))
        self.move_type, self.moveColor = objs[objIdx]
        self.move_pos = objPos[objIdx]


        # Choose a target object (to put the first object next to)
        while True:
            targetIdx = self._rand_int(0, len(objs))
            if targetIdx != objIdx:
                break
        self.target_type, self.target_color = objs[targetIdx]
        self.target_pos = objPos[targetIdx]
        self.init_mov_pos = self.move_pos

        self.mission = "put the {} {} near the {} {}".format(
            self.moveColor,
            self.move_type,
            self.target_color,
            self.target_type,
        )

        self.completed_first_task = False
        self.accumulated_reward = 0

    def step(self, action):
        preCarrying = self.carrying

        reward = 0
        terminated = False
        truncated = False
        info = {}

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal" and self.completed_first_task and preCarrying:
                terminated = True
                reward = self._reward() + self.accumulated_reward
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        u, v = self.dir_vec
        ox, oy = (self.agent_pos[0] + u, self.agent_pos[1] + v)
        tx, ty = self.target_pos

        # If we picked up the wrong object, terminate the episode
        if action == self.actions.pickup and self.carrying:
            if (
                self.carrying.type != self.move_type
                or self.carrying.color != self.moveColor
            ):
                terminated = True

        # If successfully dropping an object near the target
        if action == self.actions.drop and preCarrying:
            if self.grid.get(ox, oy) is preCarrying:
                if abs(ox - tx) <= 1 and abs(oy - ty) <= 1 and not self.completed_first_task:
                    reward = self._reward()
                    self.accumulated_reward = reward
                    self.move_type = self.target_type
                    self.moveColor = self.target_color
                    self.target_color = "green"
                    self.target_type = "goal"
                    self.target_pos = self.goal_pos

                    self.mission = "put the {} {} at the {} {}".format(
                                                                        self.moveColor,
                                                                        self.move_type,
                                                                        self.target_color,
                                                                        self.target_type,
                                                                        )
                    self.completed_first_task = True
        reward = min(reward, self.reward_range[1])
        return obs, reward, terminated, truncated, info


def register_sequential_put_near():
    # PutNear
    # ----------------------------------------

    register(
        id="MiniGrid-SequentialPutNear-6x6-N2-v0",
        entry_point="env.minigrid:SequentialPutNearEnv",
    )

    register(
        id="MiniGrid-SequentialPutNear-8x8-N3-v0",
        entry_point="env.minigrid:SequentialPutNearEnv",
        kwargs={"size": 8, "numObjs": 3},
    )