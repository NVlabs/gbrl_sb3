##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
from __future__ import annotations

from operator import add
from typing import Any, SupportsFloat
import itertools as itt

import numpy as np
from gymnasium.core import ActType, ObsType
from minigrid.core.constants import STATE_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from gymnasium.spaces import Discrete
from minigrid.core.world_object import Ball, Goal, Lava
from minigrid.minigrid_env import TILE_PIXELS, MiniGridEnv

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}

from enum import IntEnum

class SpecialActions(IntEnum):
    # Turn left, turn right, move forward, do nothing
    left = 0
    right = 1
    forward = 2
    wait = 3


class DynamicCrossing(MiniGridEnv):
    """
    ## Description

    Depending on the `obstacle_type` parameter:
    - `Lava` - The agent has to reach the green goal square on the other corner
        of the room while avoiding rivers of deadly lava which terminate the
        episode in failure. Each lava stream runs across the room either
        horizontally or vertically, and has a single crossing point which can be
        safely used; Luckily, a path to the goal is guaranteed to exist. This
        environment is useful for studying safety and safe exploration.
    - otherwise - Similar to the `LavaCrossing` environment, the agent has to
        reach the green goal square on the other corner of the room, however
        lava is replaced by walls. This MDP is therefore much easier and maybe
        useful for quickly testing your algorithms.

    ## Mission Space
    Depending on the `obstacle_type` parameter:
    - `Lava` - "avoid the lava and get to the green goal square"
    - otherwise - "find the opening and get to the green goal square"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. The agent falls into lava.
    3. Timeout (see `max_steps`).

    ## Registered Configurations

    S: size of the map SxS.
    N: number of valid crossings across lava or walls from the starting position
    to the goal

    - `Lava` :
        - `MiniGrid-LavaCrossingS9N1-v0`
        - `MiniGrid-LavaCrossingS9N2-v0`
        - `MiniGrid-LavaCrossingS9N3-v0`
        - `MiniGrid-LavaCrossingS11N5-v0`

    - otherwise :
        - `MiniGrid-SimpleCrossingS9N1-v0`
        - `MiniGrid-SimpleCrossingS9N2-v0`
        - `MiniGrid-SimpleCrossingS9N3-v0`
        - `MiniGrid-SimpleCrossingS11N5-v0`

    """

    def __init__(
        self,
        size=9,
        num_crossings=1,
        obstacle_type=Lava,
        n_obstacles=1,
        reward_penalty: bool = False,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.num_crossings = num_crossings
        self.obstacle_type = obstacle_type

        if obstacle_type == Lava:
            mission_space = MissionSpace(mission_func=self._gen_mission_lava)
        else:
            mission_space = MissionSpace(mission_func=self._gen_mission)

        # Reduce obstacles if there are too many
        if n_obstacles <= size / 2 + 1:
            self.n_obstacles = int(n_obstacles)
        else:
            self.n_obstacles = int(size / 2)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=False,  # Set this to True for maximum speed
            max_steps=max_steps,
            **kwargs,
        )
        self.actions = SpecialActions
        self.reward_penalty = reward_penalty
        # Allow only 3 actions permitted: left, right, forward, pickup which means do nothing
        self.action_space = Discrete(self.actions.wait + 1)
        # self.action_space = Discrete(3) # no wait
        self.reward_range = (-1, 1)

    @staticmethod
    def _gen_mission_lava():
        return "avoid the lava and the ball get to the green goal square"

    @staticmethod
    def _gen_mission():
        return "find the opening and get to the green goal square"
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type:ignore
        obs, info = super().reset(seed=seed, options=options)
        
        self.last_grid = self.grid.copy()
        self.last_agent_pos = tuple(self.agent_pos)
        self.last_agent_dir = self.agent_dir
        self.last_action = None

        self.prev_potential = self.get_potential(self.agent_pos)

        return obs, info

    def _gen_grid(self, width, height):
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Middle column for lava
        mid_x = width // 2

        # Place vertical lava river in the middle, except for the opening
        # Opening is never adjacent to the wall (so between 2 and height-3)
        opening_y = self.np_random.choice(range(1, height - 1))
        for y in range(1, height - 1):
            if y != opening_y:
                self.put_obj(self.obstacle_type(), mid_x, y)

        # Place the agent in the top-left corner
        top_left = np.array((1, 1))
        bottom_right = np.array((width - 2, height - 2))
        self.agent_pos = np.array((1, height - 2))
        self.agent_dir = 0

        self.max_distance = np.linalg.norm(top_left - bottom_right)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        self.goal_pos = bottom_right

        # Place a single obstacle at (mid_x + 1, opening_y + 1)
        # It moves vertically down, resets at bottom
        self.obstacles = []
        obstacle_start_x = mid_x + 1
        obstacle_start_y_lower = opening_y - 2
        obstacle_start_y_upper = opening_y + 2

        if obstacle_start_y_lower < 1:
            obstacle_start_y_lower = 1  # wrap if out of bounds
        
        if obstacle_start_y_upper > height - 2:
            obstacle_start_y_upper = height - 2  # wrap if out of bounds
        
        self.obstacles.append(Ball())

        self.grid.set(obstacle_start_x, obstacle_start_y_lower, self.obstacles[0])
        self.obstacles[0].cur_pos = (obstacle_start_x, obstacle_start_y_lower)
        self.obstacle_start_x = obstacle_start_x
        
        self.obstacle_min_y = obstacle_start_y_lower
        self.obstacle_max_y = obstacle_start_y_upper

        self.obstacle_min_y = opening_y - 1
        self.obstacle_max_y = opening_y + 1

        self.mission = "avoid the lava and the ball get to the green goal square"

    def step(self, action: ActType
        ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        self.last_grid = self.grid.copy()
        self.last_agent_pos = tuple(self.agent_pos)
        self.last_agent_dir = self.agent_dir
        self.last_action = action

        # Check if there is an obstacle in front of the agent
        fx, fy = self.front_pos
        if 0 <= fx < self.width and 0 <= fy < self.height:
            front_cell = self.grid.get(fx, fy)
        else:
            front_cell = None
        not_clear = front_cell is not None and front_cell.type != "goal"
        unsafe = front_cell is not None and front_cell.type in ['lava', 'ball']
        info = {'cost': 0.0, 'safety_label': int(unsafe)}

        # Update obstacle positions
        if self.obstacles:
            old_x, old_y = self.obstacles[0].cur_pos
            new_y = old_y + 1
            if new_y > self.obstacle_max_y:
                new_y = self.obstacle_min_y
            self.grid.set(old_x, old_y, None)
            self.grid.set(self.obstacle_start_x, new_y, self.obstacles[0])
            self.obstacles[0].cur_pos = (self.obstacle_start_x, new_y)

        potential = self.get_potential(self.agent_pos)

        reward = potential - self.prev_potential

        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fx, fy = fwd_pos
        if 0 <= fx < self.width and 0 <= fy < self.height:
            fwd_cell = self.grid.get(fx, fy)
        else:
            fwd_cell = None

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
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            # if fwd_cell is not None and fwd_cell.type == "lava":
                # terminated = True

        # Wait action (not used by default)
        elif action == self.actions.wait:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        # If the agent tried to walk over an obstacle or wall
        if action == self.actions.forward and not_clear:
            if self.reward_penalty:
                reward -= 1
            # terminated = True
            info['cost'] += 1.0
            if front_cell.type == 'ball':
                terminated = True

        self.prev_potential = potential

        return obs, reward, terminated, truncated, info

    def get_potential(self, pos):
        dist = np.linalg.norm(np.array(pos) - np.array(self.goal_pos))
        return np.clip(1.0 - (dist / self.max_distance), 0.0, 1.0)

    def get_frame(
        self,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        """Returns an RGB image using the SAVED state from before the last step."""
        
        # Temporarily swap to the saved state
        current_grid = self.grid
        current_agent_pos = self.agent_pos
        current_agent_dir = self.agent_dir
        current_carrying = self.carrying
        
        self.grid = self.last_grid
        self.agent_pos = self.last_agent_pos
        self.agent_dir = self.last_agent_dir
        
        # Render with the saved state
        if agent_pov:
            img = self.get_pov_render(tile_size)
        else:
            img = self.get_full_render(highlight, tile_size)
        
        # Restore current state
        self.grid = current_grid
        self.agent_pos = current_agent_pos
        self.agent_dir = current_agent_dir
        self.carrying = current_carrying
        
        return img
    
    def render(self):
        import cv2

        # Get the original frame
        frame = self.get_frame(self.highlight, self.tile_size, self.agent_pov)

        if frame is None:
            return frame
        
        frame_height, frame_width, _ = frame.shape
        
        # Add guidance information at the top of the frame
        # Create a banner at the top for guidance info
        banner_height = 100
        banner = np.ones((banner_height, frame_width, 3), dtype=np.uint8) * 240  # Light gray background

        # Format the actual action taken
        if self.last_action is not None:
            action_names = ['Left', 'Right', 'Fwd', 'Wait']
            taken_text = f"Action Taken: {action_names[self.last_action]}"
        else:
            taken_text = "Action Taken: None"
        
        # Add text to banner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_color = (0, 0, 0)  # Black text
        
        # Line 4: Action taken
        cv2.putText(banner, taken_text, (10, 80), font, font_scale, text_color, thickness, cv2.LINE_AA)
        # Concatenate banner on top of frame
        frame_with_banner = np.concatenate((banner, frame), axis=0)
        
        return frame_with_banner
        