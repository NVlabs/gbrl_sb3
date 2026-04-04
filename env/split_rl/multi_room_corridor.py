##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
from __future__ import annotations

from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import ActType, ObsType
from minigrid.core.constants import STATE_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv


class MultiRoomCorridorEnv(MiniGridEnv):
    """
    Multi-room environment with 4 rooms connected via a narrow corridor.

    Layout:
    - 3 rooms at the bottom arranged horizontally (left, middle, right)
    - A narrow corridor connects the middle room upward to a 4th room at the top
    - Top room: contains the GOAL behind a locked door (e.g. green)
    - Bottom-right room: contains a BOX with the same color key as the top door
      (green key in green box). Connected to middle room by a different colored
      locked door (e.g. purple).
    - Bottom-left room: contains the key (purple) to unlock the bottom-right door.
      Entrance blocked by a blue ball that must be moved.
    - Bottom-middle room: starting room. Agent spawns here.

    Solution sequence:
    1. Move the blue ball blocking the bottom-left room entrance
    2. Enter bottom-left room, pick up the purple key
    3. Use purple key to unlock door to bottom-right room
    4. Open the box in bottom-right room to get the green key
    5. Travel through corridor to top room
    6. Use green key to unlock the top room door
    7. Reach the goal
    """

    def __init__(self, width=19, height=19, max_steps=None,
                 top_door_color="green", right_door_color="purple",
                 subtask_bonus=0.5, **kwargs):
        self.top_door_color = top_door_color
        self.right_door_color = right_door_color
        self.subtask_bonus = subtask_bonus

        if max_steps is None:
            max_steps = 4 * width * height

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "move the ball, collect keys, unlock doors, and reach the goal"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Fill everything with walls first
        self.grid.wall_rect(0, 0, width, height)

        # ============================================================
        # Define geometry
        # ============================================================
        # We split the grid into a top section (top room + corridor)
        # and bottom section (3 rooms).
        #
        # Bottom rooms occupy the lower portion of the grid.
        # The corridor is a narrow vertical passage from the middle
        # bottom room up to the top room.
        #
        # Heights:
        #   top room:    rows 1 .. (top_room_h)
        #   corridor:    rows (top_room_h+1) .. (corridor_end)
        #   bottom rooms: rows (bottom_start) .. (height-2)
        #
        # The corridor is 1 tile wide centered on the grid.

        corridor_x = width // 2  # center column for corridor
        corridor_width = 1  # single tile wide

        # Vertical splits:
        # top room: rows 1 to top_room_bottom (exclusive)
        # corridor: rows corridor_top to corridor_bottom (inclusive)
        # bottom rooms: rows bottom_top to height-2 (inclusive)

        top_room_bottom = height // 3       # wall row separating top room from corridor
        bottom_top = 2 * height // 3        # wall row separating corridor from bottom rooms
        corridor_top = top_room_bottom + 1
        corridor_bottom = bottom_top - 1

        # Horizontal splits for bottom rooms:
        # left room: cols 1 to left_wall (exclusive)
        # middle room: cols left_wall+1 to right_wall (exclusive)
        # right room: cols right_wall+1 to width-2 (inclusive)
        left_wall_x = width // 3
        right_wall_x = 2 * width // 3

        # ============================================================
        # Carve out the TOP ROOM (rows 1..top_room_bottom-1)
        # ============================================================
        for y in range(1, top_room_bottom):
            for x in range(1, width - 1):
                self.grid.set(x, y, None)

        # Build the wall row at top_room_bottom
        for x in range(0, width):
            self.grid.set(x, top_room_bottom, Wall())

        # ============================================================
        # Carve out the CORRIDOR (single column)
        # ============================================================
        for y in range(corridor_top, corridor_bottom + 1):
            self.grid.set(corridor_x, y, None)

        # ============================================================
        # Build the wall row at bottom_top
        # ============================================================
        for x in range(0, width):
            self.grid.set(x, bottom_top, Wall())

        # ============================================================
        # Carve out the 3 BOTTOM ROOMS
        # ============================================================
        for y in range(bottom_top + 1, height - 1):
            for x in range(1, width - 1):
                self.grid.set(x, y, None)

        # Build internal vertical walls for bottom rooms
        # Left wall (separating left and middle rooms)
        for y in range(bottom_top, height):
            self.grid.set(left_wall_x, y, Wall())

        # Right wall (separating middle and right rooms)
        for y in range(bottom_top, height):
            self.grid.set(right_wall_x, y, Wall())

        # ============================================================
        # DOORS
        # ============================================================

        # 1. Door from corridor into top room (locked, top_door_color)
        #    Opening in the top_room_bottom wall at corridor_x
        top_door = Door(self.top_door_color, is_locked=True)
        self.grid.set(corridor_x, top_room_bottom, top_door)
        self.top_door_pos = (corridor_x, top_room_bottom)

        # 2. Door from corridor into middle bottom room
        #    Opening in the bottom_top wall at corridor_x (open passage)
        self.grid.set(corridor_x, bottom_top, None)

        # 3. Door from middle room to left room (passage blocked by ball)
        #    Opening in the left_wall at a middle row
        left_door_y = (bottom_top + 1 + height - 2) // 2
        self.grid.set(left_wall_x, left_door_y, None)  # open passage
        self.left_entrance_pos = (left_wall_x, left_door_y)

        # 4. Door from middle room to right room (locked, right_door_color)
        right_door_y = (bottom_top + 1 + height - 2) // 2
        right_door = Door(self.right_door_color, is_locked=True)
        self.grid.set(right_wall_x, right_door_y, right_door)
        self.right_door_pos = (right_wall_x, right_door_y)

        # ============================================================
        # OBJECTS
        # ============================================================

        # Blue ball blocking the entrance to the left room
        # Place it on the middle-room side of the entrance (one tile right of wall)
        ball_x = left_wall_x + 1
        ball_y = left_door_y
        blue_ball = Ball("blue")
        self.grid.set(ball_x, ball_y, blue_ball)

        # Purple key in the bottom-left room (unlocks right room door)
        left_room_center_x = (1 + left_wall_x) // 2
        left_room_center_y = (bottom_top + 1 + height - 2) // 2
        right_key = Key(self.right_door_color)
        self.put_obj(right_key, left_room_center_x, left_room_center_y)

        # Green box with green key inside, in the bottom-right room
        right_room_center_x = (right_wall_x + 1 + width - 2) // 2
        right_room_center_y = (bottom_top + 1 + height - 2) // 2
        top_key = Key(self.top_door_color)
        green_box = Box(self.top_door_color)
        green_box.contains = top_key
        self.put_obj(green_box, right_room_center_x, right_room_center_y)

        # Goal in the top room
        goal_x = width // 2
        goal_y = 1
        self.put_obj(Goal(), goal_x, goal_y)

        # ============================================================
        # AGENT start in the middle bottom room
        # ============================================================
        agent_x = (left_wall_x + 1 + right_wall_x) // 2
        agent_y = (bottom_top + 1 + height - 2) // 2
        self.agent_pos = np.array([agent_x, agent_y])
        self.agent_dir = self._rand_int(0, 4)  # random facing direction

        self.mission = "move the ball, collect keys, unlock doors, and reach the goal"

        # Store goal position for distance-based reward
        self.goal_pos = np.array([goal_x, goal_y], dtype=np.float64)

        # Store object positions for subtask detection
        self.ball_init_pos = (ball_x, ball_y)
        self._right_key_pos = (left_room_center_x, left_room_center_y)
        self._box_pos = (right_room_center_x, right_room_center_y)

    def _distance_to_goal(self):
        return np.linalg.norm(self.agent_pos - self.goal_pos)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.prev_distance = self._distance_to_goal()
        # One-time subtask completion flags
        self._ball_cleared = False
        self._purple_key_picked = False
        self._right_door_opened = False
        self._box_opened = False
        self._green_key_picked = False
        self._top_door_opened = False
        return obs, info

    def step(self, action):
        # Snapshot state before the step
        was_carrying = self.carrying

        obs, reward, terminated, truncated, info = super().step(action)

        # Dense distance-delta reward: positive when getting closer to goal
        curr_distance = self._distance_to_goal()
        reward = (self.prev_distance - curr_distance) * 0.1
        self.prev_distance = curr_distance

        bonus = self.subtask_bonus

        # --- One-time subtask completion bonuses ---

        # 1. Ball cleared: ball no longer at its init position
        if not self._ball_cleared:
            cell = self.grid.get(*self.ball_init_pos)
            if cell is None or cell.type != 'ball':
                self._ball_cleared = True
                reward += bonus

        # 2. Purple key picked up
        if not self._purple_key_picked and self.carrying is not None:
            if isinstance(self.carrying, Key) and self.carrying.color == self.right_door_color:
                self._purple_key_picked = True
                reward += bonus

        # 3. Right door (purple) opened
        if not self._right_door_opened:
            cell = self.grid.get(*self.right_door_pos)
            if isinstance(cell, Door) and cell.is_open:
                self._right_door_opened = True
                reward += bonus

        # 4. Box opened
        if not self._box_opened:
            cell = self.grid.get(*self._box_pos)
            if cell is None or (isinstance(cell, Box) and cell.contains is None):
                # Box was toggled (key spilled out, or box replaced by key on grid)
                # Check if a key appeared nearby or box.contains is gone
                if cell is None or not isinstance(cell, Box):
                    self._box_opened = True
                    reward += bonus
                elif cell.contains is None:
                    self._box_opened = True
                    reward += bonus

        # 5. Green key picked up
        if not self._green_key_picked and self.carrying is not None:
            if isinstance(self.carrying, Key) and self.carrying.color == self.top_door_color:
                self._green_key_picked = True
                reward += bonus

        # 6. Top door (green) opened
        if not self._top_door_opened:
            cell = self.grid.get(*self.top_door_pos)
            if isinstance(cell, Door) and cell.is_open:
                self._top_door_opened = True
                reward += bonus

        # Terminal reward on reaching the goal
        if terminated and not truncated:
            reward += 1.0 - min(self.step_count / self.max_steps, 1.0)

        return obs, reward, terminated, truncated, info


class MoveBallEnv(MiniGridEnv):
    """
    Subtask 1: Move a blue ball that obstructs an entrance.

    A small room with a doorway. A blue ball blocks the doorway.
    The agent must pick up the ball, move it elsewhere, and walk
    through the doorway to reach a goal on the other side.
    """

    def __init__(self, width=8, height=8, min_size=6, max_size=12, max_steps=None, **kwargs):
        self.min_size = min_size
        self.max_size = max_size

        if max_steps is None:
            max_steps = 4 * max_size * max_size

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            width=max_size,
            height=max_size,
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "move the blue ball blocking the entrance and reach the goal"

    def reset(self, **kwargs):
        # Randomize grid dimensions each episode
        self.width = self._rand_int(self.min_size, self.max_size + 1)
        self.height = self._rand_int(self.min_size, self.max_size + 1)
        return super().reset(**kwargs)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Outer walls
        self.grid.wall_rect(0, 0, width, height)

        # Randomly choose wall orientation: vertical or horizontal
        is_vertical = self._rand_bool()
        # Randomly choose which side the agent starts on
        flip_sides = self._rand_bool()

        if is_vertical:
            # Vertical wall splitting left / right
            wall_pos = width // 2
            for y in range(0, height):
                self.grid.set(wall_pos, y, Wall())

            # Randomize doorway position along the wall
            door_along = self._rand_int(2, height - 2)
            self.grid.set(wall_pos, door_along, None)
            door_xy = (wall_pos, door_along)

            # Define the two room regions
            region_a_top, region_a_size = (1, 1), (wall_pos - 2, height - 2)
            region_b_top, region_b_size = (wall_pos + 1, 1), (width - 2 - wall_pos, height - 2)

            # Ball position: always adjacent to doorway on the agent's side
            if not flip_sides:
                ball_xy = (wall_pos - 1, door_along)
                obstructing = {
                    (wall_pos - 1, door_along),
                    (wall_pos, door_along),
                    (wall_pos + 1, door_along),
                }
            else:
                ball_xy = (wall_pos + 1, door_along)
                obstructing = {
                    (wall_pos + 1, door_along),
                    (wall_pos, door_along),
                    (wall_pos - 1, door_along),
                }
        else:
            # Horizontal wall splitting top / bottom
            wall_pos = height // 2
            for x in range(0, width):
                self.grid.set(x, wall_pos, Wall())

            # Randomize doorway position along the wall
            door_along = self._rand_int(2, width - 2)
            self.grid.set(door_along, wall_pos, None)
            door_xy = (door_along, wall_pos)

            # Define the two room regions
            region_a_top, region_a_size = (1, 1), (width - 2, wall_pos - 2)
            region_b_top, region_b_size = (1, wall_pos + 1), (width - 2, height - 2 - wall_pos)

            # Ball position: always adjacent to doorway on the agent's side
            if not flip_sides:
                ball_xy = (door_along, wall_pos - 1)
                obstructing = {
                    (door_along, wall_pos - 1),
                    (door_along, wall_pos),
                    (door_along, wall_pos + 1),
                }
            else:
                ball_xy = (door_along, wall_pos + 1)
                obstructing = {
                    (door_along, wall_pos + 1),
                    (door_along, wall_pos),
                    (door_along, wall_pos - 1),
                }

        # Assign agent / goal rooms based on flip
        if not flip_sides:
            agent_top, agent_size = region_a_top, region_a_size
            goal_top, goal_size = region_b_top, region_b_size
        else:
            agent_top, agent_size = region_b_top, region_b_size
            goal_top, goal_size = region_a_top, region_a_size

        # Blue ball blocking the doorway (on agent's side)
        ball = Ball("blue")
        self.grid.set(*ball_xy, ball)

        # Store positions for reward shaping
        self.ball_init_pos = ball_xy
        self.door_pos = door_xy
        self.obstructing_cells = obstructing
        self.ball_dropped_reward_given = False
        self.ball_cleared = False

        # Goal in the far room
        self.place_obj(Goal(), top=goal_top, size=goal_size)

        # Agent starts at a random position in the near room
        self.place_agent(top=agent_top, size=agent_size)
        self.agent_dir = self._rand_int(0, 4)

        self.mission = "move the blue ball blocking the entrance and reach the goal"

    def step(self, action):
        self.step_count += 1

        # Check if agent was carrying the ball before this step
        was_carrying_ball = self.carrying is not None and isinstance(self.carrying, Ball)

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
            if fwd_cell is not None and fwd_cell.type == "goal":
                if self.carrying is not None:
                    reward = -0.5  # must drop before finishing
                else:
                    terminated = True
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


        # Detect pickup: wasn't carrying → now carrying
        is_carrying_ball = self.carrying is not None and isinstance(self.carrying, Ball)
        if not was_carrying_ball and is_carrying_ball and self.ball_dropped_reward_given and not self.ball_cleared:
            # Penalty for picking ball back up after a bad drop (still blocking)
            reward = -0.5
        elif not was_carrying_ball and is_carrying_ball and self.ball_cleared:
            # Penalty for picking ball back up after successfully clearing the doorway
            reward = -0.5
            self.ball_cleared = False
        elif not was_carrying_ball and is_carrying_ball:
            # Bonus for initially picking up the ball
            reward = 0.5

        # One-time reward/penalty when the ball is dropped
        if was_carrying_ball and not is_carrying_ball and not self.ball_dropped_reward_given:
            self.ball_dropped_reward_given = True
            # Find where the ball ended up (the cell the agent was facing when it dropped)
            fwd_pos = self.front_pos
            drop_pos = (fwd_pos[0], fwd_pos[1])
            if drop_pos in self.obstructing_cells:
                # Penalty: dropped ball back in a position that blocks the doorway
                reward = -0.5
            else:
                # Bonus: dropped ball somewhere that clears the path
                reward = 0.5
                self.ball_cleared = True

        # Terminal reward on reaching the goal
        if terminated and not truncated:
            reward += 1.0 - min(self.step_count / self.max_steps, 1.0)

        return obs, reward, terminated, truncated, info


class KeyDoorEnv(MiniGridEnv):
    """
    Subtask 2: Pick up a key and use it to unlock a door.

    A room split by a locked door. The key is on the agent's
    side. The agent must pick up the key, unlock the door, and
    reach the goal on the other side.
    """

    def __init__(self, width=8, height=8, min_size=6, max_size=12, max_steps=None,
                 door_color="purple", **kwargs):
        self.door_color = door_color
        self.min_size = min_size
        self.max_size = max_size

        if max_steps is None:
            max_steps = 4 * max_size * max_size

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            width=max_size,
            height=max_size,
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "pick up the key, unlock the door, and reach the goal"

    def reset(self, **kwargs):
        # Randomize grid dimensions each episode
        self.width = self._rand_int(self.min_size, self.max_size + 1)
        self.height = self._rand_int(self.min_size, self.max_size + 1)
        return super().reset(**kwargs)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Outer walls
        self.grid.wall_rect(0, 0, width, height)

        # Randomly choose wall orientation and which side agent starts on
        is_vertical = self._rand_bool()
        flip_sides = self._rand_bool()

        if is_vertical:
            wall_pos = width // 2
            for y in range(0, height):
                self.grid.set(wall_pos, y, Wall())

            door_along = self._rand_int(2, height - 2)
            door = Door(self.door_color, is_locked=True)
            self.grid.set(wall_pos, door_along, door)

            region_a_top, region_a_size = (1, 1), (wall_pos - 2, height - 2)
            region_b_top, region_b_size = (wall_pos + 1, 1), (width - 2 - wall_pos, height - 2)
        else:
            wall_pos = height // 2
            for x in range(0, width):
                self.grid.set(x, wall_pos, Wall())

            door_along = self._rand_int(2, width - 2)
            door = Door(self.door_color, is_locked=True)
            self.grid.set(door_along, wall_pos, door)

            region_a_top, region_a_size = (1, 1), (width - 2, wall_pos - 2)
            region_b_top, region_b_size = (1, wall_pos + 1), (width - 2, height - 2 - wall_pos)

        if not flip_sides:
            agent_top, agent_size = region_a_top, region_a_size
            goal_top, goal_size = region_b_top, region_b_size
        else:
            agent_top, agent_size = region_b_top, region_b_size
            goal_top, goal_size = region_a_top, region_a_size

        # Key at a random position on agent's side
        self.place_obj(Key(self.door_color), top=agent_top, size=agent_size)

        # Goal on the far side (not adjacent to door)
        if is_vertical:
            door_pos = (wall_pos, door_along)
        else:
            door_pos = (door_along, wall_pos)

        def not_adjacent_to_door(env, pos):
            return abs(pos[0] - door_pos[0]) <= 1 and abs(pos[1] - door_pos[1]) <= 1
        goal_pos = self.place_obj(Goal(), top=goal_top, size=goal_size,
                       reject_fn=not_adjacent_to_door)

        # Agent starts at a random position on its side, random direction
        self.place_agent(top=agent_top, size=agent_size)
        self.agent_dir = self._rand_int(0, 4)

        # Track state for reward shaping
        self._has_key = False
        self._door_opened = False
        self._key_dropped = False
        self.door_pos = np.array(door_pos, dtype=np.float64)
        self.goal_pos = np.array(goal_pos, dtype=np.float64)
        self._prev_dist_to_door = None
        self._prev_dist_to_goal = None

        self.mission = "pick up the key, unlock the door, and reach the goal"

    def step(self, action):
        self.step_count += 1
        was_carrying_key = self.carrying is not None and isinstance(self.carrying, Key)

        reward = 0
        terminated = False
        truncated = False
        info = {}

        # Get the position in front of the agent (captured before action)
        fwd_pos = self.front_pos
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
            if fwd_cell is not None and fwd_cell.type == "goal":
                if self.carrying is not None:
                    reward = -0.5  # must drop before finishing
                else:
                    terminated = True
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

        is_carrying_key = self.carrying is not None and isinstance(self.carrying, Key)

        # Bonus for picking up the key
        if not was_carrying_key and is_carrying_key:
            if not self._has_key:
                reward = 0.5  # first pickup
                self._has_key = True
            elif self._key_dropped:
                reward = -0.5  # penalty for re-picking after drop
                self._key_dropped = False

        # Key no longer carried — distinguish drop vs door consumption
        if was_carrying_key and not is_carrying_key:
            if action == self.actions.drop:
                reward = -0.5  # penalty for dropping key
                self._key_dropped = True
            elif action == self.actions.toggle:
                self._door_opened = True  # key consumed by door

        # Penalty for trying to toggle the locked door without holding the key
        if action == self.actions.toggle and not is_carrying_key and not self._door_opened:
            if isinstance(fwd_cell, Door) and fwd_cell.is_locked:
                reward = -0.5

        # Distance-based shaping toward door when carrying the key
        if is_carrying_key and not self._door_opened:
            dist = np.linalg.norm(np.array(self.agent_pos, dtype=np.float64) - self.door_pos)
            if self._prev_dist_to_door is not None:
                reward += (self._prev_dist_to_door - dist) * 0.2
            self._prev_dist_to_door = dist
            self._prev_dist_to_goal = None
        elif self._door_opened:
            # After door is opened, shape toward the goal
            dist = np.linalg.norm(np.array(self.agent_pos, dtype=np.float64) - self.goal_pos)
            if self._prev_dist_to_goal is not None:
                reward += (self._prev_dist_to_goal - dist) * 0.2
            self._prev_dist_to_goal = dist
            self._prev_dist_to_door = None
        else:
            self._prev_dist_to_door = None
            self._prev_dist_to_goal = None

        # Terminal reward on reaching the goal
        if terminated and not truncated:
            reward += 1.0 - min(self.step_count / self.max_steps, 1.0)

        return obs, reward, terminated, truncated, info


class BoxKeyEnv(MiniGridEnv):
    """
    Subtask 3: Open a box and pick up a key inside.

    A small room with a colored box. Inside the box is a key
    of the same color. The agent must open (toggle) the box to
    reveal the key, pick it up, and then reach a locked door
    to complete the task.
    """

    def __init__(self, width=8, height=8, min_size=6, max_size=12, max_steps=None,
                 box_color="green", **kwargs):
        self.box_color = box_color
        self.min_size = min_size
        self.max_size = max_size

        if max_steps is None:
            max_steps = 4 * max_size * max_size

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            width=max_size,
            height=max_size,
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "open the box, pick up the key, unlock the door, and reach the goal"

    def reset(self, **kwargs):
        # Randomize grid dimensions each episode
        self.width = self._rand_int(self.min_size, self.max_size + 1)
        self.height = self._rand_int(self.min_size, self.max_size + 1)
        return super().reset(**kwargs)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Outer walls
        self.grid.wall_rect(0, 0, width, height)

        # Randomly choose wall orientation and which side the agent starts on
        is_vertical = self._rand_bool()
        flip_sides = self._rand_bool()

        if is_vertical:
            wall_pos = width // 2
            for y in range(0, height):
                self.grid.set(wall_pos, y, Wall())

            door_along = self._rand_int(2, height - 2)
            door = Door(self.box_color, is_locked=True)
            self.grid.set(wall_pos, door_along, door)

            region_a_top, region_a_size = (1, 1), (wall_pos - 2, height - 2)
            region_b_top, region_b_size = (wall_pos + 1, 1), (width - 2 - wall_pos, height - 2)
            door_pos_xy = (wall_pos, door_along)
        else:
            wall_pos = height // 2
            for x in range(0, width):
                self.grid.set(x, wall_pos, Wall())

            door_along = self._rand_int(2, width - 2)
            door = Door(self.box_color, is_locked=True)
            self.grid.set(door_along, wall_pos, door)

            region_a_top, region_a_size = (1, 1), (width - 2, wall_pos - 2)
            region_b_top, region_b_size = (1, wall_pos + 1), (width - 2, height - 2 - wall_pos)
            door_pos_xy = (door_along, wall_pos)

        # Assign agent room (with box) and goal room
        if not flip_sides:
            agent_top, agent_size = region_a_top, region_a_size
            goal_top, goal_size = region_b_top, region_b_size
        else:
            agent_top, agent_size = region_b_top, region_b_size
            goal_top, goal_size = region_a_top, region_a_size

        # Box with key inside — random position in agent's room, not directly
        # adjacent to the door (so it doesn't block the approach)
        key = Key(self.box_color)
        box = Box(self.box_color)
        box.contains = key
        dx, dy = door_pos_xy
        self.place_obj(
            box, top=agent_top, size=agent_size,
            reject_fn=lambda env, pos: abs(int(pos[0]) - dx) <= 1 and abs(int(pos[1]) - dy) <= 1,
        )

        # Goal in the far room
        goal_pos = self.place_obj(Goal(), top=goal_top, size=goal_size)

        # Agent starts at a random position in the near room
        self.place_agent(top=agent_top, size=agent_size)
        self.agent_dir = self._rand_int(0, 4)

        # Track state for reward shaping
        self._box_opened = False
        self._has_key = False
        self._door_opened = False
        self._key_dropped = False
        if is_vertical:
            self.door_pos = np.array([wall_pos, door_along], dtype=np.float64)
        else:
            self.door_pos = np.array([door_along, wall_pos], dtype=np.float64)
        self.goal_pos = np.array(goal_pos, dtype=np.float64)
        self._prev_dist_to_door = None
        self._prev_dist_to_goal = None
        self._prev_dist_to_door = None
        self._prev_dist_to_goal = None

        self.mission = "open the box, pick up the key, unlock the door, and reach the goal"

    def step(self, action):
        self.step_count += 1
        was_carrying_key = self.carrying is not None and isinstance(self.carrying, Key)

        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # Get the position in front of the agent (captured before action)
        fwd_pos = self.front_pos
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
            if fwd_cell is not None and fwd_cell.type == "goal":
                if self.carrying is not None:
                    reward = -0.5  # must drop before finishing
                else:
                    terminated = True
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

        is_carrying_key = self.carrying is not None and isinstance(self.carrying, Key)

        # Bonus for opening the box
        if action == self.actions.toggle and isinstance(fwd_cell, Box) and not self._box_opened:
            self._box_opened = True
            reward = 0.5

        # Bonus for picking up the key
        if not was_carrying_key and is_carrying_key:
            if not self._has_key:
                reward = 0.5  # first pickup
                self._has_key = True
            elif self._key_dropped:
                reward = -0.5  # penalty for re-picking after drop
                self._key_dropped = False

        # Key no longer carried — distinguish drop vs door consumption
        if was_carrying_key and not is_carrying_key:
            if action == self.actions.drop:
                reward = -0.5  # penalty for dropping key
                self._key_dropped = True
            elif action == self.actions.toggle:
                self._door_opened = True  # key consumed by door

        # Distance-based shaping toward door when carrying the key
        if is_carrying_key and not self._door_opened:
            dist = np.linalg.norm(np.array(self.agent_pos, dtype=np.float64) - self.door_pos)
            if self._prev_dist_to_door is not None:
                reward += (self._prev_dist_to_door - dist) * 0.2
            self._prev_dist_to_door = dist
            self._prev_dist_to_goal = None
        elif self._door_opened:
            # After door is opened, shape toward the goal
            dist = np.linalg.norm(np.array(self.agent_pos, dtype=np.float64) - self.goal_pos)
            if self._prev_dist_to_goal is not None:
                reward += (self._prev_dist_to_goal - dist) * 0.2
            self._prev_dist_to_goal = dist
            self._prev_dist_to_door = None
        else:
            self._prev_dist_to_door = None
            self._prev_dist_to_goal = None

        # Terminal reward on reaching the goal
        if terminated and not truncated:
            reward += 1.0 - min(self.step_count / self.max_steps, 1.0)

        return obs, reward, terminated, truncated, info
