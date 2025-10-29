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
from gbrl.common.utils import categorical_dtype
from gymnasium.core import ActType, ObsType
from gymnasium.envs.registration import register
from minigrid.core.constants import (IDX_TO_COLOR, IDX_TO_OBJECT,
                                     STATE_TO_IDX)
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Key
from minigrid.envs.obstructedmaze import ObstructedMazeEnv
from minigrid.minigrid_env import TILE_PIXELS, MiniGridEnv

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}


def categorical_obs_encoding(observation, image_shape, flattened_image_shape, FullyObsWrapper=False, is_mixed=False):
    # Transform the observation in some way
    categorical_array = np.empty(flattened_image_shape, dtype=categorical_dtype if not is_mixed else object)
    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            if FullyObsWrapper and \
                str(IDX_TO_OBJECT[observation['image'][i, j, 0]]) == 'agent':
                category = f"{str(IDX_TO_OBJECT[observation['image'][i, j, 0]])}," + \
                            f"{str(IDX_TO_COLOR[observation['image'][i, j, 1]])}," + \
                            f"{str(observation['image'][i, j, 2])}"
            else:
                category = f"{str(IDX_TO_OBJECT[observation['image'][i, j, 0]])}," \
                            f"{str(IDX_TO_COLOR[observation['image'][i, j, 1]])}," \
                            f"{str(IDX_TO_STATE[observation['image'][i, j, 2]])}"
            categorical_array[i*image_shape[1] + j] = category.encode('utf-8')
    categorical_array[image_shape[0]*image_shape[1]] = str(observation['direction']).encode('utf-8')
    categorical_array[image_shape[0]*image_shape[1] + 1] = observation['mission'].encode('utf-8')
    return np.ascontiguousarray(categorical_array)


def spurious_box_compliance(fwd_obj):
    if fwd_obj is not None and fwd_obj.type == 'box' and fwd_obj.color == 'red':
        return 1
    return 0


def simple_fetch_compliance(agent_pos, ball_pos, start_pos, move_left, from_back):
    ax, ay = agent_pos
    bx, by = ball_pos
    sx, sy = start_pos

    if from_back:
        # Get all four adjacent positions around the ball
        neighbors = [(bx - 1, by), (bx + 1, by), (bx, by - 1), (bx, by + 1)]

        # Determine the "back" position based on start and ball
        back = (sx, sy)
        dx = bx - sx
        dy = by - sy
        back_pos = (bx, by - 1)

        if agent_pos in neighbors:
            if agent_pos == back_pos:
                return 0  # approaching from back is compliant
            else:
                return 1  # adjacent but not from back = non-compliant
        if ax == sx and ay != sy and agent_pos != back_pos:
            return 1
        return 0  # not adjacent = compliant
    # non-compliant if:  (a) still in or right of centre  AND  (b) has climbed > 1 row above start
    # if (ax > bx) or (ax == bx and (sy - ay > 1)):
    if (ax == sx and ay == sy):
        return 0
    elif (move_left and ax >= bx) or (not move_left and ax <= bx):
        return 1
    return 0


def obstacle_fetch_compliance(agent_pos, ball_pos, start_pos):
    ax, ay = agent_pos
    bx, by = ball_pos
    sx, sy = start_pos

    # non-compliant if:  (a) still in or right of centre  AND  (b) has climbed > 1 row above start
    # if (ax > bx) or (ax == bx and (sy - ay > 1)):
    if (ax > bx):
        return 1
    return 0


class SpuriousFetchEnv(MiniGridEnv):
    def __init__(self, size=8, numObjs=3, max_steps: int | None = None, use_box: bool = True, randomize: bool = False,
                 purple_ball: bool = False, purple_box: bool = False, add_red_ball: bool = False,
                 grey_ball: bool = False, mission_based: bool = True, test_box_idx: int = None,
                 compliance: bool = False, **kwargs):
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
        self.compliance = compliance
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
        self.place_obj(obj, top=(target_pos[0]-1, target_pos[1] - 1), size=(3, 3))

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
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

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

        # if self.compliance:
        #     spurious_compliance(obs, self.observation_space)
                # Get the position in front of the agent
        if self.compliance:
            info['compliance'] = spurious_box_compliance(fwd_cell)

        return obs, reward, terminated, truncated, info


class ObstructedMazeCompliance_1Dl(ObstructedMazeEnv):
    """
    A blue ball is hidden in a 2x1 maze. A locked door separates
    rooms. Doors are obstructed by a ball and keys are hidden in boxes.
    """

    def __init__(self, key_in_box=True, blocked=True, compliance=True, guided_reward=False, **kwargs):
        self.key_in_box = key_in_box
        self.blocked = blocked
        self.guided_reward = guided_reward
        self.still_blocking = True

        super().__init__(num_rows=1, num_cols=2, num_rooms_visited=2, **kwargs)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        self.door, self.door_pos = self.add_door(
            0,
            0,
            door_idx=0,
            color=self.door_colors[0],
            locked=True,
            key_in_box=self.key_in_box,
            blocked=self.blocked,
        )

        self.obj, _ = self.add_object(1, 0, "ball", color=self.ball_to_find_color)
        self.place_agent(0, 0)

    def _are_adjacent(self, pos_a, pos_b):
        """Check if Two positions are adjacent to one another."""
        ax, ay = pos_a
        bx, by = pos_b

        # Check if they are adjacent (including diagonals)
        return (abs(ax - bx) == 1 and ay == by) or (abs(ay - by) == 1 and ax == bx)

    def get_guidance(self, action):
        def action_to_onehot(action_idx):
            if action_idx is None:
                return [0, 0, 0, 0, 0, 0, 0]
            onehot = [0, 0, 0, 0, 0, 0, 0]
            onehot[action_idx] = 1
            return onehot

        # Get current target information
        target_pos, target_type, correct_action_idx = self._get_current_target()

        required_action = correct_action_idx
        if target_pos is None:
            # print(f'door opened: {self.door_opened}, key_found: {self.key_found} target not found  action: {action}, current action: {correct_action_idx}')
            if action == required_action:
                return 0, action_to_onehot(None)
            return 1, action_to_onehot(required_action)
            # Special case: Handle drop action (before calling agent_sees)

        # 1. If target not visible, compliant with None action - let the reward guide this
        if not self.agent_sees(*target_pos):
            return 0, action_to_onehot(None)

        # 2. If facing target, perform correct action
        fwd_cell = self.grid.get(*self.front_pos)
        if fwd_cell is not None and fwd_cell.type == target_type:
            if action == required_action:
                return 0, action_to_onehot(None)
            return 1, action_to_onehot(required_action)

        # 3. Target is visible - move to target
        required_action = self._navigate_to_target(target_pos)

        # Always: if action == required_action then compliant
        if action == required_action:
            return 0, action_to_onehot(None)
        return 1, action_to_onehot(required_action)

    def _get_current_target(self):
        """Determine what the agent should be targeting based on current phase."""
        key_pos = door_pos = box_pos = blocking_ball_pos = None
        key_in_box = False
        goal_pos = self.obj.cur_pos
        for i in range(self.width):
            for j in range(self.height):
                cell = self.grid.get(i, j)
                if cell is not None:
                    if cell.type == 'key':
                        key_pos = (i, j)
                    elif cell.type == 'door':
                        door_pos = (i, j)
                    elif cell.type == 'box':
                        box_pos = (i, j)
                        # Check if box contains key
                        if cell.contains is not None and cell.contains.type == 'key':
                            key_in_box = True

        # Identify blocking ball: any ball at (door_x - 1, door_y)
        if self.blocked and door_pos is not None and self.still_blocking:
            candidate_pos = (door_pos[0] - 1, door_pos[1])
            cell = self.grid.get(*candidate_pos)
            if cell is not None and cell.type == 'ball':
                blocking_ball_pos = candidate_pos

            if blocking_ball_pos is not None:
                # print(f'Going to pick up ball {blocking_ball_pos}, door_pos: {door_pos}, box_pos: {box_pos}, key_pos: {key_pos}, goal_pos: {goal_pos}')
                return blocking_ball_pos, 'ball', self.actions.pickup
            elif self.carrying and self.carrying.type == 'ball':
                drop_action = self._find_valid_drop_position_for_blocking_ball(box_pos, door_pos)
                # print(f'Carrying ball {blocking_ball_pos}, door_pos: {door_pos}, box_pos: {box_pos}, key_pos: {key_pos}, goal_pos: {goal_pos}')
                return None, 'drop', drop_action
            elif blocking_ball_pos is not None and blocking_ball_pos[0] != door_pos[0] - 1:
                self.still_blocking = False
                # print(f'Blocking ball moved away from door at {blocking_ball_pos}, door_pos: {door_pos}, box_pos: {box_pos}, key_pos: {key_pos}, goal_pos: {goal_pos}')

        if key_in_box:
            # Must open box first
            return box_pos, 'box', self.actions.toggle
        if not self.key_found:
            return key_pos, 'key', self.actions.pickup
        elif not self.door_opened:
            return door_pos, 'door', self.actions.toggle
        elif self.carrying and self.carrying.type == 'key':
            drop_action = self._find_valid_key_drop_position()
            return None, 'drop', drop_action
        else:
            return goal_pos, 'ball', self.actions.pickup

    def _find_valid_drop_position_for_blocking_ball(self, box_pos, door_pos):
        """
        Find a valid position to drop the blocking ball:
        - In the initial room (left of the door)
        - Prefer near the box
        - Not adjacent to the door
        - Prefer empty cells
        """
        candidates = []
        for x in range(self.width):
            for y in range(self.height):
                pos = (x, y)
                # Only allow positions in the initial room (assume initial room is left of the door)
                if pos[0] >= door_pos[0]:
                    continue
                if pos[0] == door_pos[0] - 1 and pos[1] == door_pos[1]:
                    continue
                # Not adjacent to the door
                if self._are_adjacent(pos, door_pos):
                    continue
                # Prefer near the box
                if box_pos and self._are_adjacent(pos, box_pos):
                    cell = self.grid.get(x, y)
                    if cell is None:
                        candidates.append(pos)
        # If no adjacent-to-box positions, allow any empty cell in initial room not adjacent to door
        if not candidates:
            for x in range(self.width):
                for y in range(self.height):
                    pos = (x, y)
                    if pos[0] >= door_pos[0]:
                        continue
                    if self._are_adjacent(pos, door_pos):
                        continue
                    if pos[0] == door_pos[0] - 1 and pos[1] == door_pos[1]:
                        continue
                    cell = self.grid.get(x, y)
                    if cell is None:
                        candidates.append(pos)
        # Choose the first candidate that is in front of the agent, otherwise navigate to the first
        for pos in candidates:
            if (self.front_pos[0], self.front_pos[1]) == pos:
                return self.actions.drop
        if candidates:
            best_direction = self._find_best_direction(candidates[0])
            if self.agent_dir == best_direction:
                fwd_cell = self.grid.get(*self.front_pos)
                if fwd_cell is None:
                    return self.actions.forward
            return self._turn_to_direction(best_direction)
        # If no good position, just drop in front if possible
        fwd_cell = self.grid.get(*self.front_pos)
        if fwd_cell is None:
            return self.actions.drop
        return None

    def _find_valid_key_drop_position(self):
        """Find a valid position and direction to drop the key, return (agent_pos, drop_pos, required_dir)."""

        # Get ball position
        ball_pos = self.obj.cur_pos
        if ball_pos is None:
            return None
        ball_x, ball_y = ball_pos

        # Get door position
        door_pos = None
        for i in range(self.width):
            for j in range(self.height):
                cell = self.grid.get(i, j)
                if cell is not None and cell.type == 'door':
                    door_pos = (i, j)
                    break
        if door_pos is None:
            return None
        door_x, door_y = door_pos
        # Check if ball is blocking the entrance (adjacent to door)
        ball_blocks_entrance = self._are_adjacent(ball_pos, door_pos)

        if ball_blocks_entrance:
            # Drop key in first room (around the door, but in starting room)
            potential_positions = [
                (door_x - 1, door_y),      # left of door (starting room side)
                (door_x - 2, door_y),      # 2 cells left of door
                (door_x - 1, door_y - 1),  # diagonal from door
                (door_x - 1, door_y + 1),  # diagonal from door
            ]
        else:
            # Ball doesn't block - drop in ball's room as before
            # Try positions in the ball's room that are away from both ball and door
            potential_positions = [
                (ball_x + 1, ball_y + 1),  # diagonal from ball
                (ball_x - 1, ball_y + 1),  # diagonal from ball
                (ball_x + 2, ball_y),      # 2 cells away from ball
                (ball_x, ball_y + 2),      # 2 cells away from ball
                (ball_x + 1, ball_y),      # 1 cell away from ball
                (ball_x, ball_y + 1),      # 1 cell away from ball
            ]

        for drop_target in potential_positions:
            x, y = drop_target

            # Check if position is in bounds and empty
            if (0 <= x < self.width and 0 <= y < self.height):
                cell = self.grid.get(x, y)
                if cell is None:
                    # Check it's not too close to door (unless we're intentionally dropping near door)
                    if not ball_blocks_entrance and door_pos is not None and self._are_adjacent(drop_target, door_pos):
                        continue
                    # Check it's not too close to ball
                    if not self._are_adjacent(drop_target, ball_pos):
                        # Found a good spot - now check if we can drop there
                        if (self.front_pos[0] == x and self.front_pos[1] == y):
                            # Facing the drop spot - can drop
                            return self.actions.drop
                        else:
                            # Navigate to position where we can drop there
                            best_direction = self._find_best_direction(drop_target)
                            if self.agent_dir == best_direction:
                                fwd_cell = self.grid.get(*self.front_pos)
                                if fwd_cell is None:
                                    return self.actions.forward
                            return self._turn_to_direction(best_direction)
        # If no good positions found, just drop in front wherever we are
        fwd_cell = self.grid.get(*self.front_pos)
        if fwd_cell is None:
            return self.actions.drop

        return None

    def _navigate_to_target(self, target_pos):
        """Navigate to target when it's visible."""        
        # Find the best direction among 4 possible positions
        best_direction = self._find_best_direction(target_pos)

        # If already facing the best direction, move forward
        if self.agent_dir == best_direction:
            return self.actions.forward
        else:
            # Turn towards the best direction
            return self._turn_to_direction(best_direction)

    def _find_best_direction(self, target_pos):
        """Find direction that gets agent closest to target among 4 possible positions."""
        agent_x, agent_y = self.agent_pos
        target_x, target_y = target_pos

        # 4 possible adjacent positions: right, down, left, up
        possible_positions = [
            (agent_x + 1, agent_y),  # right (dir 0)
            (agent_x, agent_y + 1),  # down (dir 1)
            (agent_x - 1, agent_y),  # left (dir 2)
            (agent_x, agent_y - 1)   # up (dir 3)
        ]

        distances = []
        valid_directions = []

        for i, pos in enumerate(possible_positions):
            if self._is_valid_position(pos, target_pos):
                # Calculate Euclidean distance from this position to target
                dist = np.sqrt((target_x - pos[0])**2 + (target_y - pos[1])**2)
                distances.append(dist)
                valid_directions.append(i)

        # If no valid directions, default to right (or any direction)
        if not valid_directions:
            return 0  # Default direction

        # Find direction with minimum distance
        min_dist_idx = distances.index(min(distances))
        return valid_directions[min_dist_idx]

    def _is_valid_position(self, pos, target_pos):
        """Check if a position is valid (in bounds and walkable)."""
        x, y = pos

        # Check bounds
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False

        cell = self.grid.get(x, y)
        # Can move to empty cells, target cell, or open doors
        return (cell is None or
                (x, y) == target_pos or
                (cell.type == 'door' and cell.is_open))  # type:ignore

    def _turn_to_direction(self, required_dir):
        """Calculate which action (left/right) to turn towards required direction."""
        angle_diff = (required_dir - self.agent_dir) % 4

        if angle_diff == 1:  # 90 degrees clockwise
            return self.actions.right
        elif angle_diff == 2:  # 180 degrees (choose right)
            return self.actions.right
        elif angle_diff == 3:  # 270 degrees clockwise = 90 degrees counter-clockwise
            return self.actions.left
        else:  # angle_diff == 0 (already facing correct direction)
            return self.actions.forward

    def step(self, action):
        guidance_label, user_action_onehot = self.get_guidance(action)
        # print(compliance, user_action_onehot)
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying.type == 'key':
                self.key_found = True

        if not self.door.is_locked:
            self.door_opened = True

        info['compliance'] = guidance_label
        info['user_actions'] = user_action_onehot

        if self.guided_reward:
            # if terminated or truncated:
            #     reward -= self.accum_reward  # type: ignore
            if guidance_label == 0:
                reward += 1.0 / self.max_steps  # type: ignore
                # self.accum_reward += 0.1  # type: ignore

        # print(f'reward: {reward}, compliance: {guidance_label}, accum_reward: {self.accum_reward}')

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type:ignore
        obs, info = super().reset(seed=seed, options=options)
        self.key_found = False
        self.door_opened = False
        self.start_pos = self.agent_pos
        # print('new episode')

        info['compliance'] = 0
        info['user_actions'] = [0, 0, 0, 0, 0, 0, 0]

        for i in range(self.width):
            for j in range(self.height):
                cell = self.grid.get(i, j)
                if cell is not None:
                    if cell.type == 'key':
                        key_pos = (i, j)
                        self.key_pos = key_pos

        self.accum_reward = 0.0
        return obs, info


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
    register(
        id="MiniGrid-ObstructedMazeCompliance_1Dlhb-v0",
        entry_point="env.minigrid:ObstructedMazeCompliance_1Dl",
        kwargs={"key_in_box": True, "blocked": True, "guided_reward": False},
    )
    register(
        id="MiniGrid-ObstructedMazeCompliance_1Dlh-v0",
        entry_point="env.minigrid:ObstructedMazeCompliance_1Dl",
        kwargs={"key_in_box": True, "blocked": False, "guided_reward": False},
    )
    register(
        id="MiniGrid-ObstructedMazeCompliance_1Dl-v0",
        entry_point="env.minigrid:ObstructedMazeCompliance_1Dl",
        kwargs={"key_in_box": False, "blocked": False, "guided_reward": False},
    )
    register(
        id="MiniGrid-ObstructedMazeCompliance_1Dlhb-v1",
        entry_point="env.minigrid:ObstructedMazeCompliance_1Dl",
        kwargs={"key_in_box": True, "blocked": True, "guided_reward": True},
    )
    register(
        id="MiniGrid-ObstructedMazeCompliance_1Dlh-v1",
        entry_point="env.minigrid:ObstructedMazeCompliance_1Dl",
        kwargs={"key_in_box": True, "blocked": False, "guided_reward": True},
    )
    register(
        id="MiniGrid-ObstructedMazeCompliance_1Dl-v1",
        entry_point="env.minigrid:ObstructedMazeCompliance_1Dl",
        kwargs={"key_in_box": False, "blocked": False, "guided_reward": True},
    )
