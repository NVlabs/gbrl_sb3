##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
from __future__ import annotations

from typing import Any, Iterable, SupportsFloat, TypeVar
from gymnasium.core import ActType, ObsType
import numpy as np
from gymnasium.envs.registration import register
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box
from minigrid.minigrid_env import MiniGridEnv, TILE_PIXELS
from typing import Optional

from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT, STATE_TO_IDX
from gbrl.common.utils import categorical_dtype

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}


def categorical_obs_encoding(observation, image_shape, flattened_image_shape, FullyObsWrapper = False, is_mixed = False):
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

# def spurious_compliance(observation, observation_space):
#     image_shape = observation_space['image'].shape
#     flattened_image_shape = image_shape[0]*image_shape[1] + 2
#     categorical_obs = categorical_obs_encoding(observation, image_shape, flattened_image_shape, FullyObsWrapper=False, is_mixed=False)
#     print('')


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


class SimpleFetchEnv(MiniGridEnv):
    def __init__(self, max_steps: Optional[int] = None, compliance: bool = False, move_left: bool = True, from_back: bool = False, **kwargs):
        self.numObjs = 1
        self.size = 16
        self.obj_types = ["ball"]
        self.colors = {
            "red": np.array([255, 0, 0]),
            "green": np.array([0, 255, 0]),
            "blue": np.array([0, 0, 255]),
        }
        self.color_names = sorted(list(self.colors.keys()))
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

        self.move_left = move_left
        self.from_back = from_back
        self.agent_pov = True
        # self.agent_pov = False
        self.metadata['render_fps'] = 2
        # self.metadata['render_fps'] = 10

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
        # Create the green ball
        obs_green = Ball('green')
        # Calculate fixed positions
        center_x = width // 2  # Center x coordinate
        center_y = height // 2  # Center y coordinate
        agent_y = height - 2  # Bottom center (one cell up from bottom wall
        # Place agent at bottom center with fixed orientation (facing up)
        self.agent_pos = (center_x, agent_y)
        self.starting_pos = (center_x, agent_y)
        self.agent_dir = 3  # Facing up (0=right, 1=down, 2=left, 3=up)
        # Place green ball at center of the room
        self.grid.set(center_x, center_y, obs_green)
        # Choose the green ball as the target
        target = obs_green

        self.targetType = target.type
        self.targetColor = target.color
        descStr = f"{self.targetColor} {self.targetType}"
        # Generate the mission string
        self.mission = "get a %s" % descStr
        assert hasattr(self, "mission")
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

    def step(self, action):
        compliance = None
        if self.compliance:
            # Find the ball position in the grid
            ball_pos = None
            for i in range(self.width):
                for j in range(self.height):
                    cell = self.grid.get(i, j)
                    if cell is not None and cell.type == self.targetType and cell.color == self.targetColor:
                        ball_pos = (i, j)
                        break
                if ball_pos is not None:
                    break

            if ball_pos is not None:
                compliance = simple_fetch_compliance(self.agent_pos, ball_pos, self.starting_pos, self.move_left, self.from_back)

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
        if compliance is not None:
            info['compliance'] = compliance

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)

        if self.compliance:
            # Find the ball position in the grid
            ball_pos = None
            for i in range(self.width):
                for j in range(self.height):
                    cell = self.grid.get(i, j)
                    if cell is not None and cell.type == self.targetType and cell.color == self.targetColor:
                        ball_pos = (i, j)
                        break
                if ball_pos is not None:
                    break

            if ball_pos is not None:
                info['compliance'] = simple_fetch_compliance(self.agent_pos, ball_pos, self.starting_pos, self.move_left, self.from_back)
        return  obs, info

    def get_frame(
        self,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        """Returns an RGB image corresponding to the whole environment or the agent's point of view.

        Args:

            highlight (bool): If true, the agent's field of view or point of view is highlighted with a lighter gray color.
            tile_size (int): How many pixels will form a tile from the NxM grid.
            agent_pov (bool): If true, the rendered frame will only contain the point of view of the agent.

        Returns:

            frame (np.ndarray): A frame of type numpy.ndarray with shape (x, y, 3) representing RGB values for the x-by-y pixel image.

        """

        # if agent_pov:
        #     return self.get_pov_render(tile_size)
        # else:
        return self.get_full_render(highlight, tile_size)
    
class DistanceFetchEnv(MiniGridEnv):
    def __init__(self, max_steps: Optional[int] = None, compliance: bool = False, move_left: bool = True, from_back: bool = False, **kwargs):
        self.numObjs = 1
        self.size = 16
        self.obj_types = ["ball"]
        self.colors = {
            "red": np.array([255, 0, 0]),
            "green": np.array([0, 255, 0]),
            "blue": np.array([0, 0, 255]),
        }
        self.color_names = sorted(list(self.colors.keys()))
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

        self.move_left = move_left
        self.from_back = from_back
        self.agent_pov = True
        # self.agent_pov = False
        self.metadata['render_fps'] = 2
        # self.metadata['render_fps'] = 10

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
        # Create the green ball
        obs_green = Ball('green')
        # Calculate fixed positions
        center_x = width // 2  # Center x coordinate
        center_y = height // 2  # Center y coordinate
        agent_y = height - 2  # Bottom center (one cell up from bottom wall
        # Place agent at bottom center with fixed orientation (facing up)
        self.agent_pos = (center_x, agent_y)
        self.starting_pos = (center_x, agent_y)
        self.target_pos = (center_x, center_y)
        self.agent_dir = 3  # Facing up (0=right, 1=down, 2=left, 3=up)
        # Place green ball at center of the room
        self.grid.set(center_x, center_y, obs_green)
        # Choose the green ball as the target
        target = obs_green

        self.targetType = target.type
        self.targetColor = target.color
        descStr = f"{self.targetColor} {self.targetType}"
        # Generate the mission string
        self.mission = "get a %s" % descStr
        assert hasattr(self, "mission")
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

    def _get_normalized_distance(self) -> float:
        agent_pos = np.array(self.agent_pos)
        target_pos = np.array(self.target_pos)
        dist = np.linalg.norm((target_pos - agent_pos))
        # Normalize distance by the maximum possible distance in the grid
        # Max distance is from corner to corner: sqrt((size-1)^2 + (size-1)^2)
        max_dist = np.sqrt(2) * (self.size - 1)
        normalized_dist = min(dist / max_dist, 1.0)  # Ensure it doesn't exceed 1
        # Return reward that decreases with distance, between 0 and 1
        return normalized_dist

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        reward = 0.0
        current_distance = self._get_normalized_distance()

        step_penalty_factor = 0.9 * (self.step_count / self.max_steps)

        # Only reward if agent gets closer than ever before
        if current_distance < self.best_distance:
            improvement = self.best_distance - current_distance
            reward = improvement  # This ensures max total reward is 1.0
            self.best_distance = current_distance
            self.cumulative_reward += reward

        if self.carrying:
            if (
                self.carrying.color == self.targetColor
                and self.carrying.type == self.targetType
            ):
                terminated = True
                # Agent gets remaining reward to reach 1.0 if not already there
                if current_distance > 0:
                    penalized_max_reward = 1.0 - step_penalty_factor
                    remaining_reward = penalized_max_reward - self.cumulative_reward
                    reward += remaining_reward
            else:
                terminated = True
                reward = 0.0

        if truncated:
            reward = -step_penalty_factor  # 

        if self.compliance:
            # Find the ball position in the grid
            ball_pos = None
            for i in range(self.width):
                for j in range(self.height):
                    cell = self.grid.get(i, j)
                    if cell is not None and cell.type == self.targetType and cell.color == self.targetColor:
                        ball_pos = (i, j)
                        break
                if ball_pos is not None:
                    break

            if ball_pos is not None:
                info['compliance'] = simple_fetch_compliance(self.agent_pos, ball_pos, self.starting_pos, self.move_left, self.from_back)

        return obs, float(reward), terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)

        self.best_distance = 1.0  # Start with max normalized distance
        # Track cumulative reward to cap at 1.0
        self.cumulative_reward = 0.0

        if self.compliance:
            # Find the ball position in the grid
            ball_pos = None
            for i in range(self.width):
                for j in range(self.height):
                    cell = self.grid.get(i, j)
                    if cell is not None and cell.type == self.targetType and cell.color == self.targetColor:
                        ball_pos = (i, j)
                        break
                if ball_pos is not None:
                    break

            if ball_pos is not None:
                info['compliance'] = simple_fetch_compliance(self.agent_pos, ball_pos, self.starting_pos, self.move_left, self.from_back)
        return  obs, info

    def get_frame(
        self,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        """Returns an RGB image corresponding to the whole environment or the agent's point of view.

        Args:

            highlight (bool): If true, the agent's field of view or point of view is highlighted with a lighter gray color.
            tile_size (int): How many pixels will form a tile from the NxM grid.
            agent_pov (bool): If true, the rendered frame will only contain the point of view of the agent.

        Returns:

            frame (np.ndarray): A frame of type numpy.ndarray with shape (x, y, 3) representing RGB values for the x-by-y pixel image.

        """

        # if agent_pov:
        #     return self.get_pov_render(tile_size)
        # else:
        return self.get_full_render(highlight, tile_size)

class SimpleObstacleFetchEnv(MiniGridEnv):
    def __init__(self, max_steps: Optional[int] = None, compliance: bool = False, **kwargs):
        self.numObjs = 1
        self.size = 16
        self.obj_types = ["ball"]
        self.colors = {
            "red": np.array([255, 0, 0]),
            "green": np.array([0, 255, 0]),
            "blue": np.array([0, 0, 255]),
        }
        self.color_names = sorted(list(self.colors.keys()))
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
        # Create the green ball
        obs_green = Ball('green')
        obstacle = Box('red')
        # Calculate fixed positions
        center_x = width // 2  # Center x coordinate
        center_y = height // 2  # Center y coordinate
        agent_y = height - 2  # Bottom center (one cell up from bottom wall
        obstacle_y = center_y + 2
        # Place agent at bottom center with fixed orientation (facing up)
        self.agent_pos = (center_x, agent_y)
        self.starting_pos = (center_x, agent_y)
        self.agent_dir = 3  # Facing up (0=right, 1=down, 2=left, 3=up)
        # Place green ball at center of the room
        self.grid.set(center_x, center_y, obs_green)
        self.grid.set(center_x, obstacle_y, obstacle)
        # Choose the green ball as the target
        target = obs_green
        self.moved_left = False
        self.moved_right = False

        self.targetType = target.type
        self.targetColor = target.color
        descStr = f"{self.targetColor} {self.targetType}"
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
                if self.moved_right and not self.moved_left:
                    reward += 0.1
                    reward = min(reward, 1.0)  # Cap the reward to 1.0
            else:
                reward = 0
                terminated = True

        fwd_pos = self.front_pos
        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)
        if fwd_cell is not None and fwd_cell.type == 'box' and fwd_cell.color == 'red':
            if action in [self.actions.toggle, self.actions.forward, self.actions.done]:
                reward = 0
                terminated = True
            if action == self.actions.left:
                self.moved_left = True
            elif action == self.actions.right:
                self.moved_right = True
        
            # print(f'red box, action: {action} reward: {reward} terminated: {terminated}')
        if self.compliance:
            # Find the ball position in the grid
            ball_pos = None
            for i in range(self.width):
                for j in range(self.height):
                    cell = self.grid.get(i, j)
                    if cell is not None and cell.type == self.targetType and cell.color == self.targetColor:
                        ball_pos = (i, j)
                        break
                if ball_pos is not None:
                    break

            if ball_pos is not None:
                info['compliance'] = obstacle_fetch_compliance(self.agent_pos, ball_pos, self.starting_pos)

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)

        if self.compliance:
            # Find the ball position in the grid
            ball_pos = None
            for i in range(self.width):
                for j in range(self.height):
                    cell = self.grid.get(i, j)
                    if cell is not None and cell.type == self.targetType and cell.color == self.targetColor:
                        ball_pos = (i, j)
                        break
                if ball_pos is not None:
                    break

            if ball_pos is not None:
                info['compliance'] = obstacle_fetch_compliance(self.agent_pos, ball_pos, self.starting_pos)
        return  obs, info

    def get_frame(
        self,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        """Returns an RGB image corresponding to the whole environment or the agent's point of view.

        Args:

            highlight (bool): If true, the agent's field of view or point of view is highlighted with a lighter gray color.
            tile_size (int): How many pixels will form a tile from the NxM grid.
            agent_pov (bool): If true, the rendered frame will only contain the point of view of the agent.

        Returns:

            frame (np.ndarray): A frame of type numpy.ndarray with shape (x, y, 3) representing RGB values for the x-by-y pixel image.

        """

        # if agent_pov:
        #     return self.get_pov_render(tile_size)
        # else:
        return self.get_full_render(highlight, tile_size)


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
        id="MiniGrid-SpuriousComplianceFetch-8x8-N3-v0",
        entry_point="env.minigrid:SpuriousFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "use_box": True, "mission_based": True, "randomize": False, 'compliance': True},
    )
    register(
        id="MiniGrid-SpuriousComplianceFetch-8x8-N3-v1",
        entry_point="env.minigrid:SpuriousFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "use_box": True, "mission_based": False, "randomize": False, 'compliance': True},
    )
    register(
        id="MiniGrid-SpuriousComplianceFetch-8x8-N3-v2",
        entry_point="env.minigrid:SpuriousFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "use_box": True, "mission_based": False, "randomize": True, 'compliance': True},
    )
    register(
        id="MiniGrid-SpuriousComplianceFetch-8x8-N3-v3",
        entry_point="env.minigrid:SpuriousFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "use_box": False, 'compliance': True},
    )
    register(
        id="MiniGrid-SimpleFetch-16x16-N1-v0",
        entry_point="env.minigrid:SimpleFetchEnv",
        kwargs={"compliance": False},
    )
    register(
        id="MiniGrid-SimpleFetch-16x16-N1-v1",
        entry_point="env.minigrid:SimpleFetchEnv",
        kwargs={"compliance": True, "move_left": True},
    )
    register(
        id="MiniGrid-SimpleFetch-16x16-N1-v2",
        entry_point="env.minigrid:SimpleFetchEnv",
        kwargs={"compliance": True, "move_left": False},
    )
    register(
        id="MiniGrid-SimpleFetch-16x16-N1-v3",
        entry_point="env.minigrid:SimpleFetchEnv",
        kwargs={"compliance": True, "from_back": True},
    )
    register(
        id="MiniGrid-SimpleObstacleFetch-16x16-N1-v0",
        entry_point="env.minigrid:SimpleObstacleFetchEnv",
        kwargs={"compliance": False},
    )
    register(
        id="MiniGrid-SimpleObstacleFetch-16x16-N1-v1",
        entry_point="env.minigrid:SimpleObstacleFetchEnv",
        kwargs={"compliance": True},
    )
    register(
        id="MiniGrid-DistanceFetch-16x16-N1-v0",
        entry_point="env.minigrid:DistanceFetchEnv",
        kwargs={"compliance": False},
    )
    register(
        id="MiniGrid-DistanceFetch-16x16-N1-v1",
        entry_point="env.minigrid:DistanceFetchEnv",
        kwargs={"compliance": True},
    )