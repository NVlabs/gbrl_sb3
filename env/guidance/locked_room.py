##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
from __future__ import annotations

import re
from typing import Any

import numpy as np
from gbrl.common.utils import numerical_dtype
from gymnasium.core import ActType, ObsType
from gymnasium.envs.registration import register
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT, STATE_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Key
from minigrid.envs.lockedroom import LockedRoomEnv
from minigrid.minigrid_env import TILE_PIXELS, MiniGridEnv

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}


class GuidedLockedRoomEnv(LockedRoomEnv):
    """
    ## Description

    The environment has six rooms, one of which is locked. The agent receives
    a textual mission string as input, telling it which room to go to in order
    a textual mission string as input, telling it which room to go to in order
    to get the key that opens the locked room. It then has to go into the locked
    room in order to reach the final goal. This environment is extremely
    difficult to solve with vanilla reinforcement learning alone.

    ## Mission Space

    "get the {lockedroom_color} key from the {keyroom_color} room, unlock the {door_color} door and go to the goal"

    {lockedroom_color}, {keyroom_color}, and {door_color} can be "red", "green",
    "blue", "purple", "yellow" or "grey".

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Pick up an object         |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

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
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-LockedRoom-v0`

    """

    def __init__(self, action_masking=False, guided_reward=False, **kwargs):

        self.guided_reward = guided_reward

        super().__init__(**kwargs)

    def to_prob(self, actions):
        prob_actions = np.zeros(len(self.actions), dtype=numerical_dtype)
        for a in actions:
            prob_actions[a] = 1.0 / len(actions)
        return prob_actions

    def _get_current_room(self):
        """Determine which room the agent is currently in based on position"""
        x, y = self.agent_pos

        # Check if in corridor (between lWallIdx and rWallIdx)
        lWallIdx = self.width // 2 - 2
        rWallIdx = self.width // 2 + 2
        if lWallIdx < x < rWallIdx:
            return None  # In corridor, not in any room

        # Check each room's bounds
        for room in self.rooms:
            topX, topY = room.top
            sizeX, sizeY = room.size
            if topX < x < topX + sizeX - 1 and topY < y < topY + sizeY - 1:
                return room.color

        return None  # Not in any room (e.g., on a wall)

    def get_guidance(self, action):
        def action_to_onehot(action_idx):
            if action_idx is None:
                return [0, 0, 0, 0, 0, 0, 0]
            onehot = [0, 0, 0, 0, 0, 0, 0]
            onehot[action_idx] = 1
            return onehot

        # Update current room
        self.current_room = self._get_current_room()

        # Check what's in front
        fwd_cell = self.grid.get(*self.front_pos)

        # Phase 1: Before getting the key
        if not self.key_found:
            # Check if we're about to enter a wrong room
            if fwd_cell and fwd_cell.type == 'door':
                door_color = fwd_cell.color
                # If it's not the key room, avoid entering - turn towards key room
                if door_color != self.keyroom_color:
                    self.guidance_target = 'avoid_wrong_room'
                    self.guidance_value = 1
                    # Determine which direction to turn based on key room position
                    turn_action = self._get_turn_away_from_door(self.keyroom_color)
                    self.guidance_action_probs = self.to_prob([turn_action])
                    return 1, self.guidance_action_probs
            
            # Check if we're in a wrong room and need to exit
            if self.current_room is not None and self.current_room != self.keyroom_color:
                self.guidance_target = 'exit_wrong_room'
                result = self._navigate_to_room_exit()
                self.guidance_value = result[0]
                self.guidance_action_probs = result[1]
                return result
            
            # If in the key room, navigate to key
            if self.current_room == self.keyroom_color:
                self.guidance_target = 'key'
                result = self._navigate_to_object('key', self.key_color)
                self.guidance_value = result[0]
                self.guidance_action_probs = result[1]
                return result
            
            # If in corridor, navigate to key room door
            if self.current_room is None:
                self.guidance_target = 'key_room_door'
                result = self._navigate_to_door(self.keyroom_color)
                self.guidance_value = result[0]
                self.guidance_action_probs = result[1]
                return result
        
        # Phase 2: After getting the key - only avoid wrong rooms
        else:
            # Check if we're about to enter a wrong room
            if fwd_cell and fwd_cell.type == 'door':
                door_color = fwd_cell.color
                # Avoid all rooms except the locked room
                if door_color != self.door_color:
                    self.guidance_target = 'avoid_wrong_room'
                    self.guidance_value = 1
                    # Determine which direction to turn based on locked room position
                    turn_action = self._get_turn_away_from_door(self.door_color)
                    self.guidance_action_probs = self.to_prob([turn_action])
                    return 1, self.guidance_action_probs
            
            # Check if we're in a wrong room and need to exit
            if self.current_room is not None and self.current_room != self.door_color:
                self.guidance_target = 'exit_wrong_room'
                result = self._navigate_to_room_exit()
                self.guidance_value = result[0]
                self.guidance_action_probs = result[1]
                return result
        
        # No guidance needed - let the agent figure out the rest
        self.guidance_target = 'no_guidance'
        self.guidance_value = 0
        self.guidance_action_probs = action_to_onehot(None)
        return 0, self.guidance_action_probs

    def _get_turn_away_from_door(self, target_room_color):
        """Determine which way to turn when facing a wrong door to move towards target room."""
        agent_x, agent_y = self.agent_pos
        
        # Find the target room door position
        target_door_pos = None
        for room in self.rooms:
            if room.color == target_room_color:
                target_door_pos = room.doorPos
                break
        
        if target_door_pos is None:
            # Default to turning right if we can't find target
            return self.actions.right
        
        target_x, target_y = target_door_pos
        
        # Check if we're in the corridor
        lWallIdx = self.width // 2 - 2
        rWallIdx = self.width // 2 + 2
        in_corridor = lWallIdx < agent_x < rWallIdx
        
        if not in_corridor:
            # Not in corridor, default to right
            return self.actions.right
        
        # We're in corridor, determine best turn direction
        # Based on which direction gets us closer to target_y
        if agent_y < target_y:
            # Target is below us, we want to turn to face down (direction 1)
            required_dir = 1
        elif agent_y > target_y:
            # Target is above us, we want to turn to face up (direction 3)
            required_dir = 3
        else:
            # Same y-coordinate, turn towards target x
            if agent_x < target_x:
                required_dir = 0  # right
            else:
                required_dir = 2  # left
        
        # Calculate which turn gets us to required_dir
        angle_diff = (required_dir - self.agent_dir) % 4
        
        if angle_diff == 1 or angle_diff == 2:
            # Turn right is closer (1 or 2 steps clockwise)
            return self.actions.right
        else:
            # Turn left is closer (3 steps clockwise = 1 step counter-clockwise)
            return self.actions.left

    def _navigate_to_object(self, obj_type, obj_color):
        """Navigate to a specific object (key or goal) in the current room."""
        # Find the object
        target_pos = None
        for i in range(self.width):
            for j in range(self.height):
                cell = self.grid.get(i, j)
                if cell is not None and cell.type == obj_type:
                    if obj_color is None or cell.color == obj_color:
                        target_pos = (i, j)
                        break
            if target_pos:
                break
        
        if target_pos is None:
            return 0, self._action_to_onehot(None)
        
        # Check if facing the object
        fwd_cell = self.grid.get(*self.front_pos)
        if fwd_cell and fwd_cell.type == obj_type and (obj_color is None or fwd_cell.color == obj_color):
            # Facing the object - pick it up (or move forward if goal)
            if obj_type == 'key':
                return 1, self.to_prob([self.actions.pickup])
            else:
                return 1, self.to_prob([self.actions.forward])
        
        # Navigate to the object
        required_action = self._navigate_to_target(target_pos)
        return 1, self.to_prob([required_action])

    def _navigate_to_door(self, door_color):
        """Navigate to a door of specific color in the corridor."""
        # Find the door
        door_pos = None
        for room in self.rooms:
            if room.color == door_color:
                door_pos = room.doorPos
                break
        
        if door_pos is None:
            return 0, self._action_to_onehot(None)
        
        # Check if facing the door
        fwd_cell = self.grid.get(*self.front_pos)
        if fwd_cell and fwd_cell.type == 'door' and fwd_cell.color == door_color:
            # Facing the door - enter
            return 1, self.to_prob([self.actions.forward])
        
        # Special navigation for corridor - align y-coordinate first, then x
        agent_x, agent_y = self.agent_pos
        door_x, door_y = door_pos
        
        # Check if we're in the corridor
        lWallIdx = self.width // 2 - 2
        rWallIdx = self.width // 2 + 2
        in_corridor = lWallIdx < agent_x < rWallIdx
        
        if in_corridor:
            # First priority: align y-coordinate with the door
            if agent_y != door_y:
                # Need to move up or down
                if agent_y < door_y:
                    # Need to go down (direction 1)
                    required_dir = 1
                else:
                    # Need to go up (direction 3)
                    required_dir = 3
                
                if self.agent_dir == required_dir:
                    return 1, self.to_prob([self.actions.forward])
                else:
                    return 1, self.to_prob([self._turn_to_direction(required_dir)])
            else:
                # Y is aligned, now move towards the door (left or right)
                if agent_x < door_x:
                    # Door is to the right (direction 0)
                    required_dir = 0
                else:
                    # Door is to the left (direction 2)
                    required_dir = 2
                
                if self.agent_dir == required_dir:
                    return 1, self.to_prob([self.actions.forward])
                else:
                    return 1, self.to_prob([self._turn_to_direction(required_dir)])
        else:
            # Not in corridor, use standard navigation
            required_action = self._navigate_to_target(door_pos)
            return 1, self.to_prob([required_action])

    def _navigate_to_target(self, target_pos):
        """Navigate to target position when visible. Returns action."""
        # Check if we're in the corridor
        agent_x, agent_y = self.agent_pos
        lWallIdx = self.width // 2 - 2
        rWallIdx = self.width // 2 + 2
        in_corridor = lWallIdx < agent_x < rWallIdx
        
        if in_corridor:
            # Use corridor-specific navigation (align y first, then x)
            target_x, target_y = target_pos
            
            # First priority: align y-coordinate
            if agent_y != target_y:
                if agent_y < target_y:
                    required_dir = 1  # down
                else:
                    required_dir = 3  # up
                
                if self.agent_dir == required_dir:
                    return self.actions.forward
                else:
                    return self._turn_to_direction(required_dir)
            else:
                # Y is aligned, now move in x direction
                if agent_x < target_x:
                    required_dir = 0  # right
                else:
                    required_dir = 2  # left
                
                if self.agent_dir == required_dir:
                    return self.actions.forward
                else:
                    return self._turn_to_direction(required_dir)
        else:
            # Not in corridor, use original logic
            best_direction = self._find_best_direction(target_pos)

            if self.agent_dir == best_direction:
                return self.actions.forward
            else:
                return self._turn_to_direction(best_direction)

    def _navigate_to_room_exit(self):
        """Navigate to the door to exit the current room."""
        # Find the door of the current room
        door_pos = None
        for room in self.rooms:
            if room.color == self.current_room:
                door_pos = room.doorPos
                break

        if door_pos is None:
            return 0, self._action_to_onehot(None)

        # Check if facing the door
        fwd_cell = self.grid.get(*self.front_pos)
        if fwd_cell and fwd_cell.type == 'door' and fwd_cell.color == self.current_room:
            # Facing the door - exit
            return 1, self.to_prob([self.actions.forward])
        
        # Navigate to the door
        required_action = self._navigate_to_target(door_pos)
        return 1, self.to_prob([required_action])

    def _find_best_direction(self, target_pos):
        """Find direction that gets agent closest to target."""
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
            if self._is_valid_position(pos):
                # Calculate Manhattan distance from this position to target
                dist = abs(target_x - pos[0]) + abs(target_y - pos[1])
                distances.append(dist)
                valid_directions.append(i)

        # If no valid directions, default to right
        if not valid_directions:
            return 0

        # Find direction with minimum distance
        min_dist_idx = distances.index(min(distances))
        return valid_directions[min_dist_idx]

    def _is_valid_position(self, pos):
        """Check if a position is valid (in bounds and walkable)."""
        x, y = pos

        # Check bounds
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False

        cell = self.grid.get(x, y)
        # Can move to empty cells or open doors
        return cell is None or (cell.type == 'door' and cell.is_open)

    def _turn_to_direction(self, required_dir):
        """Calculate which action (left/right) to turn towards required direction."""
        angle_diff = (required_dir - self.agent_dir) % 4

        if angle_diff == 1:  # 90 degrees clockwise
            return self.actions.right
        elif angle_diff == 2:  # 180 degrees
            return self.actions.right
        elif angle_diff == 3:  # 270 degrees clockwise = 90 degrees counter-clockwise
            return self.actions.left
        else:  # angle_diff == 0
            return self.actions.forward

    def _action_to_onehot(self, action_idx):
        if action_idx is None:
            return [0, 0, 0, 0, 0, 0, 0]
        onehot = [0, 0, 0, 0, 0, 0, 0]
        onehot[action_idx] = 1
        return onehot

    def step(self, action):
        guidance_label, user_action_onehot = self.get_guidance(action)
        
        self.last_grid = self.grid.copy()
        self.last_agent_pos = tuple(self.agent_pos)
        self.last_agent_dir = self.agent_dir
        self.last_carrying = self.carrying
        self.last_action = action

        obs, reward, terminated, truncated, info = super().step(action)
        
        # Update key_found flag
        if self.carrying and self.carrying.type == 'key' and self.carrying.color == self.key_color:
            self.key_found = True
        
        # Update door_opened flag
        for room in self.rooms:
            if room.color == self.door_color:
                door_pos = room.doorPos
                door = self.grid.get(*door_pos)
                if door and door.type == 'door' and door.is_open:
                    self.door_opened = True
                break

        info['guidance_labels'] = guidance_label
        info['guidance_actions'] = user_action_onehot

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type:ignore
        obs, info = super().reset(seed=seed, options=options)

        info['guidance_labels'] = 0
        info['guidance_actions'] = [0, 0, 0, 0, 0, 0, 0]

        def extract_words(s):
            key = re.search(r'(\w+)\s+key', s)
            room = re.search(r'(\w+)\s+room', s)
            door = re.search(r'(\w+)\s+door', s)
            return {
                'before_key': key.group(1) if key else None,
                'before_room': room.group(1) if room else None,
                'before_door': door.group(1) if door else None,
            }
        mission_words = extract_words(self.mission)
        self.key_color = mission_words['before_key']
        self.keyroom_color = mission_words['before_room']
        self.door_color = mission_words['before_door']
        self.key_found = False
        self.door_opened = False

        self.guidance_target = 'no_guidance'  # Track what we're guiding towards
        self.guidance_value = 0  # Track if guidance is active
        self.guidance_action_probs = None  # Track guidance action distribution
        
        self.last_grid = self.grid.copy()
        self.last_agent_pos = tuple(self.agent_pos)
        self.last_agent_dir = self.agent_dir
        self.last_carrying = self.carrying
        self.last_action = None

        return obs, info
    

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
        self.carrying = self.last_carrying
        
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
        
        # Prepare text information
        guidance_text = f"Target: {self.guidance_target}"
        value_text = f"Guidance: {self.guidance_value}"
        
        # Format action probabilities
        if self.guidance_action_probs is not None:
            action_names = ['Left', 'Right', 'Fwd', 'Pick', 'Drop', 'Toggle', 'Done']
            # Find non-zero actions
            active_actions = [f"{action_names[i]}: {self.guidance_action_probs[i]:.2f}" 
                            for i in range(len(self.guidance_action_probs)) 
                            if self.guidance_action_probs[i] > 0]
            action_text = "Actions: " + ", ".join(active_actions) if active_actions else "Actions: None"
        else:
            action_text = "Actions: None"

        # Format the actual action taken
        if self.last_action is not None:
            action_names = ['Left', 'Right', 'Fwd', 'Pick', 'Drop', 'Toggle', 'Done']
            taken_text = f"Action Taken: {action_names[self.last_action]}"
        else:
            taken_text = "Action Taken: None"
        
        # Add text to banner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_color = (0, 0, 0)  # Black text
        
        # Line 1: Target
        cv2.putText(banner, guidance_text, (10, 20), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        # Line 2: Guidance value
        cv2.putText(banner, value_text, (10, 40), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        # Line 3: Actions
        cv2.putText(banner, action_text, (10, 60), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        # Line 4: Action taken
        cv2.putText(banner, taken_text, (10, 80), font, font_scale, text_color, thickness, cv2.LINE_AA)
        # Concatenate banner on top of frame
        frame_with_banner = np.concatenate((banner, frame), axis=0)
        
        return frame_with_banner
        