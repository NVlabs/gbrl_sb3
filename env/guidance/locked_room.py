##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
from __future__ import annotations

from collections import deque
from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import ActType, ObsType
from minigrid.core.constants import STATE_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import TILE_PIXELS, MiniGridEnv

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}


class GuidedLockedRoomEnv(MiniGridEnv):
    """
    ## Description

    The environment has a narrow vertical corridor in the middle with 4 rooms (2 on each side).
    Each of the 3 accessible rooms contains a key of a different color.
    The upper-right room has a locked door that requires the correct key to open.
    Inside that room is the goal. The agent must collect the right key, unlock the door,
    and reach the goal.

    ## Mission Space

    "collect the {correct_color} key and unlock the door to reach the goal"

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Pick up an object         |
    | 4   | drop         | Drop an object            |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    Potential-based reward based on distance to the goal.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    """

    def __init__(self, size=15, max_steps=None, **kwargs):
        self.size = size
        
        if max_steps is None:
            max_steps = 2 * size * size

        self.available_colors = list(['red', 'blue', 'purple', 'yellow', 'grey'])
        self.correct_color = None  # Will be set in _gen_grid()

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=False,
            **kwargs
        )

    @staticmethod
    def _gen_mission():
        return "collect the key and unlock the door to reach the goal"

    def _gen_grid(self, width, height):
        # Randomize the correct key color for this episode
        self.correct_color = self.np_random.choice(self.available_colors)
        
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Middle corridor position (single column that will remain EMPTY)
        mid_x = width // 2
        mid_y = height // 2

        # Define room boundaries - rooms do NOT include the corridor column
        # Upper-left room (left of corridor)
        ul_x_start, ul_x_end = 1, mid_x  # Does not include mid_x
        ul_y_start, ul_y_end = 1, mid_y
        
        # Lower-left room (left of corridor)
        ll_x_start, ll_x_end = 1, mid_x  # Does not include mid_x
        ll_y_start, ll_y_end = mid_y, height - 1
        
        # Upper-right room (right of corridor)
        ur_x_start, ur_x_end = mid_x + 1, width - 1  # Starts after mid_x
        ur_y_start, ur_y_end = 1, mid_y
        
        # Lower-right room (right of corridor)
        lr_x_start, lr_x_end = mid_x + 1, width - 1  # Starts after mid_x
        lr_y_start, lr_y_end = mid_y, height - 1

        # Build vertical wall on right side of left rooms (separating from corridor)
        ul_door_y = ul_y_start  # Door at highest end of upper room
        for y in range(ul_y_start, ul_y_end):
            if y != ul_door_y:
                self.grid.set(mid_x - 1, y, Wall())
        
        ll_door_y = ll_y_end - 1  # Door at lowest end of lower room
        for y in range(ll_y_start, ll_y_end):
            if y != ll_door_y:
                self.grid.set(mid_x - 1, y, Wall())
        
        # Build vertical wall on left side of right rooms (separating from corridor)
        ur_door_y = ur_y_start  # Door at highest end of upper room
        for y in range(ur_y_start, ur_y_end):
            if y != ur_door_y:
                self.grid.set(mid_x + 1, y, Wall())
        
        lr_door_y = lr_y_end - 1  # Door at lowest end of lower room
        for y in range(lr_y_start, lr_y_end):
            if y != lr_door_y:
                self.grid.set(mid_x + 1, y, Wall())
        
        # Build horizontal wall separating upper and lower left rooms
        for x in range(ul_x_start, mid_x):
            self.grid.set(x, mid_y, Wall())
        
        # Build horizontal wall separating upper and lower right rooms
        for x in range(mid_x + 1, lr_x_end):
            self.grid.set(x, mid_y, Wall())

        # Choose random key colors for the 2 accessible rooms (upper-left and lower-left only)
        # We exclude lower-right to avoid reward conflicts with reaching the goal

        available_colors = [color for color in self.available_colors if color != self.correct_color]
        
        self.np_random.shuffle(available_colors)
        colors = available_colors[:1] + [self.correct_color]
        self.np_random.shuffle(colors)    
        
        ul_key_color = colors[0]
        ll_key_color = colors[1]
        
        def place_key(color, x_start, x_end, y_start, y_end):
            key_x = self.np_random.integers(x_start, x_end)
            key_y = self.np_random.integers(y_start, y_end)
            self.put_obj(Key(color), key_x, key_y)
            if color == self.correct_color:
                self.target_key_pos = (key_x, key_y)

        # Place keys only in upper-left and lower-left rooms
        place_key(ul_key_color, ul_x_start + 1, ul_x_end - 2, ul_y_start + 1, ul_y_end - 2)
        place_key(ll_key_color, ll_x_start + 1, ll_x_end - 2, ll_y_start + 2, ll_y_end - 1)

        # Verify target key was set and determine which room it's in
        assert hasattr(self, 'target_key_pos'), f"Target key not set! Correct color: {self.correct_color}, Keys: ul={ul_key_color}, ll={ll_key_color}"
        
        # Determine room name based on which color matches
        if ul_key_color == self.correct_color:
            self.target_room = "upper-left"
        elif ll_key_color == self.correct_color:
            self.target_room = "lower-left"
        else:
            self.target_room = "unknown"
        
        # Place locked door at upper-right room entrance (from corridor)
        locked_door = Door(self.correct_color, is_locked=True)
        self.grid.set(mid_x + 1, ur_door_y, locked_door)
        self.locked_door_pos = (mid_x + 1, ur_door_y)

        # Place goal at the top-right corner of upper-right room
        goal_x = ur_x_end - 1
        goal_y = ur_y_start
        self.put_obj(Goal(), goal_x, goal_y)
        self.goal_pos = np.array([goal_x, goal_y])

        # Place agent in the exact middle of the corridor
        self.agent_pos = np.array([mid_x, height // 2])
        self.agent_dir = 0  # Facing right

        # Store the mission
        self.mission = f"collect the key from the {self.target_room} room to unlock the door and reach the goal"

        # Calculate max distance for potential
        start_pos = self.agent_pos
        self.max_distance = np.linalg.norm(start_pos - self.goal_pos)

    def _get_expert_action(self):
        """
        Calculates the optimal action to retrieve the correct key.
        Returns: (action, is_active)
        """
        # 1. Condition to stop guidance: Key is already collected
        if self.carrying and self.carrying.type == 'key' and self.carrying.color == self.correct_color:
            return self.actions.done, 0

        # Current state key
        state_key = (self.agent_pos[0], self.agent_pos[1], self.agent_dir)
        
        # Check if we are strictly on the expert path
        if state_key in self.expert_trajectory_map:
            return self.expert_trajectory_map[state_key], 1
        
        # Off-path / Deviation - compute path dynamically from current state
        expert_action = self._compute_next_action_to_key()
        if expert_action is not None:
            return expert_action, 1
        
        # If no path found (shouldn't happen), return done with guidance inactive
        return self.actions.done, 0

    def _dir_vec(self, direction):
        """
        Duplicate of Minigrid logic to get vector from direction index.
        0: right, 1: down, 2: left, 3: up
        """
        if direction == 0: return (1, 0)
        elif direction == 1: return (0, 1)
        elif direction == 2: return (-1, 0)
        elif direction == 3: return (0, -1)
        return (0, 0)
    
    def _compute_next_action_to_key(self):
        """
        Compute the next optimal action from current state to reach the target key.
        Uses BFS to find the shortest path.
        Returns: next action to take, or None if no path exists
        """
        start_state = (self.agent_pos[0], self.agent_pos[1], self.agent_dir)
        key_x, key_y = self.target_key_pos
        
        queue = deque([(start_state, [])])  # (state, list_of_actions)
        visited = {start_state}
        
        # Run BFS to find the sequence of actions
        while queue:
            (cx, cy, cdir), actions = queue.popleft()
            
            # Check if we are facing the key (Pickup condition)
            dx, dy = self._dir_vec(cdir)
            if (cx + dx, cy + dy) == (key_x, key_y):
                # Found the key! Return the first action in the path
                if actions:
                    return actions[0]
                else:
                    return self.actions.pickup
            
            # Explore neighbors
            # Left
            ndir = (cdir - 1) % 4
            if (cx, cy, ndir) not in visited:
                visited.add((cx, cy, ndir))
                queue.append(((cx, cy, ndir), actions + [self.actions.left]))
            
            # Right
            ndir = (cdir + 1) % 4
            if (cx, cy, ndir) not in visited:
                visited.add((cx, cy, ndir))
                queue.append(((cx, cy, ndir), actions + [self.actions.right]))
                
            # Forward
            fx, fy = cx + dx, cy + dy
            if 0 <= fx < self.width and 0 <= fy < self.height:
                cell = self.grid.get(fx, fy)
                # Walkable if empty, goal, open door
                can_walk = (cell is None) or (cell.type == 'goal') or (cell.type == 'door' and cell.is_open)
                
                if can_walk and (fx, fy, cdir) not in visited:
                    visited.add((fx, fy, cdir))
                    queue.append(((fx, fy, cdir), actions + [self.actions.forward]))
        
        # No path found
        return None
    
    def get_potential(self, pos):
        """Calculate potential based on distance to goal."""
        dist = np.linalg.norm(np.array(pos) - self.goal_pos)
        return np.clip(1.0 - (dist / self.max_distance), 0.0, 1.0)

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # Save state for rendering
        # --- PRE-STEP: Calculate Expert Action based on CURRENT state ---
        expert_action, guidance_active = self._get_expert_action()
        self.step_count += 1
        self.last_grid = self.grid.copy()
        self.last_agent_pos = tuple(self.agent_pos)
        self.last_agent_dir = self.agent_dir
        self.last_carrying = self.carrying
        self.last_action = action

        # Take the action
        terminated = False
        truncated = False
        
        fwd_pos = self.front_pos
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

        # Calculate potential after action is executed
        potential_after = self.get_potential(self.agent_pos)
        if not terminated:
            reward = potential_after - self.prev_potential
        self.prev_potential = potential_after
        
        
        info = {'expert_action': self.to_one_hot(expert_action), 'guidance_active': guidance_active}
        self.last_expert_action = expert_action
        return obs, reward, terminated, truncated, info
    
    def _precompute_expert_trajectory(self):
        """
        Runs BFS once at Reset to find the golden path from Start -> Key.
        Populates self.expert_trajectory_map = { (x, y, dir): action }
        """
        self.expert_trajectory_map = {}
        
        start_state = (self.agent_pos[0], self.agent_pos[1], self.agent_dir)
        key_x, key_y = self.target_key_pos
        
        queue = deque([(start_state, [])]) # (state, list_of_actions)
        visited = {start_state}
        
        final_path_actions = None
        final_state = None

        # 1. Run BFS to find the sequence of actions
        while queue:
            (cx, cy, cdir), actions = queue.popleft()
            
            # Check if we are facing the key (Pickup condition)
            dx, dy = self._dir_vec(cdir)
            if (cx + dx, cy + dy) == (key_x, key_y):
                final_path_actions = actions + [self.actions.pickup]
                final_state = (cx, cy, cdir)
                break
            
            # Explore neighbors
            # Left
            ndir = (cdir - 1) % 4
            if (cx, cy, ndir) not in visited:
                visited.add((cx, cy, ndir))
                queue.append(((cx, cy, ndir), actions + [self.actions.left]))
            
            # Right
            ndir = (cdir + 1) % 4
            if (cx, cy, ndir) not in visited:
                visited.add((cx, cy, ndir))
                queue.append(((cx, cy, ndir), actions + [self.actions.right]))
                
            # Forward
            fx, fy = cx + dx, cy + dy
            if 0 <= fx < self.width and 0 <= fy < self.height:
                cell = self.grid.get(fx, fy)
                # Walkable if empty, goal, open door. (Keys block movement, but we stop BEFORE key)
                can_walk = (cell is None) or (cell.type == 'goal') or (cell.type == 'door' and cell.is_open)
                
                if can_walk and (fx, fy, cdir) not in visited:
                    visited.add((fx, fy, cdir))
                    queue.append(((fx, fy, cdir), actions + [self.actions.forward]))

        # 2. Replay the actions to map State -> Action
        if final_path_actions:
            curr_x, curr_y = self.agent_pos
            curr_dir = self.agent_dir
            
            for action in final_path_actions:
                # Map current state to the action required
                state_key = (curr_x, curr_y, curr_dir)
                self.expert_trajectory_map[state_key] = action
                
                # Simulate the move to update state for next iteration
                if action == self.actions.left:
                    curr_dir = (curr_dir - 1) % 4
                elif action == self.actions.right:
                    curr_dir = (curr_dir + 1) % 4
                elif action == self.actions.forward:
                    dx, dy = self._dir_vec(curr_dir)
                    curr_x += dx
                    curr_y += dy
                    
    def to_one_hot(self, action):
        one_hot = np.zeros(self.action_space.n, dtype=np.float32)
        one_hot[action] = 1.0
        return one_hot

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        
        # Debug: Print which key is the target
        mid_x = self.width // 2
        mid_y = self.height // 2
        key_x, key_y = self.target_key_pos
        if key_x < mid_x and key_y < mid_y:
            room = "upper-left"
        elif key_x < mid_x and key_y >= mid_y:
            room = "lower-left"
        elif key_x > mid_x and key_y >= mid_y:
            room = "lower-right"
        else:
            room = "unknown"
        # print(f"[DEBUG] Correct key: {self.correct_color} at {self.target_key_pos} in {room} room")
        
        # CRITICAL: Precompute the trajectory once at reset
        self._precompute_expert_trajectory()

        # Initialize state tracking for rendering
        self.last_grid = self.grid.copy()
        self.last_agent_pos = tuple(self.agent_pos)
        self.last_agent_dir = self.agent_dir
        self.last_carrying = self.carrying
        self.last_action = None
        

        # Initialize potential
        self.prev_potential = self.get_potential(self.agent_pos)
        expert_action, guidance_active = self._get_expert_action()
        info['expert_action'] = self.to_one_hot(expert_action)
        info['guidance_active'] = guidance_active
        self.last_expert_action = expert_action
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
        
        # Add info banner at the top of the frame
        banner_height = 80
        banner = np.ones((banner_height, frame_width, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Prepare text information
        mission_text = self.mission
        carrying_text = f"Carrying: {self.carrying.color + ' ' + self.carrying.type if self.carrying else 'Nothing'}"
        
        # Format the actual action taken
        if self.last_action is not None:
            action_names = ['Left', 'Right', 'Fwd', 'Pick', 'Drop', 'Toggle', 'Done']
            taken_text = f"Action: {action_names[self.last_action]}"
        else:
            taken_text = "Action: None"
        
        action_names = ['Left', 'Right', 'Fwd', 'Pick', 'Drop', 'Toggle', 'Done']
        taken_text += f" Expert Action: {action_names[self.last_expert_action]}"
        
        # Add text to banner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_color = (0, 0, 0)  # Black text
        
        # Line 1: Mission
        cv2.putText(banner, mission_text, (10, 20), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        # Line 2: Carrying
        cv2.putText(banner, carrying_text, (10, 45), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        # Line 3: Action taken
        cv2.putText(banner, taken_text, (10, 70), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        # Concatenate banner on top of frame
        frame_with_banner = np.concatenate((banner, frame), axis=0)
        
        return frame_with_banner

        