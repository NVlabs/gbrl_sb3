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
import cv2
from gymnasium.core import ActType, ObsType
from minigrid.core.constants import STATE_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import TILE_PIXELS, MiniGridEnv
from gymnasium.spaces import Discrete
IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}


class GuidedLockedRoomEnv(MiniGridEnv):
    """
    ## Description
    
    A split-RL compatible environment.
    - Expert Guidance is provided strictly for the optimal path to the key.
    - If the agent deviates from this pre-calculated path, guidance turns off (0).
    - Once the key is collected, guidance turns off (0).
    """

    def __init__(self, size=15, max_steps=None, **kwargs):
        self.size = size
        
        if max_steps is None:
            max_steps = 2 * size * size

        self.available_colors = list(['red', 'blue', 'purple', 'yellow', 'grey'])
        self.correct_color = None  # Will be set in _gen_grid()
        self.expert_trajectory_map = {}

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=False,
            **kwargs
        )
        
        self.action_space = Discrete(self.actions.toggle + 1)

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

        # Define room boundaries
        ul_x_start, ul_x_end = 1, mid_x
        ul_y_start, ul_y_end = 1, mid_y
        
        ll_x_start, ll_x_end = 1, mid_x
        ll_y_start, ll_y_end = mid_y, height - 1
        
        ur_x_start, ur_x_end = mid_x + 1, width - 1
        ur_y_start, ur_y_end = 1, mid_y
        
        lr_x_start, lr_x_end = mid_x + 1, width - 1
        lr_y_start, lr_y_end = mid_y, height - 1

        # Walls Logic
        ul_door_y = ul_y_start
        for y in range(ul_y_start, ul_y_end):
            if y != ul_door_y: self.grid.set(mid_x - 1, y, Wall())
        
        ll_door_y = ll_y_end - 1
        for y in range(ll_y_start, ll_y_end):
            if y != ll_door_y: self.grid.set(mid_x - 1, y, Wall())
        
        ur_door_y = ur_y_start
        for y in range(ur_y_start, ur_y_end):
            if y != ur_door_y: self.grid.set(mid_x + 1, y, Wall())
        
        lr_door_y = lr_y_end - 1
        for y in range(lr_y_start, lr_y_end):
            if y != lr_door_y: self.grid.set(mid_x + 1, y, Wall())
        
        for x in range(ul_x_start, mid_x):
            self.grid.set(x, mid_y, Wall())
        
        for x in range(mid_x + 1, lr_x_end):
            self.grid.set(x, mid_y, Wall())

        # Key Placement
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

        place_key(ul_key_color, ul_x_start + 1, ul_x_end - 2, ul_y_start + 1, ul_y_end - 2)
        place_key(ll_key_color, ll_x_start + 1, ll_x_end - 2, ll_y_start + 2, ll_y_end - 1)

        # Determine room name
        if ul_key_color == self.correct_color:
            self.target_room = "upper-left"
        elif ll_key_color == self.correct_color:
            self.target_room = "lower-left"
        else:
            self.target_room = "unknown"
        
        # Locked Door & Goal
        locked_door = Door(self.correct_color, is_locked=True)
        self.grid.set(mid_x + 1, ur_door_y, locked_door)
        self.locked_door_pos = (mid_x + 1, ur_door_y)

        goal_x = ur_x_end - 1
        goal_y = ur_y_start
        self.put_obj(Goal(), goal_x, goal_y)
        self.goal_pos = np.array([goal_x, goal_y])

        # Agent Start
        self.agent_pos = np.array([mid_x, height // 2])
        self.agent_dir = 0  # Facing right

        self.mission = f"collect the key from the {self.target_room} room to unlock the door and reach the goal"

        start_pos = self.agent_pos
        self.max_distance = np.linalg.norm(start_pos - self.goal_pos)

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
        
        # 1. Run BFS to find the sequence of actions
        while queue:
            (cx, cy, cdir), actions = queue.popleft()
            
            # Check if we are facing the key (Pickup condition)
            dx, dy = self._dir_vec(cdir)
            
            # Are we facing the key?
            if (cx + dx, cy + dy) == (key_x, key_y):
                # We found it. The full path ends with pickup.
                final_path_actions = actions + [self.actions.pickup]
                break
            
            # Helper to add neighbors
            def try_add(nx, ny, ndir, act):
                if (nx, ny, ndir) not in visited:
                    visited.add((nx, ny, ndir))
                    queue.append(((nx, ny, ndir), actions + [act]))

            # Left
            try_add(cx, cy, (cdir - 1) % 4, self.actions.left)
            # Right
            try_add(cx, cy, (cdir + 1) % 4, self.actions.right)
            
            # Forward
            fx, fy = cx + dx, cy + dy
            if 0 <= fx < self.width and 0 <= fy < self.height:
                cell = self.grid.get(fx, fy)
                # Walkable if empty, goal, open door. 
                # Note: We cannot walk INTO the key, we walk TO it.
                can_walk = (cell is None) or (cell.type == 'goal') or (cell.type == 'door' and cell.is_open)
                if can_walk:
                    try_add(fx, fy, cdir, self.actions.forward)

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
                # Pickup is the last action, no state update needed after it

    def _get_expert_action(self):
        """
        Strict Trajectory Lookup:
        1. If holding key -> Guidance OFF (0)
        2. If current state in expert_map -> Guidance ON (1), return action
        3. If current state NOT in map -> Guidance OFF (0), return dummy
        """
        # 1. Stop condition: We have the key
        if self.carrying and self.carrying.type == 'key' and self.carrying.color == self.correct_color:
            return self.actions.done, 0

        # 2. State Lookup
        state_key = (self.agent_pos[0], self.agent_pos[1], self.agent_dir)
        
        # 3. Strict Check
        if state_key in self.expert_trajectory_map:
            return self.expert_trajectory_map[state_key], 1
        
        # 4. Off-distribution / Deviation
        return self.actions.done, 0

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # --- PRE-STEP: Calculate Expert Action based on CURRENT state ---
        expert_action, guidance_active = self._get_expert_action()
        
        # Save state for rendering
        self.step_count += 1
        self.last_grid = self.grid.copy()
        self.last_agent_pos = tuple(self.agent_pos)
        self.last_agent_dir = self.agent_dir
        self.last_carrying = self.carrying
        self.last_action = action
        self.last_expert_action = expert_action
        self.last_label = guidance_active

        # Standard MiniGrid Step Logic
        terminated = False
        truncated = False
        
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        if action == self.actions.left:
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)
        elif action == self.actions.done:
            pass

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()
        
        # Reward Calculation
        potential_after = self.get_potential(self.agent_pos)
        reward = 0
        if not terminated:
            reward = potential_after - self.prev_potential
        self.prev_potential = potential_after

        # INFO INJECTION
        info = {
            'expert_action': self.to_one_hot(expert_action),
            'guidance_active': guidance_active
        }

        return obs, reward, terminated, truncated, info
    
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        # CRITICAL: Precompute the single golden trajectory
        self._precompute_expert_trajectory()
        
        # Initialize tracking vars
        self.last_grid = self.grid.copy()
        self.last_agent_pos = tuple(self.agent_pos)
        self.last_agent_dir = self.agent_dir
        self.last_carrying = self.carrying
        self.last_action = None
        self.prev_potential = self.get_potential(self.agent_pos)
        
        # Initial Expert Info
        expert_action, guidance_active = self._get_expert_action()
        info['expert_action'] = self.to_one_hot(expert_action)
        info['guidance_active'] = guidance_active
        self.last_expert_action = expert_action
        self.last_label = guidance_active
        
        return obs, info

    def get_potential(self, pos):
        dist = np.linalg.norm(np.array(pos) - self.goal_pos)
        return np.clip(1.0 - (dist / self.max_distance), 0.0, 1.0)

    def to_one_hot(self, action):
        one_hot = np.zeros(self.action_space.n, dtype=np.float32)
        if action != self.actions.done:
            one_hot[action] = 1.0
        return one_hot

    def get_frame(self, highlight=True, tile_size=TILE_PIXELS, agent_pov=False):
        # Swap current state with saved pre-step state
        curr_grid, curr_pos, curr_dir, curr_carry = self.grid, self.agent_pos, self.agent_dir, self.carrying
        
        self.grid, self.agent_pos, self.agent_dir, self.carrying = \
            self.last_grid, self.last_agent_pos, self.last_agent_dir, self.last_carrying
            
        if agent_pov:
            img = self.get_pov_render(tile_size)
        else:
            img = self.get_full_render(highlight, tile_size)
            
        # Restore real state
        self.grid, self.agent_pos, self.agent_dir, self.carrying = curr_grid, curr_pos, curr_dir, curr_carry
        return img
    
    def render(self):
        frame = self.get_frame(self.highlight, self.tile_size, self.agent_pov)
        if frame is None: return None
        
        frame_height, frame_width, _ = frame.shape
        banner_height = 80
        banner = np.ones((banner_height, frame_width, 3), dtype=np.uint8) * 240
        
        # Text Info
        mission_text = self.mission
        carrying_text = f"Carrying: {self.carrying.color + ' ' + self.carrying.type if self.carrying else 'Nothing'}"
        
        action_names = ['Left', 'Right', 'Fwd', 'Pick', 'Drop', 'Toggle', 'Done']
        act_name = action_names[self.last_action] if self.last_action is not None else "None"
        exp_name = action_names[self.last_expert_action] if self.last_expert_action is not None else "None"
        
        # Detailed Status Line
        status_text = f"Act: {act_name} | Expert: {exp_name} | Guided: {'ON' if self.last_label else 'OFF'}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(banner, mission_text, (10, 20), font, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(banner, carrying_text, (10, 45), font, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(banner, status_text, (10, 70), font, 0.5, (0,0,0), 1, cv2.LINE_AA)
        
        return np.concatenate((banner, frame), axis=0)