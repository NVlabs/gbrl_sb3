##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
import gymnasium as gym 
import gymnasium.spaces as spaces
import numpy as np
import networkx as nx
from gymnasium.envs.registration import register
import random


class PipelineSchedulingEnv(gym.Env):
    """
    Enhanced Pipeline Scheduling Environment
    - MultiBinary action space (choose which tasks to schedule each step)
    - Task Types: CPU, IO, MEMORY
    - Non-uniform sampling for task properties
    - Sparse final reward (normalized by n_tasks, +1 bonus if all tasks are done)
    - Temporal and resource-based traps
    - Option for One-Hot Encoding of Task Types
    """

    def __init__(self, n_tasks=40, max_resources=8, max_duration=4, one_hot_task_types=False):
        super(PipelineSchedulingEnv, self).__init__()
        
        self.n_tasks = n_tasks
        self.max_resources = max_resources
        self.max_duration = max_duration
        self.task_types = ['CPU', 'IO', 'MEMORY']
        self.one_hot_task_types = one_hot_task_types
        self.is_mixed = not one_hot_task_types
        # Action space: For each of the n_tasks, pick (0 or 1) -> schedule or not
        self.action_space = spaces.Discrete(self.n_tasks + 1)
        # Observation space (same structure as your original):
        # Per task: 1 (or len(self.task_types) if one-hot) + 6
        # Global: 2
        task_features = 1 if not self.one_hot_task_types else len(self.task_types)
        obs_dim = (task_features + 9) * self.n_tasks + 10
        self.observation_space = spaces.Box(
            low=0,
            high=max(self.max_resources, self.max_duration),
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.reset()

    def _generate_tasks(self):
        """Generate tasks with types, durations, and resource requirements."""
        # Duration: Pareto, then clamp
        self.task_durations = np.zeros(self.n_tasks)
        # Resource usage: normal around max_resources/3, clamp
        self.task_cpu_resources = np.ones(self.n_tasks)
        self.task_io_resources = np.ones(self.n_tasks)
        self.task_mem_resources = np.ones(self.n_tasks)
        # Assign task types randomly
        self.task_types_list = random.choices(self.task_types, k=self.n_tasks)
        # Enforce some correlations
        for i, task_type in enumerate(self.task_types_list):
            self.task_durations[i] = np.random.randint(1, self.max_duration)

    def _get_observation(self):
        """Flattened observation of task states + global state."""
        obs = []
        for i in range(self.n_tasks):
            if self.one_hot_task_types:
                one_hot = [0] * len(self.task_types)
                one_hot[self.task_types.index(self.task_types_list[i])] = 1
                obs.extend(one_hot)
            else:
                obs.append(self.task_types_list[i].encode('utf-8'))

            obs.extend([
                self.task_durations[i],
                self.task_cpu_resources[i],
                self.task_mem_resources[i],
                self.task_io_resources[i],
                int(self.task_io_resources[i] >= self.io_available),
                int(self.task_mem_resources[i] >= self.mem_available),
                int(self.task_cpu_resources[i] >= self.cpu_available),
                int(i in self.running_tasks),
                int(i in self.completed_tasks)
            ])
        obs.append(self.time_remaining)
        obs.append(self.io_available)
        obs.append(self.mem_available)
        obs.append(self.cpu_available)
        obs.append(int(self.io_available < self.max_resources * 0.2))
        obs.append(int(self.mem_available < self.max_resources * 0.2))
        obs.append(int(self.cpu_available < self.max_resources * 0.2))
        running_task_types = [self.task_types_list[task_i] for task_i in self.running_tasks]

        count_cpu = running_task_types.count('CPU')
        count_mem = running_task_types.count('MEMORY')
        count_io  = running_task_types.count('IO')
        obs.append(count_cpu)
        obs.append(count_mem)
        obs.append(count_io)
        
        return np.array(obs, dtype=object if self.is_mixed else np.float32)

    def reset(self, seed=None, options=None):
        """Reset environment state."""
        super().reset(seed=seed)
        self._generate_tasks()
        
        self.time_remaining = 25
        self.cpu_available = self.max_resources
        self.mem_available = self.max_resources
        self.io_available = self.max_resources
        self.running_tasks = set()
        self.completed_tasks = set()
        return self._get_observation(), {}

    def step(self, action):
        """
        action: MultiBinary vector of length n_tasks (1 = "schedule this task").
        1. Identify which tasks are newly chosen (not completed/running yet).
        2. Sum total resource usage of newly chosen tasks + running tasks.
        3. If the sum exceeds resources_available, penalize + skip scheduling.
        Otherwise, schedule them all.
        4. Decrement durations, check completions, update time, etc.
        """
        reward = 0
        terminated = False
        truncated = False
        info = {}

        running_task_types = [self.task_types_list[task_i] for task_i in self.running_tasks]

        count_cpu = running_task_types.count('CPU')
        count_mem = running_task_types.count('MEMORY')
        count_io  = running_task_types.count('IO')

        if count_cpu == 0:
            self.cpu_available = self.max_resources
        if count_mem == 0:
            self.mem_available = self.max_resources
        if count_io == 0:
            self.io_available = self.max_resources

        if action != self.n_tasks and action not in self.completed_tasks and action not in self.running_tasks:
            valid_task = False
            if self.task_types_list[action] == 'CPU' and self.cpu_available >= self.task_cpu_resources[action]:
                valid_task = True
                self.cpu_available -= self.task_cpu_resources[action]
                if count_cpu > 0:
                    self.cpu_available -= count_cpu
                    self.cpu_available = max(self.cpu_available, 0)
            elif self.task_types_list[action] == 'MEMORY' and self.mem_available >= self.task_mem_resources[action]:
                valid_task = True
                self.mem_available -= self.task_mem_resources[action]
                if count_mem > 0:
                    self.mem_available -= count_mem
                    self.mem_available = max(self.mem_available, 0)
                if self.cpu_available < self.max_resources * 0.2:
                    reward += 0.1  # Encourage MEM scheduling under CPU saturation
            elif self.task_types_list[action] == 'IO' and self.io_available >= self.task_io_resources[action]:
                valid_task = True
                self.io_available -= self.task_io_resources[action]
                if count_io > 0:
                    self.io_available -= count_io
                    self.io_available = max(self.io_available, 0)
            if valid_task:
                self.running_tasks.add(action)
        
        tasks_finished = []
        for task in self.running_tasks:
            self.task_durations[task] -= 1
            if self.task_durations[task] <= 0:
                tasks_finished.append(task)

        for task_i in tasks_finished:
            self.running_tasks.remove(task_i)
            self.completed_tasks.add(task_i)
            if self.task_types_list[task_i] == 'CPU':
                self.cpu_available += self.task_cpu_resources[task_i]
            elif self.task_types_list[task_i] == 'MEMORY':
                self.mem_available += self.task_mem_resources[task_i]
            elif self.task_types_list[task_i] == 'IO':
                self.io_available += self.task_io_resources[task_i]
            
        # Decrement global time
        self.time_remaining -= 1
        # Check termination
        if len(self.completed_tasks) == self.n_tasks:
            terminated = True
        elif self.time_remaining <= 0:
            terminated = True

        if terminated and not truncated:
            reward += self._compute_final_reward()

        return self._get_observation(), reward, terminated, truncated, info


    def _compute_final_reward(self):
        """
        Sparse final reward:
          - partial = (#completed_tasks / n_tasks)
          - if all completed => +1 bonus
          => maximum total = 2
        """
        partial = len(self.completed_tasks) / self.n_tasks
        bonus = 1.0 if len(self.completed_tasks) == self.n_tasks and self.time_remaining > 0 else 0.0
        return partial + bonus

    def render(self, mode='human'):
        print(f"Scheduled Tasks: {self.scheduled_tasks}")
        print(f"Running Tasks: {self.running_tasks}")
        print(f"Completed Tasks: {self.completed_tasks}")
        print(f"Time Remaining: {self.time_remaining}")
        print(f"Resources Available: {self.resources_available}")


class HPCSchedulingEnv(gym.Env):
    """
    Realistic Pipeline Scheduling Environment
    - Tasks consume CPU, MEM, and IO resources
    - Random cooldown timer for tasks
    - Idle penalty applies only when no tasks are running and none are schedulable
    - Sparse rewards (only for task completion)
    """
    
    def __init__(self, n_tasks: int=10, max_time: int=25, one_hot_task_types: bool = False):
        super(HPCSchedulingEnv, self).__init__()
        
        # Fixed global resources
        self.max_cpu = 4
        self.max_mem = 16
        self.max_io = 8
        
        self.n_tasks = n_tasks
        self.max_time = max_time
        self.is_mixed = not one_hot_task_types
        
        # Action space: Choose a task or No-Op
        self.action_space = spaces.Discrete(self.n_tasks + 1)  # Tasks + No-Op
        
        # Observation space
        # Global State: cpu_available, mem_available, io_available, time_remaining
        # Task-Level: duration, required_CPU, required_MEM, required_IO, is_running, is_completed, cooldown_timer, cooldown_active
        task_features = 9 + (self.n_tasks + 1) if not self.is_mixed else 10
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(4 + self.n_tasks * task_features,),  # 4 global + 8 per task
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment state."""
        super().reset(seed=seed)
        self.cpu = self.max_cpu
        self.mem = self.max_mem
        self.io = self.max_io
        self.time_remaining = self.max_time
        
        self.task_durations = np.random.randint(1, 5, size=self.n_tasks)
        self.task_cpu = np.random.randint(1, self.max_cpu + 1, size=self.n_tasks)
        self.task_mem = np.random.randint(1, self.max_mem + 1, size=self.n_tasks)
        self.task_io = np.random.randint(1, self.max_io + 1, size=self.n_tasks)
        self.task_cooldowns = np.zeros(self.n_tasks, dtype=int)
        self.task_running = np.zeros(self.n_tasks, dtype=bool)
        self.task_completed = np.zeros(self.n_tasks, dtype=bool)
        self.task_dependencies = np.random.choice([-1] + list(range(self.n_tasks)), size=self.n_tasks)
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Construct the observation vector."""
        obs = [
            self.cpu,
            self.mem,
            self.io,
            self.time_remaining
        ]
        
        for i in range(self.n_tasks):
            has_resource = self.task_cpu[i] <= self.cpu and self.task_mem[i] <= self.mem and self.task_io[i] <= self.io
            is_schedulable = has_resource and not self.task_running[i] and not self.task_completed[i]
            depends_on = self.task_dependencies[i]
            dependency_completed = (depends_on == -1) or self.task_completed[depends_on]
            if not self.is_mixed:
                one_hot = [0] * (self.n_tasks + 1)
                if depends_on == -1:
                    one_hot[-1] = 1
                else:
                    one_hot[depends_on] = 1
                obs.extend([
                    self.task_durations[i],
                    self.task_cpu[i],
                    self.task_mem[i],
                    self.task_io[i],
                    int(self.task_running[i]),
                    int(self.task_completed[i]),
                    int(is_schedulable),
                    self.task_cooldowns[i],
                    *one_hot,
                    int(dependency_completed)
                ])
            else:
                obs.extend([
                    self.task_durations[i],
                    self.task_cpu[i],
                    self.task_mem[i],
                    self.task_io[i],
                    str(bool(self.task_running[i])).encode('utf-8'),
                    str(bool(self.task_completed[i])).encode('utf-8'),
                    str(bool(is_schedulable)).encode('utf-8'),
                    self.task_cooldowns[i],
                    str(depends_on).encode('utf-8'),
                    str(dependency_completed).encode('utf-8'),
                ])
        
        return np.array(obs, dtype=np.float32 if not self.is_mixed else object)
    
    def step(self, action):
        """
        Perform an action.
        - If action < n_tasks: Attempt to schedule a task
        - If action == n_tasks: No-Op
        """
        reward = 0
        terminated = False
        info = {}
        
        no_schedulable_tasks = True
        
        # Check if any task is schedulable
        for i in range(self.n_tasks):
            has_resource = self.task_cpu[i] <= self.cpu and self.task_mem[i] <= self.mem and self.task_io[i] <= self.io
            depends_on = self.task_dependencies[i]
            dependency_completed = (depends_on == -1) or self.task_completed[depends_on]
            if (has_resource and not self.task_running[i] and dependency_completed and not self.task_completed[i]):
                no_schedulable_tasks = False
                break
        
        # Handle Task Scheduling
        if action < self.n_tasks and not self.task_running[action] and not self.task_completed[action]:
            has_resource = self.task_cpu[action] <= self.cpu and self.task_mem[action] <= self.mem and self.task_io[action] <= self.io
            depends_on = self.task_dependencies[action]
            dependency_completed = (depends_on == -1) or self.task_completed[depends_on]
            if has_resource and dependency_completed:
                self.cpu -= self.task_cpu[action]
                self.mem -= self.task_mem[action]
                self.io -= self.task_io[action]
                self.task_running[action] = True
        
        # Task Progression
        finished_tasks = []
        for i in range(self.n_tasks):
            if self.task_running[i]:
                self.task_durations[i] -= 1
                if self.task_durations[i] <= 0:
                    finished_tasks.append(i)
        
        for task in finished_tasks:
            self.task_running[task] = False
            self.task_completed[task] = True
            # self.task_cooldowns[task] = np.random.randint(0, 5)  # Assign cooldown time
            self.task_cooldowns[task] = int(np.mean([self.task_cpu[task], self.task_mem[task], self.task_io[task]]) / 2) + np.random.randint(0, 2)
        
        # Cooldown Recovery
        for i in range(self.n_tasks):
            if self.task_cooldowns[i] > 0:
                self.task_cooldowns[i] -= 1
                if self.task_cooldowns[i] == 0:
                    self.cpu += self.task_cpu[i]
                    self.mem += self.task_mem[i]
                    self.io += self.task_io[i]
                    self.cpu = min(self.cpu, self.max_cpu)
                    self.mem = min(self.mem, self.max_mem)
                    self.io = min(self.io, self.max_io)
        
        # Apply Idle Penalty
        if action == self.n_tasks and len(np.where(self.task_running)[0]) == 0 and no_schedulable_tasks:
            reward -= 0.01  # Idle penalty only applies when no tasks are running and none are schedulable
        
        # Time Decrement
        self.time_remaining -= 1
        if self.time_remaining <= 0 or all(self.task_completed):
            terminated = True
        
        if terminated:
            reward += np.sum(self.task_completed) / self.n_tasks
            bonus = 1 if self.time_remaining > 0 and np.sum(self.task_completed) == self.n_tasks else 0 
            reward += bonus
        
        return self._get_observation(), reward, terminated, False, info
    
    def render(self, mode='human'):
        """Display the current state."""
        print(f"Time Remaining: {self.time_remaining}")
        print(f"Resources: CPU={self.cpu}, MEM={self.mem}, IO={self.io}")
        print(f"Running Tasks: {np.where(self.task_running)[0]}")
        print(f"Completed Tasks: {np.where(self.task_completed)[0]}")
        print(f"Cooldowns: {self.task_cooldowns}")

def register_pipeline_opt_tests():
    # PutNear
    # ----------------------------------------

    register(
        id="pipeline-v0",
        entry_point="env.pipeline_opt:PipelineSchedulingEnv",
        kwargs={'n_tasks': 40, 'max_resources': 8, 'max_duration': 4},
    )
    register(
        id="pipeline-large-v0",
        entry_point="env.pipeline_opt:PipelineSchedulingEnv",
        kwargs={'n_tasks': 50, 'max_resources': 6, 'max_duration': 4},
    )
    register(
        id="pipeline-v1",
        entry_point="env.pipeline_opt:HPCSchedulingEnv",
        kwargs={'n_tasks': 10},
    )
