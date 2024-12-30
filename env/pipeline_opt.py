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
from gymnasium.envs.registration import register


class HPCSchedulingEnv(gym.Env):
    """
    Realistic Pipeline Scheduling Environment
    - Tasks consume CPU, MEM, and IO resources
    - Random cooldown timer for tasks
    - Idle penalty applies only when no tasks are running and none are schedulable
    - Sparse rewards (only for task completion)
    """
    
    def __init__(self, n_tasks: int=5, max_time: int=5, one_hot_task_types: bool = False):
        super(HPCSchedulingEnv, self).__init__()
        
        # Fixed global resources
        self.max_cpu = 10
        self.max_mem = 16
        self.max_io = 10
        
        self.n_tasks = n_tasks
        self.max_time = max_time
        self.is_mixed = not one_hot_task_types
        self.schedule_status = {0: 'running', 1: 'completed', 2: 'schedulable', 3: 'no_resources'}
        # feature categories:
        # run_status: running, completed, idle
        # status: schedulable, dependency unsatisfied, resources unavailable, cooldown_active
        # Action space: Choose a task or No-Op
        self.action_space = spaces.Discrete(self.n_tasks + 1)  # Tasks + No-Op
        # Observation space
        # Global State: cpu_available, mem_available, io_available, time_remaining
        # Task-Level: duration, required_CPU, required_MEM, required_IO, is_running, is_completed, cooldown_timer, cooldown_active
        global_features = 6
        task_features =  len(self.schedule_status) + 8 if not self.is_mixed else 9
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(global_features + self.n_tasks * task_features,),  # 4 global + 8 per task
            dtype=np.float32
        )
        
    def _generate_tasks(self):
        """
        Rule-based task generation ensuring solvability and structural consistency.
        
        Args:
            use_cooldowns (bool): Whether to include cooldowns.
        """
        rule = np.random.choice([1, 2, 3])  # Randomly select a rule
        
        if rule == 1:  # Sequential via Dependencies
            task_inds = list(range(self.n_tasks))
            task_inds[0] = -1
            np.random.shuffle(task_inds)
            resource_pool = []
            cpu_sum = 0
            mem_sum = 0
            io_sum = 0
            for i in range(self.n_tasks):
                if i < self.n_tasks - 1:
                    cpu = np.random.randint(1, int(self.max_cpu // self.n_tasks))
                    mem = np.random.randint(1, int(self.max_mem // self.n_tasks))
                    io = np.random.randint(1, int(self.max_io // self.n_tasks))
                    cpu_sum += cpu
                    mem_sum += mem
                    io_sum += io
                else:
                    cpu = self.max_cpu - cpu_sum
                    mem = self.max_mem - mem_sum
                    io = self.max_io - io_sum
                duration = np.random.randint(2, self.max_time)
                resource_pool.append((cpu, mem, io, duration))
            np.random.shuffle(resource_pool)
            for i, (cpu, mem, io, duration) in enumerate(resource_pool):
                self.task_cpu[i] = cpu
                self.task_mem[i] = mem
                self.task_io[i] = io
                self.task_durations[i] = duration
                self.task_dependencies[i] = task_inds[i]
                self.task_cooldowns[i] = np.random.randint(1, 3)
            
        elif rule == 2:  # Parallel x-Track
            x = np.random.choice([2, 3])
            track_resources = [self.max_cpu // x, self.max_mem // x, self.max_io // x]
            task_inds = list(range(self.n_tasks))
            np.random.shuffle(task_inds)
            parallel_inds = task_inds[:x]
            other_tasks = [t for t in task_inds if t not in parallel_inds]
            order_ = np.random.choice([False, True])
            for i in range(self.n_tasks):
                self.task_cpu[i] = track_resources[0] if i in parallel_inds else self.max_cpu - 1
                self.task_mem[i] = track_resources[1] if i in parallel_inds else self.max_mem - 1
                self.task_io[i] = track_resources[2] if i in parallel_inds else self.max_io - 1
                self.task_durations[i] = np.random.randint(2, self.max_time)
                if order_:
                    self.task_dependencies[i] = parallel_inds[0] if i not in parallel_inds else -1
                else:
                    self.task_dependencies[i] = other_tasks[0] if i in parallel_inds else -1
                if i not in parallel_inds:
                    self.task_priorities[i] = np.random.choice([False, True])
                self.task_cooldowns[i] = (self.task_cpu[i] + self.task_mem[i] + self.task_io[i]) // 3
        
        elif rule == 3:  # Sequential via Resources (Strict Bottleneck)
            resource_pool = []
            for i in range(self.n_tasks):
                idx = np.random.choice([0 ,1, 2])
                cpu = self.max_cpu - 1 if idx == 0 else np.random.randint(1, self.max_cpu - 1)
                mem = self.max_mem - 1 if idx == 1 else np.random.randint(1, self.max_mem - 1)
                io = self.max_io - 1 if idx == 2 else np.random.randint(1, self.max_io - 1)
                duration = np.random.randint(3, self.max_time)
                priority = np.random.choice([False , True])
                resource_pool.append((cpu, mem, io, duration, priority))
            np.random.shuffle(resource_pool)
            
            for i, (cpu, mem, io, duration, priority) in enumerate(resource_pool):
                self.task_cpu[i] = cpu
                self.task_mem[i] = mem
                self.task_io[i] = io
                self.task_durations[i] = duration
                self.task_dependencies[i] = -1
                self.task_priorities[i] = priority
                self.task_cooldowns[i] = (cpu + mem + io) // 2
        self.time_remaining = np.sum(self.task_cooldowns) + np.sum(self.task_durations)
        
    def reset(self, seed=None, options=None):
        """Reset environment state."""
        super().reset(seed=seed)
        self.cpu = self.max_cpu
        self.mem = self.max_mem
        self.io = self.max_io

        self.task_durations = np.zeros(self.n_tasks, dtype=int)
        self.task_cpu = np.zeros(self.n_tasks, dtype=int)
        self.task_mem = np.zeros(self.n_tasks, dtype=int)
        self.task_io = np.zeros(self.n_tasks, dtype=int)
        self.task_cooldowns = np.zeros(self.n_tasks, dtype=int)
        self.task_dependencies = np.zeros(self.n_tasks, dtype=int)
        self.task_priorities = np.zeros(self.n_tasks, dtype=bool)
        self._generate_tasks()
        self.task_running = np.zeros(self.n_tasks, dtype=bool)
        self.task_completed = np.zeros(self.n_tasks, dtype=bool)
        
        self.prev_action = 0
        self.schedulable_steps = 0
        self.idle_steps = 0
        self.priority_steps = 0
        self.miss_prioritization = 0
        
        return self._get_observation(), {}
    
    def _get_schedule_status(self, task_i):
        has_resource = self.task_cpu[task_i] <= self.cpu and self.task_mem[task_i] <= self.mem and self.task_io[task_i] <= self.io
        completed = self.task_completed[task_i]
        running = self.task_running[task_i]
        if running:
            return 0
        elif completed:
            return 1
        elif has_resource:
            return 2
        else:
            return 3

    def _get_observation(self):
        """Construct the observation vector."""
        obs = [
            self.cpu,
            self.mem,
            self.io,
            self.time_remaining
        ]

        if not self.is_mixed:
            obs.append(self.prev_action)
        else:
            obs.append(str(self.prev_action).encode('utf-8'))

        n_schedulable_tasks = 0
        
        for i in range(self.n_tasks):
            depends_on = self.task_dependencies[i]
            dependency_completed = (depends_on == -1) or self.task_completed[depends_on]
            schedule_status = self._get_schedule_status(i) 
            in_cooldown = self.task_completed[i] and self.task_cooldowns[i] > 0
            if schedule_status == 1:
                n_schedulable_tasks += 1
            if not self.is_mixed:
                one_hot_schedule = [0] * len(self.schedule_status)
                one_hot_schedule[schedule_status] = 1
                obs.extend([
                    self.task_durations[i] / self.max_time,
                    self.task_cpu[i] / self.max_cpu,
                    self.task_mem[i] / self.max_mem,
                    self.task_io[i] / self.max_io,
                    int(self.task_priorities[i]),
                    int(dependency_completed),
                    int(in_cooldown),
                    *one_hot_schedule,
                    depends_on,
                ])
            else:
                obs.extend([
                    self.task_durations[i] / self.max_time,
                    self.task_cpu[i] / self.max_cpu,
                    self.task_mem[i] / self.max_mem,
                    self.task_io[i] / self.max_io,
                    str(self.task_priorities[i]).encode('utf-8'),
                    str(dependency_completed).encode('utf-8'),
                    str(in_cooldown).encode('utf-8'),
                    self.schedule_status[schedule_status].encode('utf-8'),
                    str(depends_on).encode('utf-8'),
                ])
        obs.append(n_schedulable_tasks / self.n_tasks)
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
        
        schedulable_tasks = False
        has_priority = False
        
        # Check if any task is schedulable
        for i in range(self.n_tasks):
            has_resource = self.task_cpu[i] <= self.cpu and self.task_mem[i] <= self.mem and self.task_io[i] <= self.io
            if (has_resource and not self.task_running[i] and not self.task_completed[i]):
                schedulable_tasks = True
                if self.task_priorities[i]:
                    has_priority = True    

        if schedulable_tasks:
            self.schedulable_steps += 1

        # Handle Task Scheduling
        if action < self.n_tasks and not self.task_running[action] and not self.task_completed[action]:
            has_resource = self.task_cpu[action] <= self.cpu and self.task_mem[action] <= self.mem and self.task_io[action] <= self.io
            if has_resource:
                self.cpu -= self.task_cpu[action]
                self.mem -= self.task_mem[action]
                self.io -= self.task_io[action]
                self.task_running[action] = True
            
            if has_priority:
                self.priority_steps += 1
                if not self.task_priorities[action]:
                    self.miss_prioritization += 1

        
        # Task Progression
        finished_tasks = []
        for i in range(self.n_tasks):
            depends_on = self.task_dependencies[i]
            dependency_completed = (depends_on == -1) or self.task_completed[depends_on]
            if self.task_running[i]:
                self.task_durations[i] -= 1
                if self.task_durations[i] <= 0 and dependency_completed:
                    finished_tasks.append(i)
            if self.task_completed[i] and self.task_cooldowns[i] > 0:
                self.task_cooldowns[i] -= 1
                if self.task_cooldowns[i] <= 0:
                    self.cpu = min(self.cpu + self.task_cpu[i], self.max_cpu)
                    self.mem = min(self.mem + self.task_mem[i], self.max_mem)
                    self.io = min(self.io + self.task_io[i], self.max_io)
        
        for task in finished_tasks:
            self.task_running[task] = False
            self.task_completed[task] = True

        # Apply Idle Penalty
        if action == self.n_tasks and len(np.where(self.task_running)[0]) == 0 and schedulable_tasks:
            self.idle_steps += 1
        
        # Time Decrement
        self.time_remaining -= 1
        if self.time_remaining <= 0 or all(self.task_completed):
            terminated = True
        
        if terminated:
            reward = np.sum(self.task_completed) / self.n_tasks
            if self.schedulable_steps > 0:
                reward -= 0.9 * self.idle_steps / self.schedulable_steps
            if self.priority_steps > 0:
                reward -= 0.9 * self.miss_prioritization / self.priority_steps
        
        obs = self._get_observation()
        self.prev_action = action
        
        return obs, reward, terminated, False, info
    
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
        id="pipeline-v1",
        entry_point="env.pipeline_opt:HPCSchedulingEnv",
        kwargs={'n_tasks': 5},
    )
    register(
        id="pipeline-v2",
        entry_point="env.pipeline_opt:HPCSchedulingEnv",
        kwargs={'n_tasks': 5},
    )
