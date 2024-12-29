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
    
    def __init__(self, n_tasks: int=5, max_time: int=25, one_hot_task_types: bool = False):
        super(HPCSchedulingEnv, self).__init__()
        
        # Fixed global resources
        self.max_cpu = 10
        self.max_mem = 16
        self.max_io = 10
        
        self.n_tasks = n_tasks
        self.max_time = max_time
        self.is_mixed = not one_hot_task_types
        self.run_status = {0: 'running', 1: 'completed', 2: 'idle'}
        self.schedule_status = {0: 'finished', 1: 'schedulable', 2: 'no_resources', 3: 'dependency_incomplete', 4: 'cooldown_active'}

        # feature categories:
        # run_status: running, completed, idle
        # status: schedulable, dependency unsatisfied, resources unavailable, cooldown_active
        
        # Action space: Choose a task or No-Op
        self.action_space = spaces.Discrete(self.n_tasks + 1)  # Tasks + No-Op
        
        # Observation space
        # Global State: cpu_available, mem_available, io_available, time_remaining
        # Task-Level: duration, required_CPU, required_MEM, required_IO, is_running, is_completed, cooldown_timer, cooldown_active
        global_features = 6
        task_features = len(self.run_status) + len(self.schedule_status) + 5 if not self.is_mixed else 7
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
        rule = np.random.choice([2, 3, 4])  # Randomly select a rule
        # rule = np.random.choice([1, 2, 3])  # Randomly select a rule
        # rule = 3 # Randomly select a rule
        
        if rule == 1:  # Parallel Everything Immediately
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
                duration = np.random.randint(2, 5)
                resource_pool.append((cpu, mem, io, duration))
            np.random.shuffle(resource_pool)
            
            for i, (cpu, mem, io, duration) in enumerate(resource_pool):
                self.task_cpu[i] = cpu
                self.task_mem[i] = mem
                self.task_io[i] = io
                self.task_durations[i] = duration
                self.task_cooldowns[i] = (cpu + mem + io) // 3
                self.task_dependencies[i] = -1
            # self.time_remaining = 8
        
        elif rule == 2:  # Sequential via Dependencies
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
                duration = np.random.randint(2, 5)
                resource_pool.append((cpu, mem, io, duration))
            np.random.shuffle(resource_pool)
            for i, (cpu, mem, io, duration) in enumerate(resource_pool):
                self.task_cpu[i] = cpu
                self.task_mem[i] = mem
                self.task_io[i] = io
                self.task_durations[i] = duration
                self.task_dependencies[i] = task_inds[i]
                self.task_cooldowns[i] = np.random.randint(1, 3)
            
                # self.time_remaining = np.sum(self.task_cooldowns) + np.sum(self.task_durations)
        
        elif rule == 3:  # Parallel x-Track
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
                self.task_durations[i] = np.random.randint(2, 5)
                if order_:
                    self.task_dependencies[i] = parallel_inds[0] if i not in parallel_inds else -1
                else:
                    self.task_dependencies[i] = other_tasks[0] if i in parallel_inds else -1
                self.task_cooldowns[i] = (self.task_cpu[i] + self.task_mem[i] + self.task_io[i]) // 3
            # self.time_remaining = np.sum(self.task_cooldowns) + np.sum(self.task_durations) - np.random.randint(0, 5)
        
        elif rule == 4:  # Sequential via Resources (Strict Bottleneck)
            resource_pool = []
            for i in range(self.n_tasks):
                cpu = self.max_cpu - 1
                mem = self.max_mem - 1
                io = self.max_io - 1
                duration = np.random.randint(3, 6)
                resource_pool.append((cpu, mem, io, duration))
            np.random.shuffle(resource_pool)
            
            for i, (cpu, mem, io, duration) in enumerate(resource_pool):
                self.task_cpu[i] = cpu
                self.task_mem[i] = mem
                self.task_io[i] = io
                self.task_durations[i] = duration
                self.task_dependencies[i] = -1
                self.task_cooldowns[i] = (cpu + mem + io) // 2
        self.time_remaining = np.sum(self.task_cooldowns) + np.sum(self.task_durations) 
        
    def reset(self, seed=None, options=None):
        """Reset environment state."""
        super().reset(seed=seed)
        self.cpu = self.max_cpu
        self.mem = self.max_mem
        self.io = self.max_io
        # self.time_remaining = self.max_time
        
        self.task_durations = np.zeros(self.n_tasks, dtype=int)
        self.task_cpu = np.zeros(self.n_tasks, dtype=int)
        self.task_mem = np.zeros(self.n_tasks, dtype=int)
        self.task_io = np.zeros(self.n_tasks, dtype=int)
        self.task_cooldowns = np.zeros(self.n_tasks, dtype=int)
        self.task_dependencies = np.zeros(self.n_tasks, dtype=int)
        # self.task_durations = np.random.randint(1, 5, size=self.n_tasks)
        # self.task_cpu = np.random.randint(1, self.max_cpu  // 2, size=self.n_tasks)
        # self.task_mem = np.random.randint(1, self.max_mem  // 2, size=self.n_tasks)
        # self.task_io = np.random.randint(1, self.max_io // 2, size=self.n_tasks)
        # self.task_cooldowns = np.zeros(self.n_tasks, dtype=int)
        # self.task_dependencies = np.random.choice([-1] + list(range(self.n_tasks)), size=self.n_tasks)
        self._generate_tasks()
        self.task_running = np.zeros(self.n_tasks, dtype=bool)
        self.task_completed = np.zeros(self.n_tasks, dtype=bool)
        
        self.prev_action = 0
        
        return self._get_observation(), {}
    
    def _get_run_status(self, task_i):
        if self.task_running[task_i]:
             return 0
        if self.task_completed[task_i]:
            return 1 
        return 2
    
    def _get_schedule_status(self, task_i):
        has_resource = self.task_cpu[task_i] <= self.cpu and self.task_mem[task_i] <= self.mem and self.task_io[task_i] <= self.io
        has_cooldown = self.task_cooldowns[task_i] > 0
        completed = self.task_completed[task_i]
        running = self.task_running[task_i]
        depends_on = self.task_dependencies[task_i]
        dependency_completed = (depends_on == -1) or self.task_completed[depends_on]
        if running or completed:
            return 0
        if has_resource and dependency_completed and not has_cooldown:
            return 1
        elif not has_resource and dependency_completed and not has_cooldown:
            return 2
        elif has_resource and not dependency_completed:
            return 3
        else:
            return 4
    
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
            run_status = self._get_run_status(i)
            schedule_status = self._get_schedule_status(i) 
            if schedule_status == 1:
                n_schedulable_tasks += 1
            if not self.is_mixed:
                one_hot_run = [0] * len(self.run_status)
                one_hot_run[run_status] = 1
                one_hot_schedule = [0] * len(self.schedule_status)
                one_hot_schedule[schedule_status] = 1
                obs.extend([
                    self.task_durations[i],
                    self.task_cpu[i],
                    self.task_mem[i],
                    self.task_io[i],
                    *one_hot_run,
                    *one_hot_schedule,
                    depends_on,
                ])
            else:
                obs.extend([
                    self.task_durations[i],
                    self.task_cpu[i],
                    self.task_mem[i],
                    self.task_io[i],
                    self.run_status[run_status].encode('utf-8'),
                    self.schedule_status[schedule_status].encode('utf-8'),
                    str(depends_on).encode('utf-8'),
                ])
        obs.append(n_schedulable_tasks)
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
        
        schedulable_tasks = True
        
        # Check if any task is schedulable
        for i in range(self.n_tasks):
            has_resource = self.task_cpu[i] <= self.cpu and self.task_mem[i] <= self.mem and self.task_io[i] <= self.io
            depends_on = self.task_dependencies[i]
            dependency_completed = (depends_on == -1) or self.task_completed[depends_on]
            if (has_resource and not self.task_running[i] and dependency_completed and not self.task_completed[i]):
                schedulable_tasks = False
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
            # self.task_cooldowns[task] = int(np.mean([self.task_cpu[task], self.task_mem[task], self.task_io[task]]) / 3) + np.random.randint(0, 2)
        
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
        
        # # Apply Idle Penalty
        if action == self.n_tasks and len(np.where(self.task_running)[0]) == 0 and schedulable_tasks:
            reward -= 0.1  # Idle penalty only applies when no tasks are running and none are schedulable
        
        # Time Decrement
        self.time_remaining -= 1
        if self.time_remaining <= 0 or all(self.task_completed):
            terminated = True
        
        if terminated:
            reward = np.sum(self.task_completed) / self.n_tasks
            bonus = 1 if self.time_remaining > 0 and np.sum(self.task_completed) == self.n_tasks else 0 
            reward += bonus
        
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
        kwargs={'n_tasks': 5, 'max_time': 40},
    )
