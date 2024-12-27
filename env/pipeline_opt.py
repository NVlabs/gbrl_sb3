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


# class PipelineSchedulingEnv(gym.Env):
#     """
#     Pipeline Scheduling Environment
#     The agent must schedule tasks respecting dependencies and resource constraints
#     to minimize makespan.
#     Supports parallel task scheduling with clear distinction between scheduled, running, and completed tasks.
#     """
#     def __init__(self, n_tasks=5, max_resources=10, max_duration=10):
#         super(PipelineSchedulingEnv, self).__init__()
        
#         self.n_tasks = n_tasks
#         self.max_resources = max_resources
#         self.max_duration = max_duration
        
#         # Action space: Select tasks to schedule next
#         self.action_space = spaces.MultiBinary(self.n_tasks)  # Binary vector for task selection
        
#         # Observation space
#         # [Task Duration, Resource Required, Dependency Fulfilled, Task Scheduled, Task Running] * n_tasks + [Time Remaining, Resource Available]
#         self.observation_space = spaces.Box(
#             low=0,
#             high=max(self.max_resources, self.max_duration),
#             shape=(5 * self.n_tasks + 2,),
#             dtype=np.float32
#         )
        
        
#     def _generate_tasks(self):
#         """Generate task properties and dependency graph."""
#         # valid_resources = False
#         # while not valid_resources:
#         self.task_durations = np.random.randint(1, self.max_duration + 1, size=self.n_tasks)
#         self.task_resources = np.random.randint(1, self.max_resources // 2, size=self.n_tasks)
            
#             # # Ensure total resource feasibility
#             # if np.sum(self.task_resources) <= self.max_resources:
#             #     valid_resources = True
#             # print(valid_resources)
        
#         # # Generate random dependencies (ensure DAG)
#         # self.task_dependencies = nx.DiGraph()
#         # self.task_dependencies.add_nodes_from(range(self.n_tasks))
#         # for i in range(self.n_tasks):
#         #     for j in range(i+1, self.n_tasks):
#         #         if np.random.rand() > 0.5:
#         # print('gothere')
#         #             self.task_dependencies.add_edge(i, j)
#         self.task_dependencies = self.generate_random_dag()
        
#         if not nx.is_directed_acyclic_graph(self.task_dependencies):
#             # self._generate_tasks()
#             self.task_dependencies = self.generate_random_dag()
#         # self.print_dependencies()

#     def generate_random_dag(self, edge_probability=0.5):
#             """
#             Generates a random DAG using topological sort.

#             Args:
#                 n_tasks: Number of tasks.
#                 edge_probability: Probability of creating an edge between two nodes.

#             Returns:
#                 A networkx DiGraph object representing the DAG.
#             """
#             G = nx.DiGraph()
#             G.add_nodes_from(range(self.n_tasks))

#             # Generate a random topological order
#             topological_order = list(range(self.n_tasks))
#             random.shuffle(topological_order)

#             for i, source in enumerate(topological_order):
#                 for j in range(i + 1, self.n_tasks):
#                     if np.random.rand() < edge_probability:
#                         G.add_edge(source, topological_order[j])

#             return G
    
#     def _get_observation(self):
#         """Construct observation vector."""
#         obs = []
#         for i in range(self.n_tasks):
#             dependency_fulfilled = all(dep in self.completed_tasks for dep in self.task_dependencies.predecessors(i))
#             obs.extend([
#                 self.task_durations[i],
#                 self.task_resources[i],
#                 int(dependency_fulfilled),
#                 int(i in self.scheduled_tasks),
#                 int(i in self.running_tasks)
#             ])
#         obs.append(self.time_remaining)
#         obs.append(self.resources_available)
#         return np.array(obs, dtype=np.float32)

#     def print_dependencies(self):
#         """
#         Prints the dependencies in a readable format.

#         Args:
#             dependencies: The networkx DiGraph object representing task dependencies.
#         """
#         for node in self.task_dependencies.nodes():
#             predecessors = list(self.task_dependencies.predecessors(node))
#             if predecessors:
#                 print(f"Task {node} depends on: {predecessors}")
    
#     def reset(self, seed=None, options=None):
#         self._generate_tasks()
#         # self.time_remaining = self.max_duration * self.n_tasks * 0.3
#         self.time_remaining = self.max_duration * 2
#         self.resources_available = self.max_resources
#         self.scheduled_tasks = set()
#         self.running_tasks = set()
#         self.completed_tasks = set()
#         self.rewarded_tasks = set()
#         self.invalid_actions = 0
#         return self._get_observation(), {}
    
#     def step(self, action):
#         reward = 0
#         terminated = False
#         truncated = False
#         info = {}
        
#         selected_tasks = np.where(action == 1)[0]
#         current_resource_usage = sum(self.task_resources[task] for task in selected_tasks if task not in self.running_tasks)
        
#         if current_resource_usage > self.resources_available:
#             # Invalid due to resource constraint, early termination
#             reward = -1
#             terminated = True
#         else:
#             for task in selected_tasks:
#                 if task in self.scheduled_tasks or task in self.running_tasks or task in self.completed_tasks:
#                     # Ignore already processed tasks
#                     continue
                
#                 dependency_fulfilled = all(dep in self.completed_tasks for dep in self.task_dependencies.predecessors(task))
#                 if dependency_fulfilled:
#                     self.scheduled_tasks.add(task)
#                     self.running_tasks.add(task)
#                     self.resources_available -= self.task_resources[task]
#                 else:
#                     reward = -1
#                     terminated = True

        
#         # Update running tasks
#         tasks_to_complete = set()
#         for task in self.running_tasks:
#             self.task_durations[task] -= 1
#             if self.task_durations[task] <= 0:
#                 tasks_to_complete.add(task)
        
#         for task in tasks_to_complete:
#             self.running_tasks.remove(task)
#             self.completed_tasks.add(task)
        
#         self.time_remaining -= 1
    
#         if self.time_remaining <= 0:
#             reward = len(self.completed_tasks) / self.n_tasks
#             terminated = True
        
#         return self._get_observation(), reward, terminated, truncated, info
    
#     def render(self, mode='human'):
#         print(f"Scheduled Tasks: {self.scheduled_tasks}")
#         print(f"Running Tasks: {self.running_tasks}")
#         print(f"Completed Tasks: {self.completed_tasks}")
#         print(f"Time Remaining: {self.time_remaining}")
#         print(f"Resources Available: {self.resources_available}")
#         print(f"Invalid Actions: {self.invalid_actions}")
    

class PipelineSchedulingEnv(gym.Env):
    """
    Enhanced Pipeline Scheduling Environment
    - MultiBinary action space (choose which tasks to schedule each step)
    - Task Types: CPU, IO, MEMORY
    - Non-uniform sampling for task properties
    - Sparse final reward (normalized by n_tasks, +1 bonus if all tasks are done)
    - Temporal and resource-based traps
    - DAG dependencies must be fulfilled before scheduling a task
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
        #     if task_type == 'CPU':
        #         self.task_cpu_resources[i] = np.random.randint(1, self.max_resources)
        #     elif task_type == 'IO':
        #         self.task_io_resources[i] = np.random.randint(1, self.max_resources)
        #     elif task_type == 'MEMORY':
        #         self.task_mem_resources[i] = np.random.randint(1, self.max_resources)

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
