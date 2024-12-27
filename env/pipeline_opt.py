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
    - Task Types: CPU, IO, MEMORY
    - Non-uniform sampling for task properties
    - Sparse milestone rewards
    - Temporal and Resource-Based traps
    - Option for One-Hot Encoding of Task Types
    """
    def __init__(self, n_tasks=5, max_resources=10, max_duration=10, one_hot_task_types=False):
        super(PipelineSchedulingEnv, self).__init__()
        
        self.n_tasks = n_tasks
        self.max_resources = max_resources
        self.max_duration = max_duration
        self.task_types = ['CPU', 'IO', 'MEMORY']
        self.one_hot_task_types = one_hot_task_types
        self.is_mixed = not one_hot_task_types
        # Observation space per task:
        # 1. Task Type (One-hot or String Encoded)
        # 2. Task Duration (int)
        # 3. Task Resource Requirement (int)
        # 4. Dependency Fulfilled (bool)
        # 5. Scheduled Status (bool)
        # 6. Running Status (bool)
        # 7. Completed Status (bool)
        # Global features:
        # 8. Time Remaining (int)
        # 9. Resources Available (int)
        # Action space: Select tasks and priorities in a single multidiscrete space
        self.action_space = spaces.MultiDiscrete([self.n_tasks, 5])
        
        # Observation space
        task_features = 1 if not self.one_hot_task_types else len(self.task_types)
        self.observation_space = spaces.Box(
            low=0,
            high=max(self.max_resources, self.max_duration),
            shape=((task_features + 6) * self.n_tasks + 2,),
            dtype=np.float32
        )
        
        self.reset()

    def _generate_tasks(self):
        """Generate tasks with types, durations, and resource requirements."""
        # Duration: Longer durations for IO tasks, shorter for CPU
        self.task_durations = np.random.pareto(2, self.n_tasks).clip(1, self.max_duration).astype(int)
        
        # Resource: Higher for MEMORY tasks, variable for others
        self.task_resources = np.random.normal(self.max_resources / 3, 1, self.n_tasks).clip(1, self.max_resources).astype(int)
        
        self.task_types_list = random.choices(self.task_types, k=self.n_tasks)
        
        # Enforce some task-type-specific correlations
        for i, task_type in enumerate(self.task_types_list):
            if task_type == 'CPU':
                self.task_durations[i] = np.random.randint(1, self.max_duration // 2)
                self.task_resources[i] = np.random.randint(1, self.max_resources // 3)
            elif task_type == 'IO':
                self.task_durations[i] = np.random.randint(self.max_duration // 2, self.max_duration)
                self.task_resources[i] = np.random.randint(self.max_resources // 4, self.max_resources // 2)
            elif task_type == 'MEMORY':
                self.task_resources[i] = np.random.randint(self.max_resources // 2, self.max_resources)
        
        self.task_dependencies = self.generate_random_dag()
        
        # Ensure DAG validity
        while not nx.is_directed_acyclic_graph(self.task_dependencies):
            print('fixing dag')
            self.task_dependencies = self.generate_random_dag()

    def generate_random_dag(self, edge_probability=0.5):
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n_tasks))
        topological_order = list(range(self.n_tasks))
        random.shuffle(topological_order)

        for i, source in enumerate(topological_order):
            for j in range(i + 1, self.n_tasks):
                if np.random.rand() < edge_probability:
                    G.add_edge(source, topological_order[j])
        return G

    def _get_observation(self):
        obs = []
        for i in range(self.n_tasks):
            dependency_fulfilled = all(dep in self.completed_tasks for dep in self.task_dependencies.predecessors(i))
            
            if self.one_hot_task_types:
                task_type_one_hot = [0] * len(self.task_types)
                task_type_one_hot[self.task_types.index(self.task_types_list[i])] = 1
                obs.extend(task_type_one_hot)
            else:
                obs.append(self.task_types_list[i].encode('utf-8'))
            
            obs.extend([
                self.task_durations[i],
                self.task_resources[i],
                int(dependency_fulfilled),
                int(i in self.scheduled_tasks),
                int(i in self.running_tasks),
                int(i in self.completed_tasks)
            ])
        obs.append(self.time_remaining)
        obs.append(self.resources_available)
        return np.array(obs, dtype=object if self.is_mixed else np.float32)

    def reset(self, seed=None, options=None):
        self._generate_tasks()
        self.time_remaining = self.max_duration * 2
        self.resources_available = self.max_resources
        self.scheduled_tasks = set()
        self.running_tasks = set()
        self.completed_tasks = set()
        self.rewarded_tasks = set()
        return self._get_observation(), {}

    def step(self, action):
        task_index, priority = action
        reward = 0
        terminated = False
        truncated = False
        info = {}
        
        if task_index >= self.n_tasks or task_index in self.completed_tasks:
            reward = -1  # Invalid action
        else:
            dependency_fulfilled = all(dep in self.completed_tasks for dep in self.task_dependencies.predecessors(task_index))
            if dependency_fulfilled:
                if task_index not in self.scheduled_tasks:
                    self.scheduled_tasks.add(task_index)
                    self.running_tasks.add(task_index)
                    self.resources_available -= self.task_resources[task_index]
                    reward += 0.5  # Reward for valid scheduling
                    if self.task_types_list[task_index] == 'CPU' and priority > 2:
                        reward += 0.5  # Bonus for prioritizing CPU tasks correctly
            else:
                reward = -1
        
        tasks_to_complete = set()
        for task in self.running_tasks:
            self.task_durations[task] -= 1
            if self.task_durations[task] <= 0:
                tasks_to_complete.add(task)
        
        for task in tasks_to_complete:
            self.running_tasks.remove(task)
            self.completed_tasks.add(task)
            reward += 1
        
        self.time_remaining -= 1
        if self.time_remaining <= 0:
            terminated = True
            reward -= 1
        
        if self.resources_available < 0:
            reward -= 2
            terminated = True
        
        return self._get_observation(), reward, terminated, truncated, info

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
        kwargs={'n_tasks': 5, 'max_resources': 10, 'max_duration': 10},
    )
    register(
        id="pipeline-large-v0",
        entry_point="env.pipeline_opt:PipelineSchedulingEnv",
        kwargs={'n_tasks': 10, 'max_resources': 10, 'max_duration': 10},
    )
    register(
        id="pipeline-low-v0",
        entry_point="env.pipeline_opt:PipelineSchedulingEnv",
        kwargs={'n_tasks': 10, 'max_resources': 5, 'max_duration': 10},
    )
    register(
        id="pipeline-large-low-v0",
        entry_point="env.pipeline_opt:PipelineSchedulingEnv",
        kwargs={'n_tasks': 10, 'max_resources': 5, 'max_duration': 10},
    )