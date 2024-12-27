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

    def __init__(self, n_tasks=5, max_resources=10, max_duration=10, one_hot_task_types=False):
        super(PipelineSchedulingEnv, self).__init__()
        
        self.n_tasks = n_tasks
        self.max_resources = max_resources
        self.max_duration = max_duration
        self.task_types = ['CPU', 'IO', 'MEMORY']
        self.one_hot_task_types = one_hot_task_types
        self.is_mixed = not one_hot_task_types

        # Action space: For each of the n_tasks, pick (0 or 1) -> schedule or not
        self.action_space = spaces.MultiBinary(self.n_tasks)

        # Observation space (same structure as your original):
        # Per task: 1 (or len(self.task_types) if one-hot) + 6
        # Global: 2
        task_features = 1 if not self.one_hot_task_types else len(self.task_types)
        obs_dim = (task_features + 6) * self.n_tasks + 2
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
        self.task_durations = np.random.pareto(2, self.n_tasks).clip(1, self.max_duration).astype(int)
        
        # Resource usage: normal around max_resources/3, clamp
        self.task_resources = np.random.normal(
            self.max_resources / 3, 1, self.n_tasks
        ).clip(1, self.max_resources).astype(int)
        
        # Assign task types randomly
        self.task_types_list = random.choices(self.task_types, k=self.n_tasks)
        
        # Enforce some correlations
        for i, task_type in enumerate(self.task_types_list):
            if task_type == 'CPU':
                self.task_durations[i] = np.random.randint(1, max(1, self.max_duration // 2))
                self.task_resources[i] = np.random.randint(1, max(1, self.max_resources // 3))
            elif task_type == 'IO':
                self.task_durations[i] = np.random.randint(
                    max(1, self.max_duration // 2), self.max_duration
                )
                self.task_resources[i] = np.random.randint(
                    max(1, self.max_resources // 4), max(1, self.max_resources // 2)
                )
            elif task_type == 'MEMORY':
                self.task_resources[i] = np.random.randint(
                    max(1, self.max_resources // 2), self.max_resources
                )
        
        # DAG dependencies
        self.task_dependencies = self.generate_random_dag()
        while not nx.is_directed_acyclic_graph(self.task_dependencies):
            self.task_dependencies = self.generate_random_dag()

    def generate_random_dag(self, edge_probability=0.5):
        """Random DAG for dependencies."""
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
        """Flattened observation of task states + global state."""
        obs = []
        for i in range(self.n_tasks):
            dependency_fulfilled = all(
                dep in self.completed_tasks for dep in self.task_dependencies.predecessors(i)
            )
            
            if self.one_hot_task_types:
                one_hot = [0] * len(self.task_types)
                one_hot[self.task_types.index(self.task_types_list[i])] = 1
                obs.extend(one_hot)
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
        """Reset environment state."""
        super().reset(seed=seed)
        self._generate_tasks()
        
        self.time_remaining = sum(self.task_durations)
        self.resources_available = self.max_resources
        self.scheduled_tasks = set()
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
        chosen_tasks = np.where(action == 1)[0]
        reward = 0
        terminated = False
        truncated = False
        info = {}

        # 1) Filter out tasks that are already completed or already running
        #    We'll treat them as "invalid requests" if the agent tries again.
        valid_new_tasks = []
        for task_i in chosen_tasks:
            if task_i in self.completed_tasks:
                reward -= 0.1  # penalty: task is already completed
            elif task_i in self.running_tasks:
                reward -= 0.1  # penalty: task is already running
            else:
                # Check DAG dependencies
                deps_ok = all(d in self.completed_tasks 
                            for d in self.task_dependencies.predecessors(task_i))
                if not deps_ok:
                    reward -= 0.1  # penalty: unmet dependencies
                else:
                    # This is a genuinely "new" valid scheduling request
                    valid_new_tasks.append(task_i)
        
        # 2) Summation-based resource check
        #    Current usage from tasks that are already running:
        current_usage = sum(self.task_resources[t] for t in self.running_tasks)
        #    Additional usage from newly requested tasks:
        new_usage = sum(self.task_resources[t] for t in valid_new_tasks)
        
        total_usage_if_scheduled = current_usage + new_usage

        if new_usage > 0:  # i.e., at least one new task is requested
            if total_usage_if_scheduled > self.resources_available:
                # 3A) If the sum exceeds available resources, penalize
                reward -= 0.2  
                # Possibly skip scheduling them entirely
                # valid_new_tasks = []
                # Or you could do partial scheduling logic if you want
            else:
                # 3B) Otherwise, schedule them all
                for task_i in valid_new_tasks:
                    self.running_tasks.add(task_i)
                    # Deduct the resource usage for each
                    self.resources_available -= self.task_resources[task_i]
        else:
            # If agent didn't choose any valid tasks at all
            # maybe truncated or some penalty
            if len(chosen_tasks) == 0:
                truncated = True
                reward -= 0.1

        # 4) Now decrement durations of running tasks by min_duration
        #    (Same logic you already have)
        if self.running_tasks:
            min_duration = min(self.task_durations[t] for t in self.running_tasks)
        else:
            # If nothing is running, define min_duration as 1 or skip
            min_duration = 1
        
        tasks_finished = []
        for task in self.running_tasks:
            self.task_durations[task] -= min_duration
            if self.task_durations[task] <= 0:
                tasks_finished.append(task)

        for task in tasks_finished:
            self.running_tasks.remove(task)
            self.completed_tasks.add(task)
            # Free resources
            self.resources_available += self.task_resources[task]

        # Decrement global time
        self.time_remaining -= min_duration

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
        kwargs={'n_tasks': 10, 'max_resources': 10, 'max_duration': 10},
    )
    register(
        id="pipeline-large-v0",
        entry_point="env.pipeline_opt:PipelineSchedulingEnv",
        kwargs={'n_tasks': 20, 'max_resources': 10, 'max_duration': 10},
    )
