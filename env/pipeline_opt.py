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


class PipelineSchedulingEnv(gym.Env):
    """
    Pipeline Scheduling Environment
    The agent must schedule tasks respecting dependencies and resource constraints
    to minimize makespan.
    Supports parallel task scheduling with clear distinction between scheduled, running, and completed tasks.
    """
    def __init__(self, n_tasks=8, max_resources=10, max_duration=10):
        super(PipelineSchedulingEnv, self).__init__()
        
        self.n_tasks = n_tasks
        self.max_resources = max_resources
        self.max_duration = max_duration
        
        # Action space: Select tasks to schedule next
        self.action_space = spaces.MultiBinary(self.n_tasks)  # Binary vector for task selection
        
        # Observation space
        # [Task Duration, Resource Required, Dependency Fulfilled, Task Scheduled, Task Running] * n_tasks + [Time Remaining, Resource Available]
        self.observation_space = spaces.Box(
            low=0,
            high=max(self.max_resources, self.max_duration),
            shape=(5 * self.n_tasks + 2,),
            dtype=np.float32
        )
        
        
    def _generate_tasks(self):
        """Generate task properties and dependency graph."""
        valid_resources = False
        while not valid_resources:
            self.task_durations = np.random.randint(1, self.max_duration + 1, size=self.n_tasks)
            self.task_resources = np.random.randint(1, self.max_resources // 2, size=self.n_tasks)
            
            # Ensure total resource feasibility
            if np.sum(self.task_resources) <= self.max_resources:
                valid_resources = True
        
        # Generate random dependencies (ensure DAG)
        self.task_dependencies = nx.DiGraph()
        self.task_dependencies.add_nodes_from(range(self.n_tasks))
        for i in range(self.n_tasks):
            for j in range(i+1, self.n_tasks):
                if np.random.rand() > 0.5:
                    self.task_dependencies.add_edge(i, j)
        
        if not nx.is_directed_acyclic_graph(self.task_dependencies):
            self._generate_tasks()
    
    def _get_observation(self):
        """Construct observation vector."""
        obs = []
        for i in range(self.n_tasks):
            dependency_fulfilled = all(dep in self.completed_tasks for dep in self.task_dependencies.predecessors(i))
            obs.extend([
                self.task_durations[i],
                self.task_resources[i],
                int(dependency_fulfilled),
                int(i in self.scheduled_tasks),
                int(i in self.running_tasks)
            ])
        obs.append(self.time_remaining)
        obs.append(self.resources_available)
        return np.array(obs, dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        self._generate_tasks()
        self.time_remaining = self.max_duration * self.n_tasks
        self.resources_available = self.max_resources
        self.scheduled_tasks = set()
        self.running_tasks = set()
        self.completed_tasks = set()
        self.rewarded_tasks = set()
        self.invalid_actions = 0
        return self._get_observation(), {}
    
    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        info = {}
        
        selected_tasks = np.where(action == 1)[0]
        current_resource_usage = sum(self.task_resources[task] for task in selected_tasks if task not in self.running_tasks)
        
        if current_resource_usage > self.resources_available:
            # Invalid due to resource constraint, early termination
            terminated = True
        else:
            for task in selected_tasks:
                if task in self.scheduled_tasks or task in self.running_tasks or task in self.completed_tasks:
                    # Ignore already processed tasks
                    continue
                
                dependency_fulfilled = all(dep in self.completed_tasks for dep in self.task_dependencies.predecessors(task))
                if dependency_fulfilled:
                    self.scheduled_tasks.add(task)
                    self.running_tasks.add(task)
                    self.resources_available -= self.task_resources[task]
                else:
                    terminated = True

        
        # Update running tasks
        tasks_to_complete = set()
        for task in self.running_tasks:
            self.task_durations[task] -= 1
            if self.task_durations[task] <= 0:
                tasks_to_complete.add(task)
        
        for task in tasks_to_complete:
            self.running_tasks.remove(task)
            self.completed_tasks.add(task)
            if task not in self.rewarded_tasks:
                reward += 1 / self.n_tasks  # Reward only for completed tasks
                self.rewarded_tasks.add(task)
        
        self.time_remaining -= 1
        
        if len(self.completed_tasks) == self.n_tasks:
            terminated = True
            # makespan_penalty = self.max_duration * self.n_tasks - self.time_remaining
        
        if self.time_remaining <= 0:
            truncated = True
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def render(self, mode='human'):
        print(f"Scheduled Tasks: {self.scheduled_tasks}")
        print(f"Running Tasks: {self.running_tasks}")
        print(f"Completed Tasks: {self.completed_tasks}")
        print(f"Time Remaining: {self.time_remaining}")
        print(f"Resources Available: {self.resources_available}")
        print(f"Invalid Actions: {self.invalid_actions}")
    


def register_pipeline_opt_tests():
    # PutNear
    # ----------------------------------------

    register(
        id="pipeline-v0",
        entry_point="env.pipeline_opt:PipelineSchedulingEnv",
        kwargs={'n_tasks': 5, 'max_resources': 10, 'max_duration': 10},
    )