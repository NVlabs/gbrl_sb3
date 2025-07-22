import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register


class RicketyBridgeEnv(gym.Env):
    """
    Custom Gym Environment for the "Rickety Bridge" problem.

    - A 1D world with 5 states: {0, 1, 2, 3, 4}.
    - The agent starts at state 2.
    - The goal is to reach state 4.
    - State 1 is a "rickety board" and is forbidden.
    """

    def __init__(self):
        super(RicketyBridgeEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 for LEFT, 1 for RIGHT
        self.action_space = spaces.Discrete(2)
        # Observations: The 5 states on the bridge
        self.observation_space = spaces.Box(low=0, high=4, shape=(1,), dtype=np.int32)

        # Environment parameters
        self.start_state = 2
        self.goal_state = 4
        self.forbidden_state = 1
        self.max_steps = 10 # To prevent infinite loops

        # Initialize state
        self.agent_position = self.start_state
        self.step_count = 0

    def compliance(self):
        if self.agent_position == self.forbidden_state:
            return 1, [0, 1]
        return 0, [0, 0]

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        super().reset(seed=seed, options=options)
        self.agent_position = self.start_state
        self.step_count = 0
        # The state is returned as a numpy array to match the Box space
        return np.array([self.agent_position], dtype=np.int32), {}

    def step(self, action):
        """Take an action and return the next state, reward, and done flag."""
        # info dictionary can be used for debugging or expert advice
        info = {}
        
        # Ensure the action is valid
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # Update agent position based on the action
        if action == 0:  # LEFT
            self.agent_position -= 1
        elif action == 1:  # RIGHT
            self.agent_position += 1

        info['compliance'], info['user_actions'] = self.compliance()

        print(f"Action taken: {action}, Current position: {self.agent_position}, Compliance: {info['compliance']}")

        # Clip the position to stay within the bridge boundaries [0, 4]
        self.agent_position = np.clip(self.agent_position, 0, 4)
        
        self.step_count += 1

        # --- Determine reward and termination ---
        terminated = False
        reward = 0.0

        if self.agent_position == self.goal_state:
            # Agent reached the goal
            reward = 1.0
            terminated = True
        elif self.agent_position == self.forbidden_state:
            # Agent stepped on the rickety board
            reward = -1.0
            terminated = True
        
        # Check for truncation (max steps reached)
        truncated = False
        if self.step_count >= self.max_steps:
            truncated = True

        return np.array([self.agent_position], dtype=np.int32), reward, terminated, truncated, info

    def render(self, mode='human'):
        """Render the environment to the console."""
        # Create a visual representation of the bridge
        bridge = ['_'] * 5
        # Place the agent 'A' on the bridge
        bridge[self.agent_position] = 'A'
        # Mark the goal 'G' and forbidden state 'X'
        bridge[self.goal_state] = 'G'
        bridge[self.forbidden_state] = 'X'
        print(f"Step: {self.step_count} | Position: {self.agent_position} | {' '.join(bridge)}")


def register_rickety_bridge_tests():
    register(
        id="RicketyBridge-v0",
        entry_point="env.rickety_bridge:RicketyBridgeEnv",
    )