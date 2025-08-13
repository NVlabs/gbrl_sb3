##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
import gymnasium as gym
import matplotlib
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register

def act_to_str(action):
        """
        Convert action list [action_type, action_number, sign_idx] to a string representation.
        
        Args:
            action (list): Action in the form [action_type, action_number, sign_idx]
        
        Returns:
            str: String representation of the action
        """
        if action is None:
            action = [0, 0, 0]
        action_type, action_number, sign_idx = action

        value = action_number + 1
        action_sign = ''
        if action_type == 0:
            if sign_idx == 1:
                action_op = "+"
            else:
                action_op = " -"
                action_sign = ''
        elif action_type == 1:
            action_op = "/"
            action_sign = ' -' if sign_idx == 0 else ''
        elif action_type == 2:
            action_op = "X"
            action_sign = ' -' if sign_idx == 0 else ''
        else:
            if sign_idx == 1:
                action_op = "+ "
                action_sign = ""
            else:
                action_op = "-"
                action_sign = ''
            value = f'{action_number + 1}x' if action_number > 0 else 'x'


        return f"{action_op}{action_sign}{value}"
    
def linear_equation_compliance(state, agent_action, digits, target_coef_idx, target_coef_value, unwanted_indices):
    """
    Simple compliance function for linear equation environments.
    
    Args:
        state: Current state array
        agent_action: [action_type, action_number, sign_idx] that agent took
        digits: Number of digits available (usually 9)
        target_coef_idx: Index that should equal target_coef_value
        target_coef_value: Value the target coefficient should be (usually 1)
        unwanted_indices: List of indices that should be 0
    
    Returns:
        (compliance_status, correct_action)
    """
    
    # Find what needs to be fixed first (priority order)
    for idx, action_type in unwanted_indices:
        if abs(state[idx]) > 1e-3:
            # Need to eliminate this coefficient
            value = state[idx]
            action_number = min(max(int(abs(value)), 1), digits)
            sign = 0 if value > 0 else 1  # subtract positive, add negative
            correct_action = [action_type, action_number - 1, sign]

            # if list(agent_action) == correct_action:
            #     compliance = 0
            #     correct_action = None
            # else:
            compliance = 1
            return compliance, correct_action
    
    current_value = state[target_coef_idx]
    abs_current_value = abs(current_value)
    
    # All unwanted coefficients are 0, now check target coefficient
    if abs(current_value - target_coef_value) > 1e-6:
        
        # Check if current value is close to an integer
        is_integer = abs(abs_current_value - round(abs_current_value)) < 1e-6
        integer_value = round(abs_current_value) if is_integer else 0
        
        # Check if we can multiply by a single digit to make it an integer
        can_multiply_to_int = False
        multiplier = None
        for i in range(2, digits + 1):  # Start from 2, not 1!
            if abs((abs_current_value * i) - round(abs_current_value * i)) < 1e-6:
                can_multiply_to_int = True
                multiplier = i
                break

        if current_value < 0 and target_coef_value > 0:
            # Multiply by -1 to make it positive
            correct_action = [2, 0, 0]

        elif is_integer:  # Changed from > 1 to >= 2
            # Current value is an integer >= 2, divide it
            divisor = int(integer_value)
            action_number = min(max(divisor, 1), digits)
            sign = 1 if current_value > 0 else 0
            correct_action = [1, action_number - 1, sign]
            # print('can divide int:', act_to_str(correct_action))
            
        elif can_multiply_to_int and multiplier is not None:
            # Can multiply by a single digit (2-9) to make it integer
            action_number = min(max(multiplier, 1), digits)
            sign = 1 if current_value > 0 else 0
            correct_action = [2, action_number - 1, sign]
            # print('can multiply to int:', act_to_str(correct_action))
            
        else:
            # Can't fix this float with simple operations
            return 2, None
            
        # if list(agent_action) == correct_action:
        #     compliance = 0
        #     correct_action = None
        # else:
        compliance = 2
        return compliance, correct_action
    
    # Already at goal state
    return 0, None

class LinearEquationEnv(gym.Env):
    """
    Custom Gym Environment for isolating a linear equation: start with: ax + b = c
    Goal: Isolate x
    """
    metadata = {"render_modes": ["human", "rgb_array"],
                "render_fps": 10,}  # Add this line

    def __init__(self, with_history: bool = False, is_mixed: bool = False, compliance: bool = False, render_mode: str = None):
        super(LinearEquationEnv, self).__init__()

        # actions are add/ subtract/ divide/ multiply and multiply by -1

        self.n_action_types = 3
        self.digits = 9
        self.sign_type = 2
        self.with_history = with_history

        self.coef = 3
        shape = self.coef
        if self.with_history:
            if is_mixed:
                shape = self.coef + 3
            else:
                shape = self.coef + self.n_action_types + self.digits + self.sign_type
        self.action_space = spaces.MultiDiscrete([self.n_action_types, self.digits, self.sign_type])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(shape, ), dtype=float)
        self.step_count = 0
        self.max_steps = 50
        self.is_mixed = is_mixed
        self.compliance = compliance
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        info = {}
        nums = []
        for _ in range(3):
            digit = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
            sign = np.random.choice([-1, 1])
            nums.append(digit*sign)
        if self.with_history:
            if self.is_mixed:
                nums.append('None'.encode('utf-8'))
                nums.append(0)
                nums.append('None'.encode('utf-8'))
            else:
                one_hot = [0] * self.n_action_types
                nums.extend(one_hot)
                one_hot = [0] * self.digits
                nums.extend(one_hot)
                one_hot = [0] * self.sign_type
                nums.extend(one_hot)
        self.step_count = 0
        self.state = np.array(nums, dtype=object if self.is_mixed else np.single)

        if self.compliance:
            info['compliance'] = 0
            info['user_actions'] = self._action_to_onehot(None)

        return self.state, info

    def _gen_state(self, action):
        state = self.state
        action_type, action_number, sign_idx = action
        sign = 2*sign_idx - 1
        if action_type == 0:
            state[1] += sign*(action_number + 1)
            state[2] += sign*(action_number + 1)
        elif action_type == 1:
            state[:self.coef] = sign*state[:self.coef] / (action_number + 1)
        else:
            state[:self.coef] = sign*state[:self.coef] * (action_number + 1)

        if self.with_history:
            if self.is_mixed:
                state[self.coef] = str(action_type).encode('utf-8')
                state[self.coef + 1] = action_number + 1
                state[self.coef + 2] = str(bool(sign_idx)).encode('utf-8')
            else:
                state[self.coef:] = 0
                state[self.coef + action_type] = 1
                state[self.coef + self.n_action_types + action_number] = 1
                state[self.coef + self.n_action_types + self.digits + sign_idx] = 1

        self.state = state
        return state
    
    def _action_to_onehot(self, action):
        """Convert action list [action_type, action_number, sign_idx] to one-hot encoding"""
        # Create one-hot encoding
        onehot = [0] * (self.n_action_types + self.digits + self.sign_type)
        if action is None:
            return onehot
        
        action_type, action_number, sign_idx = action

        # One-hot for action type
        if 0 <= action_type < self.n_action_types:
            onehot[action_type] = 1
        
        # One-hot for action number (digits)
        if 0 <= action_number < self.digits:
            onehot[self.n_action_types + action_number] = 1
        
        # One-hot for sign
        if 0 <= sign_idx < self.sign_type:
            onehot[self.n_action_types + self.digits + sign_idx] = 1
        
        return onehot

    def step(self, action):
        """Take an action and return the next state, reward, and done flag."""

        reward = 0
        terminated = False
        truncated = False
        info = {}

        compliance, correct_action = linear_equation_compliance(
            self.state, action, self.digits, 
            target_coef_idx=0, target_coef_value=1, unwanted_indices=[(1, 0)]
        )

        state = self._gen_state(action)
        if state[0] == 1 and state[1] == 0:  # Isolating x condition
            reward = 1.0 - 0.9 * (self.step_count / self.max_steps)
            terminated = True

        if not self.is_mixed and (np.isnan(state).any() or np.isinf(state).any()):
            terminated = True

        if state[0] == 0:
            terminated = True

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True

        if self.compliance:
            info['compliance'] = compliance
            info['user_actions'] = self._action_to_onehot(correct_action)
        
        self.last_action = self.action_to_str(action)

        return state, reward, terminated, truncated, info

    def action_to_str(self, action):
        """
        Convert action list [action_type, action_number, sign_idx] to a string representation.
        
        Args:
            action (list): Action in the form [action_type, action_number, sign_idx]
        
        Returns:
            str: String representation of the action
        """
        action_type, action_number, sign_idx = action
        if action_type == 0:
            if sign_idx == 1:
                action_op = "+"
            else:
                action_op = "-"
            action_sign = ''
        elif action_type == 1:
            action_op = "/"
            action_sign = '-' if sign_idx == 0 else ''
        else:
            action_op = "X"
            action_sign = '-' if sign_idx == 0 else ''


        return f"{action_op} {action_sign}{action_number + 1}"
    
    def render(self):
        """
        Render the current state of the environment.
        
        Args:
            mode (str): 'human' for console output, 'rgb_array' for video recording
        
        Returns:
            None for 'human' mode, numpy array for 'rgb_array' mode
        """
        if self.render_mode == 'human':
            self._render_text()
            return None
        elif self.render_mode == 'rgb_array':
            return self._render_image()
        else:
            raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def _render_text(self):
        """Print the current equation state to console"""
        # Implementation depends on the specific environment
        equation = self._get_equation_string()
        print(f"Step {self.step_count}: {equation}", end="")
        if hasattr(self, 'last_action'):
            print(f" - Last action: {self.last_action}")
        else:
            print()

    def _render_image(self):
        """Create an RGB image of the current state for video recording"""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        from io import BytesIO
        import numpy as np
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Main equation
        equation_text = self._get_equation_string()
        ax.text(0.5, 0.7, equation_text, fontsize=24, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        # Step info
        ax.text(0.5, 0.1, f"Step: {self.step_count}/{self.max_steps}", 
                fontsize=14, ha='center', va='center')
        
        # Last action if available
        if hasattr(self, 'last_action'):
            action_text = f"Last action: {self.last_action}"
            ax.text(0.5, 0.5, action_text, fontsize=18, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow"))
        
         # Goal
        ax.text(0.5, 0.3, "Goal: x = ?", fontsize=24, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title("Linear Equation Environment", fontsize=20, pad=20)
        
        fig.canvas.draw()
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        
        import PIL.Image
        img = PIL.Image.open(buf)
        rgb_array = np.array(img)[:, :, :3]
        
        plt.close(fig)
        buf.close()
        return rgb_array

    def _get_equation_string(self):
        a, b, c = self.state[:3]

        def format_num(num, is_variable=False):
            if num == 0 or (is_variable and num == 1):
                return ""
            if abs(num - round(num)) < 1e-6:  # Close to integer
                return f"{int(round(num))}"
            else:
                return f"{num:.2f}"
        
        if b >= 0:
            return f"{format_num(a, True)}x + {format_num(b)} = {format_num(c)}"
        else:
            return f"{format_num(a, True)}x - {format_num(abs(b))} = {format_num(c)}"


class BalancedTwoVariableLinearEquationEnv(gym.Env):
    """
    Custom Gym Environment for isolating a linear equation: start with: ax + b = cy + d
    Goal: isolate y to the form = y = (d - b - ax) / c
    """

    def __init__(self, compliance: bool = False):
        super(BalancedTwoVariableLinearEquationEnv, self).__init__()

        # actions are add/ subtract/ divide/ multiply /add x / subtract x /add y /subtract y and multiply by -1
        self.indices = {'a': 0, 'b': 1, 'y': 2, 'c': 3, 'd': 4, 'x': 5}
        self.n_action_types = 5
        self.digits = 9
        self.sign_type = 2
        self.coef = 6
        shape = self.coef

        self.action_space = spaces.MultiDiscrete([self.n_action_types, self.digits, self.sign_type])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(shape, ), dtype=float)
        self.step_count = 0
        self.max_steps = 50
        self.compliance = compliance

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        nums = []
        info = {}
        for i in range(6):
            digit = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
            sign = np.random.choice([-1, 1])
            if i == 2 or i == 5:
                nums.append(0)
            else:
                nums.append(digit*sign)

        self.step_count = 0
        self.state = np.array(nums, dtype=np.single)
        self.x_bonus_given = False
        self.y_bonus_given = False
        self.moved_b = False

        if self.compliance:
            info['compliance'] = 0
            info['user_actions'] = self._action_to_onehot(None)  # No action taken yet
        return self.state, info

    def _get_bool(self, value):
        if isinstance(value, str):
            return True if value == 'True' else False
        return bool(value)

    def _gen_state(self, action):
        # ax + b + y_place_holder = cy + d + x_place_holder
        indices = self.indices
        state = self.state
        self.indices = {'a': 0, 'b': 1, 'y': 2, 'c': 3, 'd': 4, 'x': 5, 'x_pos': 6, 'y_pos': 7}

        action_type, action_number, sign_type = action

        sign = 2*sign_type - 1

        if action_type == 0:
            state[indices['b']] += sign*(action_number + 1)
            state[indices['d']] += sign*(action_number + 1)
        elif action_type == 1:
            state[:self.coef] = sign*state[:self.coef] / (action_number + 1)
        elif action_type == 2:
            state[:self.coef] = sign*state[:self.coef] * (action_number + 1)
        elif action_type == 3:
            state[indices['a']] += sign*(action_number + 1)
            state[indices['x']] += sign*(action_number + 1)
        elif action_type == 4:
            state[indices['y']] += sign*(action_number + 1)
            state[indices['c']] += sign*(action_number + 1)

        self.state = state
        return state

    def _action_to_onehot(self, action):
        """Convert action list [action_type, action_number, sign_idx] to one-hot encoding"""
        onehot = [0] * (self.n_action_types + self.digits + self.sign_type)
        if action is None:
            return onehot
        
        action_type, action_number, sign_idx = action
        
        # One-hot for action type
        if 0 <= action_type < self.n_action_types:
            onehot[action_type] = 1
        
        # One-hot for action number (digits)
        if 0 <= action_number < self.digits:
            onehot[self.n_action_types + action_number] = 1
        
        # One-hot for sign
        if 0 <= sign_idx < self.sign_type:
            onehot[self.n_action_types + self.digits + sign_idx] = 1
        
        return onehot
    
    def step(self, action):
        """Take an action and return the next state, reward, and done flag."""
        indices = self.indices
        reward = 0
        terminated = False
        truncated = False
        info = {}

        compliance, correct_action = linear_equation_compliance(
            self.state, action, self.digits,
            target_coef_idx=self.indices['b'], target_coef_value=1, 
            unwanted_indices=[self.indices['a'], self.indices['c']]
        )
        state = self._gen_state(action)
        # ax + b + y_place_holder = cy + d + x_place_holder
        if state[indices['a']] == 0 and state[indices['b']] == 0 and \
                state[indices['y']] == 1 and state[indices['c']] == 0:  # Isolating x condition
            reward = 1.0 - 0.9 * (self.step_count / self.max_steps) - 0.3  # remove bonuses
            terminated = True

        if state[indices['a']] == 0 and not self.x_bonus_given:
            reward += 0.1
            self.x_bonus_given = True

        if state[indices['c']] == 0 and not self.y_bonus_given:
            reward += 0.1
            self.y_bonus_given = True

        if state[indices['b']] == 0 and not self.moved_b:
            reward += 0.1
            self.moved_b = True

        if (np.isnan(state).any() or np.isinf(state).any()):
            # reward = -1
            terminated = True

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True

        if self.compliance:
            info['compliance'] = compliance
            info['user_actions'] = self._action_to_onehot(correct_action)
        return state, reward, terminated, truncated, info


class TwoVariableLinearEquationEnv(gym.Env):
    """
    Custom Gym Environment for isolating a linear equation: start with: ax + by + c = d
    Goal: Isolate x
    """
    metadata = {"render_modes": ["human", "rgb_array"],
                "render_fps": 10,}  # Add this line
    
    def __init__(self, compliance: bool = False, render_mode: str = None):
        super(TwoVariableLinearEquationEnv, self).__init__()

        # actions are add/ subtract/ divide/ multiply /add x / subtract x and multiply by -1

        self.n_action_types = 4
        self.digits = 9
        self.sign_type = 2
        self.coef = 5
        shape = self.coef
        self.indices = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'x': 4}

        self.action_space = spaces.MultiDiscrete([self.n_action_types, self.digits, self.sign_type])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(shape, ), dtype=float)
        self.step_count = 0
        self.max_steps = 50
        self.compliance = compliance
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        nums = []
        info = {}
        for _ in range(4):
            digit = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
            sign = np.random.choice([-1, 1])
            nums.append(digit*sign)

        nums.append(0)

        self.step_count = 0
        self.state = np.array(nums, dtype=np.single)
        self.moved_x = False
        self.moved_c = False

        if self.compliance:
            info['compliance'] = 0
            info['user_actions'] = self._action_to_onehot(None)  # No action taken yet
        return self.state, info
    
    def _gen_state(self, action):
        # ax + by + c = d + place_holder_x
        state = self.state.copy()
        action_type, action_number, sign_type = action
        sign = 2*sign_type - 1
        if action_type == 0:
            state[self.indices['c']] += sign*(action_number + 1)
            state[self.indices['d']] += sign*(action_number + 1)
        elif action_type == 1:
            state = sign*state / (action_number + 1)
        elif action_type == 2:
            state = sign*state * (action_number + 1)
        elif action_type == 3:
            state[self.indices['a']] += sign*(action_number + 1)
            state[self.indices['x']] += sign*(action_number + 1)
        self.state = state.copy()
        return state

    def _action_to_onehot(self, action):
        """Convert action list [action_type, action_number, sign_idx] to one-hot encoding"""
        onehot = [0] * (self.n_action_types + self.digits + self.sign_type)
        if action is None:
            return onehot
        
        action_type, action_number, sign_idx = action

        # One-hot for action type
        if 0 <= action_type < self.n_action_types:
            onehot[action_type] = 1
        
        # One-hot for action number (digits)
        if 0 <= action_number < self.digits:
            onehot[self.n_action_types + action_number] = 1
        
        # One-hot for sign
        if 0 <= sign_idx < self.sign_type:
            onehot[self.n_action_types + self.digits + sign_idx] = 1
        
        return onehot

    def step(self, action):
        """Take an action and return the next state, reward, and done flag."""

        reward = 0
        terminated = False
        truncated = False
        info = {}
        self.old_state = self.state.copy()
        compliance, correct_action = 0, None

        if self.compliance:
            compliance, correct_action = linear_equation_compliance(
                self.state, action, self.digits,
                target_coef_idx=self.indices['b'], target_coef_value=1,
                unwanted_indices=[(self.indices['c'], 0), (self.indices['a'], 3)]
            )
        state = self._gen_state(action)

        if state[self.indices['a']] == 0 and not self.moved_x:
            reward += 0.1
            self.moved_x = True
        if state[self.indices['c']] == 0 and not self.moved_c:
            reward += 0.1
            self.moved_c = True

        # Isolating x condition
        if state[self.indices['a']] == 0 and state[self.indices['b']] == 1 and state[self.indices['c']] == 0:
            reward = 1.0 - 0.9 * (self.step_count / self.max_steps) - 0.2  # remove bonus
            terminated = True

        if (np.isnan(state).any() or np.isinf(state).any()):
            terminated = True

        if state[self.indices['b']] == 0:
            terminated = True

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True

        if self.compliance:
            info['compliance'] = compliance
            info['user_actions'] = self._action_to_onehot(correct_action)

        self.last_action = self.action_to_str(action)
        self.last_correct_action = self.action_to_str(correct_action)
        self.last_compliance = compliance

        # equation, finished = self._get_equation_string()
        # print(f"Step {self.step_count}: {equation}", end="")
        # if hasattr(self, 'last_action'):
        #     print(f" - Last action: {self.last_action}, correct_action: {self.last_correct_action}, compliance: {self.last_compliance}")
        return state, reward, terminated, truncated, info
    
    def action_to_str(self, action):
        """
        Convert action list [action_type, action_number, sign_idx] to a string representation.
        
        Args:
            action (list): Action in the form [action_type, action_number, sign_idx]
        
        Returns:
            str: String representation of the action
        """
        if action is None:
            action = [0, 0, 0]
        action_type, action_number, sign_idx = action

        value = action_number + 1
        action_sign = ''
        if action_type == 0:
            if sign_idx == 1:
                action_op = "+"
            else:
                action_op = " -"
                action_sign = ''
        elif action_type == 1:
            action_op = "/"
            action_sign = ' -' if sign_idx == 0 else ''
        elif action_type == 2:
            action_op = "X"
            action_sign = ' -' if sign_idx == 0 else ''
        else:
            if sign_idx == 1:
                action_op = "+"
                action_sign = ""
            else:
                action_op = ""
                action_sign = '-'
            value = f'{action_number + 1}x' if action_number > 0 else 'x'

        return f"{action_op} {action_sign}{value}"
    
    def render(self):
        """
        Render the current state of the environment.
        
        Args:
            mode (str): 'human' for console output, 'rgb_array' for video recording
        
        Returns:
            None for 'human' mode, numpy array for 'rgb_array' mode
        """
        if self.render_mode == 'human':
            self._render_text()
            return None
        elif self.render_mode == 'rgb_array':
            return self._render_image()
        else:
            raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def _render_text(self):
        """Print the current equation state to console"""
        # Implementation depends on the specific environment
        equation, finished = self._get_equation_string()
        print(f"Step {self.step_count}: {equation}", end="")
        if hasattr(self, 'last_action'):
            print(f" - Last action: {self.last_action}")
        else:
            print()

    def _render_image(self):
        """Create an RGB image of the current state for video recording"""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        from io import BytesIO
        import numpy as np
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
         
        # Main equation with monospace font for consistent character width
        equation_text, finished = self._get_equation_string()
        ax.text(0.5, 0.7, equation_text, fontsize=18, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
                family='monospace')  # Monospace ensures consistent character width
        
        # Step info
        ax.text(0.5, 0.1, f"Step: {self.step_count:3d}/{self.max_steps}", 
                fontsize=14, ha='center', va='center', family='monospace')
        
        # Last action if available - with consistent width
        if hasattr(self, 'last_action'):
            action_text = f"Last action: {self.last_action:<20}"
            ax.text(0.5, 0.5, action_text, fontsize=14, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow"),
                    family='monospace')

        if hasattr(self, 'last_correct_action'):
            action_text = f"Correct: {str(self.last_correct_action):<20}"
            ax.text(0.75, 0.4, action_text, fontsize=12, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcyan"),
                    family='monospace')

        if hasattr(self, 'last_compliance'):
            compliance_text = f"Compliance: {self.last_compliance}"
            ax.text(0.25, 0.4, compliance_text, fontsize=12, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightpink"),
                    family='monospace')

        # State info with consistent formatting
        state_text = f"State: [{', '.join([f'{x:6.2f}' for x in self.state])}]"
        ax.text(0.5, 0.6, state_text, fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow"),
                family='monospace')
        
        # Goal
        if not finished:
            ax.text(0.5, 0.3, "Goal: y = ?", fontsize=20, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        else:
            ax.text(0.5, 0.3, "Goal Achieved!", fontsize=20, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="green"))
        
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title("Linear Equation Environment", fontsize=20, pad=20)
        
        fig.canvas.draw()
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        
        import PIL.Image
        img = PIL.Image.open(buf)
        rgb_array = np.array(img)[:, :, :3]
        
        plt.close(fig)
        buf.close()
        return rgb_array

    def _get_equation_string(self):
        # ax + by + c = d
        state = self.state.copy() if not hasattr(self, 'old_state') else self.old_state.copy()
        a, b, c , d, x = state
            
        def format_num(num, is_variable=False, width=3):
            if num == 0:
                if is_variable:
                    return f"{'0':>{width}}"
                else:
                    return f"{'0':>{width}}"

                
            if abs(num - round(num)) < 1e-6:  # Close to integer
                res = int(round(num))
                if abs(res) == 1 and is_variable:
                    return f"{'1' if res > 0 else '-1':>{width}}"
                return f"{abs(res):>{width}}"
            else:
                # For decimals, format to 2 decimal places with consistent width
                return f"{abs(num):>{width}.1f}"
            

        d_sign = '' if d > 0 else '-'
        d_sign = d_sign if d != 0 else ''
        c_sign = ' + ' if c > 0 else ' - '
        c_sign = c_sign if c != 0 else ''
        b_sign = ' + ' if b > 0 else ' - '
        b_sign = b_sign if b != 0 else ''
        a_sign = '' if a > 0 else '-'
        a_sign = a_sign if a != 0 else ''
        x_sign = ' + ' if x > 0 else ' - '
        x_sign = x_sign if x != 0 else ''

        space = ' ' if c != 0 else ' '
        x_value = 'x' if x != 0 else ''
        a_value = 'x' if a != 0 else ''

        if a_value == '':
            if b_sign == ' + ':
                b_sign = ''
            elif b_sign == ' - ':
                b_sign = '-'

        finished = True if a == 0 and b == 1 and c == 0 else False

        return f"{a_sign}{format_num(a, True)}{a_value}{b_sign}{format_num(b, True)}y{c_sign}{format_num(c)}{space}= {d_sign}{format_num(d)}{x_sign}{format_num(x, True)}{x_value}", finished
        

def register_equation_tests():
    # PutNear
    # ----------------------------------------

    register(
        id="LinearEquation-v0",
        entry_point="env.equation:LinearEquationEnv",
        kwargs={},
    )
    register(
        id="LinearEquation-v1",
        entry_point="env.equation:LinearEquationEnv",
        kwargs={"compliance": True},
    )
    register(
        id="StrLinearEquation-v0",
        entry_point="env.equation:StrLinearEquationEnv",
        kwargs={},
    )
    register(
        id="TwoVariableLinearEquation-v0",
        entry_point="env.equation:TwoVariableLinearEquationEnv",
        kwargs={},
    )
    register(
        id="TwoVariableLinearEquation-v1",
        entry_point="env.equation:TwoVariableLinearEquationEnv",
        kwargs={"compliance": True},
    )
    register(
        id="BalancedTwoVariableLinearEquation-v0",
        entry_point="env.equation:BalancedTwoVariableLinearEquationEnv",
        kwargs={},
    )
    register(
        id="BalancedTwoVariableLinearEquation-v1",
        entry_point="env.equation:BalancedTwoVariableLinearEquationEnv",
        kwargs={"compliance": True},
    )
