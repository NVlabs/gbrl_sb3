##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import gymnasium
import gym
from collections import deque
from gymnasium.spaces import Box, Discrete



from gfootball.env import config
from gfootball.env import football_env
from gfootball.env import observation_preprocessing, _process_representation_wrappers
from gfootball.env import wrappers

from stable_baselines3.common.env_checker import check_env

class FootballEnvWrapper(football_env.FootballEnv):
    def __init__(self, config):
        super().__init__(config)

    def reset(self, seed=None, **kwargs):
        obs = super().reset(**kwargs)
        return obs, {}  # Return observation and an empty dictionary

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, float(reward), done, False, info
    
class FootballEnvSpuriousWrapper(gym.ObservationWrapper):
    def __init__(self, env):
      gym.ObservationWrapper.__init__(self, env)
      action_shape = np.shape(self.env.action_space)
      shape = (action_shape[0] if len(action_shape) else 1, 119)
      self.observation_space = gymnasium.spaces.Box(
          low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

    def get_stadium_info(self):
        grass_idx = np.random.choice([0, 1, 2])
        grass_type = np.zeros(3, dtype=float)
        grass_type[grass_idx] = 1
        is_raining = np.array([np.random.choice([0.0, 1.0])])
        # number_of_people = np.array([np.random.randint(50000)], dtype=float)
        # self.stadium_info = np.concatenate([grass_type, is_raining, number_of_people], axis=0)
        self.stadium_info = np.concatenate([grass_type, is_raining], axis=0)

    def reset(self, **kwargs):
        """Resets the environment, returning a modified observation using :meth:`self.observation`."""
        obs, info = self.env.reset(**kwargs)
        self.get_stadium_info()
        return self.observation(obs), info

    def step(self, action):
        """Returns a modified observation using :meth:`self.observation` after calling :meth:`env.step`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
           self.get_stadium_info()
        return self.observation(observation), reward, terminated, truncated, info
    
    def observation(self, observation):
        """Returns a modified observation."""
        return np.concatenate([observation, self.stadium_info], axis=0)
    

class MultiAgentToSingleAgentWrapper(wrappers.MultiAgentToSingleAgent):
    def __init__(self, env, left_players, right_players):
        super().__init__(env, left_players, right_players)

    def reset(self, seed=None, **kwargs):
        obs = super().reset(**kwargs)
        return obs  # Return observation and an empty dictionary

class PeriodicDumpWriterWrapper(wrappers.PeriodicDumpWriter):
    def __init__(self, env, dump_frequency, render=False):
        super().__init__(env, dump_frequency, render)

    def reset(self, seed=None, **kwargs):
        obs = super().reset(**kwargs)
        return obs  # Return observation and an empty dictionary

class CheckpointRewardWrapper(wrappers.CheckpointRewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None, **kwargs):
        obs = super().reset(**kwargs)
        return obs  # Return observation and an empty dictionary

    def reward(self, reward):
        if isinstance(reward, float):
            reward = np.array([reward], dtype=np.float32)
        return super().reward(reward)
    
class SingleAgentRewardWrapper(wrappers.SingleAgentRewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return float(super().reward(reward))

def _process_reward_wrappers(env, rewards):
  assert 'scoring' in rewards.split(',')
  if 'checkpoints' in rewards.split(','):
    env = CheckpointRewardWrapper(env)
  return env
    
def _apply_output_wrappers(env, rewards, representation, channel_dimensions,
                           apply_single_agent_wrappers, stacked):
  """Wraps with necessary wrappers modifying the output of the environment.

  Args:
    env: A GFootball gym environment.
    rewards: What rewards to apply.
    representation: See create_environment.representation comment.
    channel_dimensions: (width, height) tuple that represents the dimensions of
       SMM or pixels representation.
    apply_single_agent_wrappers: Whether to reduce output to single agent case.
    stacked: Should observations be stacked.
  Returns:
    Google Research Football environment.
  """
  env = _process_reward_wrappers(env, rewards)
  env = _process_representation_wrappers(env, representation,
                                         channel_dimensions)
  if apply_single_agent_wrappers:
    if representation != 'raw':
      env = wrappers.SingleAgentObservationWrapper(env)
    env = SingleAgentRewardWrapper(env)
  if stacked:
    env = wrappers.FrameStack(env, 4)
  env = wrappers.GetStateWrapper(env)
  return env
      

def create_environment_sb3(env_name='',
                       stacked=False,
                       representation='extracted',
                       rewards='scoring',
                       write_goal_dumps=False,
                       write_full_episode_dumps=False,
                       render=False,
                       write_video=False,
                       dump_frequency=1,
                       logdir='',
                       extra_players=None,
                       number_of_left_players_agent_controls=1,
                       number_of_right_players_agent_controls=0,
                       channel_dimensions=(
                           observation_preprocessing.SMM_WIDTH,
                           observation_preprocessing.SMM_HEIGHT),
                       other_config_options={},
                       **kwargs):
  """Creates a Google Research Football environment.

  Args:
    env_name: a name of a scenario to run, e.g. "11_vs_11_stochastic".
      The list of scenarios can be found in directory "scenarios".
    stacked: If True, stack 4 observations, otherwise, only the last
      observation is returned by the environment.
      Stacking is only possible when representation is one of the following:
      "pixels", "pixels_gray" or "extracted".
      In that case, the stacking is done along the last (i.e. channel)
      dimension.
    representation: String to define the representation used to build
      the observation. It can be one of the following:
      'pixels': the observation is the rendered view of the football field
        downsampled to 'channel_dimensions'. The observation size is:
        'channel_dimensions'x3 (or 'channel_dimensions'x12 when "stacked" is
        True).
      'pixels_gray': the observation is the rendered view of the football field
        in gray scale and downsampled to 'channel_dimensions'. The observation
        size is 'channel_dimensions'x1 (or 'channel_dimensions'x4 when stacked
        is True).
      'extracted': also referred to as super minimap. The observation is
        composed of 4 planes of size 'channel_dimensions'.
        Its size is then 'channel_dimensions'x4 (or 'channel_dimensions'x16 when
        stacked is True).
        The first plane P holds the position of players on the left
        team, P[y,x] is 255 if there is a player at position (x,y), otherwise,
        its value is 0.
        The second plane holds in the same way the position of players
        on the right team.
        The third plane holds the position of the ball.
        The last plane holds the active player.
      'simple115'/'simple115v2': the observation is a vector of size 115.
        It holds:
         - the ball_position and the ball_direction as (x,y,z)
         - one hot encoding of who controls the ball.
           [1, 0, 0]: nobody, [0, 1, 0]: left team, [0, 0, 1]: right team.
         - one hot encoding of size 11 to indicate who is the active player
           in the left team.
         - 11 (x,y) positions for each player of the left team.
         - 11 (x,y) motion vectors for each player of the left team.
         - 11 (x,y) positions for each player of the right team.
         - 11 (x,y) motion vectors for each player of the right team.
         - one hot encoding of the game mode. Vector of size 7 with the
           following meaning:
           {NormalMode, KickOffMode, GoalKickMode, FreeKickMode,
            CornerMode, ThrowInMode, PenaltyMode}.
         Can only be used when the scenario is a flavor of normal game
         (i.e. 11 versus 11 players).
    rewards: Comma separated list of rewards to be added.
       Currently supported rewards are 'scoring' and 'checkpoints'.
    write_goal_dumps: whether to dump traces up to 200 frames before goals.
    write_full_episode_dumps: whether to dump traces for every episode.
    render: whether to render game frames.
       Must be enable when rendering videos or when using pixels
       representation.
    write_video: whether to dump videos when a trace is dumped.
    dump_frequency: how often to write dumps/videos (in terms of # of episodes)
      Sub-sample the episodes for which we dump videos to save some disk space.
    logdir: directory holding the logs.
    extra_players: A list of extra players to use in the environment.
        Each player is defined by a string like:
        "$player_name:left_players=?,right_players=?,$param1=?,$param2=?...."
    number_of_left_players_agent_controls: Number of left players an agent
        controls.
    number_of_right_players_agent_controls: Number of right players an agent
        controls.
    channel_dimensions: (width, height) tuple that represents the dimensions of
       SMM or pixels representation.
    other_config_options: dict that allows directly setting other options in
       the Config
  Returns:
    Google Research Football environment.
  """
  assert env_name

  scenario_config = config.Config({'level': env_name}).ScenarioConfig()
  players = [('agent:left_players=%d,right_players=%d' % (
      number_of_left_players_agent_controls,
      number_of_right_players_agent_controls))]

  # Enable MultiAgentToSingleAgent wrapper?
  multiagent_to_singleagent = False
  if scenario_config.control_all_players:
    if (number_of_left_players_agent_controls in [0, 1] and
        number_of_right_players_agent_controls in [0, 1]):
      multiagent_to_singleagent = True
      players = [('agent:left_players=%d,right_players=%d' %
                  (scenario_config.controllable_left_players
                   if number_of_left_players_agent_controls else 0,
                   scenario_config.controllable_right_players
                   if number_of_right_players_agent_controls else 0))]

  if extra_players is not None:
    players.extend(extra_players)
  config_values = {
      'dump_full_episodes': write_full_episode_dumps,
      'dump_scores': write_goal_dumps,
      'players': players,
      'level': env_name,
      'tracesdir': logdir,
      'write_video': write_video,
  }
  config_values.update(other_config_options)
  c = config.Config(config_values)

  env = FootballEnvWrapper(c)
  if multiagent_to_singleagent:
    env = wrappers.MultiAgentToSingleAgentWrapper(
        env, number_of_left_players_agent_controls,
        number_of_right_players_agent_controls)
  if dump_frequency > 1:
    env = wrappers.PeriodicDumpWriterWrapper(env, dump_frequency, render)
  elif render:
    env.render()
  env = _apply_output_wrappers(
      env, rewards, representation, channel_dimensions,
      (number_of_left_players_agent_controls +
       number_of_right_players_agent_controls == 1), stacked)
  if kwargs.get('spurious', False):
      env = FootballEnvSpuriousWrapper(env)
  return env

class FootballGymSB3(gymnasium.Env):
    spec = None
    metadata = None
    
    def __init__(self, env_name="11_vs_11_easy_stochastic", rewards="scoring,checkpoints", 
                 representation="simple115v2", **kwargs,
                ):
        super(FootballGymSB3, self).__init__()
        self.env = create_environment_sb3(
            env_name=env_name,
            stacked=False,
            representation=representation,
            rewards = rewards,
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            render=False,
            write_video=False,
            dump_frequency=1,
            logdir=".",
            extra_players=None,
            number_of_left_players_agent_controls=1,
            number_of_right_players_agent_controls=0,
            **kwargs)  
        self.action_space = Discrete(19)
        # self.observation_space = Box(low=0, high=255, shape=(72, 96, 16), dtype=np.uint8)
        shape_size = 119 if kwargs.get('spurious', False) else 115
        self.observation_space = Box(low=float('-inf'), high=float('inf') , shape=(shape_size,), dtype=np.float32)
        self.reward_range = (-1, 1)
        
    def transform_obs(self, raw_obs):
        obs = raw_obs[0]
        obs = observation_preprocessing.generate_smm([obs])
        if not self.obs_stack:
            self.obs_stack.extend([obs] * 4)
        else:
            self.obs_stack.append(obs)
        obs = np.concatenate(list(self.obs_stack), axis=-1)
        obs = np.squeeze(obs)
        return obs

    def reset(self, seed=None):
        return self.env.reset(seed=seed)
    
    def step(self, action):
        return self.env.step([action])
    
check_env(env=FootballGymSB3(), warn=True)