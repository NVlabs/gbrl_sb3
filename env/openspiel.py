##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
import gymnasium as gym
import numpy as np

try:
    from shimmy import OpenSpielCompatibilityV0
except:
    OpenSpielCompatibilityV0 = None

MIXED_SIZES = {'chess': 134}

def process_openspiel_kwargs(algo_kwargs):
    if 'use_sde' in algo_kwargs:
        del algo_kwargs['use_sde']
    if 'sde_sample_freq' in algo_kwargs:
        del algo_kwargs['sde_sample_freq']
    


def chess_wrapper(obs: np.ndarray, player: str):
    obs_positions = obs[:13].astype(bool)
    categorical_board = np.empty(8*8, dtype=object)
    i = 0
    if player == 'player_0':
        player_is_black = True 
    else:
        player_is_black = False 
    for pawn_type in ['king', 'queen', 'rook', 'bishop', 'knight', 'pawn']:
        black_str = f'black {pawn_type}'
        white_str = f'white {pawn_type}'
        black_str += ' (me)' if player_is_black else ' (opponent)'
        white_str += ' (me)' if not player_is_black else ' (opponent)'
        categorical_board[obs_positions[i].flatten()] = white_str
        categorical_board[obs_positions[i+1].flatten()] = black_str
        i += 2

    categorical_board[obs_positions[i].flatten()] = 'empty'
    repetitions = obs[13].copy().flatten()
    color_to_play = 'white' if (obs[14, 0, 0] == 1) else 'black'
    color_to_play += ' (me)' if player_is_black and (obs[14, 0, 0] == 0) else ' (opponent)'
    n_irreversible_moves = obs[15, 0, 0]
    white_cast_queenside = 'True' if (obs[16, 0, 0] == 1) else 'False' 
    white_cast_kingside = 'True' if (obs[17, 0, 0] == 1) else 'False' 
    
    black_cast_queenside = 'True' if (obs[18, 0, 0] == 1) else 'False' 
    black_cast_kingside = 'True' if (obs[19, 0, 0] == 1) else 'False'

    black_cast_kingside +=  ' (me)' if player_is_black else ' (opponent)'
    black_cast_queenside +=  ' (me)' if player_is_black else ' (opponent)'
    black_cast_kingside +=  ' (me)' if not player_is_black else ' (opponent)'
    white_cast_queenside +=  ' (me)' if not player_is_black else ' (opponent)'
    
    chess_obs = np.concatenate([categorical_board, repetitions, np.array([color_to_play, n_irreversible_moves, 
                                                                          white_cast_queenside, white_cast_kingside,
                                                                          black_cast_queenside, black_cast_kingside])], axis=0, dtype=object)
    return chess_obs


OBS_WRAPPERS = {'chess': chess_wrapper}


class OpenSpielGymEnv(gym.Env):
    def __init__(self, env_id: str, is_mixed:bool = True, **env_kwargs):

        self.env = OpenSpielCompatibilityV0(game_name=env_id, **env_kwargs)  # type: ignore[arg-type]
        self.observation_space = self.env.observation_space(self.env.possible_agents[0])
        self.action_space = self.env.action_space(self.env.possible_agents[0])
        self.num_envs = 1
        self.env_id = env_id
        is_mixed = False
        self.is_mixed = is_mixed
        if is_mixed:
            self.observation_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(MIXED_SIZES.get(env_id, 0), ), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        self.env.reset(seed, options)
        obs = self.observe(self.env.agent_selection)
        if self.is_mixed:
            obs = OBS_WRAPPERS[self.env_id](obs, self.env.agent_selection)
        return obs, {'player': self.env.agent_selection}
    
    def seed(self, seed):
        pass
    
    def observe(self, agent):
        """Return only raw observation, removing action mask."""
        return self.env.observe(agent)

    def render(self):
        return self.env.render()
    
    def step(self, action):
        """Gymnasium-like step function, returning observation, reward, done, info."""
        self.env.step(action)
        new_obs, reward, terminated, truncated, info = self.env.last()
        if self.is_mixed:
            new_obs = OBS_WRAPPERS[self.env_id](new_obs, self.env.agent_selection)
        # done = terminated or truncated
        # info['TimeLimit.truncated'] = truncated
        # info['terminal_observation'] = new_obs if done else None
        return new_obs, reward, terminated, truncated, info
    
    def action_mask(self):
        """Separate function used in order to access the action mask."""
        return self.env.infos[self.env.agent_selection]["action_mask"]
    
    def close(self):
        return self.env.close()
    

